"""Optional XGBoost baselines using the released original feature/decoder code.

Training outputs are data-derived and must not be added to the public package.
This single-split adapter does not reproduce the private paper folds.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .pipeline import load_directory, read_document, write_json
from .core import run, external_baselines as ext, xgboost_bus_ranker as ranker


def train(train_dir, validation_dir, output, *, seed=7, depths=(3, 6, 9), rounds=(40, 80, 160, 240), candidate_ranker=False):
    import xgboost as xgb
    training, validation = load_directory(train_dir), load_directory(validation_dir)
    if {d['group_id'] for d in training.values()} & {d['group_id'] for d in validation.values()} or set(training) & set(validation):
        raise ValueError('Training and validation must be disjoint by physical station.')
    if not depths or not rounds or min(depths) < 1 or min(rounds) < 1:
        raise ValueError('Depths and boosting rounds must be positive.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    types = sorted({n['features']['type_code'] for d in training.values() for n in d['nodes']})
    run.USE_BUS = True
    cache = {}
    for key, doc in {**training, **validation}.items():
        b = ranker.inputs(doc, types) if candidate_ranker else ext.input_batch(doc, types)
        if candidate_ranker:
            ranker.attach(b, doc)
        else:
            ext.attach_labels(b, doc)
        cache[key] = b
    tr, va = list(training), list(validation)
    xx, yy = ext.sample_training(cache, tr, seed)
    dt = xgb.DMatrix(xx, label=yy, nthread=2)
    dv = {f: xgb.DMatrix(cache[f]['x'], nthread=2) for f in va}
    selected, best = None, -1.
    for depth in depths:
        params = {'objective': 'binary:logistic', 'tree_method': 'hist', 'max_depth': depth,
                  'eta': .05, 'min_child_weight': 1, 'lambda': 1, 'subsample': .9,
                  'colsample_bytree': .9, 'nthread': 2, 'seed': seed}
        model = xgb.train(params, dt, num_boost_round=max(rounds))
        for count in rounds:
            scores = {f: model.predict(dv[f], output_margin=True, iteration_range=(0, count)) for f in va}
            settings = ext.select(cache, va, scores, 'basic')
            if settings['validation_f1'] > best:
                best = settings['validation_f1']
                selected = {'depth': depth, 'rounds': count, **settings}
                model.save_model(output / 'connection.ubj')
    rank_settings = None
    if candidate_ranker:
        rt = ranker.matrix(cache, tr, True)
        rv = ranker.matrix(cache, va, False)
        best_key = None
        for depth in depths:
            params = {'objective': 'rank:pairwise', 'tree_method': 'hist', 'eta': .05,
                      'max_depth': depth, 'subsample': .9, 'colsample_bytree': .9, 'nthread': 2, 'seed': seed}
            model = xgb.train(params, rt, num_boost_round=max(rounds))
            for count in rounds:
                _, key = ranker.choices(cache, va, model.predict(rv, iteration_range=(0, count)))
                if best_key is None or key < best_key:
                    best_key = key
                    rank_settings = {'depth': depth, 'rounds': count, 'validation_error_counts': list(key)}
                    model.save_model(output / 'ranker.ubj')
    metadata = {'method': 'XGB-C' if candidate_ranker else 'XGB-R', 'types': types,
                'seed': seed, 'connection': selected, 'ranker': rank_settings,
                'train_group_ids': [d['group_id'] for d in training.values()],
                'validation_group_ids': [d['group_id'] for d in validation.values()],
                'synthetic_only': all(d['synthetic'] for d in [*training.values(), *validation.values()]),
                'notice': 'Private-data training outputs are not cleared for publication.'}
    write_json(output / 'metadata.json', metadata)
    return metadata


def infer(bundle, document):
    import json
    import xgboost as xgb
    from .pipeline import validate_document
    bundle = Path(bundle)
    meta = json.loads((bundle / 'metadata.json').read_text(encoding='utf8'))
    doc = validate_document(document, labels=False)
    run.USE_BUS = True
    b = ranker.inputs(doc, meta['types']) if meta['ranker'] else ext.input_batch(doc, meta['types'])
    model = xgb.Booster({'nthread': 2})
    model.load_model(bundle / 'connection.ubj')
    cfg = meta['connection']
    z = model.predict(xgb.DMatrix(b['x'], nthread=2), output_margin=True, iteration_range=(0, cfg['rounds']))
    p = ext.decoded_scores(b, z, cfg['alpha'], 'basic') >= cfg['threshold']
    if meta['ranker']:
        model.load_model(bundle / 'ranker.ubj')
        zz = model.predict(ranker.matrix({'query': b}, ['query'], False), iteration_range=(0, meta['ranker']['rounds']))
        p[b['bus_mask']] = False
        offset = 0
        for group in b['groups']:
            local = zz[offset:offset + len(group['x'])]
            offset += len(local)
            winners = np.flatnonzero(np.isclose(local, local.max(), rtol=0, atol=1e-9))
            if len(winners) == 1 and winners[0] < len(group['indices']):
                p[group['indices'][winners[0]]] = True
    return {'station_id': doc['station_id'], 'synthetic': doc['synthetic'], 'method': meta['method'],
            'predicted_edges': [{'source': doc['nodes'][int(b['i'][k])]['node_id'],
                                 'target': doc['nodes'][int(b['j'][k])]['node_id']} for k in np.flatnonzero(p)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--train-dir', required=True)
    p.add_argument('--validation-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--candidate-ranker', action='store_true', help='XGB-C instead of XGB-R')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--depths', nargs='+', type=int, default=[3, 6, 9])
    p.add_argument('--rounds', nargs='+', type=int, default=[40, 80, 160, 240])
    p = sub.add_parser('infer')
    p.add_argument('--bundle', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'train':
        train(args.train_dir, args.validation_dir, args.output, seed=args.seed, depths=args.depths,
              rounds=args.rounds, candidate_ranker=args.candidate_ranker)
    else:
        target = Path(args.output)
        if target.exists():
            raise FileExistsError('Choose a new output path.')
        value = infer(args.bundle, read_document(args.input))
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, value)


if __name__ == '__main__':
    main()
