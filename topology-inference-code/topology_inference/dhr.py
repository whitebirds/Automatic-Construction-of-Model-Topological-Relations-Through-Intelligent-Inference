"""Direct historical adjacency reuse with optional role constraints (DH-R).

This is an adapted baseline, not the paper's learned historical matcher. It
selects one reference by equipment-type/voltage counts and transfers known
reference edges through deterministic name/code correspondences. The matching
functions are retained from the experiment implementation; filesystem access,
private split files, source hashes, and result files are not included.

Only input attributes are read from a query document. Reference labels must be
provided explicitly by the caller, using training references only. Data in this
module's tests are fabricated and are not a transformed real substation.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from functools import lru_cache

from .core.naming import parse_name
from .core.topology_structure_rules import equipment_role

METHODS = ('DH-code', 'DH-role')

def normalized_name(name):
    return re.sub(r'\s+', '', name).upper().replace('ＫＶ', 'KV')

def extract_code(name):
    s = re.sub(r'\d+(?:\.\d+)?(?:KV|千伏)', '', normalized_name(name))
    found = list(re.finditer(r'(?<![A-Z0-9])[0-9][A-Z0-9]{2,}[ⅠⅡⅢⅣⅤⅥⅦⅧ]?', s))
    if found:
        m = min(found, key=lambda z: (-len(z[0]), z.start()))
        return m[0], 'alphanumeric' if any(c.isalpha() or c in 'ⅠⅡⅢⅣⅤⅥⅦⅧ' for c in m[0]) else 'numeric'
    ordinal = re.search(r'#?(\d+)号?(?=主变|母联)', s)
    if ordinal:
        return '#' + ordinal[1], 'ordinal'
    return None, 'missing'

def make_node(raw):
    f = raw['features']
    # Only declared input attributes are read; extra metadata is ignored.
    v = f.get('voltage_kv')
    wind = tuple(sorted(float(x) for x in f.get('winding_voltage_kv', []) if x is not None and float(x) > 0))
    node = {'id': str(raw['node_id']), 'type': str(f['type_code']), 'name': f['name'],
            'voltage': None if v is None else float(v), 'windings': wind}
    p = parse_name(node['name'], node['type'])
    role = equipment_role({'name': node['name'], 'type_code': node['type']})
    s = normalized_name(node['name'])
    kind = ('transformer' if '主变' in s else 'coupler' if '母联' in s else
            'section' if '分段' in s or '母分' in s else
            'line' if any(x in s for x in ('线路',)) or node['type'] == '415' else None)
    attrs = {'role': (role,) if role else (), 'family': tuple(p['bus_family_candidates']),
             'section': tuple(p['bus_section_candidates']), 'side': tuple(p['device_side_candidates']),
             'kind': (kind,) if kind else ()}
    node['attrs'] = attrs
    node['tokens'] = frozenset(f'{key}:{value}' for key, vals in attrs.items() for value in vals)
    node['code'], node['code_status'] = extract_code(node['name'])
    node['voltages'] = frozenset(wind if node['type'] == '416' and wind else
                                [node['voltage']] if node['voltage'] is not None and node['voltage'] > 0 else [])
    node['signature'] = (node['type'], normalized_name(node['name']),
                         str(node['voltage']), repr(wind))
    return node

def inventory(document):
    return tuple(make_node(n) for n in document['nodes'])

def known_edges(document):
    return {tuple(sorted((str(e['source']), str(e['target'])))) for e in document['labels']['edges']}

def family_ambiguous(nodes):
    out = []
    for n in nodes:
        fam = n['attrs']['family']
        if n['type'] != '408' or len(fam) != 1 or n['attrs']['section']:
            continue
        bs = [b['id'] for b in nodes if b['type'] == '410' and b['voltage'] == n['voltage'] and b['attrs']['family'] == fam]
        if len(bs) >= 2:
            out.append((n['id'], bs))
    return out

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.1
    return len(a & b) / len(a | b)

@lru_cache(maxsize=200000)
def code_similarity(a, b):
    score = 1.0
    for k in range(max(len(a), len(b))):
        x = a[k] if k < len(a) else ''
        y = b[k] if k < len(b) else ''
        if x and y and x.isascii() and y.isascii() and x.isdigit() and y.isdigit():
            score *= 1 - min(9, abs(int(x) - int(y))) / 10
        else:
            score *= 1.0 if x == y and x else 0.1
    return score

def sim1(a, b):
    return code_similarity(a['code'], b['code']) if a['code'] and b['code'] else jaccard(a['tokens'], b['tokens'])

@lru_cache(maxsize=200000)
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = row
    return prev[-1]

def sim2(qn, rn, qr, rr):
    score = sim1(qn, rn)
    if all(n['code'] for n in (qn, rn, qr, rr)):
        a = edit_distance(qn['code'], qr['code'])
        b = edit_distance(rn['code'], rr['code'])
        if max(a, b):
            score *= 1 - abs(a - b) / max(a, b)
    return score

def compatible(a, b, method):
    if a['type'] != b['type']:
        return False
    if a['voltages'] and b['voltages'] and a['voltages'].isdisjoint(b['voltages']):
        return False
    if method == 'DH-role':
        for key in ('role', 'family', 'section', 'side', 'kind'):
            aa, bb = a['attrs'][key], b['attrs'][key]
            if aa and bb and aa != bb:
                return False
    return True

def choose(query, candidates, method, root=None):
    choices = []
    for n in candidates:
        if not compatible(query, n, method):
            continue
        score = sim1(query, n) if root is None else sim2(n, query, root[0], root[1])
        if score <= 0:
            continue
        choices.append((score, jaccard(query['tokens'], n['tokens']), n))
    if not choices:
        return None, {'reason': 'no_positive_compatible_match', 'score': None, 'top_ties': 0}
    best_score = max(x[0] for x in choices)
    choices = [x for x in choices if math.isclose(x[0], best_score, rel_tol=0, abs_tol=1e-12)]
    descriptor_best = max(x[1] for x in choices)
    choices = [x for x in choices if x[1] == descriptor_best]
    ties = len(choices)
    sig = min(x[2]['signature'] for x in choices)
    best = [x[2] for x in choices if x[2]['signature'] == sig]
    if len(best) > 1:
        return None, {'reason': 'indistinguishable_visible_signatures', 'score': best_score, 'top_ties': ties}
    return best[0], {'reason': 'unique_max' if ties == 1 else 'visible_name_tiebreak', 'score': best_score, 'top_ties': ties}

def count_signature(nodes):
    # Transformer contributes by declared winding voltages, otherwise its voltage.
    c = Counter()
    for n in nodes:
        volts = sorted(n['voltages']) or [0.0]
        for v in volts:
            c[(n['type'], v)] += 1
    return c

def transfer(query, reference_nodes, historical_edges, method):
    """Transfer a reference graph onto attribute-only query nodes.

    ``method`` is ``DH-role`` (role constraints) or ``DH-code`` (ablation).
    Query nodes are produced by :func:`inventory`; query labels are not accepted.
    Returned traces contain only caller-supplied IDs and matching diagnostics.
    """
    if method not in METHODS:
        raise ValueError(f'Unsupported method: {method}')
    rb = {n['id']: n for n in reference_nodes}
    neighbors = defaultdict(set)
    for a, b in historical_edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    predicted = set()
    traces = []
    for q in sorted(query, key=lambda n: n['signature']):
        r, info = choose(q, reference_nodes, method)
        trace = {'query_id': q['id'], 'reference_id': r['id'] if r else None,
                 'forward': info, 'neighbor_transfers': []}
        if r is not None:
            for rid in sorted(neighbors[r['id']], key=lambda i: rb[i]['signature']):
                rn = rb[rid]
                target, rev = choose(rn, [n for n in query if n['id'] != q['id']], method, (q, r))
                trace['neighbor_transfers'].append({'reference_neighbor_id': rid,
                    'query_neighbor_id': target['id'] if target else None, **rev})
                if target is not None:
                    predicted.add(tuple(sorted((q['id'], target['id']))))
        traces.append(trace)
    return predicted, traces

def edge_metrics(truth, pred):
    tp, fp, fn = len(truth & pred), len(pred - truth), len(truth - pred)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1), 'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'error_edges': fp + fn, 'true_edges': len(truth), 'predicted_edges': len(pred)}

def evaluate(nodes, truth, pred):
    byid = {n['id']: n for n in nodes}
    def bus_edge(e):
        return {byid[x]['type'] for x in e} == {'408', '410'}
    metrics = {'full_graph': edge_metrics(truth, pred),
               'bus_ds': edge_metrics({e for e in truth if bus_edge(e)}, {e for e in pred if bus_edge(e)})}
    buses = {n['id'] for n in nodes if n['type'] == '410'}
    out = []
    for device, candidates in family_ambiguous(nodes):
        def neighborhood(edges):
            return {b if a == device else a for a, b in edges if device in (a, b)} & buses
        y, p = neighborhood(truth), neighborhood(pred)
        status = 'exact' if y == p else 'multiple' if len(p) > 1 else 'missing' if not p else 'wrong'
        out.append({'device_id': device, 'candidate_bus_ids': sorted(candidates),
                    'recorded_bus_ids': sorted(y), 'predicted_bus_ids': sorted(p), 'status': status})
    metrics['hard'] = dict(Counter(x['status'] for x in out))
    metrics['hard']['total'] = len(out)
    return metrics, out

def select_reference(query, bank, *, query_group_id):
    """Choose an eligible reference using visible equipment counts.

    ``bank`` maps an opaque public reference key to ``nodes``, ``counts``,
    ``inventory_signature``, ``edges``, and an explicit ``group_id``. All exports
    or aliases of one physical station must share a group ID. References from
    the query group are excluded. The caller must restrict the bank to training
    stations; group exclusion alone does not establish a train/test split.
    """
    eligible = [key for key in bank if bank[key]['group_id'] != query_group_id]
    if not eligible:
        raise ValueError('No eligible training reference remains after group exclusion.')
    keys = sorted({k for key in eligible for k in bank[key]['counts']})
    scales = {k: max(1.0, sum(bank[key]['counts'][k] for key in eligible) / len(eligible))
              for k in keys}
    qc = count_signature(query)
    for k in qc:
        if k not in scales:
            keys.append(k)
            scales[k] = 1.0

    def distance(key):
        rc = bank[key]['counts']
        basic = sum(abs(qc[k] - rc[k]) / scales[k] for k in keys if k[0] in ('410', '416'))
        alltypes = sum(abs(qc[k] - rc[k]) / scales[k] for k in keys)
        return basic, alltypes, bank[key]['inventory_signature'], key

    selected = sorted(eligible, key=distance)[0]
    d = distance(selected)
    ties = sum(distance(key)[:2] == d[:2] for key in eligible)
    return selected, {'primary_count_distance': d[0], 'secondary_count_distance': d[1],
                      'count_distance_ties': ties, 'eligible_reference_count': len(eligible)}


def _validate_nodes(nodes):
    ids = [node['id'] for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError('Node IDs must be unique within each station.')
    if not ids:
        raise ValueError('An equipment inventory must contain at least one node.')


def predict(query_document, reference_documents, *, query_group_id, roles_filter=True):
    """Predict query edges from explicitly supplied training reference documents.

    Query schema::

        {"nodes": [{"node_id": "q1", "features": {
            "type_code": "407", "name": "test 265 breaker", "voltage_kv": 220,
            "winding_voltage_kv": []}}]}

    A reference adds nonempty ``station_id`` and ``group_id`` strings and
    ``labels: {edges: [{source: ..., target: ...}]}``. IDs are opaque identifiers,
    not features. ``query_group_id`` is mandatory to support alias exclusion;
    use the same group ID for all versions of one station. The query's optional
    labels and all extra source fields are ignored. Reference labels are known
    historical connections and are used only for adjacency transfer.

    ``roles_filter=False`` gives the code-based ablation (DH-code). Otherwise,
    compatible equipment role, bus family/section, side and circuit kind are
    additionally required when both sides expose these attributes. The result
    contains predicted edges and matching diagnostics; it contains no truth
    metrics. Evaluate it separately with held-out labels if available.

    Callers are responsible for supplying training references only. No private
    data, trained weights, results, or network requests are loaded by this API.
    """
    if not isinstance(query_group_id, str) or not query_group_id:
        raise ValueError('query_group_id must be a nonempty opaque string.')
    query = inventory(query_document)
    _validate_nodes(query)
    bank = {}
    for document in reference_documents:
        key = document.get('station_id')
        group_id = document.get('group_id')
        if not isinstance(key, str) or not key or not isinstance(group_id, str) or not group_id:
            raise ValueError('Every reference needs nonempty station_id and group_id strings.')
        if key in bank:
            raise ValueError('Reference station_id values must be unique.')
        nodes = inventory(document)
        _validate_nodes(nodes)
        edges = known_edges(document)
        ids = {node['id'] for node in nodes}
        if any(a == b or a not in ids or b not in ids for a, b in edges):
            raise ValueError('Reference edges must join distinct declared equipment IDs.')
        bank[key] = {'group_id': group_id, 'nodes': nodes,
                     'counts': count_signature(nodes),
                     'inventory_signature': tuple(sorted(n['signature'] for n in nodes)),
                     'edges': edges}
    reference_key, reference_selection = select_reference(query, bank, query_group_id=query_group_id)
    method = 'DH-role' if roles_filter else 'DH-code'
    predicted, traces = transfer(query, bank[reference_key]['nodes'], bank[reference_key]['edges'], method)
    return {'method': 'DH-R' if roles_filter else 'DH-code',
            'reference_station': reference_key, 'reference_selection': reference_selection,
            'predicted_edges': [list(edge) for edge in sorted(predicted)],
            'correspondences': traces}
