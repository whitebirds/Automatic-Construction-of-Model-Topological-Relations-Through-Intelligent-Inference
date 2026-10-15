"""Portable training/inference harness around the released research modules.

The synthetic demo is an execution check, not a reproduction of paper scores.
Training with confidential inputs produces confidential bundles: reference
inventories and their labels are needed by historical matching. Do not publish
these generated bundles. NPZ loading always disables pickle.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .core import run
from .core import whole_station_study as whole
from .core import hierarchical_network as hierarchy
from .core import character_hierarchical_network as character
from .core import topology_structure_rules as role
from .core import expanded_structure_study as template
from .core import dedicated_bus_study as bus
from .core.dedicated_bus_network import BusNet
from .core import station_energy_study as energy
from .core import learned_reference_study as memory
from .synthetic import write_dataset


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_document(path, labels=False):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_document(doc, labels=labels)


def validate_document(doc, labels=False):
    """Allowlist input attributes, retaining targets only when requested."""
    sid = doc.get("station_id")
    if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", sid):
        raise ValueError("station_id must be a stable alphanumeric inventory identifier.")
    group_id = doc.get("group_id")
    if not isinstance(group_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", group_id):
        raise ValueError("group_id must identify the physical station, shared by all aliases/exports.")
    nodes = doc.get("nodes", [])
    if len(nodes) < 2:
        raise ValueError("A station requires at least two equipment objects.")
    output = {"station_id": sid, "group_id": group_id, "synthetic": doc.get("synthetic") is True, "nodes": []}
    ids = set()
    for node in nodes:
        nid = node.get("node_id")
        if not isinstance(nid, str) or not nid or nid in ids:
            raise ValueError("node_id values must be nonempty and unique within a station.")
        ids.add(nid)
        f = node.get("features", {})
        if not isinstance(f.get("name"), str) or not isinstance(f.get("type_code"), str):
            raise ValueError("Each equipment object needs string name and type_code attributes.")
        voltage = f.get("voltage_kv")
        windings = f.get("winding_voltage_kv", [])
        if not isinstance(windings, list):
            raise ValueError("winding_voltage_kv must be a list.")
        for value in [voltage, *windings]:
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0):
                raise ValueError("Voltages must be finite nonnegative numbers or null.")
        output["nodes"].append({"node_id": nid, "features": {
            "name": f["name"], "type_code": f["type_code"],
            "voltage_kv": voltage, "winding_voltage_kv": copy.deepcopy(windings),
        }})
    if labels:
        raw_edges = doc.get("labels", {}).get("edges")
        if not isinstance(raw_edges, list):
            raise ValueError("Training/evaluation documents require labels.edges.")
        edges, seen = [], set()
        for e in raw_edges:
            a, b = e.get("source"), e.get("target")
            if a not in ids or b not in ids or a == b:
                raise ValueError("Every target edge must join distinct listed equipment objects.")
            key = tuple(sorted((a, b)))
            if key in seen:
                raise ValueError("Duplicate undirected target edge.")
            seen.add(key)
            edges.append({"source": a, "target": b})
        output["labels"] = {"edges": edges}
    return output


def load_directory(path, labels=True):
    files = sorted(Path(path).glob("*.json"))
    if not files:
        raise ValueError(f"No station JSON documents in {path}.")
    docs, groups = {}, set()
    for path in files:
        d = read_document(path, labels)
        key = "station." + d["station_id"] + ".json"
        if key in docs:
            raise ValueError("Duplicate station_id; aliases must share one split.")
        if d["group_id"] in groups:
            raise ValueError("Choose one inventory per physical group_id before training/evaluation.")
        groups.add(d["group_id"])
        docs[key] = d
    return docs


def normalize_stats(arrays):
    values = np.concatenate(arrays)
    mean, std = values.mean(0), values.std(0)
    std[std < .05] = 1
    return mean, std


def save_params(path, params, **arrays):
    np.savez_compressed(path, **{f"p{k}": p for k, p in enumerate(params)}, **arrays)


def load_params(path, model):
    with np.load(path, allow_pickle=False) as data:
        params = [data[f"p{k}"].copy() for k in range(len(model.p))]
        if any(a.shape != b.shape for a, b in zip(params, model.p)):
            raise ValueError("Checkpoint architecture mismatch.")
        model.p = params


def train(train_dir, validation_dir, output, epochs=None, seed=19):
    """Train each original model stage without reading private saved artifacts.

    This portable single-split runner reuses the actual architecture, losses,
    template extraction and historical matching. It is not a byte-for-byte
    replay of private five-fold selection histories. All selection uses only
    the explicitly supplied validation stations.
    """
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be positive.")
    budgets = {"mlp": 24, "hierarchy": 36, "ecpn": 24, "bcpn": 40, "hrm": 25}
    if epochs is not None:
        budgets = {key: epochs for key in budgets}
    train_docs, val_docs = load_directory(train_dir), load_directory(validation_dir)
    if set(train_docs) & set(val_docs):
        raise ValueError("Training and validation station IDs must be disjoint.")
    if {d["group_id"] for d in train_docs.values()} & {d["group_id"] for d in val_docs.values()}:
        raise ValueError("Physical station group_id values must not overlap between splits.")
    if len(train_docs) < 3:
        raise ValueError("At least three training stations are required for templates.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train_files, val_files = list(train_docs), list(val_docs)
    documents = {**train_docs, **val_docs}
    types = sorted({n["features"]["type_code"] for d in train_docs.values() for n in d["nodes"]})
    run.USE_BUS = True
    cache = {}
    for f, doc in documents.items():
        b = whole.make_input(doc, types)
        whole.attach_labels(b, doc)
        fs = [n["features"]["type_code"] for n in b["d"]["nodes"]]
        # Original ECPN auxiliary count target: bus counts include DS only.
        for ni, kind in enumerate(fs):
            if kind == "410":
                degree = sum(bool(y) and ((i == ni and fs[j] == "408") or (j == ni and fs[i] == "408"))
                             for i, j, y in zip(b["i"], b["j"], b["y"]))
                b["target_degree"][ni] = np.log1p(degree) / 3
        b["layout"] = role.input_layout(b)
        b["extended"] = template.groups(b)
        cache[f] = b
    if not any(cache[f]["y"].sum() for f in train_files):
        raise ValueError("Training data require positive target relations.")
    rng = np.random.default_rng(seed)
    sampled = {}
    for f in train_files:
        b = cache[f]
        pos, neg = np.flatnonzero(b["y"]), np.flatnonzero(1 - b["y"])
        sampled[f] = np.r_[pos, rng.choice(neg, min(len(neg), 5 * len(pos)), replace=False)]
    pm, ps = normalize_stats([cache[f]["x"][sampled[f]] for f in train_files])
    xx = np.concatenate([(cache[f]["x"][sampled[f]] - pm) / ps for f in train_files])
    yy = np.concatenate([cache[f]["y"][sampled[f]] for f in train_files])
    vy = np.concatenate([cache[f]["y"] for f in val_files])
    mlp = run.MLP(len(pm), seed)
    best, chosen, base_history = None, None, []
    print("Training initial relation MLP", flush=True)
    for epoch in range(budgets["mlp"] + 1):
        if epoch == 0:
            continue
        if epoch:
            order = rng.permutation(len(yy))
            for start in range(0, len(order), 512):
                ix = order[start:start + 512]
                mlp.step(xx[ix], yy[ix])
        scores = np.concatenate([mlp.predict((cache[f]["x"] - pm) / ps) for f in val_files])
        threshold, f1 = run.best_threshold(vy, scores)
        base_history.append({"epoch": epoch, "validation_f1": f1})
        if best is None or f1 > best:
            best, chosen = f1, ([p.copy() for p in mlp.p], threshold, epoch)
    mlp.p, base_threshold, base_epoch = chosen
    save_params(output / "initial_mlp.npz", mlp.p, mean=pm, std=ps)
    tm, ts = normalize_stats([cache[f]["tokens"] for f in train_files])
    vocab = character.vocabulary([cache[f]["d"] for f in train_files])
    prepared = {}
    for f, b in cache.items():
        x = (b["x"] - pm) / ps
        graph = hierarchy.structure(b, mlp.predict(x), base_threshold)
        graph["text_ids"] = character.encode(b["d"], vocab)
        prepared[f] = {"x": x, "tokens": (b["tokens"] - tm) / ts, "graph": graph,
                       "zero": np.zeros(len(b["y"]), np.float32)}
    for f in train_files:
        b, p = cache[f], prepared[f]
        idx = np.unique(np.r_[sampled[f], *[g["pair_indices"] for g in b["templates"]]])
        lookup = {int(k): j for j, k in enumerate(idx)}
        p["idx"] = idx
        p["groups"] = [(np.array([lookup[int(k)] for k in g["pair_indices"]]), b["y"][g["pair_indices"]])
                       for g in b["templates"] if b["y"][g["pair_indices"]].sum() == 1]

    def train_graph(model, name, budget, learning_rate, include_initial=True):
        best_score, best_weights, best_epoch, history = None, None, 0, []
        for epoch in range(budget + 1):
            if epoch == 0 and not include_initial:
                continue
            if epoch:
                for index in rng.permutation(len(train_files)):
                    f = train_files[index]
                    b, v = cache[f], prepared[f]
                    ix = v["idx"]
                    model.step((v["tokens"], v["x"][ix], b["i"][ix], b["j"][ix], v["zero"][ix],
                                v["graph"], b["y"][ix], v["groups"], b["target_degree"]), lr=learning_rate)
            scores = np.concatenate([model.predict(prepared[f]["tokens"], prepared[f]["x"],
                                    cache[f]["i"], cache[f]["j"], prepared[f]["zero"], prepared[f]["graph"])
                                     for f in val_files])
            _, f1 = run.best_threshold(vy, scores)
            history.append({"epoch": epoch, "validation_f1": f1})
            if best_score is None or f1 > best_score:
                best_score, best_epoch, best_weights = f1, epoch, [p.copy() for p in model.p]
        model.p = best_weights
        print(f"Selected {name} checkpoint at epoch {best_epoch}", flush=True)
        return best_epoch, history

    warm = hierarchy.HierNet(len(pm), len(tm), seed)
    warm_epoch, warm_history = train_graph(warm, "hierarchical warm stage", budgets["hierarchy"], .001, include_initial=False)
    graph_model = character.CharacterHierNet(len(pm), len(tm), len(vocab), seed, "name_code", warm.p)
    graph_epoch, graph_history = train_graph(graph_model, "ECPN", budgets["ecpn"], .0001)
    save_params(output / "ecpn.npz", graph_model.p, pair_mean=pm, pair_std=ps, token_mean=tm, token_std=ts)
    graph_scores = {f: graph_model.predict(v["tokens"], v["x"], cache[f]["i"], cache[f]["j"], v["zero"], v["graph"])
                    for f, v in prepared.items()}
    rules = whole.choose_rules(cache, graph_scores, val_files)
    line_templates, _ = role.learn_templates(cache, train_files)
    line_trials = []
    for strength in (-1, 0, .5, 1, 2, 4, 8):
        all_true, all_pred = [], []
        for f in val_files:
            b = cache[f]
            z = graph_scores[f] + rules["alpha"] * b["name_hint"]
            p = role.enforce(b, run.structure_mask(b, z) & (z >= rules["threshold"]), z)
            if strength >= 0:
                p, _ = role.apply_templates(b, p, z, rules["threshold"], strength,
                    choices=[role.candidate_templates(g, line_templates) for g in b["layout"]["groups"]])
            all_true.append(b["y"])
            all_pred.append(p)
        line_trials.append({"strength": strength, "f1": run.metrics(np.concatenate(all_true), np.concatenate(all_pred))["f1"]})
    line_strength = max(line_trials, key=lambda t: t["f1"])["strength"]
    templates = template.library(cache, train_files)
    trials = []
    for strength in (-1, 0, .5, 2):
        all_true, all_pred = [], []
        for f in val_files:
            b = cache[f]
            z = graph_scores[f] + rules["alpha"] * b["name_hint"]
            p = role.enforce(b, run.structure_mask(b, z) & (z >= rules["threshold"]), z)
            if line_strength >= 0:
                p, _ = role.apply_templates(b, p, z, rules["threshold"], line_strength,
                    choices=[role.candidate_templates(g, line_templates) for g in b["layout"]["groups"]])
            p = template.decode(b, p, z, rules["threshold"], strength,
                                [template.choices(g, templates) for g in b["extended"]])
            all_true.append(b["y"])
            all_pred.append(p)
        trials.append({"strength": strength, "f1": run.metrics(np.concatenate(all_true), np.concatenate(all_pred))["f1"]})
    template_strength = max(trials, key=lambda t: t["f1"])["strength"]
    write_json(output / "line_templates.json", line_templates)
    write_json(output / "templates.json", templates)
    write_json(output / "vocabulary.json", vocab)

    dedicated = {}
    for f, doc in documents.items():
        d = bus.inputs(doc, types, vocab)
        bus.targets(d, doc)
        dedicated[f] = d
    if not any(len(dedicated[f]["groups"]) for f in train_files):
        raise ValueError("BCPN training requires disconnector objects (type 408).")
    nm, ns = normalize_stats([dedicated[f]["raw_node"] for f in train_files])
    bm, bs = normalize_stats([dedicated[f]["raw_pair"] for f in train_files])
    for d in dedicated.values():
        d["node_x"], d["pair_x"] = (d["raw_node"] - nm) / ns, (d["raw_pair"] - bm) / bs
    encoders, bus_history, bus_epochs = [], [], []
    for offset in (0, 1000):
        net = BusNet(len(nm), len(bm), len(vocab), seed + offset, "independent")
        best_key, weights, selected_epoch, history = None, None, 0, []
        brng = np.random.default_rng(seed + offset)
        for epoch in range(budgets["bcpn"] + 1):
            if epoch:
                for index in brng.permutation(len(train_files)):
                    d = dedicated[train_files[index]]
                    if d["groups"]:
                        net.step(d)
            counts = Counter()
            for f in val_files:
                d = dedicated[f]
                if d["groups"]:
                    counts.update(bus.measures(d, net.forward(d)[0]))
            key = bus.key(counts)
            history.append({"epoch": epoch, "validation_selection": list(key)})
            if best_key is None or key < best_key:
                best_key, weights, selected_epoch = key, [p.copy() for p in net.p], epoch
        net.p = weights
        norm = {"node_mean": nm, "node_std": ns, "pair_mean": bm, "pair_std": bs}
        save_params(output / f"bcpn_{offset}.npz", net.p, **norm)
        encoders.append((net, norm))
        bus_history.append(history)
        bus_epochs.append(selected_epoch)
        print(f"Selected BCPN {offset} at epoch {selected_epoch}", flush=True)

    enc = ({"types": types}, vocab, encoders)
    stage = (mlp, pm, ps, base_threshold)
    states = {}
    for f, doc in documents.items():
        s = memory.enrich(energy.prepare(doc, enc), stage)
        bus.targets(s["d"], doc)
        s["y"] = np.array([g["target"] for g in s["d"]["groups"]])
        if f in train_docs:
            memory.history(s)
        states[f] = s
    scale = np.array([states[f]["sig"] for f in train_files]).std(0)
    scale[scale < .05] = 1
    physical_ids = {f: d["group_id"] for f, d in documents.items()}
    tasks = {f: memory.build(s, f, states, train_files, scale, physical_ids=physical_ids,
                            reference_count=8, equipment_count=8) for f, s in states.items()}
    example = next((t for f in train_files for t in tasks[f] if t["refs"]), None)
    memory_meta = {"enabled": example is not None, "strength": 0.0, "epoch": 0, "reference_loss": .5}
    memory_history = []
    if example is not None:
        r = example["refs"][0]
        dims = [len(r["station"]), r["structure"].shape[1], r["transfer"].shape[-1]]
        net = memory.MemoryNet(dims, seed, "structure")
        training = [(f, t) for f in train_files for t in tasks[f]]
        best_key, selected = None, None
        for epoch in range(budgets["hrm"] + 1):
            if epoch:
                for index in rng.permutation(len(training)):
                    f, task = training[index]
                    net.step(task, int(states[f]["y"][task["group"]]), reference_loss=.5)
            if epoch % 5:
                continue
            for strength in (0, .1, .3, .5, .7, 1.):
                if epoch == 0 and strength:
                    continue
                counts = Counter()
                for f in val_files:
                    assignment, _ = memory.evaluate(states[f], tasks[f], net, strength)
                    counts.update(energy.stats(states[f], assignment, states[f]["y"]))
                key = (counts["hard_errors"], counts["errors"], strength, epoch)
                if best_key is None or key < best_key:
                    best_key = key
                    selected = ([p.copy() for p in net.params()], strength, epoch)
            memory_history.append({"epoch": epoch, "validation_selection": list(best_key)})
        weights, strength, memory_epoch = selected
        save_params(output / "hrm.npz", weights, scale=scale)
        memory_meta.update(dims=dims, strength=strength, epoch=memory_epoch)
        print(f"Selected HRM at epoch {memory_epoch}, mixture {strength}", flush=True)
    else:
        print("No compatible ambiguous historical tasks; BCPN remains the fallback.", flush=True)
    write_json(output / "references.json", train_docs)
    metadata = {
        "format_version": 1, "types": types, "seed": seed, "epochs_per_stage": budgets,
        "smoke_epoch_override": epochs,
        "optimizer": {"mlp_learning_rate": .001, "mlp_batch_size": 512, "hierarchy_learning_rate": .001,
                      "ecpn_learning_rate": .0001, "bcpn_learning_rate": .0003,
                      "hrm_learning_rate": .001, "hrm_validation_interval": 5},
        "initial_mlp_threshold": base_threshold, "initial_mlp_epoch": base_epoch,
        "hierarchy_epoch": warm_epoch, "ecpn_epoch": graph_epoch, "bcpn_epochs": bus_epochs,
        "rules": rules, "line_template_strength": line_strength, "template_strength": template_strength, "hrm": memory_meta,
        "train_station_ids": [d["station_id"] for d in train_docs.values()],
        "validation_station_ids": [d["station_id"] for d in val_docs.values()],
        "train_group_ids": [d["group_id"] for d in train_docs.values()],
        "validation_group_ids": [d["group_id"] for d in val_docs.values()],
        "synthetic_only": all(d["synthetic"] for d in documents.values()),
        "scope": "Portable single-split training; not a replay of the paper's private experiments.",
        "bundle_notice": "Contains training-reference inventories and labels. Treat as confidential when trained on confidential data.",
    }
    write_json(output / "bundle.json", metadata)
    write_json(output / "training_history.json", {
        "initial_mlp": base_history, "hierarchy": warm_history, "ecpn": graph_history,
        "bcpn": bus_history, "line_templates": line_trials, "templates": trials, "hrm": memory_history,
    })
    return metadata


class Predictor:
    """Load generated NPZ/JSON artifacts and infer from equipment attributes."""
    def __init__(self, bundle):
        folder = Path(bundle)
        self.meta = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
        if self.meta["format_version"] != 1:
            raise ValueError("Unsupported bundle format.")
        self.vocab = json.loads((folder / "vocabulary.json").read_text(encoding="utf-8"))
        self.templates = json.loads((folder / "templates.json").read_text(encoding="utf-8"))
        self.line_templates = json.loads((folder / "line_templates.json").read_text(encoding="utf-8"))
        with np.load(folder / "initial_mlp.npz", allow_pickle=False) as d:
            self.pm, self.ps = d["mean"].copy(), d["std"].copy()
        self.mlp = run.MLP(len(self.pm), 0)
        load_params(folder / "initial_mlp.npz", self.mlp)
        with np.load(folder / "ecpn.npz", allow_pickle=False) as d:
            self.tm, self.ts = d["token_mean"].copy(), d["token_std"].copy()
        self.ecpn = character.CharacterHierNet(len(self.pm), len(self.tm), len(self.vocab), 0, "name_code")
        load_params(folder / "ecpn.npz", self.ecpn)
        encoders = []
        for offset in (0, 1000):
            with np.load(folder / f"bcpn_{offset}.npz", allow_pickle=False) as d:
                norm = {key: d[key].copy() for key in ("node_mean", "node_std", "pair_mean", "pair_std")}
            model = BusNet(len(norm["node_mean"]), len(norm["pair_mean"]), len(self.vocab), 0, "independent")
            load_params(folder / f"bcpn_{offset}.npz", model)
            encoders.append((model, norm))
        self.enc = ({"types": self.meta["types"]}, self.vocab, encoders)
        self.stage = (self.mlp, self.pm, self.ps, self.meta["initial_mlp_threshold"])
        self.net, self.bank, self.physical_ids = None, {}, {}
        if self.meta["hrm"]["enabled"]:
            self.net = memory.MemoryNet(self.meta["hrm"]["dims"], 0, "structure")
            with np.load(folder / "hrm.npz", allow_pickle=False) as d:
                weights = [d[f"p{k}"].copy() for k in range(len(self.net.params()))]
                if any(a.shape != b.shape for a, b in zip(weights, self.net.params())):
                    raise ValueError("HRM checkpoint architecture mismatch.")
                self.net.restore(weights)
                self.scale = d["scale"].copy()
            references = json.loads((folder / "references.json").read_text(encoding="utf-8"))
            for key, raw in references.items():
                doc = validate_document(raw, labels=True)
                if key != "station." + doc["station_id"] + ".json":
                    raise ValueError("Reference identity mismatch.")
                s = memory.enrich(energy.prepare(doc, self.enc), self.stage)
                bus.targets(s["d"], doc)
                s["y"] = np.array([g["target"] for g in s["d"]["groups"]])
                memory.history(s)
                self.bank[key] = s
                self.physical_ids[key] = doc["group_id"]

    def predict(self, raw):
        doc = validate_document(raw, labels=False)
        run.USE_BUS = True
        b = whole.make_input(doc, self.meta["types"])
        x = (b["x"] - self.pm) / self.ps
        graph = hierarchy.structure(b, self.mlp.predict(x), self.meta["initial_mlp_threshold"])
        graph["text_ids"] = character.encode(b["d"], self.vocab)
        z = self.ecpn.predict((b["tokens"] - self.tm) / self.ts, x, b["i"], b["j"], np.zeros(len(b["i"])), graph)
        rules = self.meta["rules"]
        z = z + rules["alpha"] * b["name_hint"]
        b["layout"], b["extended"] = role.input_layout(b), template.groups(b)
        p = role.enforce(b, run.structure_mask(b, z) & (z >= rules["threshold"]), z)
        if self.meta["line_template_strength"] >= 0:
            p, _ = role.apply_templates(b, p, z, rules["threshold"], self.meta["line_template_strength"],
                choices=[role.candidate_templates(g, self.line_templates) for g in b["layout"]["groups"]])
        p = template.decode(b, p, z, rules["threshold"], self.meta["template_strength"],
                            [template.choices(g, self.templates) for g in b["extended"]])
        state = memory.enrich(energy.prepare(doc, self.enc), self.stage)
        assignment = state["base"].copy()
        tasks, traces = [], []
        if self.net is not None:
            key = "station." + doc["station_id"] + ".json"
            physical_ids = {**self.physical_ids, key: doc["group_id"]}
            tasks = memory.build(state, key, self.bank, list(self.bank), self.scale,
                                 physical_ids=physical_ids, reference_count=8, equipment_count=8)
            assignment, traces = memory.evaluate(state, tasks, self.net, self.meta["hrm"]["strength"], details=True)
        d = state["d"]
        p[d["all_bus_pair_indices"]] = False
        selections = []
        for index, g in enumerate(d["groups"]):
            chosen = int(assignment[index])
            target = None
            if chosen < len(g["indices"]):
                p[g["indices"][chosen]] = True
                target = doc["nodes"][g["buses"][chosen]]["node_id"]
            selections.append({"device_id": doc["nodes"][g["device"]]["node_id"],
                               "bus_id": target, "ambiguous": bool(g["hard"])})
        edges = [{"source": doc["nodes"][int(b["i"][k])]["node_id"],
                  "target": doc["nodes"][int(b["j"][k])]["node_id"]} for k in np.flatnonzero(p)]
        return {"station_id": doc["station_id"], "group_id": doc["group_id"], "synthetic": doc["synthetic"],
                "predicted_edges": edges, "bus_assignments": selections,
                "historical_tasks": len(tasks), "historical_trace": traces,
                "notice": "Predictions require engineering review; reference weights are not calibrated confidence."}


def evaluate(predictor, directory):
    docs = load_directory(directory, labels=True)
    used = set(predictor.meta["train_station_ids"]) | set(predictor.meta["validation_station_ids"])
    used_groups = set(predictor.meta["train_group_ids"]) | set(predictor.meta["validation_group_ids"])
    if any(d["station_id"] in used or d["group_id"] in used_groups for d in docs.values()):
        raise ValueError("Evaluation stations must be disjoint from training and validation.")
    counts = Counter()
    for doc in docs.values():
        pred = predictor.predict(doc)
        truth = {tuple(sorted((e["source"], e["target"]))) for e in doc["labels"]["edges"]}
        guessed = {tuple(sorted((e["source"], e["target"]))) for e in pred["predicted_edges"]}
        counts.update(tp=len(truth & guessed), fp=len(guessed - truth), fn=len(truth - guessed))
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {"stations": len(docs), **counts,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "synthetic_only": all(d["synthetic"] for d in docs.values()),
            "notice": "Locally computed demonstration metrics; not paper results."}


def demo(output, epochs=2, seed=19):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_dataset(output / "data")
    meta = train(output / "data/train", output / "data/validation", output / "bundle", epochs=epochs, seed=seed)
    predictor = Predictor(output / "bundle")
    prediction = predictor.predict(read_document(output / "data/query.json"))
    write_json(output / "prediction.json", prediction)
    result = evaluate(predictor, output / "data/evaluation")
    write_json(output / "metrics.json", result)
    print(json.dumps({"synthetic_only": True, "modules": ["MLP", "HierNet", "ECPN", "rules/templates", "BCPN ensemble", "HRM"],
                      "hrm_available": meta["hrm"]["enabled"], "metrics": result}, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Code-only topology inference research release")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="Generate invented data and exercise actual model stages")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=19)
    p = sub.add_parser("generate", help="Generate entirely invented inventories")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stations", type=int, default=12)
    p = sub.add_parser("train", help="Train on explicitly supplied authorized datasets")
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--validation-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=None,
                   help="Optional short smoke-test override for every stage; omit for stage-specific budgets")
    p.add_argument("--seed", type=int, default=19)
    p = sub.add_parser("infer", help="Infer without reading any query labels")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate", help="Score a disjoint, labelled evaluation directory")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "demo":
        demo(args.output, args.epochs, args.seed)
    elif args.command == "generate":
        write_dataset(args.output, args.stations)
    elif args.command == "train":
        train(args.train_dir, args.validation_dir, args.output, args.epochs, args.seed)
    else:
        if args.output.exists():
            raise FileExistsError("Choose a new output path; existing files are not overwritten.")
        predictor = Predictor(args.bundle)
        value = predictor.predict(read_document(args.input)) if args.command == "infer" else evaluate(predictor, args.evaluation_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, value)
