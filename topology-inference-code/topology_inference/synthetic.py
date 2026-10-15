"""Invented toy inventories for execution checks; never derived from real data."""
from __future__ import annotations

import copy
import json
from pathlib import Path


def make_station(number: int, circuits: int = 3) -> dict:
    """Make a schematic equipment-relation graph, not an engineering design.

    All identifiers, names, and connections are generated here. The examples
    intentionally use Chinese equipment-role words supported by the original
    name parser. They are neither anonymized records nor benchmark samples.
    """
    sid = f"SYN_{number:03d}"
    nodes, edges = [], []

    def node(key, name, kind):
        nid = f"{sid}_{key}"
        nodes.append({"node_id": nid, "features": {
            "name": name, "type_code": kind, "voltage_kv": 220.0,
            "winding_voltage_kv": [],
        }})
        return nid

    def edge(a, b):
        edges.append({"source": a, "target": b})

    buses = {}
    for family in ("正母", "副母"):
        for section, roman in ((1, "Ⅰ"), (2, "Ⅱ")):
            buses[family, section] = node(
                f"BUS_{'MAIN' if family == '正母' else 'AUX'}_{section}",
                f"220kV{family}{roman}段母线", "410")
    for c in range(circuits):
        # This toy relationship is deliberately simple and not a learned result.
        section = 1 + (c % 2)
        prefix = f"测试{c + 1:02d}线9Z{c + 1:02d}"
        main = node(f"C{c}_DS_MAIN", prefix + "正母闸刀", "408")
        auxiliary = node(f"C{c}_DS_AUX", prefix + "副母闸刀", "408")
        breaker = node(f"C{c}_CB", prefix + "开关", "407")
        line = node(f"C{c}_DS_LINE", prefix + "线路闸刀", "408")
        ground = node(f"C{c}_ES", prefix + "线路接地闸刀", "409")
        terminal = node(f"C{c}_LINE", prefix + "线路", "415")
        edge(buses["正母", section], main)
        edge(buses["副母", section], auxiliary)
        edge(main, breaker)
        edge(auxiliary, breaker)
        edge(breaker, line)
        edge(line, ground)
        edge(ground, terminal)
    # Rotate storage order: identifiers and ordering must not encode features.
    shift = number % len(nodes)
    nodes = nodes[shift:] + nodes[:shift]
    return {"station_id": sid, "group_id": sid, "synthetic": True, "nodes": nodes,
            "labels": {"edges": edges}}


def write_dataset(output: Path, count: int = 12) -> dict:
    if count < 6:
        raise ValueError("Use at least six invented stations.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    n_train = max(3, count - 3)
    plan = {"train": [], "validation": [], "evaluation": []}
    for part in plan:
        (output / part).mkdir()
    for i in range(count):
        part = "train" if i < n_train else "validation" if i < count - 1 else "evaluation"
        doc = make_station(i + 1)
        fn = doc["station_id"] + ".json"
        (output / part / fn).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        plan[part].append(fn)
    query = copy.deepcopy(doc)
    query.pop("labels")
    (output / "query.json").write_text(json.dumps(query, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "synthetic_notice.json").write_text(json.dumps({
        "synthetic": True,
        "purpose": "Execution demonstration only; not paper data or benchmark evidence.",
        "provenance": "Generated from literal schematic rules in synthetic.py; no source data read.",
        "split": plan,
    }, indent=2), encoding="utf-8")
    return plan
