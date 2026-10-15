# Equipment-inventory data interface

Use only data that you are authorized to process. The distributed examples are
synthetic. They contain deliberately invented identifiers and names, rather than
renamed records or topology extracted from a real station.

## Inventory document

The numerical core accepts a UTF-8 JSON document with `nodes`. Each equipment
object has a station-local `node_id` and a `features` dictionary:

```json
{
  "station_id": "synthetic-station-a-export-1",
  "group_id": "synthetic-station-a",
  "nodes": [
    {
      "node_id": "synthetic-bus-a",
      "features": {
        "name": "示例正母I段母线",
        "type_code": "410",
        "voltage_kv": 220,
        "winding_voltage_kv": []
      }
    },
    {
      "node_id": "synthetic-disconnector-a",
      "features": {
        "name": "220kV测试甲线90001正母闸刀",
        "type_code": "408",
        "voltage_kv": 220,
        "winding_voltage_kv": []
      }
    }
  ]
}
```

This two-object fragment illustrates the schema only. It is not a complete
training dataset or a physical station design. Use the synthetic generator in
the root README for a complete runnable example.

- `station_id`: opaque identifier for this station document/export.
- `group_id`: opaque physical-station identifier. All exports of the same
  physical station use the same value. It controls splits and self-exclusion,
  not equipment prediction features.
- `node_id`: unique within the document; opaque output identifier. Preserve a
  consistent correspondence between labels and identifiers.
- `features.name`: equipment name. The parser contains Chinese naming tokens
  used by this method. Arbitrarily replacing names with unrelated English labels
  changes the model inputs and disables those naming cues.
- `features.type_code`: string equipment-type code. These codes are conventions
  of the implementation, not a claimed universal coding standard.
- `features.voltage_kv`: positive numeric voltage in kilovolts, or `null` for
  unknown voltage. Do not replace an unknown voltage with a guessed value.
- `features.winding_voltage_kv`: list of positive winding voltages for a
  transformer; use an empty list for other objects or unavailable values.

The public runner accepts only letters, digits, underscores, and hyphens in
`station_id` and `group_id`, with a maximum length of 80 characters. It requires
at least two equipment objects. Its loader keeps one inventory per `group_id` in
each directory; select a representative export before supplying multiple aliases
of the same physical station.

The recognized type labels used by the main rules are `407` (circuit breaker),
`408` (disconnector), `409` (earthing switch), `410` (busbar), `415` (line object),
`416` (transformer), `463` (voltage transformer), and `515` (surge arrester).
The historical feature vocabulary also has codes `411` and `421`; this release
does not assign them an unsupported English equipment definition. Equipment types
outside a trained vocabulary use the unknown-type features where implemented;
they do not acquire invented role-specific rules.

## Historical labels

For supervised training or evaluation, add a separate label field:

```json
{
  "labels": {
    "edges": [
      {
        "source": "synthetic-bus-a",
        "target": "synthetic-disconnector-a"
      }
    ]
  }
}
```

Every endpoint must exist in `nodes`. Edges are undirected: do not include both
orders of the same pair, duplicate edges, or self-loops. These labels describe
equipment-model relations. A projected relation should not be interpreted as
proof of a series current path through the devices.

Training labels supervise losses and training-reference relations. Evaluation
labels are used only after predicting the held-out inventory. A query document
does not need a `labels` field for inference. Connection-point fields, true
neighbor counts, connected-bus identifiers, and topology-derived bay membership
are not predictive inputs.

## Physical-station identity and splits

Manage physical-station identity explicitly when forming splits and reference
libraries. Different exports of one station must share the same `group_id` even
when they have different `station_id` values. Core adapters may alternatively
accept a `physical_ids` mapping from document keys to those group identities.
Neither form becomes a learned feature. The public runner's accepted manifest
and command-line details are documented in the root README.

Do not infer independence from different filenames. The original private
experiment harness had a dataset-specific filename convention; public data
adapters must use explicit grouping metadata instead of relying on that private
convention. A synthetic split supplied by this release is only a demonstration.

## Parser scope

Name parsing separates circuit stems, busbar category, section indications,
equipment roles, and name-derived code candidates. Matching code fragments are
weak association features, not proof that an edge exists. Unknown or ambiguous
fields remain unresolved. Applying the rules to another naming system requires
validation and, where appropriate, a separately documented parser adaptation.
