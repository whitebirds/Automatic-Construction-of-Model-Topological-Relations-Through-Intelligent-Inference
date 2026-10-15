# Equipment Model Topology Inference

Research implementation accompanying **Automatic Construction of Model
Topological Relations Through Intelligent Inference**.

This is a **code-only release**. It includes the numerical model components,
structural rules, historical matching, baseline adapters, and a fully synthetic
execution example. It contains **no real substation inventories, connections,
reference libraries, trained research weights, private split membership, or
prediction logs**. There are no network calls in the supplied running code.

Real operational data cannot be distributed because of confidentiality
restrictions. The synthetic workflow checks that the implementation runs; its
scores are **not the paper's experimental results**. Without the restricted
dataset and its exact partition, this package does not independently reproduce
those numerical results.

## Quick start

Use Python 3.12 and a new virtual environment. From this directory:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then:

```sh
python -m pip install -r requirements.txt
python -m topology_inference demo --output outputs/synthetic-demo --epochs 2
```

The demo generates twelve invented station inventories, trains the actual
numerical modules on nine, selects settings on two, and evaluates on one. It
also predicts a separate **label-free** copy of the evaluation inventory. The
short epoch override is for an execution check, not a scientific comparison.
Choose a new output directory for each run; existing runs are not overwritten.

Outputs include:

- `data/`: generated inventories and a label-free `query.json`.
- `bundle/`: locally trained demonstration weights and synthetic references.
- `prediction.json`: predicted equipment connections and historical-reference
  traces where applicable.
- `metrics.json`: metrics computed from the synthetic evaluation targets.

The default demonstration includes ambiguous busbar candidates. Validation may
choose zero historical fusion on a small synthetic set; this is a valid
selection outcome, not evidence of the method's effectiveness or ineffectiveness
on the restricted dataset.

## Train and infer with an authorized local dataset

Prepare one JSON document per station following [the schema](docs/data_schema.md).
Training and validation documents need `labels.edges`; inference does not.
`group_id` identifies the physical station across aliases or exports. Choose one
representative inventory per physical station, and keep physical groups
disjoint between training, validation and evaluation.

```sh
python -m topology_inference train --train-dir private/train --validation-dir private/validation --output outputs/my-bundle
python -m topology_inference infer --bundle outputs/my-bundle --input private/query.json --output outputs/query-prediction.json
python -m topology_inference evaluate --bundle outputs/my-bundle --evaluation-dir private/evaluation --output outputs/evaluation.json
```

The inference boundary retains only the documented equipment attributes and
identifiers. Query labels, connection-point fields and derived truth fields are
discarded. Known connections of **training references** are deliberately used
by historical matching. Identifiers support output lookup and station exclusion;
they are not predictive features.

**Do not upload outputs produced from real data.** Model bundles contain
training-reference inventories and connections, learned name vocabularies,
normalizers, parameters and identifiers. Prediction traces can identify matched
reference equipment. These outputs remain confidential even if their containing
directory has a different name. The provided `.gitignore` is only a guard
against accidental inclusion, not a confidentiality guarantee.

## Methods and paper correspondence

- **ECPN**: equipment connection prediction with character encoding and
  hierarchical context, initialized through the base prediction stages.
- **Role constraints and templates**: correct incompatible relationships and
  use patterns learned from the training stations.
- **BCPN**: two independent busbar-candidate prediction networks, averaged in
  probability space, including a no-direct-busbar option.
- **HRM**: trainable historical-reference weighting, equipment correspondence
  and busbar mapping, using attributes and local structural descriptors.
- **DH-R**: direct historical adjacency transfer with role constraints.
- **XGB-R / XGB-C**: optional connection prediction with basic rules, with an
  optional circuit-context busbar-candidate ranker.

See [paper protocol](docs/paper_protocol.md) for the method-to-table mapping,
training order, selection rules and metric definitions, and
[paper settings](configs/paper_settings.json) for the recorded public
hyperparameters. That JSON is a descriptive record, **not a CLI configuration
file**. The portable runner is a single-split implementation; it does not
replay the private five-fold experiment harness. Its differences and the
extracted components are described in [implementation notes](docs/implementation_notes.md).

The original neural implementation uses NumPy and CPU execution, rather than a
GPU framework. Equipment-name parsing uses generic Chinese role terms and
numeric type codes from the original modeling convention; English-only naming
or another equipment taxonomy needs a corresponding parser adaptation.

## Direct-history baseline

DH-R needs no trained neural checkpoint. A small Python example, using only
the generated demonstration documents:

```python
import json
from pathlib import Path
from topology_inference.dhr import predict

root = Path("outputs/synthetic-demo/data")
query = json.loads((root / "query.json").read_text(encoding="utf-8"))
references = [json.loads(p.read_text(encoding="utf-8"))
              for p in sorted((root / "train").glob("*.json"))]
result = predict(query, references, query_group_id=query["group_id"])
print(result)
```

Only print or share these outputs when the input is synthetic or independently
cleared for disclosure. DH-R is different from the fixed-weight HRM ablation.

## Optional XGBoost baselines

In a separate virtual environment, install the tested optional dependencies:

```sh
python -m pip install -r requirements-xgboost.txt
python -m topology_inference.baselines train --train-dir outputs/synthetic-demo/data/train --validation-dir outputs/synthetic-demo/data/validation --output outputs/xgb-r
python -m topology_inference.baselines infer --bundle outputs/xgb-r --input outputs/synthetic-demo/data/query.json --output outputs/xgb-r-prediction.json
```

Add `--candidate-ranker` to the training command and choose a fresh output path
to run XGB-C. `--depths 3 --rounds 2` gives a fast execution check; omit these
overrides for the recorded depth/round search. Both adapters select only on the
supplied validation set. They do not read the private experiment caches.

## Tests and environment

```sh
python -m unittest discover -s tests -v
```

Tests cover the synthetic end-to-end workflow, label exclusion, split
separation, one selected busbar per disconnector, safe numeric checkpoint
loading, direct-history transfer, and query-ID/order invariance for DH-R.
Finite-difference tests check the character/hierarchical network, busbar
candidate network, and historical scoring heads, including the null gate.
Optional baseline tests run only when XGBoost is installed.

Core verification environment: Windows, Python 3.12.14, NumPy 2.3.5. Optional
baseline environment: NumPy 2.5.3, SciPy 1.18.1 and XGBoost 3.4.1. Dependency
versions are kept separately; the demo is not a substitute for the paper's
historical runtime record. Linux/macOS commands are provided for convenience;
they have not been separately tested in this release.

For installable CLI use, `python -m pip install --no-deps -e .` exposes
`topology-inference`; running from the source directory needs no package install.

## Publication boundary

The prepared package consists of source, documentation, configuration values
and generated-example **code**, not real or renamed real example records.
Do not add the original workspace, experiment directories, Git history,
checkpoints or logs when publishing it. The repository does not contact GitHub
or upload anything automatically.
