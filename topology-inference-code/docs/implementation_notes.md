# Implementation and release boundaries

## What this package contains

The numerical modules are extracted from the research implementation. They
contain NumPy-based neural layers, losses, feature construction, naming rules,
template operations, candidate scoring, and historical matching. XGBoost is an
optional dependency for the corresponding traditional machine-learning
baselines. No GPU framework is required by the neural implementation.

Extraction removes research-harness entry points and their assumptions about
private paths, saved station inventories, intermediate checkpoints, and
experiment-output directories. Public wrappers expose an explicit data and
execution interface. The root README states which wrappers are runnable and
which modules are provided as inspectable computational components.

The extracted core retains original computations rather than substituting a
different framework or newly trained published model. Public wrappers and the
synthetic generator are release-specific additions. Configuration extraction and
path adaptation are not a claim of bitwise equivalence to every historical
experiment run. The paper protocol document identifies the intended method
variants and the controls needed for an equivalent experiment.

## Portable runner versus paper experiment harness

The public `train` command is a single-split execution interface. It does not load
`configs/paper_settings.json`, generate the paper's five folds, or automatically
run the Table 1--3 comparison grids. Its default stage budgets are 24 initial-MLP
epochs, 36 hierarchical epochs, 24 character-training epochs, 40 BCPN epochs, and
25 HRM epochs. Supplying `--epochs` explicitly overrides each stage with a shared
budget. The `demo` command uses this override for a short smoke test.

Default learning rates and the MLP batch size follow the documented paper
settings, and HRM is validated every five epochs. The wrapper still uses one
user-supplied seed rather than replaying the paper's fold-specific initialization
schedule and exact random-number streams. It does not recreate the earlier
intermediate bus-candidate checkpoint used during the original sequence of
structure-refinement experiments; the final busbar edges come from the BCPN/HRM
stage. It uses the actual core networks and losses but does not automatically
replay every original training and selection decision.

The package includes lower-level functions for the ablation variants and
baselines. Running the full research protocol on a new authorized dataset
requires a harness that implements its splits, separate training budgets,
selection grids, and controlled comparisons. The public demo's output should
therefore be described as execution of the released method components, not a
reproduction of the paper tables.

## Variant choices that matter

- BCPN in the paper is an ensemble of two **independent** scorers. A class may
  also contain a joint-attention mode; that mode is not the main BCPN result.
- The final HRM configuration includes both attribute and local-structure
  features and uses reference-branch loss weight `0.5`. Attribute-only,
  unsupervised-reference, and fixed-history variants are ablations.
- The main HRM output uses initialization setting `19`. Initializations `7` and
  `19` are both needed to reconstruct the reported module-repeat comparison.
- The active MLP naming association implementation uses exact eligible
  alphanumeric tokens. It should not be replaced with an earlier prefix-only
  comparator when recreating the baseline.
- Coarse historical retrieval is fixed. The three small HRM scoring networks
  learn reference-station weighting, equipment correspondence, and busbar
  mapping; the earlier encoders and retrieval features remain fixed.
- A query's local structural descriptors come from predicted non-busbar
  connections. A training reference may use its known connections. Feeding the
  query's label graph into this step changes the task and is not permitted.
- Station/reference weights are not calibrated correctness probabilities.
  Automatic per-device rejection is not a validated capability of this release.

## Data and artifacts intentionally excluded

The package contains no real station data, renamed real station records,
historical reference libraries, trained research checkpoints, learned
vocabularies, real-data normalizers, prediction logs, station-level split
membership, data hashes, source diagrams, author contact details, or Git history.
Synthetic examples and their locally generated outputs must remain clearly
identified as synthetic.

Outputs from training on another dataset can themselves contain input-derived
names, vocabularies, learned parameters, and prediction details. The fact that
this package is prepared for sharing does not make later user-generated outputs
safe to publish. Review those artifacts separately before adding them to a
repository. Ignore rules reduce accidental inclusion but are not an access
control mechanism.

## Environment and tests

Use the root dependency files and README for the tested release environment.
Historical runs and a local release smoke test are different records. Installing
an available library version or completing the synthetic test does not establish
that it matches the original experimental environment.

The release checks should cover imports, the synthetic workflow, label-free
query inference, and representative loss/gradient behavior where tests are
included. The final release report records what was actually run. Tests of
synthetic inputs support software validation; they cannot validate a real
station's topology or reproduce the paper's confidential-data measurements.

## Ownership and publication

This package does not infer publication rights or apply a new global software
license on the authors' behalf. Repository visibility and licensing must be
chosen by the rights holder. No upload is performed by this package preparation.
