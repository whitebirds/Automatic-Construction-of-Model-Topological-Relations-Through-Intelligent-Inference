# Paper methods and experiment protocol

This document describes the settings of the paper, not a claim that the synthetic
example reproduces its results. The confidential station inventories, historical
connections, station-level split membership, trained weights, learned character
vocabularies, reference libraries, and prediction logs are not included.

`configs/paper_settings.json` is a manually written record of the public method
settings. It is documentation, not an exported private experiment protocol. The
demonstration uses a smaller training budget; see the root README for the commands
and the settings actually used by that example.

The portable `train` command also differs from the original experiment harness;
it does not consume this configuration or reproduce the table grids. See
[implementation notes](implementation_notes.md#portable-runner-versus-paper-experiment-harness)
for the explicit differences in budgets, optimization, and template selection.

## Module order

1. Prepare a fixed equipment inventory and keep labels separately. Derive name
   features and circuit groups from equipment attributes.
2. Split by physical station. All exports of a physical station must stay in the
   same split. Fit vocabularies, normalizers, templates, and reference libraries
   using training stations only.
3. Train the initial multilayer perceptron (MLP) using historical edge labels.
   Freeze it before training the hierarchical equipment encoder.
4. Train the hierarchical encoder, then its character-encoding extension. This
   produces the Equipment Connection Prediction Network (ECPN).
5. Learn circuit templates from training connections and select correction
   settings on validation stations. Apply role and template refinement to ECPN.
6. Train two independent Busbar Candidate Prediction Networks (BCPNs). Average
   their candidate probabilities, including the no-direct-busbar option.
7. Freeze the two BCPNs. Construct the historical reference library from training
   stations and learn Historical Reference Matching (HRM). Query local structure
   uses initial-MLP predictions; reference local structure uses known training
   connections.
8. Replace the disconnector--busbar edges with the BCPN/HRM choices and retain
   the other refined ECPN edges. Compute metrics against held-out labels.

Historical station matching denotes statistical correspondence between separate
station graphs, not physical electrical connections between stations.

## Common data separation

The paper uses five station-level evaluation folds. Each station is evaluated
once. Every fold has distinct training, validation, reserved, and evaluation
partitions. The reserved partition is not used to train or select the methods
reported in the main comparison. The exact private station membership is not
distributed; an example split generated from synthetic stations is a new split.

Epochs, decision thresholds, rule strengths, and historical fusion are selected
on validation data. Evaluation labels must not influence those choices.
Queries must be excluded from their own reference libraries, including other
exports of the same physical station. Object identifiers are used to identify
outputs, not as predictive features.

The original research folds informed method development. Merely generating new
split files or rerunning the released code does not make those original results
an independent external validation.

## Table 1: full methods

- **DH-R (adapted):** Select one historical training station using voltage-specific
  inventory counts. Match equipment using name-derived codes and transfer known
  neighbors, subject to explicit role and attribute compatibility. No neural
  predictor is used. This is a direct-reuse adaptation to equipment-relation
  graphs, not a complete reproduction of an earlier diagram-generation system.
- **MLP-R:** Initial MLP connection scores, validation-selected name association
  weight, voltage compatibility, and the explicit line-voltage-transformer
  connection-count rule.
- **XGB-R:** XGBoost connection prediction with the same equipment-pair features
  and basic rules as MLP-R. The main comparison uses initialization setting 7.
- **XGB-C:** XGB-R with circuit context and joint busbar-candidate ranking. Its
  own XGB-R non-busbar edges are retained. Using the proposed method's non-busbar
  edges instead would be a different controlled comparison.
- **Proposed:** ECPN character-name-and-code configuration with initialization
  offset 0, role/template refinement, the independent BCPN ensemble, and
  structural HRM with reference-branch supervision and initialization setting 19.

The fixed thresholds and learned parameters are not embedded in this package.
They must be learned and selected again on an authorized training/validation set.

## Table 2: role and template correction

The four configurations are base prediction, roles only, templates only, and
roles plus templates. They share the same saved ECPN scores. All
disconnector--busbar predictions are held fixed and excluded from this table's
edge scope. Disabling role filtering does not disable role parsing for template
construction. The second initialization repeats the correction comparison with
its corresponding ECPN scores.

## Table 3: candidate prediction and historical matching

- **MLP-R** is the basic full-output baseline's candidate behavior.
- **Single-bus constraint** applies an exclusivity decision to unchanged MLP-R
  scores; it is not a newly trained candidate network.
- **BCPN (no history)** averages the two independent BCPN scorers.
- **Fixed history** uses predetermined matching scores with validation-selected
  fusion. It is different from DH-R: it starts from BCPN probabilities and uses
  multiple historical references.
- **HRM attributes / attributes plus structure** use the same BCPN starting
  probabilities and independently enable local structural descriptors and
  reference-branch supervision. Each learned configuration uses initializations
  7 and 19. The two-run summary and the main configuration are different
  aggregations; an individual main-run score must not be mislabeled as a repeat
  average.

These HRM comparisons hold the earlier encoders, reference retrieval, other
equipment edges, training budget, and selection procedure fixed. They are module
repeats, not complete end-to-end retraining.

## Metrics

Edges are undirected and identified by unordered endpoint pairs. Pool true
positives (TP), extra edges (FP), and missing edges (FN) across evaluation
stations before calculating the reported micro metrics:

- Precision = TP / (TP + FP).
- Recall = TP / (TP + FN).
- F1 = 2 TP / (2 TP + FP + FN).
- Edge-edit count = FP + FN.

Use an explicit empty-denominator convention in an implementation. Exact busbar
target accuracy compares the complete predicted busbar-neighbor set with its
label set. Wrong targets, missing targets, and multiple targets are errors. The
ambiguous subset is defined from names and candidate attributes, not selected by
which examples a method predicts incorrectly. A station with no ambiguous
devices contributes no device-level denominator to that subset.

## Reproduction scope

The release supports inspection and execution of the included implementations.
The synthetic example demonstrates the computational workflow and the separation
of inventory attributes from labels. It is not a replacement for the confidential
dataset, a new paper experiment, or evidence that the paper's numerical results
are reproduced. Access to code alone cannot independently reproduce those
specific numerical results without the original data and experimental artifacts.
