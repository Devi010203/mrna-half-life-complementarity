# Data dictionary

## Core identifiers and target

| Field | Meaning |
|---|---|
| `ENSID` | Transcript-level identifier used to align target, feature, split, and prediction records |
| `fold` | Outer test-fold assignment, indexed 0-4 |
| `HALFLIFE` | Saluki-processed normalized human mRNA half-life prediction target |
| `split` | Train, validation, or test role where explicitly stored |

The reported metrics use the processed target scale. No result in the frozen release is presented as an untransformed physical half-life in hours.

## Feature groups

| Code | Display name | Features |
|---|---|---:|
| `saluki_basic8` | Basic architecture | 8 |
| `genetic_only` | Sequence-statistical | 23,005 |
| `biochemical_371` | Compact biochemical/regulatory | 371 |
| `genetic_plus_biochemical_371` | Compact biochemical/regulatory + sequence-statistical | 23,376 |
| `all_23384` | All Saluki tabular features | 23,384 |
| `biochemical_full_no_basic` | Full regulatory/biochemical without BASIC | 551 |
| `basic_plus_biochemical_full` | Full regulatory/biochemical + BASIC | 559 |
| `genetic_plus_biochemical_full` | Full regulatory/biochemical + sequence-statistical | 23,556 |

`BASIC` represents transcript-architecture descriptors. `GENETIC` represents high-dimensional sequence-statistical descriptors. The full regulatory/biochemical block contains RIP, ENCORI, eCLIP, M6ACLIP, PARCLIP, TE, and related released annotation families.

## Prediction fields

| Field | Meaning |
|---|---|
| `y_true` | Processed held-out target |
| `y_pred_xgb` | Held-out XGBoost prediction from the main annotation-augmented feature set |
| `y_pred_mrnalm` | Held-out AggHead mRNA-LM sequence-only prediction |
| `y_pred_fusion` | Held-out prediction from validation-fitted linear fusion |
| `residual_*` | `y_true - y_pred_*` |
| `abs_err_*` | Absolute prediction error |
| `fusion_gain` | `abs_err_xgb - abs_err_fusion`; positive values indicate lower absolute error after fusion |

