# Expected release results

The following values provide a regression check for a complete reproduction. Values are mean +/- population SD across the five fixed held-out test folds (`ddof = 0`).

| Result | Expected R2 |
|---|---:|
| AggHead mRNA-LM | 0.4436 +/- 0.0112 |
| XGBoost, `basic_plus_biochemical_full` | 0.6766 +/- 0.0057 |
| ElasticNet, `genetic_plus_biochemical_full` | 0.6374 +/- 0.0141 |
| XGBoost component in paired fusion evaluation | 0.6766 +/- 0.0057 |
| Linear prediction-level fusion | 0.7008 +/- 0.0080 |
| Permuted-label XGBoost | -0.0030 +/- 0.0072 |
| Permuted-label Ridge | -0.0095 +/- 0.0101 |

The fold-wise fusion gains over XGBoost are expected to be approximately `0.0281`, `0.0220`, `0.0220`, `0.0217`, and `0.0269` for folds 0-4.

Prediction/residual correlation summaries under the same convention are expected to be: prediction Pearson `0.6752 +/- 0.0048`, prediction Spearman `0.6617 +/- 0.0102`, residual Pearson `0.5683 +/- 0.0156`, and residual Spearman `0.5444 +/- 0.0219`.

Relative to the internally retrained full family-ablation model, the expected R2 decreases are `0.0160` after removing BASIC, `0.0229` after removing ENCORI, `0.0599` after removing RIP, and `0.1167` after removing both RIP and ENCORI.

Minor last-decimal differences can arise from software and hardware versions. A material difference should be investigated at the level of identifiers, fold assignments, preprocessing, target scale, and validation-selected model state before changing a frozen result.
