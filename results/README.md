# Result directories

`frozen/` contains the compact numeric files used to reproduce the publication figures and to audit the reported values. These files should be treated as immutable release inputs.

`runs/` is the default destination for newly trained models and downstream analyses. Its contents are ignored by Git except for `.gitkeep`.

The frozen directories map to the analysis stages as follows:

| Directory | Contents |
|---|---|
| `ridge/`, `elasticnet/`, `xgboost/` | Fold-level and summary tabular metrics |
| `mrnalm_agghead/`, `mrnalm_seqhead/` | Fold-specific sequence-model summaries |
| `fusion/` | Paired predictions, coefficients, correlations, and fold metrics |
| `permutation/` | Permuted-label negative controls |
| `shap/`, `ablation/` | TreeSHAP and feature-family retraining results |
| `fusion_gain/`, `motif/`, `case_studies/` | Sample-level fusion diagnostics |
| `helped_hurt_shap/` | SHAP differences between fusion-gain groups |

