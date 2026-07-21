# Family-ablation, standard-deviation, and CI audit

## 1. Family-ablation value provenance

The currently valid human clean-rerun ablation output is:

`.\result\62374\result\xgb_family_ablation\xgb_family_ablation_basic_plus_biochemical_full_clean_codex1\ablation_summary_metrics.csv`

Its internally retrained full model has mean test R2 = 0.6766160828. Relative to this model, the clean fold-level results give the following decreases:

| Ablation | Clean mean test R2 | R2 decrease from full |
|---|---:|---:|
| remove BASIC | 0.6605683538 | 0.0160477290 (0.0160) |
| remove ENCORI | 0.6537608143 | 0.0228552685 (0.0229) |
| remove RIP | 0.6166727121 | 0.0599433707 (0.0599) |
| remove RIP + ENCORI | 0.5599114259 | 0.1167046569 (0.1167) |

The superseded values 0.0188, 0.0261, 0.0567, and 0.1145 are reproduced exactly by the historical output:

`.\result\xgb_family_ablation\xgb_family_ablation_basic_plus_biochemical_full_20260526\ablation_summary_metrics.csv`

The historical and clean runs used the same 12,981 ENSID identifiers, the same 559-feature `basic_plus_biochemical_full` manifest, and equivalent XGBoost settings. Their split files were different:

- Historical run: `data\processed\human\splits\cv_k5_seed42.csv`
- Current clean run: `temp\processed\codex_1\human\splits\cv_k5_seed42.csv`

After joining by ENSID, all 12,981 identifiers were shared, but only 3,114 retained the same fold label; 9,867 were assigned to a different fold. The old ablation values are therefore valid only for the historical split and must not be used as current clean-rerun results. The discrepancy is a split-provenance/label-mixing issue, not an error in the current family definitions, feature matrix, or ablation arithmetic.

The historical and current split files have SHA-256 values `2e82660e1373972af16be94ffb0d77e25cf3d74ffbf3c8579747fafd8c64d565` and `d47ba778ac7fb301551433bf52689a327baf38b2de3ff962d49225c4a0ad848d`, respectively. Both have fold sizes 2,600/2,600/2,600/2,600/2,581; equal fold sizes therefore do not imply equal ENSID-to-fold assignments.

## 2. Standard-deviation convention

The primary model and fusion scripts summarize the five fixed outer folds with population standard deviation (`ddof = 0`), whereas the earlier correlation table and ablation summary used sample standard deviation (`ddof = 1`). The project now uses `ddof = 0` consistently for model-performance, fusion, prediction/residual-correlation, fold-delta, permutation, and family-ablation summaries. Sample-level descriptive statistics for the complete target distribution are separate and are not fold-summary statistics.

Under `ddof = 0`, the prediction/residual correlation summaries are:

| Quantity | Metric | Mean | SD |
|---|---|---:|---:|
| prediction | Pearson | 0.6751610375 | 0.0047740699 |
| prediction | Spearman | 0.6616771669 | 0.0102417609 |
| residual | Pearson | 0.5682583475 | 0.0155504530 |
| residual | Spearman | 0.5443849531 | 0.0218745482 |

## 3. Paired-comparison confidence intervals

Supplementary Table S5 now includes a `paired_model_comparisons` sheet. For each comparison, the table reports the five fold-level R2 differences, their mean and population SD, a 95% percentile interval from 5,000 bootstrap resamples of the five paired differences (seed 42), and an exact two-sided sign-flip result. Because only five outer folds are available, these intervals and p values are supplementary descriptive summaries; the manuscript emphasizes effect size and fold-wise directional consistency.

## 4. Files updated

- `paper/manuscript_draft.md`
- `paper/manuscript_draft_zh.md`
- `paper/Supplementary Methods.md`
- `paper/table_scripts/generate_supplementary_tables_s1_s9.py`
- `script/main/24_xgb_family_ablation_noargs.py`
- `paper/mrna-half-life-complementarity/scripts/tables/generate_supplementary_tables.py`
- `paper/mrna-half-life-complementarity/scripts/analysis/feature_family_ablation.py`
