# Reproducibility workflow

## Evaluation design

All current human analyses use the same processed `HALFLIFE` target and the same five outer folds. The fold-assignment file is `data/processed/human/splits/cv_k5_seed42.csv`. Within each outer fold, model selection and early stopping use only a validation subset drawn from the outer-training records. The outer test fold remains held out until final evaluation.

Fold-level model performance, fusion, prediction/residual correlation, and family-ablation summaries use the population standard deviation across the five fixed outer folds (`ddof = 0`). Supplementary paired comparisons additionally report a 95% percentile interval from 5,000 bootstrap resamples of the five paired fold differences (seed 42) and an exact two-sided sign-flip summary. With only five folds, effect size and directional consistency remain the primary evidence.

The tabular benchmark and prediction-level fusion both contain 12,981 transcripts in the current release. They remain distinct experiments: the tabular benchmark evaluates each feature/model combination, whereas fusion pairs XGBoost and mRNA-LM predictions record by record and fits fusion coefficients only from aligned validation predictions.

## Required inputs

See `data/README.md` and `models/README.md`. Paths can be overridden with variables listed in `config/paths.env.example`. The scripts resolve the repository root from their own file locations by default, so execution does not depend on the caller's working directory.

## Model training

### mRNA-LM

Run the five folds on a GPU-capable node. The wrapper imports the main AggHead implementation from the same repository:

```bash
python scripts/models/train_mrnalm_agghead_lora.py
python scripts/analysis/merge_mrnalm_predictions.py
```

`train_mrnalm_seqhead.py` provides the SeqHead-noConv comparator. Checkpoint locations, adapter settings, tokenization, window handling, optimizer settings, and fold-specific outputs are recorded by the scripts. The pretrained weights themselves are not included.

### Ridge and XGBoost

```bash
python scripts/models/train_ridge.py
python scripts/models/train_xgboost.py
```

Both scripts evaluate the eight feature groups using the common five-fold protocol. Ridge chooses its regularization parameter on validation data. XGBoost uses validation RMSE for early stopping and evaluates the held-out test fold once.

### ElasticNet

ElasticNet is designed for one feature-group/fold pair per array task because the high-dimensional feature groups are computationally heavier:

```bash
sbatch slurm/train_elasticnet_array.slurm
python scripts/models/collect_elasticnet.py
```

For a non-SLURM run, set the same group/fold environment variables used in the template before invoking `train_elasticnet.py`.

## Downstream analyses

After fold-level validation and test predictions are available:

```bash
python scripts/analysis/late_fusion.py
python scripts/analysis/permutation_control.py
python scripts/analysis/shap_interpretability.py
python scripts/analysis/feature_family_ablation.py
python scripts/analysis/fusion_gain_profile.py
python scripts/analysis/motif_kmer_checks.py
python scripts/analysis/case_studies.py
python scripts/analysis/helped_hurt_shap.py
```

The linear fusion model is fitted independently in each fold:

```text
y_fusion = beta_0 + beta_xgb * y_xgb + beta_mrnalm * y_mrnalm
```

Only aligned validation predictions are used to estimate the intercept and coefficients. The corresponding aligned test predictions are held out for final metrics. The fact that a coefficient can always be fitted is not itself evidence of complementarity; the evidence is the consistent out-of-fold improvement over XGBoost, the positive mRNA-LM coefficient across folds, and the incomplete residual correlation.

Permutation controls independently permute training and validation labels within each fold and seed while leaving the test labels unchanged. TreeSHAP uses XGBoost contribution outputs for the fold-specific trained models. Feature-family ablation removes or isolates feature families and retrains XGBoost under the same split and training protocol.

## Figures and tables

Regenerate all active figures from frozen outputs:

```bash
python scripts/make_all_figures.py
```

Regenerate Supplementary Tables S1-S9 after all required processed inputs are installed:

```bash
python scripts/tables/generate_supplementary_tables.py
```

The committed workbooks are compound tables with logically separate sheets. This avoids sparse rectangular CSV exports in which many columns are structurally empty.

## Validation checklist

```bash
python scripts/check_setup.py
python -m compileall -q scripts
python scripts/make_all_figures.py
```

Before release, scan for machine-specific paths:

```bash
python scripts/sanitize_release_metadata.py
```

Add `--write` only when the report identifies metadata strings that should be replaced. The script does not inspect or modify binary workbooks; those should be checked separately.
