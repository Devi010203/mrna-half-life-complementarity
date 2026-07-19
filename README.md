# Pretrained mRNA predictions complement regulatory annotations for half-life prediction

This repository contains the reproducibility code, frozen analysis outputs, figures, and supplementary tables for a systematic comparison of sequence-only and annotation-augmented predictors of human mRNA half-life.

The central analysis uses the Saluki-released human mRNA half-life resource. A pretrained mRNA-LM predictor supplies sequence-only predictions, while Ridge, ElasticNet, and XGBoost use Saluki-derived transcript-architecture, sequence-statistical, biochemical, and regulatory feature groups. Prediction-level linear fusion tests whether the two prediction sources contain non-redundant information.

This is a comparison and complementarity analysis, not a new Saluki dataset release or a claim that the reported models replace Saluki. SHAP, feature-family ablation, motif/k-mer, and case-study analyses are predictive or hypothesis-generating and do not establish causal mechanisms.

## Repository layout

```text
.
|-- config/                  # Environment-variable examples
|-- data/                    # Data placement documentation and public split file
|-- models/                  # Checkpoint placement documentation
|-- results/
|   |-- frozen/              # Frozen numeric inputs used by figures and tables
|   `-- runs/                # Outputs from new local or cluster runs (ignored by Git)
|-- scripts/
|   |-- models/              # mRNA-LM and tabular model training
|   |-- analysis/            # Fusion, controls, SHAP, ablation, and profiling
|   |-- figures/             # Main and supplementary figure scripts
|   `-- tables/              # Supplementary table generation
|-- figures/                 # Publication figures and plotted-data CSV files
|-- tables/                  # Supplementary Table S1-S9 workbooks
|-- slurm/                   # Portable SLURM submission templates
`-- docs/                    # Reproducibility and data-dictionary notes
```

## Reproducibility levels

### 1. Reproduce figures from frozen outputs

The compact files under `results/frozen/` are sufficient to regenerate the main and supplementary figures without redistributing the Saluki feature matrices or pretrained checkpoints.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/check_setup.py
python scripts/make_all_figures.py
```

Each figure is written to `figures/` as PDF, PNG, and SVG, together with a `*_plotted_data.csv` audit file.

The supplied Supplementary Table S1-S9 workbooks in `tables/` are the release-ready tables generated from the frozen analysis. Regenerating all workbook sheets from raw inputs additionally requires the processed human target table described below.

### 2. Re-run models and analyses

Full model reproduction requires data and model files that are not redistributed here. Obtain them from their original resources and place them according to [data/README.md](data/README.md) and [models/README.md](models/README.md). No script contains a user-specific workstation or cluster path; paths default to locations inside this repository and can be overridden with environment variables.

Install the deep-learning dependencies when training mRNA-LM:

```bash
python -m pip install -e ".[mrnalm]"
```

The high-level execution order is:

1. Train the AggHead and optional SeqHead mRNA-LM predictors across five folds.
2. Merge fold-specific mRNA-LM validation and test predictions.
3. Train Ridge, ElasticNet, and XGBoost on the eight feature groups.
4. Fit prediction-level linear fusion using validation predictions only.
5. Run permutation controls, TreeSHAP, feature-family ablation, fusion-gain profiling, motif/k-mer checks, and case-study summaries.
6. Regenerate supplementary tables and figures.

Detailed commands and dependencies are provided in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Cluster users can adapt the templates under `slurm/` without editing Python source files.

Run the frozen-result regression tests with the standard library test runner:

```bash
python -m unittest discover -s tests -v
```

## Primary entry points

```bash
# Tabular models
python scripts/models/train_ridge.py
python scripts/models/train_xgboost.py

# Sequence-only predictor
python scripts/models/train_mrnalm_agghead_lora.py
python scripts/analysis/merge_mrnalm_predictions.py

# Fusion and interpretation
python scripts/analysis/late_fusion.py
python scripts/analysis/permutation_control.py
python scripts/analysis/shap_interpretability.py
python scripts/analysis/feature_family_ablation.py
python scripts/analysis/fusion_gain_profile.py
python scripts/analysis/motif_kmer_checks.py
python scripts/analysis/case_studies.py
python scripts/analysis/helped_hurt_shap.py
```

Within each model run, preprocessing, model selection, calibration, and fusion fitting are confined to training/validation data. Held-out test-fold predictions are used only for final evaluation.

## Frozen reference results

The included frozen summaries support the following five-fold human results:

| Analysis | Test R2, mean +/- SD |
|---|---:|
| AggHead mRNA-LM, sequence only | 0.4436 +/- 0.0112 |
| XGBoost, full regulatory/biochemical + BASIC | 0.6766 +/- 0.0057 |
| Best ElasticNet feature group | 0.6374 +/- 0.0141 |
| Linear prediction-level fusion | 0.7008 +/- 0.0080 |
| Permuted-label XGBoost control | -0.0030 +/- 0.0072 |
| Permuted-label Ridge control | -0.0095 +/- 0.0101 |

The same 12,981 transcripts and five outer folds are used in the current human analysis. Fusion remains an explicitly paired prediction experiment: coefficients are fitted on fold-aligned validation predictions and evaluated on the corresponding held-out test predictions.

## Data and checkpoint availability

The repository does not redistribute Saluki feature tables, processed sequence files, or pretrained mRNA-LM checkpoints. Their licenses and access conditions remain those of the original sources. The fold assignment used by the analysis is included at `data/processed/human/splits/cv_k5_seed42.csv`.

Before public release, update the manuscript citation, dataset URLs, checkpoint URLs, authors, and repository DOI in `CITATION.cff.template` and the associated manuscript Data Availability statement.

## License

The repository code is released under the [MIT License](LICENSE). Dataset files, pretrained checkpoints, and third-party resources remain subject to their original licenses and terms of use.
