# mRNA stability: complementarity and transfer limits

Code and saved predictions accompanying **Complementarity and transfer limits of
sequence- and feature-based mRNA stability prediction in human and mouse**.

This repository reproduces statistics, main tables and approved figures from
saved gene-level predictions. The default workflow is CPU-only and performs no
model training, weight-based inference, fusion fitting or model selection.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from
this directory:

```bash
uv sync --locked
uv run python scripts/reproduce.py
```

The locked analysis environment uses Python 3.12. Each run creates a new
`outputs/reproduction_<timestamp>/` directory. No input or reference file is
overwritten. `REPRODUCTION_REPORT.json` records executed commands, exit status,
recalculated analyses and display-only results. A failed numerical comparison
stops the workflow. The complete validation including GSE336417 took about 11 minutes on two CPU
workers; runtime varies with hardware. Allow at least 1 GB of working disk space
in addition to the Python environment. No GPU or R installation is required.

The default includes 5,000 paired bootstrap draws for human/external comparisons,
mouse/ortholog analyses, all 27 mouse measurements, all 16 human contexts, and
the matched distillation analyses, and the additional GSE336417 organoid
evaluation (including an independent recalculation of both 5,000-draw sets).
It generates Tables 1–3, three supplementary
workbooks, and Figures 1–4/S1–S2 in PNG, PDF and SVG. This workflow never overwrites
the bundled approved figures. Arial is used when installed; otherwise Matplotlib's
sans-serif fallback may change text layout, not data or statistics. Fonts are
not redistributed.

```bash
# Quick input, dependency-closure and frozen split checks (no model execution):
uv run python tests/test_release.py
# Optional faster run: context/measurement points only; their CIs remain LEVEL1.
uv run python scripts/reproduce.py --core-only
```

## Contents

| Directory | Purpose |
|---|---|
| `data/` | Canonical human (12,981 genes), external (11,381), mouse (13,744), 27 measurement targets, all saved SHAP values and 116,829 distillation records |
| `scripts/` | Statistical tests, bootstrap, main tables, approved figure renderer and optional R target construction |
| `results/` | Frozen numerical references used to check recalculation |
| `figures/` | Approved Figures 1–4/S1–S2; PNG/PDF plus SVG for revised Figure 2 and supplementary figures |
| `supplementary/` | ESM PDF and three multi-sheet XLSX workbooks for readers |
| `training/` | Optional scientific implementations/configurations; separate Python 3.11 uv lock, missing resources documented |
| `docs/` | Data dictionary, estimands, reproduction scope and per-file source/license manifest |
| `tests/` | Data/split/implementation checks and opt-in clean-run guard |

CSV/NPZ files are computational inputs/references; XLSX workbooks are the
reader-facing supplement. Repeated per-model OOF copies are not needed: all
human predictions share one table and distillation arm/seed views are generated
in memory. `Common-XGB` is the compact 1,027-feature model, never Common-Full.

## Additional organoid evaluation

[GSE336417](docs/organoid.md) adds a single patient-derived organoid shCtrl
measurement test using the same five frozen human OOF predictors. The primary
set has 4,890 genes; 4,801 share all three human targets. Supplementary Methods
S7 and Tables S13–S14 describe the evaluation, common-target comparison and
coverage diagnostics. Run it separately with:

```bash
uv run python scripts/organoid/reproduce.py --output outputs/organoid_check
```

The module uses the published processed tables, not sequencing reads. No models
are retrained, inferred, calibrated or selected; earlier inputs and results
remain unchanged. It retains all 251 half-life estimates above 12 h. This is
new-measurement evaluation in one organoid line, not unseen-gene or multi-patient
validation. Source provenance, limitations and output definitions are in the
[module guide](docs/organoid.md).

## Supplementary materials and figures

The [supplementary directory](supplementary/) contains the complete ESM_1 PDF,
three multi-sheet XLSX workbooks, editable DOCX/Markdown sources and a file guide.
It includes Supplementary Methods S1–S7, Note S1, Tables S1–S14 and Figures S1–S2.
The bundled [ESM_1 PDF](supplementary/ESM_1_Supplementary_Information.pdf) is the
author-approved submission copy dated 7 November 2026. Editable companions and
export notes are provided in [supplementary/source](supplementary/source/).

Figure 2 combines target agreement (11,381 genes), five-model performance and
paired fusion gains across three targets (the same 4,801 genes), and all 16
external contexts. Panel b shows points and 95% CIs without between-target lines.
Figure S2 contains the three 11,381-gene External-PC1 prediction scatters formerly
shown in Figure 2b–d. The 4,890-gene primary organoid evaluation remains a distinct
analysis in Table S13. See the [figure legends](docs/figure_legends.md).

To redraw figures without recalculating statistics:

```bash
uv run python scripts/figures.py --output-dir outputs/figures_redrawn
# Only the two supplementary figures:
uv run python scripts/figures.py --figures supplementary --output-dir outputs/supplement_figures
```

PNG files are exported at 600 dpi; PDF and SVG retain vector content. Existing
outputs are protected, so choose a new output directory for each run.

## Reproduction scope

**LEVEL2** means actual point estimates and the stated intervals were recomputed
from saved per-gene targets/predictions and compared with frozen references.
**LEVEL1** means a saved result was read, checked or redrawn, without regenerating
the underlying analysis. Training/inference (**LEVEL3**) is not self-contained.
See [coverage and limitations](docs/reproduction_scope.md),
[statistical definitions](docs/statistical_protocol.md) and
[data sources/dictionary](docs/data_sources.md).

SHAP uses all saved values (124-feature union, not a complete 559-feature matrix);
Figure 1d displays Top20. These attributions come from a separate interpretation
model, not the benchmark checkpoints. Negative and uncertain results, including
both Simon MEF measurements and Koszinowski, remain in the release.

External-PC1 construction from the saved processed raw matrices is an optional
[separate R workflow](environments/r/README.md). Primary, context-balanced and
no-QN targets were rebuilt in a fresh R environment and matched the frozen CSV
values exactly. The Primary target's 13,630 genes, direction, 11,381 matched
genes and specified model correlations were verified. All 14 leave-one-dataset-out
targets were generated and checked against stored gene counts and summary
statistics; original per-gene LODO references were not bundled for value-by-value
comparison. This is processed-matrix reconstruction, not sequencing-read
reanalysis, and is separate from the default Python workflow. It does not rerun
Balanced/No-QN model-performance or target-concordance intervals. Restore the
recorded preprocessCore source-build option before running the R scripts.

## Citation and licensing

Authors: Xu Jin, Bihang Cui, Ruxin Yan, Yingchen Mao and Dinglin Zhang.
Use `CITATION.cff` and cite the source datasets when reusing their data. Original software is
licensed under MIT. Author-owned new data and original Figure 1–4/S1–S2
visualization contributions are covered by CC BY 4.0 within the scope described
in [DATA_LICENSES.md](DATA_LICENSES.md); third-party components retain their
source terms. The complete manuscript and ESM_1 scholarly prose are not covered
by this data grant. See the [file manifest](docs/file_manifest.csv) and
[supplementary rights guide](supplementary/README.md) for mixed-file boundaries
and source-specific conditions. No model weights are bundled.
