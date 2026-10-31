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
stops the workflow. The full local validation took about 9 minutes on two CPU
workers; runtime varies with hardware. Allow at least 1 GB of working disk space
in addition to the Python environment. No GPU or R installation is required.

The default includes 5,000 paired bootstrap draws for human/external comparisons,
mouse/ortholog analyses, all 27 mouse measurements, all 16 human contexts, and
the matched distillation analyses. It generates Tables 1–3, three supplementary
workbooks, and Figures 1–4/S1 in PNG, PDF and SVG. The bundled approved PNG/PDF
files remain unchanged. Arial is used when installed; otherwise Matplotlib's
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
| `figures/` | Approved Figure 1–4/S1 PNG and PDF files |
| `supplementary/` | ESM PDF and three multi-sheet XLSX workbooks for readers |
| `training/` | Optional scientific implementations/configurations; separate Python 3.11 uv lock, missing resources documented |
| `docs/` | Data dictionary, estimands, reproduction scope and per-file source/license manifest |
| `tests/` | Data/split/implementation checks and opt-in clean-run guard |

CSV/NPZ files are computational inputs/references; XLSX workbooks are the
reader-facing supplement. Repeated per-model OOF copies are not needed: all
human predictions share one table and distillation arm/seed views are generated
in memory. `Common-XGB` is the compact 1,027-feature model, never Common-Full.

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

Authors: Xu Jin, Bihang Cui, Yingchen Mao and Dinglin Zhang. Use `CITATION.cff`
and cite the source datasets when reusing their data. Original software is
licensed under MIT. Author-owned new data and original Figure 1–4/S1
visualization contributions are covered by CC BY 4.0 within the scope described
in [DATA_LICENSES.md](DATA_LICENSES.md); third-party components retain their
source terms. The complete manuscript and ESM_1 scholarly prose are not covered
by this data grant. See the [file manifest](docs/file_manifest.csv) and
[supplementary rights guide](supplementary/README.md) for mixed-file boundaries
and source-specific conditions. No model weights are bundled.
