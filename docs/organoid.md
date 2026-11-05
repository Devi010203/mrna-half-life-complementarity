# Additional organoid measurement evaluation: GSE336417

This module tests the five existing human OOF predictors against published shCtrl
SLAM-seq half-lives from the MSK107Li patient-derived colorectal cancer
liver-metastasis organoid. It is an evaluation of new measurements on
benchmark-represented genes in one organoid background, not unseen-gene or
multi-patient validation. It does not add organoid measurements to External-PC1.

## Run on CPU

From the repository root, using the same locked environment as the other analyses:

```bash
uv sync --locked
uv run python scripts/organoid/reproduce.py --output outputs/organoid_check
```

The output directory must be new and below `outputs/`. The default
`uv run python scripts/reproduce.py` also executes this module. No additional
dependencies, downloads, GPU, model weights or R are needed. Two implementations
each calculate 5,000 bootstrap draws for the primary set and 5,000 for the
common-target set; neither trains, infers, fits or selects a model. Existing
Python 3.12.3 / NumPy 1.26.4 / pandas 2.1.4 / SciPy 1.11.4 / openpyxl 3.1.2
pins are unchanged. The isolated module took about 81 seconds in the integration
test; runtime depends on hardware.

## Inputs and sources

- `data/organoid/sources/GSE336417_SLAMseq_halflife.csv.gz`: unchanged GEO
  processed table, downloaded 2 November 2026. `gene_name`, `hl_exp_ctrl`
  (hours) and `r2_exp_ctrl` define the control evaluation. Knockdown columns
  are preserved in the original download but never select or label genes here.
- `data/organoid/sources/41586_2026_10890_MOESM7_ESM.xlsx`: the publisher's
  original Supplementary Table 5 workbook. Only sheet `Supplementary Table 5e`
  is checked. This small source workbook is retained intact because Table 5e
  is distributed with other sheets; no other experiments are analysed.
- Adjacent `.receipt.json` files identify exact source URLs, retrieval times,
  sizes and SHA256 hashes. No read, BAM or SRA files are included.
- `data/human/oof.csv`, `data/external/matched.csv` and
  `data/human/coverage_features.csv`: the same frozen inputs used by the
  preceding analyses. The new module does not modify or replace them.

Source publication: Jiang Q, Raghavan MS, Rodriguez AM, et al. *ZFP36L2
orchestrates stress-adaptive plasticity in regeneration and cancer*. Nature
657, 1053–1063 (2026). [doi:10.1038/s41586-026-10890-0](https://doi.org/10.1038/s41586-026-10890-0).
Accession: [GSE336417](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE336417).
See [data rights](../DATA_LICENSES.md#gse336417-and-jiang-et-al-attribution).

## Fixed design

The published table contains 5,888 unique symbols. Finite positive half-lives
and valid control decay-fit R² strictly above 0.6 retain 5,369. Exact,
case-sensitive, unambiguous symbol matching to the existing human OOF table
retains 4,890 genes; 479 quality-filtered symbols have no exact match. No aliases
are rescued and all five models share the same set. Of these, 4,801 also have
External-PC1. All 89 remaining primary genes and all 251 primary half-lives
above 12 h remain included.

The sole primary contrast is Bio-Fusion minus Bio-XGB Spearman. Spearman uses
hours; secondary Pearson uses log2 hours. Bootstrap resamples genes, paired
across models (and across targets within the common set), with 5,000 draws,
seed 42, within-draw average ranks, and linear-interpolated percentile intervals.
Neither undefined-draw replacement nor outcome-dependent filtering is allowed.
Bio-XGB minus LM and the original five-fold direction diagnostic were added in
v2 before any performance calculation. Secondary comparisons are descriptive.

The reader-facing methods are S7. Tables S13–S14 are in the extended ESM_2
workbook and the supplement PDF. All original S1–S12 data cells are retained.
Main Figure 2b,c visualizes the 4,801-gene common-target model correlations and
paired fusion gains; it does not replace the 4,890-gene primary evaluation.
Supplementary Figure S2 instead shows the earlier 11,381-gene External-PC1
prediction scatters.

## Outputs and independent checks

`results/organoid/` contains immutable full-precision references, the original
formal-results workbook and saved bootstrap arrays. A run writes new files only
to the requested output directory:

- `Primary_models.csv`, `Primary_deltas.csv`: five-model performance and paired
  contrasts on 4,890 genes; primary ΔSpearman 0.017859, CI [0.012561, 0.023250].
- `Three_target_models.csv`, `Target_concordance.csv`,
  `Three_target_deltas.csv`, `Gain_shift_descriptive.csv`: the 4,801-gene
  comparison with original Saluki, External-PC1 and organoid targets.
- `Fold_diagnostics.csv`: descriptive original fold 0–4 correlations and
  differences (all five positive); no fold reweighting or interval calculation.
- `Coverage_*.csv`: all prespecified coverage groups, original-target OOF
  correlations and 77 feature summaries. Missing coverage remains missing.
- `primary_evaluation_data.csv`, `three_target_evaluation_data.csv`: ordered
  gene-level inputs with fixed predictions, targets, folds and quality flags.
- `*_bootstrap.npz`: complete draw arrays and point matrices.
- `independent_checks/INDEPENDENT_STATISTICAL_RECHECK.json`: independent
  reconstruction, re-ranking and recalculation of all 10,000 draws using a
  different statistical implementation in the same environment.
- `REPRODUCTION_REPORT.json`: comparison against frozen CSV and array
  references, source correspondence, cohort sizes and draw-index hashes.

Absolute numerical tolerance is 5×10⁻¹²; the reference values are never edited
to obtain agreement. Historical and reproduced XLSX values are checked against
their CSV sources. `scripts/tables.py` extends the existing display workbook
from a separate organoid display source; the original workbook specification
and original display data remain unchanged.

## Frozen records and relocation

`data/organoid/frozen/` preserves the original Chinese v1/v2 protocols, the
pre-performance freeze, source/QC workbook, gene manifests, formal report and
independent acceptance record byte-for-byte. Their historical execution paths,
stop instructions and former Git HEAD describe the 2 November evaluation;
they are archival records, not commands or current repository state. The later
integration does not change their analysis rules.

The original preflight index also records acquisition scripts and GEO SOFT
files that are not redistributed here. It remains a historical record, not a
claim that every acquisition artifact is bundled. Sample/accession evidence
is retained in `SOURCE_QC_AND_MAPPING_AUDIT.xlsx`. Runtime checks validate all
five entries of the original v2 freeze and all inputs/references in the new
portable `data/organoid/input_manifest.json`. The original statistical scripts
are also retained as non-entrypoint snapshots in the frozen evaluation directory;
their portable copies change only paths,
output handling and integrity checks. Original script hashes are in the frozen
execution records. No statistical helpers or estimators were changed.

## Interpretation limits

R²_exp > 0.6 is an adopted decay-fit criterion, not a measure of half-life
precision. Peak calling, raw-read processing and decay fitting are not repeated.
Intervals condition on fixed models, target estimates and observed genes, not
training, fitting or inter-patient uncertainty. The 251 estimates beyond 12 h
retain extrapolation limits. Fold consistency is not biological replication.
Source inventories identified no overlap; reads were not deduplicated.
Coverage from the earlier external-human compendium is not organoid read depth.
All target-gain shifts and secondary contrasts retain their descriptive status.
