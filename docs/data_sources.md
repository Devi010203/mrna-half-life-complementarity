# Data dictionary and source attribution

All table paths are repository-relative. `data/input_manifest.json` protects
the statistical inputs and frozen references with SHA256; `file_manifest.csv`
provides per-file checksums, provenance and reuse classification for the public
payload. No runtime request to the original research workspace is needed.

| Canonical file | Rows and key fields | Use |
|---|---|---|
| `data/human/oof.csv` | 12,981 unique `ENSID`; `target`, `fold`, `LM`, `Bio-XGB`, `Bio-Fusion`, `Common-XGB`, `Common-Fusion`, `BASIC`, `Ridge`, `ElasticNet` | Original human score and one held-out prediction per gene/model; gene ID/order preserved |
| `data/external/matched.csv` | 11,381 unique `ENSID`; `target`=External-PC1, `original_target`, same five LM/Bio/Common predictions, original `fold`, coverage columns | Same-gene-set external comparison; no new predictions |
| `data/external/contexts.csv.gz` | 147,612 gene/context rows, all 16 contexts | Context-specific target and saved three-model OOF predictions |
| `data/external/construction/` | 62 retained sample metadata rows, raw processed matrices and coverage | 14 datasets, 11 cell lines; primary consensus contains13,630 target genes before model matching |
| `data/external/mapping/` | Frozen source/current identifier crosswalks and unmatched records | Matching without remapping or identifier rescue |
| `data/mouse/predictions.csv` | 13,744 unique `mouse_ENSID`; `mouse_HALFLIFE`, `ortholog_status`, human ortholog ID, three `pred_*_ensemble` and15 fold-specific columns | Human-trained models; ensemble is the mean of five folds |
| `data/mouse/measurements.txt.gz` | 22,499 source genes, 27 measurements; missing entries retained | Join by the source gene identifier using `measurement_manifest.csv`; formal order/orientation/seed/matched n frozen |
| `data/mouse/ortholog_manifest.csv` | 13,744 rows | `ORTHOLOG_REPRESENTED`=10,200; `NO_CURRENT_1TO1_ORTHOLOG`=3,544 |
| `data/distillation/predictions.csv` | 116,829 rows; `ENSID`, `y_true`, `prediction`, `fold`, `seed`, `arm` | 12,981 genes ×3 seeds ×3 arms; `pairing.json` defines resampling order, protocol defines practical thresholds |
| `data/interpretation/shap_values.csv.gz` | 1,305,900 saved long records,12,981 genes,124-feature union | Not a dense559-feature matrix; all259,620 Top20 gene-feature records retained for Figure1d |
| `data/human/coverage_features.csv` | Full human coverage descriptors | Matched versus unmatched descriptive/internal-OOF comparison |
| `data/sources/` | Curated accession/source crosswalks and exact display cells | S12 attribution and display-only supplement tables; no raw downloaded web caches |

`HALFLIFE` and `target` denote the supplied stability/PC1 scores, not an assertion
that all outcomes are hours. `Common-XGB` always abbreviates Common-Compact XGB
(1,027 common features). Bio-XGB uses BASIC8 plus551 biochemical/regulatory
features. Sample sizes vary by measurement; missing measurements are not imputed
for model evaluation. Duplicate study labels remain separate formal measurements.

RNAdecayCafe: [version1.1.0, DOI10.5281/zenodo.16884513](https://doi.org/10.5281/zenodo.16884513).
Saluki: [source repository](https://github.com/vagarwal87/saluki_paper) and
[associated study](https://doi.org/10.1186/s13059-022-02811-x).
The preserved `data/sources/*provenance.csv` tables give study/sample-level
accessions and distinguish evidence from unresolved overlap. Later publication
dates alone are not proof of experimental independence.

Saved predictions are study outputs; their training inputs and pretrained
weights are not included. `training/config/splits/` retains exact human split
IDs, but does not supply missing sequence/feature matrices or fit models.

## Reuse and component-level rights

Original software/configurations use MIT. Author-owned new predictions,
computed SHAP values, statistical and bootstrap results, fusion coefficients
and original data curation use CC BY 4.0. Author-owned visualization contributions
use CC BY 4.0 within the figure scope and third-party exclusions in
[DATA_LICENSES.md](../DATA_LICENSES.md). This is not a license for all source
components or the complete manuscript/ESM_1 scholarly prose.

RNAdecayCafe material retains Isaac Vock's v1.1.0 CC BY 4.0 attribution, DOI and
transformation history. Saluki repository components retain the recorded
Apache-2.0 notice and applicable source terms; Saluki article material retains
its publication-specific CC BY 4.0/article-data CC0 conditions and exceptions.

In mixed tables, the author's prediction is distinct from the supplied target
or annotation: `original_target`, `mouse_HALFLIFE`, PD4 `y_true` and SHAP
`feature_value` are not newly authored model outputs. Source IDs and public-domain
facts acquire no new restrictions. The `assignment` field of the Saluki script
crosswalk remains an Apache-2.0 code excerpt, even when included in a workbook.
The [manifest](file_manifest.csv) gives per-file provenance and component-scope
notes; the [supplementary guide](../supplementary/README.md) distinguishes the
article-source and code-excerpt sheets of ESM_3b.

## Additional organoid measurements (GSE336417)

`data/organoid/sources/` contains the original GEO processed CSV and the
publisher's Supplementary Table 5 workbook from Jiang et al. (Nature 657,
1053–1063, 2026; doi:10.1038/s41586-026-10890-0), retrieved on 2 November 2026.
Only shCtrl half-lives and fit quality from Table 5e/the corresponding CSV enter
evaluation. Adjacent receipts identify exact URLs, byte sizes and hashes.
Gene symbols (`gene_name`) match the existing human `gene_symbol` exactly;
ENSID and original fold come from `data/human/oof.csv`.

`hl_exp_ctrl` is an estimated half-life in hours, `r2_exp_ctrl` is the control
decay-fit R², and `half_life_gt_12h` is descriptive, never an exclusion flag.
`Organoid` in joined results retains the GEO hours; `log2_half_life` is used
only for Pearson. `Saluki` and `External-PC1` retain their existing target
values and orientation. The 4,890-gene primary and 4,801-gene common-target
manifests are in `data/organoid/frozen/preflight_20261102/audit/`.

Main Figure 2b,c reads `results/organoid/Three_target_models.csv` and
`Three_target_deltas.csv` directly, using the same 4,801 genes. Its CIs are stored
model and paired-contrast intervals, not newly calculated plotting quantities.
Figure 2a and Figure S2 use the 11,381 rows of `data/external/matched.csv`;
Figure 2d retains the frozen order and sample sizes in
`results/external/per_context_metrics.csv`. Their hashes are registered in
`data/figure_sources.json` and checked by the figure entrypoint.

Author-owned new calculations/curation use CC BY 4.0, original code uses MIT,
and copied measurements retain Jiang et al. attribution and source conditions.
Mixed targets/predictions retain Saluki and RNAdecayCafe conditions as applicable.
See [module documentation](organoid.md) and
[source-specific rights](../DATA_LICENSES.md#gse336417-and-jiang-et-al-attribution).
