# Software, data and figure rights

This package has component-specific licenses. Neither MIT nor the data license
is an unconditional license for the complete archive or every cell of a mixed
CSV/XLSX file. The [file manifest](docs/file_manifest.csv) records the source,
checksum and rights scope of each public file. Its scope labels describe
components; they are not whole-file SPDX expressions for mixed materials.

## Licensing scopes

### Original software: MIT

The authors' original software, tests, software configurations and associated
software documentation are licensed under the [MIT license](LICENSE). This
includes original implementations in `scripts/`, `training/` and `tests/`.
The MIT text remains unchanged. It does not relicense dependencies, upstream
code excerpts, measurements, annotations, model weights or fonts.

### Original data contributions: CC BY 4.0

The authors' original data contributions are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
to the extent that the authors hold the relevant rights. Covered contributions
include newly generated model predictions, computed SHAP values, statistical
summaries, saved bootstrap results, fitted fusion coefficients, and original
data curation, selection and table organization. Credit Xu Jin, Bihang Cui,
Yingchen Mao and Dinglin Zhang and this study, using [CITATION.cff](CITATION.cff),
and retain the applicable source-dataset attribution.

When reusing covered material, retain attribution and the license link and
indicate modifications. The [legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)
governs reuse. This grant covers only rights held by the authors: it does not
replace third-party terms, license the underlying source data on others' behalf,
or add restrictions to public-domain data, identifiers or unprotected facts.

### Original figures

The authors' original visualization contributions to Figures 1–4 and Figure S1
are licensed under CC BY 4.0 within the author-owned scope above. This is not a
blanket grant over third-party components, underlying source data or embedded
fonts, which retain their applicable terms. The figure files themselves are
unchanged.

### Scholarly prose and unbundled resources

The complete manuscript and the scholarly prose of
`supplementary/ESM_1_Supplementary_Information.pdf` are **not automatically covered**
by this data license. Original data, figures and table organization within them
have the scope stated above, with source exceptions retained. Permission to
license the complete scholarly text must be handled separately.

Optional pretrained weights and complete sequence/annotation training inputs
are not bundled. Dependency lockfiles are not redistributions of the installed
dependencies. Those resources retain their distributors' terms; no new grant
over them is made here. No standalone font package is included.

## RNAdecayCafe attribution

Source: Isaac Vock, *RNADecayCafe: a uniformly processed atlas of RNA half-life
estimates across multiple human cell lines*, version 1.1.0,
[Zenodo DOI 10.5281/zenodo.16884513](https://doi.org/10.5281/zenodo.16884513).
The official record identifies the license as **CC BY 4.0**.

RNAdecayCafe measurements, processed matrices, sample metadata and derived
targets retain that source attribution and license. Study changes include
sample/gene selection, context aggregation, identifier mapping, normalization,
PC1/sensitivity-target construction and joins to saved predictions. The authors'
new contributions to these operations use CC BY 4.0 without removing the
source attribution. Retain source notices and identify transformations when
redistributing these materials. The source provider does not endorse this study.

## Saluki source conditions

The [Saluki paper repository](https://github.com/vagarwal87/saluki_paper/tree/3aa4e56a19bbbf87ac9f5f0a251098cf749ad6bc)
is recorded at commit `3aa4e56a19bbbf87ac9f5f0a251098cf749ad6bc`. Its
[Apache-2.0 license notice](docs/licenses/SALUKI_REPOSITORY_LICENSE.txt) is
preserved verbatim. Retain applicable copyright/attribution notices and source
references, identify modifications, and follow applicable NOTICE obligations
under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).

Source measurements, targets, annotations, ortholog mappings and code excerpts
retain their applicable source terms. `data/mouse/measurements.txt.gz` is an
unchanged copy of the 27-measurement compilation. Other files join source
targets/identifiers to study predictions or select/reorganize source material;
the new author data grant does not cover the copied source components. This
repository-level notice is not a representation that every underlying
experiment's rights have been independently cleared. Retain the original study
citations and accession provenance.

The [Saluki article](https://doi.org/10.1186/s13059-022-02811-x) has a different
source-specific rights statement: article content is CC BY 4.0 and data made
available in the article carry its CC0 statement, unless a credit line states
otherwise. Preserve those exceptions and attribution. Material transcribed from
[Table 1](https://link.springer.com/article/10.1186/s13059-022-02811-x/tables/1)
is identified separately in the manifest. Do not infer that all files in the
GitHub repository are CC0, or impose new license restrictions on CC0 material.

## Mixed-file component guide

The file structure can remain intact; rights follow the components, not the
CSV/XLSX extension. These examples identify the important boundaries; detailed
file provenance is in [docs/file_manifest.csv](docs/file_manifest.csv).

| File or component | Author contribution | Source component retained |
|---|---|---|
| `data/human/oof.csv` | Model predictions and study fold assignments | Saluki `target` and source identifiers |
| `data/external/matched.csv` | Model predictions and matching/coverage curation | Saluki `original_target`; RNAdecayCafe-derived `target` and its attribution |
| `data/external/contexts.csv.gz` and sensitivity matches | Predictions, mapping decisions and new processing | RNAdecayCafe-derived context/target values and experiment metadata |
| `data/external/mapping/external_to_saluki_official_mapping.csv` | Crosswalk organization and mapping decisions | Saluki `saluki_human_PC1`; RNAdecayCafe-derived `external_PC1`; source identifiers |
| `data/human/coverage_features.csv` | New coverage/grouping and derived summaries | Saluki target/feature values; RNAdecayCafe provenance for measurement coverage |
| `data/mouse/predictions.csv` and ortholog manifest | `pred_*` and study-relative ortholog groups | Saluki `mouse_HALFLIFE`, source IDs and ortholog correspondence |
| `data/distillation/predictions.csv` | `prediction` and study fold/seed/arm organization | Saluki `y_true` and source identifiers |
| `data/interpretation/shap_values.csv.gz` | `shap_value`, derived ranks and original organization | Upstream `feature_value`, feature definitions and identifiers |
| `training/config/common_compact_features.csv` | Original feature selection/order and grouping | Source feature names/definitions and provenance |
| `data/sources/saluki_human_source_inventory.csv` | Original inventory organization | Saluki article/Table 1 material, with its publication-specific conditions |
| `data/sources/saluki_script_source_crosswalk.csv` | Original crosswalk organization | `assignment` contains Saluki R code excerpts retaining Apache-2.0 |

The three supplementary workbooks and `data/sources/supplementary_display.xlsx`
use the same component-level approach. In particular, ESM_3b's
`Saluki_Human_Sources` sheet carries the article-source conditions, whereas
`Saluki_Script_Crosswalk` contains Apache-2.0 code excerpts. The new author grant
does not replace either. See the [supplementary file and rights guide](supplementary/README.md).
