# Reproduction coverage

The tested default workflow is saved-prediction statistical/figure reproduction,
not a new experiment. Bootstrap uncertainty is conditional on fixed predictions.

| Evidence | Point estimates | Bootstrap95% CIs | Status |
|---|---|---|---|
| Table1 human benchmark and primary paired human differences | Recomputed five-fold mean/population SD; pooled OOF separately | Recomputed for registered pooled comparisons | LEVEL2 for supplied models; do not reinterpret fold SD as CI |
| Table2 five-model primary external human | Recomputed | Recomputed5,000 paired draws | LEVEL2 |
| Table3A human Common pooled OOF | Recomputed | Recomputed unified supplemental intervals | LEVEL2 |
| Table3B mouse and model contrasts | Recomputed | Recomputed5,000 paired draws | LEVEL2 |
| S1 full24-system feature-group/control grid | Display from original stored summaries; supplied eight-model OOF subset independently checked | Not applicable | Full grid LEVEL1; no retraining |
| S2 fusion coefficients | Stored coefficients checked, not refitted | Not applicable | LEVEL1 |
| S3 Balanced/No-QN sensitivities | Saved targets/predictions and reference statistics retained | Stored intervals | LEVEL1 in default entrypoint |
| S4 ortholog metrics/deltas/heterogeneity and LM/XGB complementarity | Recomputed | Recomputed original independent-within-stratum bootstrap for gains | LEVEL2 for these quantities; context/target-concordance diagnostics remain LEVEL1 |
| S5 primary distillation | Recomputed45 fold-seed-arm metrics and matched differences | Recomputed original5,000 paired draws and practical GO/STOP rules | LEVEL2 |
| S6 secondary distillation and calibration | Recomputed | Recomputed where formally defined | LEVEL2 |
| S7 all27 mouse measurements | Recomputed without omitting negative/uncertain cases | Recomputed5,000 draws per measurement, original seeds/order | LEVEL2 |
| S8 all16 human contexts | Recomputed | Recomputed5,000 draws per context, original seeds/order | LEVEL2 |
| S9–S10 unified human/external comparisons | Recomputed | Recomputed and frozen arrays checked | LEVEL2 |
| S11 coverage descriptions and internal OOF by coverage | Recomputed | No new intervals added | LEVEL2 for recorded point/descriptive statistics |
| S12 source independence/accession crosswalks | Curated evidence preserved | Not a metric | LEVEL1 |
| S13 organoid primary five-model evaluation | Reconstructed from published processed controls and fixed OOF predictions, n=4,890 | Recomputed 5,000 paired draws; every draw independently checked | LEVEL2; single organoid context |
| S14 organoid common-target, fold and coverage diagnostics | Recomputed on n=4,801 for three targets, original five folds for n=4,890, and all n=12,981 for coverage | Recomputed 5,000 paired common-target draws; no new fold/coverage intervals | LEVEL2 for recorded statistics; original peak calling and decay fitting not rerun |
| Figures 1–4 and S1–S2 | Renderers rerun with complete saved inputs; source-cell/hash QA | Annotations use frozen results, not fresh selection | Display reproduction; does not imply new SHAP/training |
| SHAP, ablation/permutation, Common-Full sensitivity | Saved evidence retained/redrawn where applicable | No new fitting or explanations | LEVEL1 |
| External-PC1 processed-matrix construction | Primary, context-balanced and no-QN target values rebuilt and matched exactly; Primary direction, 13,630 genes, 11,381 matched genes and specified model correlations checked; all 14 LODO gene counts and summary statistics checked | Not recalculated by target construction | Separate fresh R environment; not part of the default Python run; no original per-gene LODO reference comparison |
| Original training, pretrained embeddings and weight inference | Scientific implementations/configurations retained | Not executed | LEVEL3 not provided |

The default rebuilds all three XLSX workbooks from canonical sources and checks
5,311 cells in the reader-facing and source tables. The bundled ESM PDF was re-exported
from its editable source after adding Figure S2 and updating the Figure 2b,c
cross-reference. All existing tables, equations, references and Figure S1 were
preserved, as were the approved S6.4 reproduction scope and S7 organoid methods.
The statistical workflow does not regenerate or modify the ESM PDF. The three
bundled XLSX workbooks retain their existing values and structure.

## Verification and figure contract

`scripts/reproduce.py` fails on input hashes, sample/ID misalignment or numerical
mismatch. Main-table checks compare69 numeric cells with their manuscript
display strings. Independent estimand checks, reference-array comparisons and
scoped check JSONs separate point recomputation from interval recomputation and
saved-reference verification. The original bootstrap seeds, gene order, paired
sampling and percentile definitions are preserved.

Figures use the author-approved panel composition, full gene counts, full
measurement order, colors, scales and annotations. Figure 2b,c presents the saved
three-target comparisons on 4,801 common genes; Figure S2 contains the three
relocated external prediction scatters on 11,381 genes. The Python-only renderer
emits 600-dpi PNG and vector PDF/SVG with print-size checks. No downsampling or
new performance calculation is introduced. Rendering depends on installed fonts;
data equivalence is tested independently of byte/pixel identity. Display-only
updates are checked separately from the previous full statistical acceptance;
they do not imply another execution of all bootstrap analyses.

Acceptance used a freshly installed locked Python environment, cleared process
environment, offline execution, a deny-read/write Python audit hook with a
negative control, and native file-open tracing. This is a filesystem-access test,
not a claim of a functioning container or an adversarial security sandbox.

The separate R check used the recorded versions and the original preprocessCore
source build with threading disabled. Primary, context-balanced and no-QN
target values matched exactly at saved CSV precision. All 14 LODO targets were
generated and checked at the gene-count and summary-statistic level, not against
unbundled original per-gene references. Rebuilding targets does not itself
reproduce the Balanced/No-QN model-performance or target-concordance intervals.
See `environments/r/README.md` for the required source-build step and the tested
environment scope.
