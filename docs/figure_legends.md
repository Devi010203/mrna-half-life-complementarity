# Figure 2. Human prediction performance and complementarity across measurement targets

**a,** External-PC1 versus the original Saluki human stability target for 11,381
matched genes; each point represents one gene. External-PC1 was constructed from
62 samples, 14 datasets and 11 cell lines, retaining 13,630 genes before matching.
The annotation gives the saved Spearman correlation. **b,** Spearman correlations
of five frozen human out-of-fold (OOF) predictors with the original Saluki target,
External-PC1 and MSK107Li organoid shCtrl SLAM-seq half-lives on the same 4,801
genes. Points show correlations and vertical bars show percentile 95% confidence
intervals from 5,000 gene-bootstrap resamples, paired across models and targets.
Small horizontal offsets separate the intervals. Open markers identify the Common branch; Common-XGB denotes
Common-Compact XGB. **c,** Bio-Fusion minus Bio-XGB Spearman differences on the
same 4,801 genes, with paired 95% bootstrap intervals and a zero reference line.
These intervals were calculated from within-resample differences, not by
subtracting marginal intervals. **d,** Saved Spearman estimates for LM, Bio-XGB
and Bio-Fusion across 16 RNAdecayCafe dataset–cell-line contexts, with matched
gene counts. Grey lines connect models within a context; small vertical offsets
separate markers. Daggers identify the two K562 contexts whose paired fusion-gain
intervals include zero (Table S8). Gene-bootstrap intervals condition on fixed
predictions and targets; contexts are not independent biological replicates.
The 4,890-gene primary organoid evaluation is reported separately in Table S13;
the common-target analysis is detailed in Supplementary Methods S7 and Table
S14a,c. Cross-target differences do not isolate organoid biology or assay effects.

## Supplementary Figure S2. Gene-level predictions against External-PC1

a–c, External-PC1 versus frozen human OOF predictions from LM (a), Bio-XGB (b) and Bio-Fusion (c) for the same 11,381 matched genes. All genes are shown, and the three panels share identical axis limits. Annotations give the saved Spearman correlations. Blue, orange and purple identify the respective models. Model-performance estimates and paired contrasts are reported in Table 2; target concordance and context-specific comparisons are shown in Figure 2a,d.
