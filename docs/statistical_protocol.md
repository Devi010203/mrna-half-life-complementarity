# Statistical definitions

The default workflow evaluates fixed, saved predictions. It performs no training,
inference, model selection, coefficient fitting or recalibration.

- Human benchmark: 12,981 genes; five held-out folds (2,600 × 4; 2,581).
  Table 1 uses unweighted five-fold means and population SD (`ddof=0`).
  Pooled OOF metrics and intervals are separate estimands (Table 3A and S9).
- External human: the same 11,381 genes and shared resampling indices for all five
  models. Targets are saved External-PC1, not hours. Original Saluki scores remain
  available. Common-model external comparisons are post hoc.
- Gene bootstrap: 5,000 draws, `numpy.random.default_rng(42)`; duplicate sampled
  genes are re-ranked for Spearman, with average ties. Percentile 95% intervals
  use linear interpolation at 0.025 and 0.975. No draws are discarded or replaced.
- Mouse primary: 13,744 genes, equal five-human-fold ensemble predictions. Model
  differences use paired gene draws, with root seed 42.
- Ortholog strata: represented 10,200, no-current-one-to-one 3,544. The original
  `SeedSequence(42).spawn(2)` streams are preserved, represented first. Independent
  within-stratum resampling supplies the between-stratum gain heterogeneity.
- The 27 mouse measurement and 16 human context orders are frozen. Each uses
  seed 42 plus its zero-based frozen index. All models share the observed genes
  within a measurement/context. Missing measurements are not imputed. Original
  measurement orientation is preserved, never selected against model performance.
- PD4: 5 folds × seeds 42, 2026, 3407 × A/B/C; 116,829 long prediction rows. One
  global gene index vector is shared across all nine seed/arm predictions in each
  bootstrap iteration. Fold identities remain fixed; seeds are not resampled.
  The primary difference averages the 15 paired fold-by-seed R² differences.
  Pooled per-seed and equal-seed-prediction-mean results are secondary.
  Teacher–label practical magnitude is 0.010; biochemical-specific B–C magnitude
  is 0.005 with additional sign, interval and safety rules in the supplied protocol.
- SHAP values were saved from a separate same-protocol interpretation run.
  The stored long table spans 124 feature identities, not the full 559-feature
  matrix. Figure 1d uses the complete Top20 values for all 12,981 genes.

Intervals condition on fixed models, included genes and saved target construction;
they do not measure all training uncertainty or independent study replication.
No formal results are overwritten. Reproduced values are tested against frozen
references, with numerical tolerance reported in every check. Failures are not
resolved by changing reference results.
