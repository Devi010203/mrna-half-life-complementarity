# Supplementary Table S1-S9 release audit

The compound workbooks in this directory were generated from the frozen numeric outputs under `results/frozen/`. Each logically distinct table occupies its own worksheet, avoiding structurally empty columns caused by concatenating heterogeneous analyses into a single flat CSV.

| Table | Workbook | Main content | Status |
|---|---|---|---|
| S1 | `supplementary_table_s1_data_inclusion_target_fold_overview.xlsx` | Data inclusion, target scale, availability, missingness, and fold sizes | Complete |
| S2 | `supplementary_table_s2_identifier_split_overlap_audit.xlsx` | ENSID uniqueness and cross-fold overlap audit | Complete at ENSID level |
| S3 | `supplementary_table_s3_feature_group_definitions_dimensions.xlsx` | Eight feature-group definitions and dimensions | Complete |
| S4 | `supplementary_table_s4_complete_fold_level_performance_metrics.xlsx` | Fold-level and summary metrics for all model classes and fusion | Complete |
| S5 | `supplementary_table_s5_prediction_fusion_complementarity_diagnostics.xlsx` | Prediction/residual correlations, fusion coefficients, and paired delta R2 | Complete |
| S6 | `supplementary_table_s6_shap_feature_family_interpretation.xlsx` | Feature SHAP, family aggregation, normalized summaries, and ablation | Complete |
| S7 | `supplementary_table_s7_fusion_gain_sample_profile_case_studies.xlsx` | Fusion-gain groups, comparisons, sample profiles, and case studies | Complete |
| S8 | `supplementary_table_s8_motif_kmer_region_composition_checks.xlsx` | 3-mer, 4-mer, motif, region-length, and GC checks | Complete |
| S9 | `supplementary_table_s9_permutation_negative_control_metrics.xlsx` | Fold-, seed-, and overall permutation-control metrics | Complete |

## Source mapping

| Table | Frozen source directories | Generation script |
|---|---|---|
| S1-S3 | `results/frozen/xgboost/`, `data/processed/human/splits/` | `scripts/tables/generate_supplementary_tables.py` |
| S4 | `results/frozen/{mrnalm_agghead,mrnalm_seqhead,ridge,elasticnet,xgboost,fusion}/` | same |
| S5 | `results/frozen/fusion/` | same |
| S6 | `results/frozen/{shap,ablation}/` | same |
| S7 | `results/frozen/{fusion_gain,case_studies}/` | same |
| S8 | `results/frozen/motif/` | same |
| S9 | `results/frozen/permutation/` | same |

## Value checks

The checks in `supplementary_table_s1_s9_value_checks.csv` found no conflicts with the frozen manuscript values. The workbooks were also inspected for formula errors after removal of machine-specific source paths.

## Scope limitations

The identifier audit supports no detected ENSID-level cross-fold duplicates. It does not exclude biological relatedness among different isoforms when a separate gene-level or isoform-family identifier is unavailable. SHAP, feature-family ablation, motif/k-mer, region-composition, and case-study results describe predictive associations or hypothesis-generating patterns, not causal mechanisms.
