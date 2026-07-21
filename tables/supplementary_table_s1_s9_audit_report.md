# Supplementary Table S1-S9 Audit Report

This report maps existing project outputs onto the current manuscript's Supplementary Table S1-S9 order and records value checks against frozen manuscript statements.

## Current File to Target Table Mapping

| Current/source file | Target table | Use |
|---|---:|---|
| `old supplementary_table_s0_feature_group_definitions.csv` | S3 | feature group definitions and dimensions |
| `old supplementary_table_s1_complete_fold_metrics.csv` | S4 | complete fold-level metrics |
| `old supplementary_table_s2_permutation_negative_control.csv` | S9 | permutation negative controls |
| `old supplementary_table_s3_motif_kmer_region_checks.csv` | S8 | motif/k-mer and region checks |
| `old supplementary_table_s4_fusion_gain_case_studies.csv` | S7 | fusion-gain profile and case studies |
| `old supplementary_table_s5_shap_family_ablation.csv` | S6 | SHAP and feature-family ablation |
| `old supplementary_table_s6_late_fusion_details.csv` | S5 | late-fusion coefficients and delta metrics |
| `old supplementary_table_s7_prediction_residual_correlations.csv` | S5 | prediction/residual correlations |
| `old supplementary_table_s9_gene_fold_overlap_check.csv` | S2 | identifier split overlap audit |
| `normalized.csv.gz and feature_group_manifest.json` | S1 | data inclusion, target, fold and missingness overview |

## Final Outputs

| Table | Output path | Rows | Columns | Status | Required content | Missing fields | Repair needed |
|---|---|---:|---:|---|---|---|---|
| S1 | `.\paper\tables\supplementary_table_s1_data_inclusion_target_fold_overview.xlsx` | 19 | 22 | generated and complete | complete | none | none |
| S2 | `.\paper\tables\supplementary_table_s2_identifier_split_overlap_audit.xlsx` | 3 | 11 | generated and complete | complete | none for ENSID-level audit; isoform-level relatedness cannot be audited without transcript isoform IDs | none unless transcript isoform identifiers become available |
| S3 | `.\paper\tables\supplementary_table_s3_feature_group_definitions_dimensions.xlsx` | 8 | 13 | generated and complete | complete | none | none |
| S4 | `.\paper\tables\supplementary_table_s4_complete_fold_level_performance_metrics.xlsx` | 330 | 35 | generated and complete | complete | none | none |
| S5 | `.\paper\tables\supplementary_table_s5_prediction_fusion_complementarity_diagnostics.xlsx` | 43 | 36 | generated and complete | complete | none | none |
| S6 | `.\paper\tables\supplementary_table_s6_shap_feature_family_interpretation.xlsx` | 656 | 61 | generated and complete | complete | none | none |
| S7 | `.\paper\tables\supplementary_table_s7_fusion_gain_sample_profile_case_studies.xlsx` | 13064 | 127 | generated and complete | complete | none | none |
| S8 | `.\paper\tables\supplementary_table_s8_motif_kmer_region_composition_checks.xlsx` | 987 | 19 | generated and complete | complete | none | none |
| S9 | `.\paper\tables\supplementary_table_s9_permutation_negative_control_metrics.xlsx` | 70 | 23 | generated and complete | complete | none | none |

## Data Sources and Generation Script

| Table | Source status | Source files | Generation script |
|---|---|---|---|
| S1 | needed new organized table from existing target, split, and feature-manifest files | `.\temp\processed\codex_1\human\normalized.csv.gz; .\temp\processed\codex_1\human\splits\cv_k5_seed42.csv; .\result\62340\xgb_saluki_regulatory_features_human_clean_codex1\xgb_saluki_regulatory_features_human_clean_codex1_20260615_125610\feature_group_manifest.json` | `.\paper\table_scripts\generate_supplementary_tables_s1_s9.py` |
| S2 | generated from existing split files; replaces old S9 gene-fold overlap table | `.\temp\processed\codex_1\human\splits\cv_k5_seed42.csv; .\script\cv_k5_seed42.csv` | `.\paper\table_scripts\generate_supplementary_tables_s1_s9.py` |
| S3 | renamed and reorganized from old S0 plus current XGBoost feature manifest | `.\result\62340\xgb_saluki_regulatory_features_human_clean_codex1\xgb_saluki_regulatory_features_human_clean_codex1_20260615_125610\feature_group_manifest.json` | `.\paper\table_scripts\generate_supplementary_tables_s1_s9.py` |
| S4 | renamed and reorganized from old S1 plus raw model fold metrics | `.\result\62339\ridge_saluki_genetic_biochemical_human_clean_codex1\ridge_saluki_genetic_biochemical_human_clean_codex1_20260615_125539\fold_metrics.csv; .\result\elasticnet_collect_63111\elasticnet_saluki_regulatory_features_human\elasticnet_saluki_regulatory_features_human_clean_codex1_full8_array\elasticnet_fold_metrics_merged.csv; .\result\62340\xgb_saluki_regulatory_features_human_clean_codex1\xgb_saluki_regulatory_features_human_clean_codex1_20260615_125610\fold_metrics.csv; .\result\2773\mrnalm_triplet_human\mrnalm_triplet_human_matched_lora_20260425_145311; .\result\late_fusion_mrnalm_xgb\late_fusion_basicbiofull_xgb_mrnalm_clean_codex1\late_fusion_fold_metrics.csv` | `.\paper\table_scripts\generate_supplementary_tables_s1_s9.py` |
| S5 | merged from fusion predictions, prediction/residual correlations, and fold-aligned tabular metrics; paired bootstrap/sign-flip summaries added | `.\result\late_fusion_mrnalm_xgb\late_fusion_basicbiofull_xgb_mrnalm_clean_codex1\late_fusion_pred_test.csv; .\result\late_fusion_mrnalm_xgb\late_fusion_basicbiofull_xgb_mrnalm_clean_codex1\late_fusion_coefficients.csv; .\result\late_fusion_mrnalm_xgb\late_fusion_basicbiofull_xgb_mrnalm_clean_codex1\late_fusion_delta_vs_xgb.csv; .\result\62339\ridge_saluki_genetic_biochemical_human_clean_codex1\ridge_saluki_genetic_biochemical_human_clean_codex1_20260615_125539\fold_metrics.csv; .\result\elasticnet_collect_63111\elasticnet_saluki_regulatory_features_human\elasticnet_saluki_regulatory_features_human_clean_codex1_full8_array\elasticnet_fold_metrics_merged.csv; .\result\62340\xgb_saluki_regulatory_features_human_clean_codex1\xgb_saluki_regulatory_features_human_clean_codex1_20260615_125610\fold_metrics.csv` | `.\paper\table_scripts\generate_supplementary_tables_s1_s9.py` |
| S6 | renamed and merged from old S5/S12 plus raw SHAP and ablation outputs | `.\result\62367\result\xgb_shap_interpretability\xgb_shap_basic_plus_biochemical_full_clean_codex1\shap_top_features.csv; .\result\62367\result\xgb_shap_interpretability\xgb_shap_basic_plus_biochemical_full_clean_codex1\shap_family_summary.csv; .\result\62367\result\xgb_shap_interpretability\xgb_shap_basic_plus_biochemical_full_clean_codex1\shap_direction_biological_interpretation.csv; .\result\62374\result\xgb_family_ablation\xgb_family_ablation_basic_plus_biochemical_full_clean_codex1\ablation_summary_metrics.csv; .\result\62374\result\xgb_family_ablation\xgb_family_ablation_basic_plus_biochemical_full_clean_codex1\ablation_fold_metrics.csv` | `.\paper\table_scripts\generate_supplementary_tables_s1_s9.py` |
| S7 | renamed and reorganized from old S4 plus raw fusion-gain and case-study outputs | `.\result\fusion_gain_sample_profile\fusion_gain_basicbiofull_xgb_mrnalm_clean_codex1\fusion_gain_sample_profile.csv; .\result\fusion_gain_sample_profile\fusion_gain_basicbiofull_xgb_mrnalm_clean_codex1\fusion_gain_group_summary_within_fold.csv; .\result\fusion_gain_sample_profile\fusion_gain_basicbiofull_xgb_mrnalm_clean_codex1\fusion_gain_helped_vs_hurt_stats_within_fold.csv; .\result\fusion_case_studies\fusion_case_studies_basicbiofull_xgb_mrnalm_clean_codex1\fusion_case_study_combined.csv` | `.\paper\table_scripts\generate_supplementary_tables_s1_s9.py` |
| S8 | renamed and reorganized from old S3 plus raw motif/k-mer and region-composition outputs | `.\result\fusion_gain_motif_enrichment\fusion_gain_kmer_motif_clean_codex1\motif_enrichment_fusion_helped_vs_hurt.csv; .\result\fusion_gain_motif_enrichment\fusion_gain_kmer_motif_clean_codex1\region_length_gc_differences_within_fold.csv` | `.\paper\table_scripts\generate_supplementary_tables_s1_s9.py` |
| S9 | renamed and reorganized from old S2 plus raw permutation outputs | `.\result\62366\permutation_negative_control\perm_basicbiofull_clean_codex1\permutation_fold_metrics.csv; .\result\62366\permutation_negative_control\perm_basicbiofull_clean_codex1\permutation_summary_metrics.csv` | `.\paper\table_scripts\generate_supplementary_tables_s1_s9.py` |

## Key Value Checks

Full check table: `.\paper\tables\supplementary_table_s1_s9_value_checks.csv`.

No conflicts were detected between generated S1-S9 tables and the frozen key values checked by this script.

## Notes

- Tables use `full_tabular_benchmark`, `paired_xgb_mrnalm_predictions`, and `sequence_only_human_5fold` as sample-scope labels.
- Historical labels such as matched-subset, matched scope, clean run, and clean rerun are not used in the generated table fields.
- Source paths may still contain historical run directory names because they identify immutable result files.
- SHAP, feature-family ablation, motif/k-mer, and region-composition rows summarize predictive associations and hypothesis-generating checks, not causal mechanisms.
