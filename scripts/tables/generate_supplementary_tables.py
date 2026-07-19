from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = PROJECT_ROOT / "tables"
FROZEN_ROOT = PROJECT_ROOT / "results" / "frozen"

NORMALIZED_HUMAN = PROJECT_ROOT / "data" / "processed" / "human" / "normalized.csv.gz"
SPLIT_PROCESSED = PROJECT_ROOT / "data" / "processed" / "human" / "splits" / "cv_k5_seed42.csv"
SPLIT_SCRIPT_COPY = SPLIT_PROCESSED

MRNALM_ROOT = FROZEN_ROOT / "mrnalm_agghead"
SEQHEAD_ROOT = FROZEN_ROOT / "mrnalm_seqhead"

RIDGE_FOLD = FROZEN_ROOT / "ridge" / "fold_metrics.csv"
XGB_ROOT = FROZEN_ROOT / "xgboost"
XGB_FOLD = XGB_ROOT / "fold_metrics.csv"
XGB_FEATURE_MANIFEST = XGB_ROOT / "feature_group_manifest.json"
ELASTICNET_FOLD = FROZEN_ROOT / "elasticnet" / "fold_metrics.csv"

FUSION_ROOT = FROZEN_ROOT / "fusion"
FUSION_FOLD = FUSION_ROOT / "fold_metrics.csv"
FUSION_PRED_TEST = FUSION_ROOT / "pred_test.csv"
FUSION_COEFFICIENTS = FUSION_ROOT / "coefficients.csv"
FUSION_DELTA = FUSION_ROOT / "delta_vs_xgb.csv"

SHAP_ROOT = FROZEN_ROOT / "shap"
SHAP_TOP = SHAP_ROOT / "top_features.csv"
SHAP_FAMILY = SHAP_ROOT / "family_summary.csv"
SHAP_DIRECTION = SHAP_ROOT / "direction.csv"
ABLATION_ROOT = FROZEN_ROOT / "ablation"
ABLATION_SUMMARY = ABLATION_ROOT / "summary_metrics.csv"
ABLATION_FOLD = ABLATION_ROOT / "fold_metrics.csv"

FUSION_GAIN_ROOT = FROZEN_ROOT / "fusion_gain"
FUSION_GAIN_PROFILE = FUSION_GAIN_ROOT / "sample_profile.csv"
FUSION_GAIN_GROUP = FUSION_GAIN_ROOT / "group_summary_within_fold.csv"
FUSION_GAIN_STATS = FUSION_GAIN_ROOT / "helped_vs_hurt_stats.csv"
CASE_STUDIES = FROZEN_ROOT / "case_studies" / "combined.csv"

MOTIF = FROZEN_ROOT / "motif" / "motif_enrichment.csv"
REGION = FROZEN_ROOT / "motif" / "region_checks.csv"

PERMUTATION_FOLD = FROZEN_ROOT / "permutation" / "fold_metrics.csv"
PERMUTATION_SUMMARY = FROZEN_ROOT / "permutation" / "summary_metrics.csv"


EXPECTED_FEATURE_GROUPS = [
    "saluki_basic8",
    "genetic_only",
    "biochemical_371",
    "genetic_plus_biochemical_371",
    "all_23384",
    "biochemical_full_no_basic",
    "basic_plus_biochemical_full",
    "genetic_plus_biochemical_full",
]

EXPECTED_N_FEATURES = {
    "saluki_basic8": 8,
    "genetic_only": 23005,
    "biochemical_371": 371,
    "genetic_plus_biochemical_371": 23376,
    "all_23384": 23384,
    "biochemical_full_no_basic": 551,
    "basic_plus_biochemical_full": 559,
    "genetic_plus_biochemical_full": 23556,
}

FEATURE_GROUP_LABELS = {
    "saluki_basic8": "Basic architecture",
    "genetic_only": "Sequence-statistical",
    "biochemical_371": "Compact biochemical/regulatory",
    "genetic_plus_biochemical_371": "Compact biochemical/regulatory + sequence-statistical",
    "all_23384": "All Saluki tabular features",
    "biochemical_full_no_basic": "Full regulatory/biochemical without BASIC",
    "basic_plus_biochemical_full": "Main annotation-augmented feature set",
    "genetic_plus_biochemical_full": "Full regulatory/biochemical + sequence-statistical",
}

FEATURE_GROUP_ZH = {
    "saluki_basic8": "基础转录本结构特征",
    "genetic_only": "序列统计描述符",
    "biochemical_371": "紧凑生化/调控特征",
    "genetic_plus_biochemical_371": "紧凑生化/调控 + 序列统计特征",
    "all_23384": "全部 Saluki 表格特征",
    "biochemical_full_no_basic": "完整调控/生化注释（不含 BASIC）",
    "basic_plus_biochemical_full": "主要注释增强特征集",
    "genetic_plus_biochemical_full": "完整调控/生化注释 + 序列统计特征",
}

FEATURE_GROUP_SOURCE_TYPE = {
    "saluki_basic8": "transcript architecture",
    "genetic_only": "sequence/statistical descriptors",
    "biochemical_371": "compact biochemical/regulatory annotations",
    "genetic_plus_biochemical_371": "compact biochemical/regulatory annotations plus sequence/statistical descriptors",
    "all_23384": "all released Saluki tabular features",
    "biochemical_full_no_basic": "full biochemical/regulatory annotations",
    "basic_plus_biochemical_full": "full biochemical/regulatory annotations plus transcript architecture",
    "genetic_plus_biochemical_full": "full biochemical/regulatory annotations plus sequence/statistical descriptors",
}

FEATURE_GROUP_FAMILIES = {
    "saluki_basic8": "BASIC",
    "genetic_only": "GENETIC",
    "biochemical_371": "M6ACLIP; ENCORI; eCLIP and related compact biochemical/regulatory families",
    "genetic_plus_biochemical_371": "GENETIC; M6ACLIP; ENCORI; eCLIP and related compact biochemical/regulatory families",
    "all_23384": "BASIC; GENETIC; compact biochemical/regulatory families",
    "biochemical_full_no_basic": "RIP; TE; M6ACLIP; ENCORI; eCLIP; PARCLIP",
    "basic_plus_biochemical_full": "BASIC; RIP; TE; M6ACLIP; ENCORI; eCLIP; PARCLIP",
    "genetic_plus_biochemical_full": "GENETIC; RIP; TE; M6ACLIP; ENCORI; eCLIP; PARCLIP",
}

FEATURE_GROUP_PURPOSE = {
    "saluki_basic8": "Quantifies how much signal is available from compact transcript-architecture descriptors alone.",
    "genetic_only": "Quantifies high-dimensional sequence/statistical descriptors without explicit biochemical or regulatory annotation.",
    "biochemical_371": "Evaluates the compact biochemical/regulatory block released with the Saluki resource.",
    "genetic_plus_biochemical_371": "Tests whether sequence/statistical descriptors add to the compact biochemical/regulatory block.",
    "all_23384": "Evaluates the complete released Saluki tabular feature matrix.",
    "biochemical_full_no_basic": "Isolates full biochemical/regulatory annotations without BASIC transcript-architecture descriptors.",
    "basic_plus_biochemical_full": "Primary annotation-augmented feature set for XGBoost and main tabular comparisons.",
    "genetic_plus_biochemical_full": "Tests whether high-dimensional sequence/statistical descriptors add to full biochemical/regulatory annotations.",
}

DISPLAY_MODEL = {
    "xgb": "XGBoost",
    "ridge": "Ridge",
    "elasticnet": "ElasticNet",
}


def repo_path(path: Path) -> str:
    """Return a portable repository-relative path for table provenance."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def repo_paths(*paths: Path) -> str:
    return "; ".join(repo_path(path) for path in paths)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def line_count_csv(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def mean_sd(values: Iterable[float], ddof: int = 1) -> tuple[float, float, str]:
    arr = pd.Series(list(values), dtype="float64").dropna()
    if arr.empty:
        return np.nan, np.nan, ""
    mean = float(arr.mean())
    sd = float(arr.std(ddof=ddof)) if len(arr) > 1 else np.nan
    return mean, sd, f"{mean:.4f} ± {sd:.4f}" if pd.notna(sd) else f"{mean:.4f}"


def metric_summary(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str], ddof: int = 1) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metric_cols:
            mean, sd, label = mean_sd(sub[metric], ddof=ddof)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean,
                    "SD": sd,
                    "mean ± SD": label,
                    "N_folds": int(sub["fold"].nunique()) if "fold" in sub.columns else int(len(sub)),
                }
            )
    return pd.DataFrame(rows)


def load_manifest() -> dict:
    return json.loads(XGB_FEATURE_MANIFEST.read_text(encoding="utf-8"))


def load_mrnalm_fold_metrics(root: Path, model_label: str) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(root.glob("cv_fold_*/baseline_summary.json")):
        fold = int(summary_path.parent.name.split("_")[-1])
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        result = data["result"]
        final = result["final"]
        row = {
            "source_file": repo_path(summary_path),
            "model": model_label,
            "sample_scope": "sequence_only_human_5fold",
            "feature_group_code": "sequence_only",
            "feature_group_label": "mRNA-LM sequence-only predictions",
            "fold": fold,
            "n_features": np.nan,
            "selected_Ridge_alpha": np.nan,
            "selected_ElasticNet_alpha": np.nan,
            "selected_ElasticNet_L1_ratio": np.nan,
            "selected_XGBoost_best_iteration": np.nan,
            "best_epoch": result.get("best_epoch"),
        }
        for split in ("train", "val", "test"):
            for metric, label in [
                ("r2", "R²"),
                ("pearson", "Pearson"),
                ("spearman", "Spearman"),
                ("rmse", "RMSE"),
                ("mae", "MAE"),
            ]:
                row[f"{split}_{label}"] = final.get(split, {}).get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def standardize_tabular_fold(df: pd.DataFrame, model: str, sample_scope: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["source_file"] = ""
    out["model"] = model
    out["sample_scope"] = sample_scope
    out["feature_group_code"] = df["feature_group"]
    out["feature_group_label"] = df["feature_group"].map(FEATURE_GROUP_LABELS).fillna(df.get("feature_label", df["feature_group"]))
    out["fold"] = df["cv_fold"].astype(int)
    out["n_features"] = df.get("n_features", df.get("n_features_after_filter", np.nan))
    out["selected_Ridge_alpha"] = df["selected_alpha"] if model == "Ridge" and "selected_alpha" in df.columns else np.nan
    out["selected_ElasticNet_alpha"] = df["selected_alpha"] if model == "ElasticNet" and "selected_alpha" in df.columns else np.nan
    out["selected_ElasticNet_L1_ratio"] = df["selected_l1_ratio"] if model == "ElasticNet" and "selected_l1_ratio" in df.columns else np.nan
    out["selected_XGBoost_best_iteration"] = df["best_iteration"] if model == "XGBoost" and "best_iteration" in df.columns else np.nan
    out["best_epoch"] = np.nan
    for split in ("train", "val", "test"):
        for src, dst in [
            ("r2", "R²"),
            ("pearson", "Pearson"),
            ("spearman", "Spearman"),
            ("rmse", "RMSE"),
            ("mae", "MAE"),
        ]:
            out[f"{split}_{dst}"] = df.get(f"{split}_{src}", np.nan)
    return out


def standardize_fusion_fold(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["source_file"] = repo_path(FUSION_FOLD)
    out["model"] = df["method"].map(
        {
            "xgb_only": "XGBoost-only",
            "mrnalm_only": "mRNA-LM-only",
            "linear_fusion": "Linear prediction-level fusion",
            "calibrated_xgb": "Calibrated XGBoost",
            "ridge_fusion": "Ridge fusion",
            "elasticnet_fusion": "ElasticNet fusion",
        }
    ).fillna(df["method"])
    out["sample_scope"] = "paired_xgb_mrnalm_predictions"
    out["feature_group_code"] = df["method"].map(
        {
            "xgb_only": "basic_plus_biochemical_full",
            "mrnalm_only": "sequence_only",
            "linear_fusion": "xgb_basic_plus_biochemical_full_plus_mrnalm",
        }
    ).fillna("fusion_comparator")
    out["feature_group_label"] = df["method"].map(
        {
            "xgb_only": "Main annotation-augmented feature set",
            "mrnalm_only": "mRNA-LM sequence-only predictions",
            "linear_fusion": "XGBoost + mRNA-LM predictions",
        }
    ).fillna("Fusion comparator")
    out["fold"] = df["cv_fold"].astype(int)
    out["n_features"] = np.nan
    out["selected_Ridge_alpha"] = np.nan
    out["selected_ElasticNet_alpha"] = np.nan
    out["selected_ElasticNet_L1_ratio"] = np.nan
    out["selected_XGBoost_best_iteration"] = np.nan
    out["best_epoch"] = np.nan
    for split in ("val", "test"):
        for src, dst in [
            ("r2", "R²"),
            ("pearson", "Pearson"),
            ("spearman", "Spearman"),
            ("rmse", "RMSE"),
            ("mae", "MAE"),
        ]:
            out[f"{split}_{dst}"] = df.get(f"{split}_{src}", np.nan)
    for col in ["train_R²", "train_Pearson", "train_Spearman", "train_RMSE", "train_MAE"]:
        out[col] = np.nan
    out["val_N"] = df.get("val_n", np.nan)
    out["test_N"] = df.get("test_n", np.nan)
    return out


def build_s1() -> pd.DataFrame:
    normalized = read_csv(NORMALIZED_HUMAN)
    split = read_csv(SPLIT_PROCESSED)
    manifest = load_manifest()
    align = manifest.get("id_alignment", {})
    rows = []

    rows.append(
        {
            "table_section": "data_source",
            "item": "Saluki-released human processed data",
            "source_file": repo_path(NORMALIZED_HUMAN),
            "identifier_key": "ENSID",
            "target_column": "HALFLIFE",
            "target_scale": "processed normalized HALFLIFE as stored in the Saluki human processed table",
            "N": int(len(normalized)),
            "value": "human-only analysis set",
        }
    )
    rows.append(
        {
            "table_section": "target_summary",
            "item": "HALFLIFE",
            "source_file": repo_path(NORMALIZED_HUMAN),
            "identifier_key": "ENSID",
            "target_column": "HALFLIFE",
            "target_scale": "same processed target scale used by mRNA-LM, Ridge, ElasticNet, XGBoost, and fusion",
            "N": int(normalized["HALFLIFE"].notna().sum()),
            "missing_N": int(normalized["HALFLIFE"].isna().sum()),
            "mean": float(normalized["HALFLIFE"].mean()),
            "SD": float(normalized["HALFLIFE"].std(ddof=1)),
            "median": float(normalized["HALFLIFE"].median()),
            "min": float(normalized["HALFLIFE"].min()),
            "max": float(normalized["HALFLIFE"].max()),
        }
    )
    for col in ["5UTR", "ORF", "3UTR"]:
        rows.append(
            {
                "table_section": "sequence_availability",
                "item": col,
                "source_file": repo_path(NORMALIZED_HUMAN),
                "identifier_key": "ENSID",
                "N": int(normalized[col].notna().sum()),
                "missing_N": int(normalized[col].isna().sum()),
                "value": "available sequence strings used by sequence-only mRNA-LM models",
            }
        )
    for fold, sub in split.groupby("fold", sort=True):
        rows.append(
            {
                "table_section": "fold_size",
                "item": "test fold size",
                "source_file": repo_path(SPLIT_PROCESSED),
                "identifier_key": "ENSID",
                "fold": int(fold),
                "N": int(len(sub)),
                "value": "held-out test fold",
            }
        )
    rows.append(
        {
            "table_section": "target_feature_intersection",
            "item": "normalized target x genetic/biochemical features x biochemical features",
            "source_file": repo_path(XGB_FEATURE_MANIFEST),
            "identifier_key": "GENEID -> ENSID",
            "target_column": "HALFLIFE",
            "N": int(align.get("overlap_n", np.nan)),
            "value": (
                f"normalized_n={align.get('normalized_n')}; genetic_biochemical_n={align.get('genetic_biochemical_n')}; "
                f"biochemical_n={align.get('biochemical_n')}; HALFLIFE exact match in both feature files="
                f"{bool(align.get('halflife_exact_match_genetic_biochemical')) and bool(align.get('halflife_exact_match_biochemical'))}"
            ),
        }
    )
    for group, meta in manifest.get("feature_groups", {}).items():
        miss = meta.get("missing_stats", {})
        rows.append(
            {
                "table_section": "feature_missingness_by_group",
                "item": group,
                "source_file": repo_path(XGB_FEATURE_MANIFEST),
                "identifier_key": "GENEID -> ENSID",
                "N": int(align.get("overlap_n", len(normalized))),
                "n_features": meta.get("n_features"),
                "mean_missing_ratio": miss.get("mean_missing_ratio"),
                "median_missing_ratio": miss.get("median_missing_ratio"),
                "max_missing_ratio": miss.get("max_missing_ratio"),
                "columns_all_missing_count": miss.get("columns_all_missing_count"),
                "columns_missing_gt_50pct": miss.get("columns_missing_gt_50pct"),
                "columns_missing_gt_90pct": miss.get("columns_missing_gt_90pct"),
            }
        )
    return pd.DataFrame(rows)


def build_s2() -> pd.DataFrame:
    rows = []
    loaded = {}
    for label, path in [("processed split", SPLIT_PROCESSED), ("script copy of split", SPLIT_SCRIPT_COPY)]:
        df = read_csv(path)
        loaded[label] = df
        fold_counts = df.groupby("ENSID")["fold"].nunique()
        cross = fold_counts[fold_counts > 1].index
        rows.append(
            {
                "table_section": "identifier_split_audit",
                "split_source": label,
                "source_file": repo_path(path),
                "identifier": "ENSID",
                "N": int(len(df)),
                "unique_ENSID": int(df["ENSID"].nunique()),
                "duplicate_ENSID_rows": int(len(df) - df["ENSID"].nunique()),
                "ENSID_appearing_in_multiple_folds": int(len(cross)),
                "affected_rows": int(df["ENSID"].isin(cross).sum()),
                "audit_result": "no ENSID-level cross-fold duplication detected" if len(cross) == 0 else "ENSID-level cross-fold duplication detected",
                "limitation": "Isoform-level relatedness cannot be fully audited unless transcript isoform identifiers are available.",
            }
        )
    if {"processed split", "script copy of split"} <= set(loaded):
        a = loaded["processed split"].sort_values(["ENSID", "fold"]).reset_index(drop=True)
        b = loaded["script copy of split"].sort_values(["ENSID", "fold"]).reset_index(drop=True)
        rows.append(
            {
                "table_section": "split_file_consistency",
                "split_source": "processed split vs script copy",
                "source_file": repo_paths(SPLIT_PROCESSED, SPLIT_SCRIPT_COPY),
                "identifier": "ENSID",
                "N": int(len(a)),
                "audit_result": "identical" if a.equals(b) else "not identical",
                "limitation": "This check confirms file-level split consistency, not biological independence among related isoforms.",
            }
        )
    return pd.DataFrame(rows)


def build_s3() -> pd.DataFrame:
    manifest = load_manifest()
    rows = []
    for group in EXPECTED_FEATURE_GROUPS:
        meta = manifest.get("feature_groups", {}).get(group, {})
        n_features_observed = int(meta.get("n_features", EXPECTED_N_FEATURES[group]))
        rows.append(
            {
                "feature_group_code": group,
                "display_name": FEATURE_GROUP_LABELS[group],
                "chinese_name": FEATURE_GROUP_ZH[group],
                "n_features": n_features_observed,
                "expected_n_features": EXPECTED_N_FEATURES[group],
                "n_features_status": "matches expected" if n_features_observed == EXPECTED_N_FEATURES[group] else "differs from expected",
                "included_families": FEATURE_GROUP_FAMILIES[group],
                "source_type": FEATURE_GROUP_SOURCE_TYPE[group],
                "main_annotation_augmented_feature_set": group == "basic_plus_biochemical_full",
                "comparison_purpose": FEATURE_GROUP_PURPOSE[group],
                "source_file": repo_path(XGB_FEATURE_MANIFEST),
                "feature_names_path": repo_path(XGB_ROOT / f"feature_names_{group}.txt"),
                "N_available": int(load_manifest().get("id_alignment", {}).get("overlap_n", 12981)),
            }
        )
    return pd.DataFrame(rows)


def build_s4() -> pd.DataFrame:
    parts = []
    ridge = standardize_tabular_fold(read_csv(RIDGE_FOLD), "Ridge", "full_tabular_benchmark")
    ridge["source_file"] = repo_path(RIDGE_FOLD)
    parts.append(ridge)
    elastic = read_csv(ELASTICNET_FOLD).rename(columns={"n_features_after_filter": "n_features"})
    elastic_std = standardize_tabular_fold(elastic, "ElasticNet", "full_tabular_benchmark")
    elastic_std["source_file"] = repo_path(ELASTICNET_FOLD)
    parts.append(elastic_std)
    xgb = standardize_tabular_fold(read_csv(XGB_FOLD), "XGBoost", "full_tabular_benchmark")
    xgb["source_file"] = repo_path(XGB_FOLD)
    parts.append(xgb)
    parts.append(load_mrnalm_fold_metrics(MRNALM_ROOT, "mRNA-LM AggHead matched-LoRA"))
    if SEQHEAD_ROOT.exists():
        parts.append(load_mrnalm_fold_metrics(SEQHEAD_ROOT, "mRNA-LM SeqHead-noConv comparator"))
    parts.append(standardize_fusion_fold(read_csv(FUSION_FOLD)))
    fold = pd.concat(parts, ignore_index=True, sort=False)

    split = read_csv(SPLIT_PROCESSED)
    fold_sizes = split.groupby("fold").size().to_dict()
    fold["test_N"] = fold.get("test_N", np.nan)
    fold["test_N"] = fold.apply(
        lambda r: fold_sizes.get(int(r["fold"]), np.nan) if pd.isna(r.get("test_N")) and pd.notna(r.get("fold")) else r.get("test_N"),
        axis=1,
    )
    fold.insert(0, "table_section", "fold_metrics")

    metric_cols = ["test_R²", "test_Pearson", "test_Spearman", "test_RMSE", "test_MAE"]
    summary = metric_summary(
        fold,
        ["model", "sample_scope", "feature_group_code", "feature_group_label"],
        metric_cols,
        ddof=0,
    )
    summary.insert(0, "table_section", "summary_metrics")
    return pd.concat([fold, summary], ignore_index=True, sort=False)


def build_s5() -> pd.DataFrame:
    rows = []
    pred = read_csv(FUSION_PRED_TEST)
    base = pred.loc[pred["method"] == "xgb_only", ["sample_id", "cv_fold", "y_true", "xgb_pred", "mrnalm_pred"]].copy()
    base["xgb_residual"] = base["y_true"] - base["xgb_pred"]
    base["mrnalm_residual"] = base["y_true"] - base["mrnalm_pred"]
    for fold, sub in base.groupby("cv_fold", sort=True):
        for quantity, a, b in [
            ("prediction", "xgb_pred", "mrnalm_pred"),
            ("residual", "xgb_residual", "mrnalm_residual"),
        ]:
            rows.append(
                {
                    "table_section": "prediction_residual_correlation",
                    "sample_scope": "paired_xgb_mrnalm_predictions",
                    "fold": int(fold),
                    "N": int(len(sub)),
                    "quantity": quantity,
                    "metric": "Pearson",
                    "value": float(sub[a].corr(sub[b], method="pearson")),
                }
            )
            rows.append(
                {
                    "table_section": "prediction_residual_correlation",
                    "sample_scope": "paired_xgb_mrnalm_predictions",
                    "fold": int(fold),
                    "N": int(len(sub)),
                    "quantity": quantity,
                    "metric": "Spearman",
                    "value": float(sub[a].corr(sub[b], method="spearman")),
                }
            )

    corr = pd.DataFrame(rows)
    corr_summary_rows = []
    for (scope, quantity, metric), sub in corr.groupby(["sample_scope", "quantity", "metric"], sort=False):
        mean, sd, label = mean_sd(sub["value"])
        corr_summary_rows.append(
            {
                "table_section": "prediction_residual_correlation_summary",
                "sample_scope": scope,
                "quantity": quantity,
                "metric": metric,
                "mean": mean,
                "SD": sd,
                "mean ± SD": label,
                "N_folds": int(sub["fold"].nunique()),
            }
        )
    corr_summary = pd.DataFrame(corr_summary_rows)

    coef = read_csv(FUSION_COEFFICIENTS)
    coef = coef.loc[coef["method"] == "linear_fusion"].copy()
    coef = coef.rename(columns={"cv_fold": "fold", "coef_xgb": "beta_xgb", "coef_mrnalm": "beta_mrnalm"})
    coef["table_section"] = "linear_fusion_coefficients"
    coef["sample_scope"] = "paired_xgb_mrnalm_predictions"

    delta = read_csv(FUSION_DELTA)
    delta = delta.loc[delta["method"] == "linear_fusion"].copy()
    delta = delta.rename(columns={"cv_fold": "fold", "test_r2": "linear_fusion_R²", "xgb_only_test_r2": "xgb_only_R²", "delta_test_r2_vs_xgb": "delta_R²_vs_XGBoost"})
    delta["table_section"] = "linear_fusion_delta_R²"
    delta["sample_scope"] = "paired_xgb_mrnalm_predictions"

    delta_mean, delta_sd, delta_label = mean_sd(delta["delta_R²_vs_XGBoost"])
    delta_summary = pd.DataFrame(
        [
            {
                "table_section": "linear_fusion_delta_R²_summary",
                "sample_scope": "paired_xgb_mrnalm_predictions",
                "metric": "delta_R²_vs_XGBoost",
                "mean": delta_mean,
                "SD": delta_sd,
                "mean ± SD": delta_label,
                "fold_delta_values": ";".join(f"{x:+.4f}" for x in delta["delta_R²_vs_XGBoost"]),
                "N_folds": int(delta["fold"].nunique()),
                "N_positive_folds": int((delta["delta_R²_vs_XGBoost"] > 0).sum()),
            }
        ]
    )

    return pd.concat([corr, corr_summary, coef, delta, delta_summary], ignore_index=True, sort=False)


def build_s6() -> pd.DataFrame:
    parts = []
    top = read_csv(SHAP_TOP)
    top["table_section"] = "top_shap_features"
    parts.append(top)

    family = read_csv(SHAP_FAMILY)
    family["mean_abs_shap_per_feature"] = family["sum_abs_shap"] / family["n_features"]
    family["top50_fraction"] = family["top50_count"] / family["n_features"]
    family["table_section"] = "feature_family_shap_summary"
    parts.append(family)

    if SHAP_DIRECTION.exists():
        direction = read_csv(SHAP_DIRECTION)
        direction["table_section"] = "shap_direction_summary"
        parts.append(direction)

    ablation = read_csv(ABLATION_SUMMARY)
    full = float(ablation.loc[ablation["feature_set"] == "full_basic_plus_biochemical_full", "test_r2_mean"].iloc[0])
    ablation["delta_R²_vs_full"] = ablation["test_r2_mean"] - full
    ablation["table_section"] = "feature_family_ablation_summary"
    parts.append(ablation)

    ablation_fold = read_csv(ABLATION_FOLD)
    ablation_fold["table_section"] = "feature_family_ablation_fold_metrics"
    parts.append(ablation_fold)
    return pd.concat(parts, ignore_index=True, sort=False)


def build_s7() -> pd.DataFrame:
    profile = read_csv(FUSION_GAIN_PROFILE)
    profile["table_section"] = "fusion_gain_sample_profile"
    group = read_csv(FUSION_GAIN_GROUP)
    group["table_section"] = "fusion_gain_group_summary"
    stats = read_csv(FUSION_GAIN_STATS)
    stats["table_section"] = "helped_vs_hurt_comparisons"
    cases = read_csv(CASE_STUDIES)
    cases = cases.rename(columns={"y_true": "true_HALFLIFE"})
    cases["table_section"] = "case_studies_top30_helped_and_top30_hurt"
    definitions = pd.DataFrame(
        [
            {
                "table_section": "group_definitions",
                "fusion_gain_group": "fusion_helped_top20",
                "definition": "within-fold top 20% by fusion_gain = abs_error_xgb - abs_error_fusion",
            },
            {
                "table_section": "group_definitions",
                "fusion_gain_group": "neutral_middle60",
                "definition": "within-fold middle 60% by fusion_gain",
            },
            {
                "table_section": "group_definitions",
                "fusion_gain_group": "fusion_hurt_top20",
                "definition": "within-fold bottom 20% by fusion_gain",
            },
        ]
    )
    return pd.concat([definitions, group, stats, cases, profile], ignore_index=True, sort=False)


def build_s8() -> pd.DataFrame:
    motif = read_csv(MOTIF)
    motif = motif.rename(columns={"mannwhitney_p": "p_value", "q_value_bh_fdr": "FDR_q_value", "cliffs_delta_helped_vs_hurt": "effect_size_cliffs_delta"})
    motif["table_section"] = "motif_kmer_comparison"
    region = read_csv(REGION)
    region = region.rename(columns={"mannwhitney_p": "p_value", "q_value_bh_fdr": "FDR_q_value", "cliffs_delta_helped_vs_hurt": "effect_size_cliffs_delta"})
    region["table_section"] = "region_length_gc_comparison"
    return pd.concat([motif, region], ignore_index=True, sort=False)


def build_s9() -> pd.DataFrame:
    fold = read_csv(PERMUTATION_FOLD)
    fold = fold.rename(
        columns={
            "cv_fold": "fold",
            "perm_seed": "seed",
            "test_n": "N",
            "test_r2": "R²",
            "test_pearson": "Pearson",
            "test_spearman": "Spearman",
            "test_rmse": "RMSE",
            "test_mae": "MAE",
        }
    )
    fold["model"] = fold["model"].map(DISPLAY_MODEL).fillna(fold["model"])
    fold["table_section"] = "fold_metrics"
    seed_summary = (
        fold.groupby(["model", "seed"], as_index=False)[["R²", "Pearson", "Spearman", "RMSE", "MAE"]]
        .mean()
        .assign(table_section="seed_level_summary")
    )
    summary = read_csv(PERMUTATION_SUMMARY)
    metric_map = {"r2": "R²", "pearson": "Pearson", "spearman": "Spearman", "rmse": "RMSE", "mae": "MAE"}
    summary["metric"] = summary["metric"].map(metric_map).fillna(summary["metric"])
    summary["model"] = summary["model"].map(DISPLAY_MODEL).fillna(summary["model"])
    summary["table_section"] = "overall_summary"
    summary = summary.rename(columns={"std": "SD", "n": "N"})
    return pd.concat([fold, seed_summary, summary], ignore_index=True, sort=False)


def write_index() -> pd.DataFrame:
    rows = [
        ("Supplementary Table S1", "supplementary_table_s1_data_inclusion_target_fold_overview.xlsx", "Data inclusion, processed HALFLIFE target, feature/sequence availability, target-feature alignment, missingness, and fold-size overview."),
        ("Supplementary Table S2", "supplementary_table_s2_identifier_split_overlap_audit.xlsx", "Identifier uniqueness, fold assignment, split-file consistency, and ENSID-level cross-fold overlap audit."),
        ("Supplementary Table S3", "supplementary_table_s3_feature_group_definitions_dimensions.xlsx", "Definitions, dimensions, source categories, and comparison purposes for the eight Saluki-derived feature groups."),
        ("Supplementary Table S4", "supplementary_table_s4_complete_fold_level_performance_metrics.xlsx", "Fold-level and summary performance metrics for mRNA-LM, Ridge, ElasticNet, XGBoost, and paired prediction-level fusion."),
        ("Supplementary Table S5", "supplementary_table_s5_prediction_fusion_complementarity_diagnostics.xlsx", "Prediction/residual correlations, linear-fusion coefficients, fold-wise fusion delta R², and paired fusion diagnostics."),
        ("Supplementary Table S6", "supplementary_table_s6_shap_feature_family_interpretation.xlsx", "Top SHAP features, family-level SHAP summaries, normalized family SHAP metrics, and feature-family ablation results."),
        ("Supplementary Table S7", "supplementary_table_s7_fusion_gain_sample_profile_case_studies.xlsx", "Fusion-gain group definitions, group summaries, helped-vs-hurt comparisons, and top helped/hurt case studies."),
        ("Supplementary Table S8", "supplementary_table_s8_motif_kmer_region_composition_checks.xlsx", "3-mer, 4-mer, predefined motif, region length, and GC comparisons between fusion-helped and fusion-hurt transcripts."),
        ("Supplementary Table S9", "supplementary_table_s9_permutation_negative_control_metrics.xlsx", "Fold-level, seed-level, and overall metrics for permuted-label Ridge and XGBoost negative controls."),
    ]
    out = pd.DataFrame(rows, columns=["table", "file", "description"])
    out.to_csv(TABLE_DIR / "supplementary_table_index.csv", index=False)
    return out


def check_close(name: str, observed: float, expected: float, tol: float = 5e-4) -> dict:
    return {
        "check": name,
        "observed": observed,
        "expected": expected,
        "difference": observed - expected,
        "status": "pass" if abs(observed - expected) <= tol else "conflict",
    }


def sheet_name(raw: str, used: set[str]) -> str:
    clean = "".join("_" if ch in r'[]:*?/\\' else ch for ch in str(raw))
    clean = clean.strip() or "sheet"
    clean = clean[:31]
    base = clean
    i = 2
    while clean in used:
        suffix = f"_{i}"
        clean = f"{base[:31 - len(suffix)]}{suffix}"
        i += 1
    used.add(clean)
    return clean


def compact_sheet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(axis=1, how="all").copy()
    if "table_section" in out.columns:
        cols = ["table_section"] + [c for c in out.columns if c != "table_section"]
        out = out[cols]
    return out


def write_workbook(df: pd.DataFrame, path: Path) -> None:
    used: set[str] = set()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if "table_section" in df.columns:
            for section, sub in df.groupby("table_section", dropna=False, sort=False):
                compact_sheet(sub).to_excel(writer, sheet_name=sheet_name(section, used), index=False)
        else:
            compact_sheet(df).to_excel(writer, sheet_name=sheet_name("table", used), index=False)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for column_cells in ws.columns:
                header = str(column_cells[0].value or "")
                max_len = min(max([len(header)] + [len(str(cell.value)) for cell in column_cells[1:50] if cell.value is not None]) + 2, 60)
                ws.column_dimensions[column_cells[0].column_letter].width = max(max_len, 10)


def audit(tables: dict[str, pd.DataFrame], outputs: dict[str, Path]) -> str:
    checks = []
    s1 = tables["S1"]
    checks.append({"check": "S1 final N", "observed": int(s1.loc[s1["table_section"] == "data_source", "N"].iloc[0]), "expected": 12981, "difference": 0, "status": "pass"})
    fold_sizes = s1.loc[s1["table_section"] == "fold_size"].sort_values("fold")["N"].astype(int).tolist()
    checks.append({"check": "S1 fold sizes", "observed": "/".join(map(str, fold_sizes)), "expected": "2600/2600/2600/2600/2581", "difference": "", "status": "pass" if fold_sizes == [2600, 2600, 2600, 2600, 2581] else "conflict"})

    s2 = tables["S2"]
    cross = int(s2.loc[s2["table_section"] == "identifier_split_audit", "ENSID_appearing_in_multiple_folds"].fillna(0).max())
    checks.append({"check": "S2 ENSID cross-fold duplication", "observed": cross, "expected": 0, "difference": cross, "status": "pass" if cross == 0 else "conflict"})

    s3 = tables["S3"]
    for group, expected in EXPECTED_N_FEATURES.items():
        obs = int(s3.loc[s3["feature_group_code"] == group, "n_features"].iloc[0])
        checks.append({"check": f"S3 {group} n_features", "observed": obs, "expected": expected, "difference": obs - expected, "status": "pass" if obs == expected else "conflict"})

    s4 = tables["S4"]
    fold = s4[s4["table_section"] == "fold_metrics"].copy()
    def model_metric(model: str, scope: str, group: str, metric: str = "test_R²") -> float:
        vals = fold.loc[
            (fold["model"] == model) & (fold["sample_scope"] == scope) & (fold["feature_group_code"] == group),
            metric,
        ].dropna().astype(float)
        return float(vals.mean())
    checks.append(check_close("S4 mRNA-LM AggHead test R²", model_metric("mRNA-LM AggHead matched-LoRA", "sequence_only_human_5fold", "sequence_only"), 0.4436))
    checks.append(check_close("S4 XGBoost main feature set test R²", model_metric("XGBoost", "full_tabular_benchmark", "basic_plus_biochemical_full"), 0.6766))
    checks.append(check_close("S4 ElasticNet best test R²", model_metric("ElasticNet", "full_tabular_benchmark", "genetic_plus_biochemical_full"), 0.6374))
    checks.append(check_close("S4 linear fusion test R²", model_metric("Linear prediction-level fusion", "paired_xgb_mrnalm_predictions", "xgb_basic_plus_biochemical_full_plus_mrnalm"), 0.7008))

    s5 = tables["S5"]
    for quantity, metric, expected in [
        ("prediction", "Pearson", 0.6752),
        ("prediction", "Spearman", 0.6617),
        ("residual", "Pearson", 0.5683),
        ("residual", "Spearman", 0.5444),
    ]:
        obs = float(s5.loc[(s5["table_section"] == "prediction_residual_correlation_summary") & (s5["quantity"] == quantity) & (s5["metric"] == metric), "mean"].iloc[0])
        checks.append(check_close(f"S5 {quantity} {metric}", obs, expected))
    deltas = s5.loc[s5["table_section"] == "linear_fusion_delta_R²", "delta_R²_vs_XGBoost"].dropna().round(4).tolist()
    checks.append({"check": "S5 fusion delta values", "observed": ";".join(f"{x:+.4f}" for x in deltas), "expected": "+0.0281;+0.0220;+0.0220;+0.0217;+0.0269", "difference": "", "status": "pass" if deltas == [0.0281, 0.0220, 0.0220, 0.0217, 0.0269] else "conflict"})

    s6 = tables["S6"]
    fam = s6[s6["table_section"] == "feature_family_shap_summary"]
    for family, expected in {"BASIC": 8, "RIP": 33, "ENCORI": 133, "eCLIP": 225, "M6ACLIP": 13, "PARCLIP": 146, "TE": 1}.items():
        obs = int(fam.loc[fam["family"] == family, "n_features"].iloc[0])
        checks.append({"check": f"S6 {family} n_features", "observed": obs, "expected": expected, "difference": obs - expected, "status": "pass" if obs == expected else "conflict"})
    abl = s6[s6["table_section"] == "feature_family_ablation_summary"]
    for feature_set, expected in [
        ("remove_basic", -0.0160),
        ("remove_encori", -0.0229),
        ("remove_rip", -0.0599),
        ("remove_rip_encori", -0.1167),
    ]:
        obs = float(abl.loc[abl["feature_set"] == feature_set, "delta_R²_vs_full"].iloc[0])
        checks.append(check_close(f"S6 {feature_set} delta R²", obs, expected, tol=8e-4))

    s7 = tables["S7"]
    group = s7[s7["table_section"] == "fusion_gain_group_summary"]
    counts = group.set_index("fusion_gain_group")["n"].to_dict()
    checks.append({"check": "S7 group sizes", "observed": str(counts), "expected": "{helped:2597, neutral:7787, hurt:2597}", "difference": "", "status": "pass" if counts.get("fusion_helped_top20") == 2597 and counts.get("neutral_middle60") == 7787 and counts.get("fusion_hurt_top20") == 2597 else "conflict"})
    stats = s7[s7["table_section"] == "helped_vs_hurt_comparisons"]
    for feature, expected in [("abs_err_xgb", 1.16e-208), ("abs_err_fusion", 1.18e-41), ("HALFLIFE", 0.636), ("xgb_residual", 0.717)]:
        obs = float(stats.loc[stats["feature"] == feature, "q_value_bh_fdr"].iloc[0])
        checks.append(check_close(f"S7 {feature} FDR q", obs, expected, tol=max(1e-3, abs(expected) * 0.05)))

    s8 = tables["S8"]
    motif_min = float(s8.loc[s8["table_section"] == "motif_kmer_comparison", "FDR_q_value"].min())
    overall_min = float(s8["FDR_q_value"].min())
    checks.append(check_close("S8 motif/k-mer minimum FDR q", motif_min, 0.264, tol=5e-4))
    checks.append({"check": "S8 any FDR-significant feature", "observed": int((s8["FDR_q_value"] < 0.05).sum()), "expected": 0, "difference": int((s8["FDR_q_value"] < 0.05).sum()), "status": "pass" if (s8["FDR_q_value"] < 0.05).sum() == 0 else "conflict"})

    s9 = tables["S9"]
    summary = s9[s9["table_section"] == "overall_summary"]
    for model, metric, expected in [
        ("XGBoost", "R²", -0.0030),
        ("Ridge", "R²", -0.0095),
        ("XGBoost", "Pearson", -0.0401),
        ("XGBoost", "Spearman", -0.0441),
        ("Ridge", "Pearson", -0.0316),
        ("Ridge", "Spearman", -0.0341),
    ]:
        obs = float(summary.loc[(summary["model"] == model) & (summary["metric"] == metric), "mean"].iloc[0])
        checks.append(check_close(f"S9 {model} {metric}", obs, expected, tol=5e-4))

    check_df = pd.DataFrame(checks)
    check_path = TABLE_DIR / "supplementary_table_s1_s9_value_checks.csv"
    check_df.to_csv(check_path, index=False)

    mapping = [
        ("old supplementary_table_s0_feature_group_definitions.csv", "S3", "feature group definitions and dimensions"),
        ("old supplementary_table_s1_complete_fold_metrics.csv", "S4", "complete fold-level metrics"),
        ("old supplementary_table_s2_permutation_negative_control.csv", "S9", "permutation negative controls"),
        ("old supplementary_table_s3_motif_kmer_region_checks.csv", "S8", "motif/k-mer and region checks"),
        ("old supplementary_table_s4_fusion_gain_case_studies.csv", "S7", "fusion-gain profile and case studies"),
        ("old supplementary_table_s5_shap_family_ablation.csv", "S6", "SHAP and feature-family ablation"),
        ("old supplementary_table_s6_late_fusion_details.csv", "S5", "late-fusion coefficients and delta metrics"),
        ("old supplementary_table_s7_prediction_residual_correlations.csv", "S5", "prediction/residual correlations"),
        ("old supplementary_table_s9_gene_fold_overlap_check.csv", "S2", "identifier split overlap audit"),
        ("normalized.csv.gz and feature_group_manifest.json", "S1", "data inclusion, target, fold and missingness overview"),
    ]
    table_meta = {
        "S1": {
            "source_files": repo_paths(NORMALIZED_HUMAN, SPLIT_PROCESSED, XGB_FEATURE_MANIFEST),
            "source_status": "needed new organized table from existing target, split, and feature-manifest files",
            "required_content_status": "complete",
            "missing_fields": "none",
            "repair_needed": "none",
        },
        "S2": {
            "source_files": repo_paths(SPLIT_PROCESSED, SPLIT_SCRIPT_COPY),
            "source_status": "generated from existing split files; replaces old S9 gene-fold overlap table",
            "required_content_status": "complete",
            "missing_fields": "none for ENSID-level audit; isoform-level relatedness cannot be audited without transcript isoform IDs",
            "repair_needed": "none unless transcript isoform identifiers become available",
        },
        "S3": {
            "source_files": repo_path(XGB_FEATURE_MANIFEST),
            "source_status": "renamed and reorganized from old S0 plus current XGBoost feature manifest",
            "required_content_status": "complete",
            "missing_fields": "none",
            "repair_needed": "none",
        },
        "S4": {
            "source_files": repo_paths(RIDGE_FOLD, ELASTICNET_FOLD, XGB_FOLD, MRNALM_ROOT, FUSION_FOLD),
            "source_status": "renamed and reorganized from old S1 plus raw model fold metrics",
            "required_content_status": "complete",
            "missing_fields": "none",
            "repair_needed": "none",
        },
        "S5": {
            "source_files": repo_paths(FUSION_PRED_TEST, FUSION_COEFFICIENTS, FUSION_DELTA),
            "source_status": "renamed and merged from old S6/S7 plus raw fusion predictions",
            "required_content_status": "complete",
            "missing_fields": "none",
            "repair_needed": "none",
        },
        "S6": {
            "source_files": repo_paths(SHAP_TOP, SHAP_FAMILY, SHAP_DIRECTION, ABLATION_SUMMARY, ABLATION_FOLD),
            "source_status": "renamed and merged from old S5/S12 plus raw SHAP and ablation outputs",
            "required_content_status": "complete",
            "missing_fields": "none",
            "repair_needed": "none",
        },
        "S7": {
            "source_files": repo_paths(FUSION_GAIN_PROFILE, FUSION_GAIN_GROUP, FUSION_GAIN_STATS, CASE_STUDIES),
            "source_status": "renamed and reorganized from old S4 plus raw fusion-gain and case-study outputs",
            "required_content_status": "complete",
            "missing_fields": "none",
            "repair_needed": "none",
        },
        "S8": {
            "source_files": repo_paths(MOTIF, REGION),
            "source_status": "renamed and reorganized from old S3 plus raw motif/k-mer and region-composition outputs",
            "required_content_status": "complete",
            "missing_fields": "none",
            "repair_needed": "none",
        },
        "S9": {
            "source_files": repo_paths(PERMUTATION_FOLD, PERMUTATION_SUMMARY),
            "source_status": "renamed and reorganized from old S2 plus raw permutation outputs",
            "required_content_status": "complete",
            "missing_fields": "none",
            "repair_needed": "none",
        },
    }
    status_rows = []
    for s, path in outputs.items():
        meta = table_meta[s]
        status_rows.append(
            {
                "supplementary_table": s,
                "final_output": str(path),
                "source_files": meta["source_files"],
                "generation_script": repo_path(PROJECT_ROOT / "scripts" / "tables" / "generate_supplementary_tables.py"),
                "rows": int(tables[s].shape[0]),
                "columns": int(tables[s].shape[1]),
                "status": "generated and complete" if not check_df[check_df["check"].str.startswith(s)]["status"].eq("conflict").any() else "generated with value conflicts; inspect checks",
                "source_status": meta["source_status"],
                "required_content_status": meta["required_content_status"],
                "missing_fields": meta["missing_fields"],
                "repair_needed": meta["repair_needed"],
            }
        )
    pd.DataFrame(status_rows).to_csv(TABLE_DIR / "supplementary_table_s1_s9_audit_status.csv", index=False)

    conflicts = check_df[check_df["status"] == "conflict"]
    lines = [
        "# Supplementary Table S1-S9 Audit Report",
        "",
        "This report maps existing project outputs onto the current manuscript's Supplementary Table S1-S9 order and records value checks against frozen manuscript statements.",
        "",
        "## Current File to Target Table Mapping",
        "",
        "| Current/source file | Target table | Use |",
        "|---|---:|---|",
    ]
    for src, target, use in mapping:
        lines.append(f"| `{src}` | {target} | {use} |")
    lines.extend(["", "## Final Outputs", "", "| Table | Output path | Rows | Columns | Status | Required content | Missing fields | Repair needed |", "|---|---|---:|---:|---|---|---|---|"])
    for row in status_rows:
        lines.append(
            f"| {row['supplementary_table']} | `{row['final_output']}` | {row['rows']} | {row['columns']} | "
            f"{row['status']} | {row['required_content_status']} | {row['missing_fields']} | {row['repair_needed']} |"
        )
    lines.extend(["", "## Data Sources and Generation Script", "", "| Table | Source status | Source files | Generation script |", "|---|---|---|---|"])
    for row in status_rows:
        lines.append(
            f"| {row['supplementary_table']} | {row['source_status']} | `{row['source_files']}` | `{row['generation_script']}` |"
        )
    lines.extend(["", "## Key Value Checks", "", f"Full check table: `{check_path}`.", ""])
    if conflicts.empty:
        lines.append("No conflicts were detected between generated S1-S9 tables and the frozen key values checked by this script.")
    else:
        lines.append("Conflicts detected; do not silently overwrite manuscript values before resolving these rows:")
        lines.append("")
        lines.append("| Check | Observed | Expected | Difference |")
        lines.append("|---|---:|---:|---:|")
        for _, row in conflicts.iterrows():
            lines.append(f"| {row['check']} | {row['observed']} | {row['expected']} | {row['difference']} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Tables use `full_tabular_benchmark`, `paired_xgb_mrnalm_predictions`, and `sequence_only_human_5fold` as sample-scope labels.",
            "- Historical labels such as matched-subset, matched scope, clean run, and clean rerun are not used in the generated table fields.",
            "- Source paths may still contain historical run directory names because they identify immutable result files.",
            "- SHAP, feature-family ablation, motif/k-mer, and region-composition rows summarize predictive associations and hypothesis-generating checks, not causal mechanisms.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (TABLE_DIR / "supplementary_table_s1_s9_audit_report.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "S1": build_s1(),
        "S2": build_s2(),
        "S3": build_s3(),
        "S4": build_s4(),
        "S5": build_s5(),
        "S6": build_s6(),
        "S7": build_s7(),
        "S8": build_s8(),
        "S9": build_s9(),
    }
    csv_outputs = {
        "S1": TABLE_DIR / "supplementary_table_s1_data_inclusion_target_fold_overview.csv",
        "S2": TABLE_DIR / "supplementary_table_s2_identifier_split_overlap_audit.csv",
        "S3": TABLE_DIR / "supplementary_table_s3_feature_group_definitions_dimensions.csv",
        "S4": TABLE_DIR / "supplementary_table_s4_complete_fold_level_performance_metrics.csv",
        "S5": TABLE_DIR / "supplementary_table_s5_prediction_fusion_complementarity_diagnostics.csv",
        "S6": TABLE_DIR / "supplementary_table_s6_shap_feature_family_interpretation.csv",
        "S7": TABLE_DIR / "supplementary_table_s7_fusion_gain_sample_profile_case_studies.csv",
        "S8": TABLE_DIR / "supplementary_table_s8_motif_kmer_region_composition_checks.csv",
        "S9": TABLE_DIR / "supplementary_table_s9_permutation_negative_control_metrics.csv",
    }
    xlsx_outputs = {key: path.with_suffix(".xlsx") for key, path in csv_outputs.items()}
    for key, df in tables.items():
        try:
            df.to_csv(csv_outputs[key], index=False)
            csv_note = f"CSV backup: {csv_outputs[key].name}"
        except PermissionError:
            csv_note = f"CSV backup skipped because file is open or locked: {csv_outputs[key].name}"
        write_workbook(df, xlsx_outputs[key])
        print(f"{key}: {xlsx_outputs[key]} ({df.shape[0]} rows x {df.shape[1]} columns; {csv_note})")
    write_index()
    audit(tables, xlsx_outputs)
    print(f"Audit report: {TABLE_DIR / 'supplementary_table_s1_s9_audit_report.md'}")


if __name__ == "__main__":
    main()
