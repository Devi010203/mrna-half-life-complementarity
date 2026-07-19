#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No-args prediction-level late fusion for mRNA-LM and Saluki XGB outputs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


# =============================================================================
# CONFIG: paths are resolved from environment first. Fill the *_BATCH_ROOT
# values in the SLURM template before submitting a clean fusion run.
# =============================================================================
def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else Path(default)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid {name}={value!r}; expected boolean")


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else float(default)


PROJECT_ROOT = env_path("SALUKI_PROJECT_ROOT", Path(__file__).resolve().parents[2])
RUNS_ROOT = env_path("SALUKI_RUNS_ROOT", PROJECT_ROOT / "results" / "runs")

XGB_BATCH_ROOT = env_path(
    "SALUKI_FUSION_XGB_BATCH_ROOT",
    RUNS_ROOT / "xgboost",
)
XGB_FEATURE_GROUP = env_str("SALUKI_FUSION_XGB_FEATURE_GROUP", "basic_plus_biochemical_full")
XGB_VAL_PRED_PATH = env_path("SALUKI_FUSION_XGB_VAL_PRED_PATH", XGB_BATCH_ROOT / "pred_val.csv")
XGB_TEST_PRED_PATH = env_path("SALUKI_FUSION_XGB_TEST_PRED_PATH", XGB_BATCH_ROOT / "pred_test.csv")
XGB_FULL_SUMMARY_PATH = env_path("SALUKI_FUSION_XGB_FULL_SUMMARY_PATH", XGB_BATCH_ROOT / "summary_metrics.csv")

MRNALM_BATCH_ROOT = env_path(
    "SALUKI_FUSION_MRNALM_BATCH_ROOT",
    RUNS_ROOT / "mrnalm_agghead",
)
MRNALM_MODEL_NAME = env_str("SALUKI_FUSION_MRNALM_MODEL_NAME", "mrnalm_best")
MRNALM_VAL_PRED_PATH = env_path("SALUKI_FUSION_MRNALM_VAL_PRED_PATH", MRNALM_BATCH_ROOT / "pred_val.csv")
MRNALM_TEST_PRED_PATH = env_path("SALUKI_FUSION_MRNALM_TEST_PRED_PATH", MRNALM_BATCH_ROOT / "pred_test.csv")
FILTER_MRNALM_ROWS = env_bool("SALUKI_FUSION_FILTER_MRNALM_ROWS", True)
MRNALM_FILTER_COL = env_str("SALUKI_FUSION_MRNALM_FILTER_COL", "model_name")
MRNALM_FILTER_VALUE = env_str("SALUKI_FUSION_MRNALM_FILTER_VALUE", "mrnalm_best")

RUN_MATCHED_BASELINE_COMPARISON = env_bool("SALUKI_FUSION_RUN_MATCHED_BASELINE_COMPARISON", True)
RIDGE_TEST_PRED_PATH = env_path(
    "SALUKI_FUSION_RIDGE_TEST_PRED_PATH",
    RUNS_ROOT / "ridge" / "pred_test.csv",
)
RIDGE_FEATURE_GROUP = env_str("SALUKI_FUSION_RIDGE_FEATURE_GROUP", "basic_plus_biochemical_full")
ELASTICNET_TEST_PRED_PATH = env_path(
    "SALUKI_FUSION_ELASTICNET_TEST_PRED_PATH",
    RUNS_ROOT / "elasticnet" / "pred_test_merged.csv",
)
ELASTICNET_FEATURE_GROUP = env_str("SALUKI_FUSION_ELASTICNET_FEATURE_GROUP", "basic_plus_biochemical_full")

RUN_STRATIFIED_ANALYSIS = env_bool("SALUKI_FUSION_RUN_STRATIFIED_ANALYSIS", True)
BASIC_FEATURE_PATH = env_path(
    "SALUKI_FUSION_BASIC_FEATURE_PATH",
    os.environ.get(
        "SALUKI_GENETIC_BIOCHEMICAL_PATH",
        PROJECT_ROOT / "data" / "saluki_features" / "human" / "alldataGeneticBiochemical.tsv.gz",
    ),
)
BASIC_STRATIFY_FEATURES = [
    "BASIC.UTR3LEN",
    "BASIC.CDSLEN",
    "BASIC.ORFEXONDENSITY",
    "BASIC.UTR3GC",
]

OUT_BASE_DIR = env_path("SALUKI_FUSION_OUT_BASE_DIR", RUNS_ROOT)
RUN_TAG = env_str("SALUKI_FUSION_RUN_TAG", "late_fusion")
OUT_DIR = OUT_BASE_DIR / RUN_TAG

CV_FOLDS = [0, 1, 2, 3, 4]

ID_COL_CANDIDATES = ["ENSID", "GENEID", "gene_id", "id", "sample_id"]
FOLD_COL_CANDIDATES = ["cv_fold", "fold"]
Y_TRUE_COL_CANDIDATES = ["y_true", "HALFLIFE", "target"]
Y_PRED_COL_CANDIDATES = ["y_pred", "pred", "prediction"]

Y_TRUE_ATOL = env_float("SALUKI_FUSION_Y_TRUE_ATOL", 1e-5)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def config_dict() -> dict[str, Any]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "timestamp": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "xgb_batch_root": str(XGB_BATCH_ROOT),
        "xgb_feature_group": XGB_FEATURE_GROUP,
        "xgb_val_pred_path": str(XGB_VAL_PRED_PATH),
        "xgb_test_pred_path": str(XGB_TEST_PRED_PATH),
        "xgb_full_summary_path": str(XGB_FULL_SUMMARY_PATH),
        "mrnalm_batch_root": str(MRNALM_BATCH_ROOT),
        "mrnalm_model_name": MRNALM_MODEL_NAME,
        "mrnalm_val_pred_path": str(MRNALM_VAL_PRED_PATH),
        "mrnalm_test_pred_path": str(MRNALM_TEST_PRED_PATH),
        "filter_mrnalm_rows": bool(FILTER_MRNALM_ROWS),
        "mrnalm_filter_col": MRNALM_FILTER_COL,
        "mrnalm_filter_value": MRNALM_FILTER_VALUE,
        "run_matched_baseline_comparison": bool(RUN_MATCHED_BASELINE_COMPARISON),
        "ridge_test_pred_path": str(RIDGE_TEST_PRED_PATH),
        "ridge_feature_group": RIDGE_FEATURE_GROUP,
        "elasticnet_test_pred_path": str(ELASTICNET_TEST_PRED_PATH),
        "elasticnet_feature_group": ELASTICNET_FEATURE_GROUP,
        "run_stratified_analysis": bool(RUN_STRATIFIED_ANALYSIS),
        "basic_feature_path": str(BASIC_FEATURE_PATH),
        "basic_stratify_features": list(BASIC_STRATIFY_FEATURES),
        "out_base_dir": str(OUT_BASE_DIR),
        "run_tag": RUN_TAG,
        "out_dir": str(OUT_DIR),
        "cv_folds": list(CV_FOLDS),
        "id_col_candidates": list(ID_COL_CANDIDATES),
        "fold_col_candidates": list(FOLD_COL_CANDIDATES),
        "y_true_col_candidates": list(Y_TRUE_COL_CANDIDATES),
        "y_pred_col_candidates": list(Y_PRED_COL_CANDIDATES),
        "y_true_atol": Y_TRUE_ATOL,
    }


def print_config_and_raise(message: str) -> None:
    print("[CONFIG]")
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False))
    raise FileNotFoundError(message)


def validate_config_paths() -> None:
    required_paths = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "XGB_VAL_PRED_PATH": XGB_VAL_PRED_PATH,
        "XGB_TEST_PRED_PATH": XGB_TEST_PRED_PATH,
        "XGB_FULL_SUMMARY_PATH": XGB_FULL_SUMMARY_PATH,
        "MRNALM_VAL_PRED_PATH": MRNALM_VAL_PRED_PATH,
        "MRNALM_TEST_PRED_PATH": MRNALM_TEST_PRED_PATH,
    }
    if RUN_MATCHED_BASELINE_COMPARISON:
        required_paths.update(
            {
                "RIDGE_TEST_PRED_PATH": RIDGE_TEST_PRED_PATH,
                "ELASTICNET_TEST_PRED_PATH": ELASTICNET_TEST_PRED_PATH,
            }
        )
    if RUN_STRATIFIED_ANALYSIS:
        required_paths["BASIC_FEATURE_PATH"] = BASIC_FEATURE_PATH
    missing = {name: str(path) for name, path in required_paths.items() if not Path(path).exists()}
    if missing:
        print_config_and_raise(f"Missing CONFIG path(s): {missing}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def report_path_text(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def find_first_column(columns: list[str], candidates: list[str], role: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find {role} column. Candidates={candidates}, available={columns}")


def print_config() -> None:
    print("[CONFIG]")
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False))


def normalize_prediction_frame(
    path: Path,
    source_name: str,
    filter_col: str | None = None,
    filter_value: str | None = None,
    require_filter_col: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if filter_col and filter_value is not None:
        if filter_col not in df.columns:
            if require_filter_col:
                raise ValueError(f"{source_name} missing filter column {filter_col!r}: {path}")
        else:
            df = df.loc[df[filter_col].astype(str) == str(filter_value)].copy()
            if df.empty:
                raise ValueError(f"{source_name} has no rows for {filter_col}={filter_value!r}: {path}")

    columns = [str(col) for col in df.columns]
    id_col = find_first_column(columns, ID_COL_CANDIDATES, f"{source_name} sample id")
    fold_col = find_first_column(columns, FOLD_COL_CANDIDATES, f"{source_name} fold")
    y_true_col = find_first_column(columns, Y_TRUE_COL_CANDIDATES, f"{source_name} y_true")
    y_pred_col = find_first_column(columns, Y_PRED_COL_CANDIDATES, f"{source_name} y_pred")

    out = pd.DataFrame(
        {
            "sample_id": df[id_col].astype(str),
            "cv_fold": pd.to_numeric(df[fold_col], errors="raise").astype(int),
            "y_true": pd.to_numeric(df[y_true_col], errors="raise").astype(float),
            "y_pred": pd.to_numeric(df[y_pred_col], errors="raise").astype(float),
        }
    )
    duplicated = out.duplicated(["sample_id", "cv_fold"], keep=False)
    if duplicated.any():
        examples = out.loc[duplicated, ["sample_id", "cv_fold"]].head(10).to_dict("records")
        raise ValueError(f"{source_name} has duplicate sample_id/cv_fold rows after filtering: {examples}")
    return out


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    denom = np.sqrt(np.nansum(x * x)) * np.sqrt(np.nansum(y * y))
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.nansum(x * y) / denom)


def spearmanr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return pearsonr_np(rx, ry)


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "r2": r2_np(y_true, y_pred),
        "pearson": pearsonr_np(y_true, y_pred),
        "spearman": spearmanr_np(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def prefixed_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def check_y_true_alignment(merged: pd.DataFrame, split: str, fold: int) -> None:
    if not np.allclose(
        merged["y_true_xgb"].to_numpy(dtype=np.float64),
        merged["y_true_mrnalm"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=Y_TRUE_ATOL,
    ):
        max_diff = float(
            np.max(np.abs(merged["y_true_xgb"].to_numpy() - merged["y_true_mrnalm"].to_numpy()))
        )
        raise ValueError(f"y_true mismatch in {split} fold {fold}; max_abs_diff={max_diff}")


def fit_methods(
    y_val: np.ndarray,
    xgb_val: np.ndarray,
    mrnalm_val: np.ndarray,
    xgb_test: np.ndarray,
    mrnalm_test: np.ndarray,
) -> dict[str, dict[str, Any]]:
    val_two = np.column_stack([xgb_val, mrnalm_val])
    test_two = np.column_stack([xgb_test, mrnalm_test])

    xgb_cal = LinearRegression()
    xgb_cal.fit(xgb_val.reshape(-1, 1), y_val)

    mrnalm_cal = LinearRegression()
    mrnalm_cal.fit(mrnalm_val.reshape(-1, 1), y_val)

    linear_fusion = LinearRegression()
    linear_fusion.fit(val_two, y_val)

    ridge_fusion = Ridge(alpha=1.0, fit_intercept=True)
    ridge_fusion.fit(val_two, y_val)

    return {
        "xgb_only": {
            "val_pred": xgb_val,
            "test_pred": xgb_test,
            "intercept": 0.0,
            "coef_xgb": 1.0,
            "coef_mrnalm": 0.0,
        },
        "mrnalm_only": {
            "val_pred": mrnalm_val,
            "test_pred": mrnalm_test,
            "intercept": 0.0,
            "coef_xgb": 0.0,
            "coef_mrnalm": 1.0,
        },
        "xgb_calibrated": {
            "val_pred": xgb_cal.predict(xgb_val.reshape(-1, 1)),
            "test_pred": xgb_cal.predict(xgb_test.reshape(-1, 1)),
            "intercept": float(xgb_cal.intercept_),
            "coef_xgb": float(xgb_cal.coef_[0]),
            "coef_mrnalm": 0.0,
        },
        "mrnalm_calibrated": {
            "val_pred": mrnalm_cal.predict(mrnalm_val.reshape(-1, 1)),
            "test_pred": mrnalm_cal.predict(mrnalm_test.reshape(-1, 1)),
            "intercept": float(mrnalm_cal.intercept_),
            "coef_xgb": 0.0,
            "coef_mrnalm": float(mrnalm_cal.coef_[0]),
        },
        "simple_average": {
            "val_pred": 0.5 * xgb_val + 0.5 * mrnalm_val,
            "test_pred": 0.5 * xgb_test + 0.5 * mrnalm_test,
            "intercept": 0.0,
            "coef_xgb": 0.5,
            "coef_mrnalm": 0.5,
        },
        "linear_fusion": {
            "val_pred": linear_fusion.predict(val_two),
            "test_pred": linear_fusion.predict(test_two),
            "intercept": float(linear_fusion.intercept_),
            "coef_xgb": float(linear_fusion.coef_[0]),
            "coef_mrnalm": float(linear_fusion.coef_[1]),
        },
        "ridge_fusion_alpha1": {
            "val_pred": ridge_fusion.predict(val_two),
            "test_pred": ridge_fusion.predict(test_two),
            "intercept": float(ridge_fusion.intercept_),
            "coef_xgb": float(ridge_fusion.coef_[0]),
            "coef_mrnalm": float(ridge_fusion.coef_[1]),
        },
    }


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in fold_metrics.columns
        if col.startswith("val_") or col.startswith("test_")
    ]
    rows: list[dict[str, Any]] = []
    for method, sub in fold_metrics.groupby("method", sort=False):
        for col in metric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            split, metric = col.split("_", 1)
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "n_folds": int(values.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def build_per_fold_r2_coefficients(fold_metrics: pd.DataFrame, coeff: pd.DataFrame) -> pd.DataFrame:
    r2_wide = (
        fold_metrics.loc[
            fold_metrics["method"].isin(["xgb_only", "mrnalm_only", "linear_fusion"]),
            ["cv_fold", "method", "test_n", "test_r2"],
        ]
        .pivot(index="cv_fold", columns="method", values="test_r2")
        .reset_index()
    )
    n_by_fold = (
        fold_metrics.loc[fold_metrics["method"] == "xgb_only", ["cv_fold", "test_n"]]
        .drop_duplicates("cv_fold")
        .copy()
    )
    linear_coef = (
        coeff.loc[coeff["method"] == "linear_fusion", ["cv_fold", "coef_xgb", "coef_mrnalm"]]
        .drop_duplicates("cv_fold")
        .copy()
    )
    table = n_by_fold.merge(r2_wide, on="cv_fold", how="left").merge(linear_coef, on="cv_fold", how="left")
    table = table.rename(
        columns={
            "cv_fold": "fold",
            "xgb_only": "xgb_only_r2",
            "mrnalm_only": "mrnalm_r2",
            "linear_fusion": "linear_fusion_r2",
        }
    )
    table["delta_r2_vs_xgb"] = table["linear_fusion_r2"] - table["xgb_only_r2"]
    return table[
        [
            "fold",
            "test_n",
            "xgb_only_r2",
            "mrnalm_r2",
            "linear_fusion_r2",
            "delta_r2_vs_xgb",
            "coef_xgb",
            "coef_mrnalm",
        ]
    ].sort_values("fold")


def get_late_summary(summary: pd.DataFrame, method: str, metric: str) -> tuple[float, float]:
    hit = summary.loc[
        (summary["method"] == method) & (summary["split"] == "test") & (summary["metric"] == metric)
    ]
    if hit.empty:
        return float("nan"), float("nan")
    return float(hit["mean"].iloc[0]), float(hit["std"].iloc[0])


def load_xgb_full_summary_r2(path: Path) -> tuple[float, float]:
    full_summary = pd.read_csv(path)
    required = {"feature_group", "split", "metric", "mean", "std"}
    missing = required.difference(full_summary.columns)
    if missing:
        raise ValueError(f"XGB full summary missing columns {sorted(missing)}: {path}")
    hit = full_summary.loc[
        (full_summary["feature_group"].astype(str) == str(XGB_FEATURE_GROUP))
        & (full_summary["split"].astype(str) == "test")
        & (full_summary["metric"].astype(str) == "r2")
    ]
    if hit.empty:
        raise ValueError(f"No full XGB test R2 for feature_group={XGB_FEATURE_GROUP!r}: {path}")
    return float(hit["mean"].iloc[0]), float(hit["std"].iloc[0])


def build_sample_scope_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    full_xgb_mean, full_xgb_std = load_xgb_full_summary_r2(XGB_FULL_SUMMARY_PATH)
    xgb_mean, xgb_std = get_late_summary(summary, "xgb_only", "r2")
    mrnalm_mean, mrnalm_std = get_late_summary(summary, "mrnalm_only", "r2")
    fusion_mean, fusion_std = get_late_summary(summary, "linear_fusion", "r2")
    return pd.DataFrame(
        [
            {
                "setting": "full benchmark",
                "sample_scope": "all XGB test samples",
                "model": "XGB",
                "r2_mean": full_xgb_mean,
                "r2_std": full_xgb_std,
            },
            {
                "setting": "late fusion",
                "sample_scope": "matched XGB intersect mRNA-LM samples",
                "model": "XGB-only",
                "r2_mean": xgb_mean,
                "r2_std": xgb_std,
            },
            {
                "setting": "late fusion",
                "sample_scope": "matched XGB intersect mRNA-LM samples",
                "model": "mRNA-LM-only",
                "r2_mean": mrnalm_mean,
                "r2_std": mrnalm_std,
            },
            {
                "setting": "late fusion",
                "sample_scope": "matched XGB intersect mRNA-LM samples",
                "model": "linear fusion",
                "r2_mean": fusion_mean,
                "r2_std": fusion_std,
            },
        ]
    )


def summarize_residual_complementarity(residual_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in residual_df.columns
        if col not in {"cv_fold", "test_n"}
    ]
    rows: list[dict[str, Any]] = []
    for col in metric_cols:
        values = pd.to_numeric(residual_df[col], errors="coerce")
        rows.append(
            {
                "metric": col,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "n_folds": int(values.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def build_core_test_predictions(pred_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = (
        pred_test.loc[
            pred_test["method"] == "xgb_only",
            ["sample_id", "cv_fold", "y_true", "xgb_pred", "mrnalm_pred"],
        ]
        .drop_duplicates(["sample_id", "cv_fold"])
        .copy()
    )
    fusion = (
        pred_test.loc[
            pred_test["method"] == "linear_fusion",
            ["sample_id", "cv_fold", "y_pred"],
        ]
        .rename(columns={"y_pred": "linear_fusion_pred"})
        .drop_duplicates(["sample_id", "cv_fold"])
        .copy()
    )
    wide = base.merge(fusion, on=["sample_id", "cv_fold"], how="inner")
    if len(wide) != len(base):
        raise ValueError("Could not align xgb_only and linear_fusion predictions for the matched subset.")

    rows = []
    model_defs = [
        ("xgb_basic_plus_biochemical_full", "XGB basic_plus_biochemical_full", "xgb_pred"),
        ("mrnalm", "mRNA-LM", "mrnalm_pred"),
        ("late_fusion_linear", "late fusion linear", "linear_fusion_pred"),
    ]
    for model, model_label, pred_col in model_defs:
        rows.append(
            pd.DataFrame(
                {
                    "sample_id": wide["sample_id"].to_numpy(),
                    "cv_fold": wide["cv_fold"].to_numpy(dtype=int),
                    "model": model,
                    "model_label": model_label,
                    "y_true": wide["y_true"].to_numpy(dtype=np.float64),
                    "y_pred": wide[pred_col].to_numpy(dtype=np.float64),
                }
            )
        )
    return wide, pd.concat(rows, ignore_index=True)


def append_external_baseline_predictions(
    model_predictions: pd.DataFrame,
    matched_wide: pd.DataFrame,
    path: Path,
    source_name: str,
    feature_group: str,
    model: str,
    model_label: str,
) -> pd.DataFrame:
    external = normalize_prediction_frame(
        path,
        source_name,
        filter_col="feature_group",
        filter_value=feature_group,
        require_filter_col=True,
    )
    key = ["sample_id", "cv_fold"]
    matched_keys = matched_wide[key + ["y_true"]].copy()
    merged = matched_keys.merge(external, on=key, how="left", suffixes=("_matched", "_external"))
    if merged["y_pred"].isna().any():
        missing = merged.loc[merged["y_pred"].isna(), key].head(10).to_dict("records")
        raise ValueError(f"{source_name} missing predictions for matched subset examples: {missing}")
    if not np.allclose(
        merged["y_true_matched"].to_numpy(dtype=np.float64),
        merged["y_true_external"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=Y_TRUE_ATOL,
    ):
        max_diff = float(
            np.max(
                np.abs(
                    merged["y_true_matched"].to_numpy(dtype=np.float64)
                    - merged["y_true_external"].to_numpy(dtype=np.float64)
                )
            )
        )
        raise ValueError(f"{source_name} y_true mismatch on matched subset; max_abs_diff={max_diff}")

    extra = pd.DataFrame(
        {
            "sample_id": merged["sample_id"].to_numpy(),
            "cv_fold": merged["cv_fold"].to_numpy(dtype=int),
            "model": model,
            "model_label": model_label,
            "y_true": merged["y_true_matched"].to_numpy(dtype=np.float64),
            "y_pred": merged["y_pred"].to_numpy(dtype=np.float64),
        }
    )
    return pd.concat([model_predictions, extra], ignore_index=True)


def metric_rows_by_group(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(group_cols, sort=False, dropna=False)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        y_true = sub["y_true"].to_numpy(dtype=np.float64)
        y_pred = sub["y_pred"].to_numpy(dtype=np.float64)
        row["n"] = int(len(sub))
        row.update(regression_metrics(y_true, y_pred))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metric_rows(metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    candidate_metric_cols = [
        "r2",
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "delta_r2_vs_xgb",
        "delta_mae_vs_xgb",
        "mae_improvement_vs_xgb",
        "rmse_improvement_vs_xgb",
    ]
    metric_cols = [col for col in candidate_metric_cols if col in metrics.columns]
    rows: list[dict[str, Any]] = []
    for keys, sub in metrics.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {col: value for col, value in zip(group_cols, keys)}
        for metric in metric_cols:
            values = pd.to_numeric(sub[metric], errors="coerce")
            row = dict(base)
            row.update(
                {
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "n_folds": int(values.notna().sum()),
                    "n_total": int(pd.to_numeric(sub["n"], errors="coerce").sum()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def add_delta_vs_xgb(metrics: pd.DataFrame, stratum_cols: list[str]) -> pd.DataFrame:
    xgb = metrics.loc[
        metrics["model"] == "xgb_basic_plus_biochemical_full",
        stratum_cols + ["r2", "rmse", "mae"],
    ].rename(columns={"r2": "xgb_r2", "rmse": "xgb_rmse", "mae": "xgb_mae"})
    out = metrics.merge(xgb, on=stratum_cols, how="left")
    out["delta_r2_vs_xgb"] = out["r2"] - out["xgb_r2"]
    out["delta_mae_vs_xgb"] = out["mae"] - out["xgb_mae"]
    out["mae_improvement_vs_xgb"] = out["xgb_mae"] - out["mae"]
    out["rmse_improvement_vs_xgb"] = out["xgb_rmse"] - out["rmse"]
    return out


def add_tertile_by_fold(df: pd.DataFrame, value_col: str, out_col: str, labels: list[str]) -> pd.DataFrame:
    out = df.copy()
    chunks = []
    for _, sub in out.groupby("cv_fold", sort=False):
        ranked = sub[value_col].rank(method="first")
        sub = sub.copy()
        sub[out_col] = pd.qcut(ranked, q=len(labels), labels=labels).astype(str)
        chunks.append(sub)
    return pd.concat(chunks, ignore_index=True)


def add_low_high_by_fold(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    chunks = []
    for _, sub in df.groupby("cv_fold", sort=False):
        sub = sub.copy()
        valid = pd.to_numeric(sub[value_col], errors="coerce")
        if valid.notna().sum() == 0:
            sub[out_col] = "missing"
        else:
            median = float(valid.median())
            sub[out_col] = np.where(valid.isna(), "missing", np.where(valid <= median, "low", "high"))
        chunks.append(sub)
    return pd.concat(chunks, ignore_index=True)


def load_basic_feature_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    usecols = ["GENEID", *columns]
    df = pd.read_csv(path, sep="\t", compression="infer", usecols=usecols)
    missing = set(usecols).difference(df.columns)
    if missing:
        raise ValueError(f"BASIC feature table missing columns {sorted(missing)}: {path}")
    df = df.rename(columns={"GENEID": "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].duplicated().any():
        examples = df.loc[df["sample_id"].duplicated(), "sample_id"].head(10).tolist()
        raise ValueError(f"BASIC feature table has duplicate GENEID rows: {examples}")
    return df


def build_stratified_metrics(
    matched_wide: pd.DataFrame,
    core_predictions: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    xgb_error = matched_wide.copy()
    xgb_error["xgb_abs_error"] = np.abs(xgb_error["y_true"] - xgb_error["xgb_pred"])
    xgb_error = add_tertile_by_fold(
        xgb_error,
        "xgb_abs_error",
        "stratum",
        ["low_xgb_error", "medium_xgb_error", "high_xgb_error"],
    )
    joined = core_predictions.merge(xgb_error[["sample_id", "cv_fold", "stratum"]], on=["sample_id", "cv_fold"])
    metrics = metric_rows_by_group(joined, ["cv_fold", "stratum", "model", "model_label"])
    metrics = add_delta_vs_xgb(metrics, ["cv_fold", "stratum"])
    summary = summarize_metric_rows(metrics, ["stratum", "model", "model_label"])
    outputs["xgb_error"] = (metrics, summary)

    halflife = add_tertile_by_fold(
        matched_wide,
        "y_true",
        "stratum",
        ["short_halflife", "medium_halflife", "long_halflife"],
    )
    joined = core_predictions.merge(halflife[["sample_id", "cv_fold", "stratum"]], on=["sample_id", "cv_fold"])
    metrics = metric_rows_by_group(joined, ["cv_fold", "stratum", "model", "model_label"])
    metrics = add_delta_vs_xgb(metrics, ["cv_fold", "stratum"])
    summary = summarize_metric_rows(metrics, ["stratum", "model", "model_label"])
    outputs["halflife"] = (metrics, summary)

    basic = load_basic_feature_frame(BASIC_FEATURE_PATH, BASIC_STRATIFY_FEATURES)
    basic_wide = matched_wide.merge(basic, on="sample_id", how="left")
    if basic_wide[BASIC_STRATIFY_FEATURES].isna().all(axis=None):
        raise ValueError(f"No BASIC feature values matched sample_id in {BASIC_FEATURE_PATH}")

    basic_metric_chunks = []
    for feature in BASIC_STRATIFY_FEATURES:
        feature_name = feature.replace("BASIC.", "")
        tmp = add_low_high_by_fold(basic_wide, feature, "stratum")
        tmp["stratification"] = feature_name
        joined = core_predictions.merge(
            tmp[["sample_id", "cv_fold", "stratification", "stratum"]],
            on=["sample_id", "cv_fold"],
        )
        metrics = metric_rows_by_group(
            joined,
            ["cv_fold", "stratification", "stratum", "model", "model_label"],
        )
        metrics = add_delta_vs_xgb(metrics, ["cv_fold", "stratification", "stratum"])
        basic_metric_chunks.append(metrics)
    basic_metrics = pd.concat(basic_metric_chunks, ignore_index=True)
    basic_summary = summarize_metric_rows(basic_metrics, ["stratification", "stratum", "model", "model_label"])
    outputs["basic_features"] = (basic_metrics, basic_summary)
    return outputs


def write_report(
    summary: pd.DataFrame,
    delta_df: pd.DataFrame,
    per_fold_table: pd.DataFrame,
    sample_scope: pd.DataFrame,
    residual_df: pd.DataFrame,
    residual_summary: pd.DataFrame,
    matched_model_summary: pd.DataFrame | None,
    stratified_summaries: dict[str, pd.DataFrame] | None,
    path: Path,
) -> None:
    def get_summary(method: str, metric: str) -> tuple[float, float]:
        return get_late_summary(summary, method, metric)

    methods = [
        "xgb_only",
        "mrnalm_only",
        "xgb_calibrated",
        "mrnalm_calibrated",
        "simple_average",
        "linear_fusion",
        "ridge_fusion_alpha1",
    ]
    lines = [
        "# Late Fusion mRNA-LM + XGB Report",
        "",
        "## Inputs",
        f"- XGB val: `{report_path_text(XGB_VAL_PRED_PATH)}`",
        f"- XGB test: `{report_path_text(XGB_TEST_PRED_PATH)}`",
        f"- XGB feature group: `{XGB_FEATURE_GROUP}`",
        f"- mRNA-LM val: `{report_path_text(MRNALM_VAL_PRED_PATH)}`",
        f"- mRNA-LM test: `{report_path_text(MRNALM_TEST_PRED_PATH)}`",
        f"- mRNA-LM model name: `{MRNALM_MODEL_NAME}`",
    ]
    if RUN_MATCHED_BASELINE_COMPARISON:
        lines.extend(
            [
                f"- Ridge matched-subset test: `{report_path_text(RIDGE_TEST_PRED_PATH)}`",
                f"- Ridge feature group: `{RIDGE_FEATURE_GROUP}`",
                f"- ElasticNet matched-subset test: `{report_path_text(ELASTICNET_TEST_PRED_PATH)}`",
                f"- ElasticNet feature group: `{ELASTICNET_FEATURE_GROUP}`",
            ]
        )
    if RUN_STRATIFIED_ANALYSIS:
        lines.append(f"- BASIC feature table: `{report_path_text(BASIC_FEATURE_PATH)}`")
    lines.extend(
        [
            "",
            "## Test Summary",
            "",
            "| method | test R2 | test Pearson | test Spearman |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in methods:
        r2_mean, r2_std = get_summary(method, "r2")
        pear_mean, pear_std = get_summary(method, "pearson")
        spear_mean, spear_std = get_summary(method, "spearman")
        lines.append(
            f"| {method} | {r2_mean:.6f} +/- {r2_std:.6f} | "
            f"{pear_mean:.6f} +/- {pear_std:.6f} | {spear_mean:.6f} +/- {spear_std:.6f} |"
        )

    linear_delta = delta_df.loc[delta_df["method"] == "linear_fusion", "delta_test_r2_vs_xgb"]
    mean_delta = float(linear_delta.mean()) if not linear_delta.empty else float("nan")
    if np.isfinite(mean_delta) and mean_delta > 0.005:
        interpretation = "mRNA-LM may provide weak complementary information."
    else:
        interpretation = "little additional complementarity beyond annotation-augmented XGB."

    lines.extend(
        [
            "",
            "## Delta",
            f"- linear_fusion mean delta test R2 vs xgb_only: `{mean_delta:.6f}`",
            f"- Interpretation: {interpretation}",
            "",
            "## Per-Fold Late Fusion Check",
            "",
            "| fold | test_n | xgb_only R2 | mRNA-LM R2 | linear_fusion R2 | delta | coef_xgb | coef_mrnalm |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_fold_table.to_dict("records"):
        lines.append(
            f"| {int(row['fold'])} | {int(row['test_n'])} | {row['xgb_only_r2']:.6f} | "
            f"{row['mrnalm_r2']:.6f} | {row['linear_fusion_r2']:.6f} | "
            f"{row['delta_r2_vs_xgb']:.6f} | {row['coef_xgb']:.6f} | {row['coef_mrnalm']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Sample Scope",
            "",
            "Late fusion is evaluated only on the matched subset with both XGB and mRNA-LM predictions. "
            "Therefore, the xgb_only number here is not the same estimand as the full XGB benchmark.",
            "",
            "| setting | sample scope | model | test R2 |",
            "|---|---|---|---:|",
        ]
    )
    for row in sample_scope.to_dict("records"):
        lines.append(
            f"| {row['setting']} | {row['sample_scope']} | {row['model']} | "
            f"{row['r2_mean']:.6f} +/- {row['r2_std']:.6f} |"
        )

    if matched_model_summary is not None and not matched_model_summary.empty:
        lines.extend(
            [
                "",
                "## Matched-Subset Model Comparison",
                "",
                "All models in this table are re-evaluated on the same matched test subset.",
                "",
                "| model | test R2 | Pearson | Spearman | RMSE | MAE |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        metric_lookup = (
            matched_model_summary.set_index(["model_label", "metric"])[["mean", "std"]].to_dict("index")
        )
        for model_label in matched_model_summary["model_label"].drop_duplicates().tolist():
            vals = {}
            for metric in ["r2", "pearson", "spearman", "rmse", "mae"]:
                item = metric_lookup.get((model_label, metric), {"mean": float("nan"), "std": float("nan")})
                vals[metric] = f"{item['mean']:.6f} +/- {item['std']:.6f}"
            lines.append(
                f"| {model_label} | {vals['r2']} | {vals['pearson']} | {vals['spearman']} | "
                f"{vals['rmse']} | {vals['mae']} |"
            )

    lines.extend(
        [
            "",
            "## Residual Complementarity",
            "",
            "| metric | mean | std | n_folds |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in residual_summary.to_dict("records"):
        lines.append(
            f"| {row['metric']} | {row['mean']:.6f} | {row['std']:.6f} | {int(row['n_folds'])} |"
        )
    lines.extend(["", "Per-fold values are saved in `late_fusion_residual_complementarity.csv`.", ""])

    if stratified_summaries:
        xgb_error_summary = stratified_summaries.get("xgb_error")
        if xgb_error_summary is not None and not xgb_error_summary.empty:
            lines.extend(
                [
                    "",
                    "## XGB Error Stratification",
                    "",
                    "| XGB error stratum | model | MAE | MAE improvement vs XGB | R2 delta vs XGB |",
                    "|---|---|---:|---:|---:|",
                ]
            )
            for _, row in xgb_error_summary.loc[
                xgb_error_summary["metric"].isin(["mae", "mae_improvement_vs_xgb", "delta_r2_vs_xgb"])
            ].pivot_table(
                index=["stratum", "model_label"],
                columns="metric",
                values="mean",
                aggfunc="first",
            ).reset_index().iterrows():
                lines.append(
                    f"| {row['stratum']} | {row['model_label']} | {row.get('mae', np.nan):.6f} | "
                    f"{row.get('mae_improvement_vs_xgb', np.nan):.6f} | {row.get('delta_r2_vs_xgb', np.nan):.6f} |"
                )
        halflife_summary = stratified_summaries.get("halflife")
        if halflife_summary is not None and not halflife_summary.empty:
            lines.extend(
                [
                    "",
                    "## Half-Life Stratification",
                    "",
                    "| half-life stratum | model | MAE | MAE improvement vs XGB |",
                    "|---|---|---:|---:|",
                ]
            )
            for _, row in halflife_summary.loc[
                halflife_summary["metric"].isin(["mae", "mae_improvement_vs_xgb"])
            ].pivot_table(
                index=["stratum", "model_label"],
                columns="metric",
                values="mean",
                aggfunc="first",
            ).reset_index().iterrows():
                lines.append(
                    f"| {row['stratum']} | {row['model_label']} | {row.get('mae', np.nan):.6f} | "
                    f"{row.get('mae_improvement_vs_xgb', np.nan):.6f} |"
                )
        if "basic_features" in stratified_summaries:
            lines.extend(
                [
                    "",
                    "## BASIC Feature Stratification",
                    "",
                    "Detailed low/high strata for UTR3LEN, CDSLEN, ORFEXONDENSITY, and UTR3GC are saved in "
                    "`late_fusion_stratified_by_basic_features.csv` and "
                    "`late_fusion_stratified_by_basic_features_summary.csv`.",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print_config()
    validate_config_paths()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run_config = config_dict()
    write_json(OUT_DIR / "late_fusion_run_config.json", run_config)
    write_json(OUT_DIR / "run_config.json", run_config)

    xgb_val = normalize_prediction_frame(
        XGB_VAL_PRED_PATH,
        "xgb_val",
        filter_col="feature_group",
        filter_value=XGB_FEATURE_GROUP,
        require_filter_col=False,
    )
    xgb_test = normalize_prediction_frame(
        XGB_TEST_PRED_PATH,
        "xgb_test",
        filter_col="feature_group",
        filter_value=XGB_FEATURE_GROUP,
        require_filter_col=False,
    )
    mrnalm_filter_col = MRNALM_FILTER_COL if FILTER_MRNALM_ROWS else None
    mrnalm_filter_value = MRNALM_FILTER_VALUE if FILTER_MRNALM_ROWS else None
    mrnalm_val = normalize_prediction_frame(
        MRNALM_VAL_PRED_PATH,
        "mrnalm_val",
        filter_col=mrnalm_filter_col,
        filter_value=mrnalm_filter_value,
        require_filter_col=bool(FILTER_MRNALM_ROWS),
    )
    mrnalm_test = normalize_prediction_frame(
        MRNALM_TEST_PRED_PATH,
        "mrnalm_test",
        filter_col=mrnalm_filter_col,
        filter_value=mrnalm_filter_value,
        require_filter_col=bool(FILTER_MRNALM_ROWS),
    )

    fold_rows: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    val_merged_rows: list[pd.DataFrame] = []
    test_pred_rows: list[pd.DataFrame] = []

    for fold in CV_FOLDS:
        xv = xgb_val.loc[xgb_val["cv_fold"] == int(fold)].copy()
        xt = xgb_test.loc[xgb_test["cv_fold"] == int(fold)].copy()
        mv = mrnalm_val.loc[mrnalm_val["cv_fold"] == int(fold)].copy()
        mt = mrnalm_test.loc[mrnalm_test["cv_fold"] == int(fold)].copy()
        if xv.empty or xt.empty or mv.empty or mt.empty:
            raise ValueError(
                f"Fold {fold} has empty input split: "
                f"xgb_val={len(xv)}, xgb_test={len(xt)}, mrnalm_val={len(mv)}, mrnalm_test={len(mt)}"
            )

        val = xv.merge(mv, on=["sample_id", "cv_fold"], how="inner", suffixes=("_xgb", "_mrnalm"))
        test = xt.merge(mt, on=["sample_id", "cv_fold"], how="inner", suffixes=("_xgb", "_mrnalm"))
        if val.empty or test.empty:
            raise ValueError(f"Fold {fold} merge produced empty val/test frame.")
        check_y_true_alignment(val, "val", int(fold))
        check_y_true_alignment(test, "test", int(fold))

        overlap_ids = set(val["sample_id"]).intersection(set(test["sample_id"]))
        if overlap_ids:
            raise ValueError(f"Fold {fold} val/test sample_id overlap; first examples={sorted(overlap_ids)[:10]}")

        y_val = val["y_true_xgb"].to_numpy(dtype=np.float64)
        y_test = test["y_true_xgb"].to_numpy(dtype=np.float64)
        xgb_val_pred = val["y_pred_xgb"].to_numpy(dtype=np.float64)
        xgb_test_pred = test["y_pred_xgb"].to_numpy(dtype=np.float64)
        mrnalm_val_pred = val["y_pred_mrnalm"].to_numpy(dtype=np.float64)
        mrnalm_test_pred = test["y_pred_mrnalm"].to_numpy(dtype=np.float64)

        residual_xgb = y_test - xgb_test_pred
        residual_mrnalm = y_test - mrnalm_test_pred
        residual_rows.append(
            {
                "cv_fold": int(fold),
                "test_n": int(len(y_test)),
                "pearson_xgb_residual_vs_mrnalm_pred": pearsonr_np(residual_xgb, mrnalm_test_pred),
                "spearman_xgb_residual_vs_mrnalm_pred": spearmanr_np(residual_xgb, mrnalm_test_pred),
                "pearson_xgb_error_vs_mrnalm_error": pearsonr_np(residual_xgb, residual_mrnalm),
                "spearman_xgb_error_vs_mrnalm_error": spearmanr_np(residual_xgb, residual_mrnalm),
            }
        )

        methods = fit_methods(y_val, xgb_val_pred, mrnalm_val_pred, xgb_test_pred, mrnalm_test_pred)
        val_merged_rows.append(
            pd.DataFrame(
                {
                    "sample_id": val["sample_id"].to_numpy(),
                    "cv_fold": int(fold),
                    "y_true": y_val,
                    "xgb_pred": xgb_val_pred,
                    "mrnalm_pred": mrnalm_val_pred,
                }
            )
        )

        for method, payload in methods.items():
            val_pred = np.asarray(payload["val_pred"], dtype=np.float64)
            test_pred = np.asarray(payload["test_pred"], dtype=np.float64)
            row = {
                "cv_fold": int(fold),
                "method": method,
                "val_n": int(len(y_val)),
                "test_n": int(len(y_test)),
                **prefixed_metrics(regression_metrics(y_val, val_pred), "val"),
                **prefixed_metrics(regression_metrics(y_test, test_pred), "test"),
            }
            fold_rows.append(row)
            coef_rows.append(
                {
                    "cv_fold": int(fold),
                    "method": method,
                    "intercept": float(payload["intercept"]),
                    "coef_xgb": float(payload["coef_xgb"]),
                    "coef_mrnalm": float(payload["coef_mrnalm"]),
                    "val_n": int(len(y_val)),
                    "test_n": int(len(y_test)),
                }
            )
            test_pred_rows.append(
                pd.DataFrame(
                    {
                        "sample_id": test["sample_id"].to_numpy(),
                        "cv_fold": int(fold),
                        "method": method,
                        "y_true": y_test,
                        "y_pred": test_pred,
                        "xgb_pred": xgb_test_pred,
                        "mrnalm_pred": mrnalm_test_pred,
                    }
                )
            )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = summarize_fold_metrics(fold_metrics)
    coeff = pd.DataFrame(coef_rows)
    residual = pd.DataFrame(residual_rows)
    residual_summary = summarize_residual_complementarity(residual)
    pred_val_merged = pd.concat(val_merged_rows, ignore_index=True)
    pred_test = pd.concat(test_pred_rows, ignore_index=True)

    xgb_r2 = (
        fold_metrics.loc[fold_metrics["method"] == "xgb_only", ["cv_fold", "test_r2"]]
        .rename(columns={"test_r2": "xgb_only_test_r2"})
        .copy()
    )
    delta = fold_metrics[["cv_fold", "method", "test_r2"]].merge(xgb_r2, on="cv_fold", how="left")
    delta["delta_test_r2_vs_xgb"] = delta["test_r2"] - delta["xgb_only_test_r2"]
    per_fold_table = build_per_fold_r2_coefficients(fold_metrics, coeff)
    sample_scope = build_sample_scope_comparison(summary)
    matched_wide, core_test_predictions = build_core_test_predictions(pred_test)

    matched_model_metrics: pd.DataFrame | None = None
    matched_model_summary: pd.DataFrame | None = None
    if RUN_MATCHED_BASELINE_COMPARISON:
        matched_model_predictions = core_test_predictions.copy()
        matched_model_predictions = append_external_baseline_predictions(
            matched_model_predictions,
            matched_wide,
            RIDGE_TEST_PRED_PATH,
            "ridge_test",
            RIDGE_FEATURE_GROUP,
            "ridge_basic_plus_biochemical_full",
            "Ridge basic_plus_biochemical_full",
        )
        matched_model_predictions = append_external_baseline_predictions(
            matched_model_predictions,
            matched_wide,
            ELASTICNET_TEST_PRED_PATH,
            "elasticnet_test",
            ELASTICNET_FEATURE_GROUP,
            "elasticnet_basic_plus_biochemical_full",
            "ElasticNet basic_plus_biochemical_full",
        )
        matched_model_metrics = metric_rows_by_group(
            matched_model_predictions,
            ["cv_fold", "model", "model_label"],
        )
        matched_model_summary = summarize_metric_rows(matched_model_metrics, ["model", "model_label"])

    stratified_summaries: dict[str, pd.DataFrame] = {}
    if RUN_STRATIFIED_ANALYSIS:
        stratified_outputs = build_stratified_metrics(matched_wide, core_test_predictions)
        for name, (metrics_df, summary_df) in stratified_outputs.items():
            metrics_df.to_csv(OUT_DIR / f"late_fusion_stratified_by_{name}.csv", index=False)
            summary_df.to_csv(OUT_DIR / f"late_fusion_stratified_by_{name}_summary.csv", index=False)
            stratified_summaries[name] = summary_df

    fold_metrics.to_csv(OUT_DIR / "late_fusion_fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "late_fusion_summary_metrics.csv", index=False)
    coeff.to_csv(OUT_DIR / "late_fusion_coefficients.csv", index=False)
    pred_test.to_csv(OUT_DIR / "late_fusion_pred_test.csv", index=False)
    pred_val_merged.to_csv(OUT_DIR / "late_fusion_pred_val_merged.csv", index=False)
    delta.to_csv(OUT_DIR / "late_fusion_delta_vs_xgb.csv", index=False)
    per_fold_table.to_csv(OUT_DIR / "late_fusion_per_fold_r2_coefficients.csv", index=False)
    sample_scope.to_csv(OUT_DIR / "late_fusion_sample_scope_comparison.csv", index=False)
    residual.to_csv(OUT_DIR / "late_fusion_residual_complementarity.csv", index=False)
    residual_summary.to_csv(OUT_DIR / "late_fusion_residual_complementarity_summary.csv", index=False)
    if matched_model_metrics is not None:
        matched_model_metrics.to_csv(OUT_DIR / "late_fusion_matched_subset_model_metrics.csv", index=False)
    if matched_model_summary is not None:
        matched_model_summary.to_csv(OUT_DIR / "late_fusion_matched_subset_model_summary.csv", index=False)
    write_report(
        summary,
        delta,
        per_fold_table,
        sample_scope,
        residual,
        residual_summary,
        matched_model_summary,
        stratified_summaries,
        OUT_DIR / "late_fusion_report.md",
    )

    print(f"[DONE] Late fusion outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
