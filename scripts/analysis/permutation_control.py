#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No-args permutation negative control for Saluki feature pipelines."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIG: paths are resolved from environment first. SLURM templates should set
# these to the same canonical inputs used by Ridge/XGBoost/ElasticNet.
# =============================================================================
def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else Path(default)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


PROJECT_ROOT = env_path("SALUKI_PROJECT_ROOT", Path(__file__).resolve().parents[2])
RUNS_ROOT = env_path("SALUKI_RUNS_ROOT", PROJECT_ROOT / "results" / "runs")
FEATURE_ROOT = env_path("SALUKI_PERM_FEATURE_ROOT", os.environ.get("SALUKI_FEATURE_ROOT", PROJECT_ROOT / "data/saluki_features/human"))
NORMALIZED_PATH = env_path("SALUKI_PERM_NORMALIZED_PATH", os.environ.get("SALUKI_NORMALIZED_PATH", PROJECT_ROOT / "data/processed/human/normalized.csv.gz"))
SPLIT_PATH = env_path("SALUKI_PERM_SPLIT_PATH", os.environ.get("SALUKI_SPLIT_PATH", PROJECT_ROOT / "data/processed/human/splits/cv_k5_seed42.csv"))
GENETIC_BIOCHEMICAL_PATH = env_path(
    "SALUKI_PERM_GENETIC_BIOCHEMICAL_PATH",
    os.environ.get("SALUKI_GENETIC_BIOCHEMICAL_PATH", FEATURE_ROOT / "alldataGeneticBiochemical.tsv.gz"),
)
BIOCHEMICAL_FULL_PATH = env_path(
    "SALUKI_PERM_BIOCHEMICAL_FULL_PATH",
    os.environ.get("SALUKI_BIOCHEMICAL_FULL_PATH", FEATURE_ROOT / "alldataBiochemical.tsv.gz"),
)

OUT_BASE_DIR = env_path("SALUKI_PERM_OUT_BASE_DIR", RUNS_ROOT)
RUN_TAG = env_str("SALUKI_PERM_RUN_TAG", "permutation_control")
OUT_ROOT = OUT_BASE_DIR / RUN_TAG
FEATURE_GROUP = "basic_plus_biochemical_full"
CV_FOLDS = [0, 1, 2, 3, 4]
PERM_SEEDS = [1001, 1002, 1003, 1004, 1005]

RUN_XGB = True
RUN_RIDGE = True

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "max_depth": 3,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "reg_alpha": 0.1,
    "early_stopping_rounds": 100,
    "n_jobs": N_JOBS,
}

RIDGE_ALPHAS = np.logspace(-4, 4, 21)
RIDGE_CLIP_Z = 5.0
RIDGE_LOW_VAR_EPS = 1e-6
RIDGE_MAX_ITER = 20000
RIDGE_TOL = 1e-4

EXPECTED_OVERLAP = 12981
HALFLIFE_ATOL = 1e-8
INNER_VAL_RATIO = 0.10
STRATIFY_BINS = 20
SEED = 42
TARGET_STD_EPS = 1e-8


FEATURE_GROUP_META = {
    "saluki_basic8": {
        "label": "SalukiBasic8",
        "parts": ["BASIC.* from alldataGeneticBiochemical.tsv.gz"],
    },
    "genetic_only": {
        "label": "GeneticOnly",
        "parts": ["GENETIC.* from alldataGeneticBiochemical.tsv.gz"],
    },
    "biochemical_371": {
        "label": "Biochemical371",
        "parts": ["BIOCHEMICAL.* from alldataGeneticBiochemical.tsv.gz"],
    },
    "biochemical_full_no_basic": {
        "label": "BiochemicalFullNoBasic",
        "parts": ["RIP/TE/M6ACLIP/ENCORI/eCLIP/PARCLIP from alldataBiochemical.tsv.gz"],
    },
    "basic_plus_biochemical_full": {
        "label": "SalukiBasic8PlusBiochemicalFull",
        "parts": [
            "BASIC.* from alldataGeneticBiochemical.tsv.gz",
            "RIP/TE/M6ACLIP/ENCORI/eCLIP/PARCLIP from alldataBiochemical.tsv.gz",
        ],
    },
    "genetic_plus_biochemical_371": {
        "label": "GeneticPlusBiochemical371",
        "parts": [
            "GENETIC.* from alldataGeneticBiochemical.tsv.gz",
            "BIOCHEMICAL.* from alldataGeneticBiochemical.tsv.gz",
        ],
    },
    "all_23384": {
        "label": "All23384",
        "parts": [
            "BASIC.* from alldataGeneticBiochemical.tsv.gz",
            "GENETIC.* from alldataGeneticBiochemical.tsv.gz",
            "BIOCHEMICAL.* from alldataGeneticBiochemical.tsv.gz",
        ],
    },
    "genetic_plus_biochemical_full": {
        "label": "GeneticPlusBiochemicalFull",
        "parts": [
            "GENETIC.* from alldataGeneticBiochemical.tsv.gz",
            "RIP/TE/M6ACLIP/ENCORI/eCLIP/PARCLIP from alldataBiochemical.tsv.gz",
        ],
    },
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def config_dict() -> dict[str, Any]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "timestamp": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "feature_root": str(FEATURE_ROOT),
        "normalized_path": str(NORMALIZED_PATH),
        "split_path": str(SPLIT_PATH),
        "genetic_biochemical_path": str(GENETIC_BIOCHEMICAL_PATH),
        "biochemical_full_path": str(BIOCHEMICAL_FULL_PATH),
        "out_base_dir": str(OUT_BASE_DIR),
        "run_tag": RUN_TAG,
        "out_root": str(OUT_ROOT),
        "feature_group": FEATURE_GROUP,
        "cv_folds": list(CV_FOLDS),
        "perm_seeds": list(PERM_SEEDS),
        "run_xgb": bool(RUN_XGB),
        "run_ridge": bool(RUN_RIDGE),
        "n_jobs": int(N_JOBS),
        "xgb_params": dict(XGB_PARAMS),
        "ridge_alphas": [float(x) for x in RIDGE_ALPHAS],
        "ridge_clip_z": float(RIDGE_CLIP_Z),
        "ridge_low_var_eps": float(RIDGE_LOW_VAR_EPS),
        "ridge_max_iter": int(RIDGE_MAX_ITER),
        "ridge_tol": float(RIDGE_TOL),
        "expected_overlap": EXPECTED_OVERLAP,
        "halflife_atol": float(HALFLIFE_ATOL),
        "inner_val_ratio": float(INNER_VAL_RATIO),
        "stratify_bins": int(STRATIFY_BINS),
        "seed": int(SEED),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_config_and_raise(message: str) -> None:
    print("[CONFIG]")
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False))
    raise FileNotFoundError(message)


def print_config() -> None:
    print("[CONFIG]")
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False))


def validate_config_paths() -> None:
    required_paths = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "FEATURE_ROOT": FEATURE_ROOT,
        "NORMALIZED_PATH": NORMALIZED_PATH,
        "SPLIT_PATH": SPLIT_PATH,
        "GENETIC_BIOCHEMICAL_PATH": GENETIC_BIOCHEMICAL_PATH,
        "BIOCHEMICAL_FULL_PATH": BIOCHEMICAL_FULL_PATH,
    }
    missing = {name: str(path) for name, path in required_paths.items() if not Path(path).exists()}
    if missing:
        print_config_and_raise(f"Missing CONFIG path(s): {missing}")


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


def prediction_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    abs_err = np.abs(y_true - y_pred)
    return {
        "test_pred_min": float(np.min(y_pred)),
        "test_pred_max": float(np.max(y_pred)),
        "test_abs_err_max": float(np.max(abs_err)),
        "test_abs_err_p99": float(np.quantile(abs_err, 0.99)),
    }


def _make_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    uniq = np.unique(y[~np.isnan(y)])
    b = int(min(n_bins, max(2, len(uniq))))
    series = pd.Series(y)
    try:
        bins = pd.qcut(series, q=b, labels=False, duplicates="drop")
    except Exception:
        ranks = series.rank(method="average", na_option="keep")
        bins = pd.qcut(ranks, q=b, labels=False, duplicates="drop")
    arr = bins.to_numpy()
    return np.where(pd.isna(arr), -1, arr).astype(int)


def stratified_train_val(ids: list[str], y: np.ndarray, val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    ids_arr = np.asarray(ids, dtype=object)
    bins = _make_bins(y, STRATIFY_BINS)
    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    for bucket in [x for x in np.unique(bins) if x >= 0]:
        idx = np.where(bins == bucket)[0]
        block = ids_arr[idx].copy()
        rng.shuffle(block)
        n_val = int(math.floor(len(block) * val_ratio))
        train_ids.extend(block[: len(block) - n_val].tolist())
        val_ids.extend(block[len(block) - n_val :].tolist())
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


def load_normalized_table() -> pd.DataFrame:
    df = pd.read_csv(NORMALIZED_PATH, usecols=["ENSID", "HALFLIFE"])
    df["ENSID"] = df["ENSID"].astype(str)
    df["HALFLIFE"] = pd.to_numeric(df["HALFLIFE"], errors="raise").astype(float)
    if df["ENSID"].duplicated().any():
        raise ValueError("NORMALIZED_PATH has duplicate ENSID values.")
    return df


def load_split_table() -> pd.DataFrame:
    df = pd.read_csv(SPLIT_PATH)
    if "ENSID" not in df.columns:
        raise ValueError(f"SPLIT_PATH must contain ENSID. Columns={list(df.columns)}")
    fold_col = "fold" if "fold" in df.columns else "cv_fold" if "cv_fold" in df.columns else None
    if fold_col is None:
        raise ValueError(f"SPLIT_PATH must contain fold or cv_fold. Columns={list(df.columns)}")
    out = pd.DataFrame(
        {
            "ENSID": df["ENSID"].astype(str),
            "fold": pd.to_numeric(df[fold_col], errors="raise").astype(int),
        }
    )
    if out["ENSID"].duplicated().any():
        raise ValueError("SPLIT_PATH has duplicate ENSID values.")
    return out


def load_feature_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="infer")
    if "GENEID" not in df.columns or "HALFLIFE" not in df.columns:
        raise ValueError(f"{path} must contain GENEID and HALFLIFE columns.")
    df["GENEID"] = df["GENEID"].astype(str)
    df["HALFLIFE"] = pd.to_numeric(df["HALFLIFE"], errors="raise").astype(float)
    if df["GENEID"].duplicated().any():
        raise ValueError(f"{path} has duplicate GENEID values.")
    return df


def align_tables(
    normalized: pd.DataFrame,
    genetic_biochemical: pd.DataFrame,
    biochemical: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    norm = normalized.set_index("ENSID", drop=False)
    gb = genetic_biochemical.set_index("GENEID", drop=False)
    bc = biochemical.set_index("GENEID", drop=False)
    overlap = sorted(set(norm.index).intersection(gb.index).intersection(bc.index))
    if EXPECTED_OVERLAP is not None and len(overlap) != int(EXPECTED_OVERLAP):
        raise ValueError(f"Expected overlap={EXPECTED_OVERLAP}, observed={len(overlap)}")

    norm_y = norm.loc[overlap, "HALFLIFE"].to_numpy(dtype=np.float64)
    gb_y = gb.loc[overlap, "HALFLIFE"].to_numpy(dtype=np.float64)
    bc_y = bc.loc[overlap, "HALFLIFE"].to_numpy(dtype=np.float64)
    gb_diff = float(np.max(np.abs(norm_y - gb_y))) if len(overlap) else float("nan")
    bc_diff = float(np.max(np.abs(norm_y - bc_y))) if len(overlap) else float("nan")
    if not np.allclose(norm_y, gb_y, rtol=0.0, atol=HALFLIFE_ATOL):
        raise ValueError(f"HALFLIFE mismatch between normalized and genetic_biochemical; max_abs_diff={gb_diff}")
    if not np.allclose(norm_y, bc_y, rtol=0.0, atol=HALFLIFE_ATOL):
        raise ValueError(f"HALFLIFE mismatch between normalized and biochemical; max_abs_diff={bc_diff}")

    sample_df = norm.loc[overlap, ["ENSID", "HALFLIFE"]].copy()
    sample_df.index = sample_df["ENSID"].astype(str)
    source_tables = {
        "genetic_biochemical": gb.loc[overlap].copy(),
        "biochemical": bc.loc[overlap].copy(),
    }
    alignment = {
        "overlap": int(len(overlap)),
        "expected_overlap": EXPECTED_OVERLAP,
        "halflife_atol": float(HALFLIFE_ATOL),
        "halflife_max_abs_diff_genetic_biochemical": gb_diff,
        "halflife_max_abs_diff_biochemical": bc_diff,
    }
    return sample_df, source_tables, alignment


def make_feature_spec(key: str, components: list[tuple[str, list[str]]]) -> dict[str, Any]:
    feature_names: list[str] = []
    normalized_components: list[dict[str, Any]] = []
    for source, columns in components:
        cols = [str(col) for col in columns]
        normalized_components.append({"source": source, "columns": tuple(cols)})
        feature_names.extend(cols)
    meta = FEATURE_GROUP_META[key]
    return {
        "key": key,
        "label": str(meta["label"]),
        "parts": tuple(meta["parts"]),
        "components": tuple(normalized_components),
        "feature_names": tuple(feature_names),
        "n_features": int(len(feature_names)),
    }


def build_feature_specs(
    genetic_biochemical_columns: list[str],
    biochemical_columns: list[str],
) -> dict[str, dict[str, Any]]:
    forbidden = {"GENEID", "ENSID", "HALFLIFE"}
    gb_feature_cols = [str(col) for col in genetic_biochemical_columns if str(col) not in forbidden]
    bc_feature_cols = [str(col) for col in biochemical_columns if str(col) not in forbidden]
    basic_cols = [col for col in gb_feature_cols if col.startswith("BASIC.")]
    genetic_cols = [col for col in gb_feature_cols if col.startswith("GENETIC.")]
    biochemical_371_cols = [col for col in gb_feature_cols if col.startswith("BIOCHEMICAL.")]
    full_prefixes = ("RIP.", "TE.", "M6ACLIP.", "ENCORI.", "eCLIP.", "PARCLIP.")
    biochemical_full_cols = [col for col in bc_feature_cols if any(col.startswith(prefix) for prefix in full_prefixes)]
    required = {
        "basic": basic_cols,
        "genetic": genetic_cols,
        "biochemical_371": biochemical_371_cols,
        "biochemical_full_no_basic": biochemical_full_cols,
    }
    empty = [name for name, cols in required.items() if not cols]
    if empty:
        raise ValueError(f"Empty Saluki feature partitions: {empty}")

    specs = {
        "saluki_basic8": make_feature_spec("saluki_basic8", [("genetic_biochemical", basic_cols)]),
        "genetic_only": make_feature_spec("genetic_only", [("genetic_biochemical", genetic_cols)]),
        "biochemical_371": make_feature_spec(
            "biochemical_371",
            [("genetic_biochemical", biochemical_371_cols)],
        ),
        "biochemical_full_no_basic": make_feature_spec(
            "biochemical_full_no_basic",
            [("biochemical", biochemical_full_cols)],
        ),
        "basic_plus_biochemical_full": make_feature_spec(
            "basic_plus_biochemical_full",
            [("genetic_biochemical", basic_cols), ("biochemical", biochemical_full_cols)],
        ),
        "genetic_plus_biochemical_371": make_feature_spec(
            "genetic_plus_biochemical_371",
            [("genetic_biochemical", [*genetic_cols, *biochemical_371_cols])],
        ),
        "all_23384": make_feature_spec(
            "all_23384",
            [("genetic_biochemical", [*basic_cols, *genetic_cols, *biochemical_371_cols])],
        ),
        "genetic_plus_biochemical_full": make_feature_spec(
            "genetic_plus_biochemical_full",
            [("genetic_biochemical", genetic_cols), ("biochemical", biochemical_full_cols)],
        ),
    }
    forbidden_hits = {
        key: [col for col in spec["feature_names"] if col in forbidden]
        for key, spec in specs.items()
    }
    forbidden_hits = {key: value for key, value in forbidden_hits.items() if value}
    if forbidden_hits:
        raise ValueError(f"Forbidden identifier/target columns in feature specs: {forbidden_hits}")
    return specs


def materialize_feature_group(feature_spec: dict[str, Any], source_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for component in feature_spec["components"]:
        source = str(component["source"])
        columns = list(component["columns"])
        frames.append(source_tables[source].loc[:, columns].apply(pd.to_numeric, errors="coerce"))
    X = pd.concat(frames, axis=1)
    forbidden = {"GENEID", "ENSID", "HALFLIFE"}
    forbidden_present = sorted(forbidden.intersection(set(map(str, X.columns))))
    if forbidden_present:
        raise ValueError(f"Forbidden columns entered feature matrix: {forbidden_present}")
    if X.columns.duplicated().any():
        dup = X.columns[X.columns.duplicated()].astype(str).tolist()[:20]
        raise ValueError(f"Duplicate feature columns in {feature_spec['key']}: {dup}")
    X.index = X.index.astype(str)
    return X.astype(np.float32, copy=False)


def regression_split_from_fold(sample_df: pd.DataFrame, split_df: pd.DataFrame, fold: int) -> dict[str, list[str]]:
    available = set(sample_df.index.astype(str))
    test_ids = split_df.loc[split_df["fold"] == int(fold), "ENSID"].astype(str).tolist()
    pool_ids = split_df.loc[split_df["fold"] != int(fold), "ENSID"].astype(str).tolist()
    test_ids = [ensid for ensid in test_ids if ensid in available]
    pool_ids = [ensid for ensid in pool_ids if ensid in available]
    if not test_ids or not pool_ids:
        raise ValueError(f"Fold {fold} has empty test or train/val pool after feature alignment.")
    y_pool = sample_df.loc[pool_ids, "HALFLIFE"].to_numpy(dtype=np.float64)
    train_ids, val_ids = stratified_train_val(
        pool_ids,
        y_pool,
        INNER_VAL_RATIO,
        seed=SEED * 100 + int(fold),
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def fit_train_preprocessor(X: pd.DataFrame, split_ids: dict[str, list[str]]) -> dict[str, Any]:
    X_train = X.loc[split_ids["train"]].to_numpy(dtype=np.float32, copy=True)
    X_val = X.loc[split_ids["val"]].to_numpy(dtype=np.float32, copy=True)
    X_test = X.loc[split_ids["test"]].to_numpy(dtype=np.float32, copy=True)

    keep_finite = np.isfinite(X_train).any(axis=0)
    if int(keep_finite.sum()) <= 0:
        raise ValueError("All Ridge feature columns are non-finite in train.")
    X_train = X_train[:, keep_finite]
    X_val = X_val[:, keep_finite]
    X_test = X_test[:, keep_finite]

    X_train_for_median = np.where(np.isfinite(X_train), X_train, np.nan)
    with np.errstate(all="ignore"):
        train_median = np.nanmedian(X_train_for_median, axis=0).astype(np.float32)
    del X_train_for_median
    train_median[~np.isfinite(train_median)] = 0.0
    for matrix in (X_train, X_val, X_test):
        mask = ~np.isfinite(matrix)
        if np.any(mask):
            matrix[mask] = np.take(train_median, np.where(mask)[1])

    train_std = X_train.std(axis=0, ddof=0)
    keep_variable = np.isfinite(train_std) & (train_std > float(RIDGE_LOW_VAR_EPS))
    if int(keep_variable.sum()) <= 0:
        raise ValueError(f"All Ridge feature columns removed by RIDGE_LOW_VAR_EPS={RIDGE_LOW_VAR_EPS}.")
    X_train = X_train[:, keep_variable]
    X_val = X_val[:, keep_variable]
    X_test = X_test[:, keep_variable]

    scaler = StandardScaler(copy=False)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    if RIDGE_CLIP_Z is not None and float(RIDGE_CLIP_Z) > 0:
        clip_z = float(RIDGE_CLIP_Z)
        np.clip(X_train_s, -clip_z, clip_z, out=X_train_s)
        np.clip(X_val_s, -clip_z, clip_z, out=X_val_s)
        np.clip(X_test_s, -clip_z, clip_z, out=X_test_s)
    return {"X_train": X_train_s, "X_val": X_val_s, "X_test": X_test_s}


def best_iteration_from_model(model: Any) -> int:
    value = getattr(model, "best_iteration", None)
    if value is None:
        value = getattr(model, "best_iteration_", None)
    if value is None:
        return int(XGB_PARAMS["n_estimators"]) - 1
    return int(value)


def predict_with_best_iteration(model: XGBRegressor, X: np.ndarray) -> np.ndarray:
    best_iteration = best_iteration_from_model(model)
    try:
        pred = model.predict(X, iteration_range=(0, best_iteration + 1))
    except TypeError:
        pred = model.predict(X)
    return np.asarray(pred, dtype=np.float64)


def fit_xgb_permutation(
    X: pd.DataFrame,
    y: pd.Series,
    split_ids: dict[str, list[str]],
    fold: int,
    perm_seed: int,
) -> dict[str, Any]:
    from xgboost import XGBRegressor

    rng = np.random.default_rng(int(perm_seed) + int(fold) * 10000)
    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]
    X_train = X.loc[train_ids].to_numpy(dtype=np.float32, copy=True)
    X_val = X.loc[val_ids].to_numpy(dtype=np.float32, copy=True)
    X_test = X.loc[test_ids].to_numpy(dtype=np.float32, copy=True)
    y_train_perm = rng.permutation(y.loc[train_ids].to_numpy(dtype=np.float64))
    y_val_perm = rng.permutation(y.loc[val_ids].to_numpy(dtype=np.float64))
    y_test_true = y.loc[test_ids].to_numpy(dtype=np.float64)

    params = dict(XGB_PARAMS)
    params["random_state"] = int(SEED + fold + perm_seed)
    model = XGBRegressor(**params)
    model.fit(X_train, y_train_perm, eval_set=[(X_val, y_val_perm)], verbose=False)
    pred_test = predict_with_best_iteration(model, X_test)
    best_score = getattr(model, "best_score", None)
    return {
        "metrics": regression_metrics(y_test_true, pred_test),
        "prediction_diagnostics": prediction_diagnostics(y_test_true, pred_test),
        "best_iteration": best_iteration_from_model(model),
        "best_score": None if best_score is None else float(best_score),
        "test_n": int(len(test_ids)),
    }


def fit_ridge_permutation(
    X: pd.DataFrame,
    y: pd.Series,
    split_ids: dict[str, list[str]],
    fold: int,
    perm_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(perm_seed) + int(fold) * 10000)
    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]
    preproc = fit_train_preprocessor(X, split_ids)
    X_train = preproc["X_train"]
    X_val = preproc["X_val"]
    X_test = preproc["X_test"]
    y_train_perm = rng.permutation(y.loc[train_ids].to_numpy(dtype=np.float64))
    y_val_perm = rng.permutation(y.loc[val_ids].to_numpy(dtype=np.float64))
    y_test_true = y.loc[test_ids].to_numpy(dtype=np.float64)

    y_mean = float(y_train_perm.mean())
    y_std = float(y_train_perm.std(ddof=0))
    if not np.isfinite(y_std) or y_std < TARGET_STD_EPS:
        y_std = 1.0
    y_train_perm_s = (y_train_perm - y_mean) / y_std

    best_model: Ridge | None = None
    best_alpha: float | None = None
    best_val_r2 = -np.inf
    for alpha in RIDGE_ALPHAS:
        model = Ridge(
            alpha=float(alpha),
            fit_intercept=True,
            solver="auto",
            tol=float(RIDGE_TOL),
            max_iter=int(RIDGE_MAX_ITER),
        )
        model.fit(X_train, y_train_perm_s)
        pred_val_perm = model.predict(X_val) * y_std + y_mean
        val_r2 = regression_metrics(y_val_perm, pred_val_perm)["r2"]
        if val_r2 > best_val_r2:
            best_val_r2 = float(val_r2)
            best_alpha = float(alpha)
            best_model = model

    if best_model is None or best_alpha is None:
        raise RuntimeError("Ridge permutation alpha selection failed.")
    pred_test = best_model.predict(X_test) * y_std + y_mean
    return {
        "metrics": regression_metrics(y_test_true, pred_test),
        "prediction_diagnostics": prediction_diagnostics(y_test_true, pred_test),
        "selected_alpha": best_alpha,
        "val_perm_r2": best_val_r2,
        "test_n": int(len(test_ids)),
    }


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = ["test_r2", "test_pearson", "test_spearman", "test_rmse", "test_mae"]
    for model, sub in metrics.groupby("model", sort=False):
        for col in metric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            rows.append(
                {
                    "model": model,
                    "metric": col.replace("test_", ""),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "n": int(values.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Permutation Negative Control Report",
        "",
        f"- Feature group: `{FEATURE_GROUP}`",
        f"- Folds: `{CV_FOLDS}`",
        f"- Permutation seeds: `{PERM_SEEDS}`",
        "",
        "| model | mean test R2 | mean test Pearson | mean test Spearman |",
        "|---|---:|---:|---:|",
    ]
    warnings: list[str] = []
    for model in summary["model"].drop_duplicates().tolist():
        sub = summary.loc[summary["model"] == model]
        values = {row["metric"]: float(row["mean"]) for _, row in sub.iterrows()}
        r2 = values.get("r2", float("nan"))
        pearson = values.get("pearson", float("nan"))
        spearman = values.get("spearman", float("nan"))
        lines.append(f"| {model} | {r2:.6f} | {pearson:.6f} | {spearman:.6f} |")
        if np.isfinite(r2) and r2 > 0.1:
            warnings.append(f"WARNING: {model} mean permutation test R2 is {r2:.6f} > 0.1.")
    lines.append("")
    if warnings:
        lines.extend(["## Warnings", *[f"- {msg}" for msg in warnings], ""])
    else:
        lines.extend(
            [
                "## Interpretation",
                "Permutation test R2 is near zero or negative for all enabled models.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print_config()
    validate_config_paths()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_config = config_dict()
    write_json(OUT_ROOT / "permutation_run_config.json", run_config)
    write_json(OUT_ROOT / "run_config.json", run_config)

    normalized = load_normalized_table()
    split_df = load_split_table()
    genetic_biochemical = load_feature_table(GENETIC_BIOCHEMICAL_PATH)
    biochemical = load_feature_table(BIOCHEMICAL_FULL_PATH)
    sample_df, source_tables, alignment = align_tables(normalized, genetic_biochemical, biochemical)
    y = sample_df["HALFLIFE"].astype(float)
    feature_specs = build_feature_specs(list(genetic_biochemical.columns), list(biochemical.columns))
    if FEATURE_GROUP not in feature_specs:
        raise ValueError(f"Unknown FEATURE_GROUP={FEATURE_GROUP!r}. Available={sorted(feature_specs)}")
    spec = feature_specs[FEATURE_GROUP]
    X = materialize_feature_group(spec, source_tables)

    rows: list[dict[str, Any]] = []
    for fold in CV_FOLDS:
        split_ids = regression_split_from_fold(sample_df, split_df, int(fold))
        for perm_seed in PERM_SEEDS:
            print(f"[PERM] fold={fold} seed={perm_seed}")
            if RUN_XGB:
                result = fit_xgb_permutation(X, y, split_ids, int(fold), int(perm_seed))
                rows.append(
                    {
                        "model": "xgb",
                        "feature_group": FEATURE_GROUP,
                        "feature_label": spec["label"],
                        "cv_fold": int(fold),
                        "perm_seed": int(perm_seed),
                        "test_n": int(result["test_n"]),
                        "best_iteration": int(result["best_iteration"]),
                        "best_score": result["best_score"],
                        **result["prediction_diagnostics"],
                        **{f"test_{k}": v for k, v in result["metrics"].items()},
                    }
                )
            if RUN_RIDGE:
                result = fit_ridge_permutation(X, y, split_ids, int(fold), int(perm_seed))
                rows.append(
                    {
                        "model": "ridge",
                        "feature_group": FEATURE_GROUP,
                        "feature_label": spec["label"],
                        "cv_fold": int(fold),
                        "perm_seed": int(perm_seed),
                        "test_n": int(result["test_n"]),
                        "selected_alpha": float(result["selected_alpha"]),
                        "val_perm_r2": float(result["val_perm_r2"]),
                        **result["prediction_diagnostics"],
                        **{f"test_{k}": v for k, v in result["metrics"].items()},
                    }
                )

    metrics = pd.DataFrame(rows)
    summary = summarize_metrics(metrics)
    metrics.to_csv(OUT_ROOT / "permutation_fold_metrics.csv", index=False)
    summary.to_csv(OUT_ROOT / "permutation_summary_metrics.csv", index=False)
    write_report(summary, OUT_ROOT / "permutation_report.md")
    write_json(
        OUT_ROOT / "permutation_run_summary.json",
        {
            "config": run_config,
            "alignment": alignment,
            "feature_group": {
                "key": FEATURE_GROUP,
                "label": spec["label"],
                "parts": list(spec["parts"]),
                "n_features": int(spec["n_features"]),
            },
            "n_metric_rows": int(len(metrics)),
            "completed_at": now_iso(),
        },
    )
    print(f"[DONE] Permutation negative control outputs written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
