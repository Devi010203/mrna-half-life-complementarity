#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No-args ElasticNet baselines for Saluki regulatory feature groups."""

from __future__ import annotations

import gc
import json
import math
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIG: edit these paths manually before submitting SLURM jobs.
# =============================================================================
def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else Path(default)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name, "").strip()
    if not value:
        return list(default)
    folds: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        folds.append(int(token))
    if not folds:
        raise ValueError(f"{name} was set but no valid folds were provided.")
    return folds


def env_str_list(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name, "").strip()
    if not value:
        return list(default)
    groups = [token.strip() for token in value.split(",") if token.strip()]
    if not groups:
        raise ValueError(f"{name} was set but no valid groups were provided.")
    return groups


PROJECT_ROOT = env_path("SALUKI_ENET_PROJECT_ROOT", os.environ.get("SALUKI_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
FEATURE_ROOT = env_path("SALUKI_ENET_FEATURE_ROOT", os.environ.get("SALUKI_FEATURE_ROOT", PROJECT_ROOT / "data/saluki_features/human"))
NORMALIZED_PATH = env_path(
    "SALUKI_ENET_NORMALIZED_PATH",
    os.environ.get("SALUKI_NORMALIZED_PATH", PROJECT_ROOT / "data/processed/human/normalized.csv.gz"),
)
SPLIT_PATH = env_path(
    "SALUKI_ENET_SPLIT_PATH",
    os.environ.get("SALUKI_SPLIT_PATH", PROJECT_ROOT / "data/processed/human/splits/cv_k5_seed42.csv"),
)

GENETIC_BIOCHEMICAL_PATH = env_path(
    "SALUKI_ENET_GENETIC_BIOCHEMICAL_PATH",
    os.environ.get("SALUKI_GENETIC_BIOCHEMICAL_PATH", FEATURE_ROOT / "alldataGeneticBiochemical.tsv.gz"),
)
BIOCHEMICAL_FULL_PATH = env_path(
    "SALUKI_ENET_BIOCHEMICAL_FULL_PATH",
    os.environ.get("SALUKI_BIOCHEMICAL_FULL_PATH", FEATURE_ROOT / "alldataBiochemical.tsv.gz"),
)

OUT_BASE_DIR = env_path(
    "SALUKI_ENET_OUT_BASE_DIR",
    PROJECT_ROOT / "results" / "runs",
)
RUN_TAG = env_str("SALUKI_ENET_RUN_TAG", "elasticnet")
OUT_ROOT = OUT_BASE_DIR / RUN_TAG
DEFAULT_ACTIVE_GROUPS = [
    "biochemical_full_no_basic",
    "basic_plus_biochemical_full",
    "genetic_plus_biochemical_full",
    "all_23384",
]
DEFAULT_CV_FOLDS = [0, 1, 2, 3, 4]
ACTIVE_GROUPS = env_str_list("SALUKI_ENET_GROUP", DEFAULT_ACTIVE_GROUPS)
CV_FOLDS = env_int_list("SALUKI_ENET_FOLDS", DEFAULT_CV_FOLDS)

ALPHAS = np.logspace(-5, 3, 33)
L1_RATIOS = [0.1, 0.5, 0.9, 1.0]
CLIP_Z = 5.0
LOW_VAR_EPS = 1e-6

EXPECTED_OVERLAP = 12981
HALFLIFE_ATOL = 1e-8
INNER_VAL_RATIO = 0.10
STRATIFY_BINS = 20
SEED = 42
TARGET_STD_EPS = 1e-8
COEF_EPS = 1e-12


FEATURE_GROUP_META = {
    "saluki_basic8": {
        "label": "SalukiBasic8-ElasticNet",
        "parts": ["BASIC.* from alldataGeneticBiochemical.tsv.gz"],
    },
    "genetic_only": {
        "label": "GeneticOnly-ElasticNet",
        "parts": ["GENETIC.* from alldataGeneticBiochemical.tsv.gz"],
    },
    "biochemical_371": {
        "label": "Biochemical371-ElasticNet",
        "parts": ["BIOCHEMICAL.* from alldataGeneticBiochemical.tsv.gz"],
    },
    "biochemical_full_no_basic": {
        "label": "BiochemicalFullNoBasic-ElasticNet",
        "parts": ["RIP/TE/M6ACLIP/ENCORI/eCLIP/PARCLIP from alldataBiochemical.tsv.gz"],
    },
    "basic_plus_biochemical_full": {
        "label": "SalukiBasic8PlusBiochemicalFull-ElasticNet",
        "parts": [
            "BASIC.* from alldataGeneticBiochemical.tsv.gz",
            "RIP/TE/M6ACLIP/ENCORI/eCLIP/PARCLIP from alldataBiochemical.tsv.gz",
        ],
    },
    "genetic_plus_biochemical_371": {
        "label": "GeneticPlusBiochemical371-ElasticNet",
        "parts": [
            "GENETIC.* from alldataGeneticBiochemical.tsv.gz",
            "BIOCHEMICAL.* from alldataGeneticBiochemical.tsv.gz",
        ],
    },
    "all_23384": {
        "label": "All23384-ElasticNet",
        "parts": [
            "BASIC.* from alldataGeneticBiochemical.tsv.gz",
            "GENETIC.* from alldataGeneticBiochemical.tsv.gz",
            "BIOCHEMICAL.* from alldataGeneticBiochemical.tsv.gz",
        ],
    },
    "genetic_plus_biochemical_full": {
        "label": "GeneticPlusBiochemicalFull-ElasticNet",
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
        "active_groups": list(ACTIVE_GROUPS),
        "cv_folds": list(CV_FOLDS),
        "alphas": [float(x) for x in ALPHAS],
        "l1_ratios": [float(x) for x in L1_RATIOS],
        "clip_z": float(CLIP_Z),
        "low_var_eps": float(LOW_VAR_EPS),
        "expected_overlap": EXPECTED_OVERLAP,
        "halflife_atol": float(HALFLIFE_ATOL),
        "inner_val_ratio": float(INNER_VAL_RATIO),
        "stratify_bins": int(STRATIFY_BINS),
        "seed": int(SEED),
        "target_std_eps": float(TARGET_STD_EPS),
        "coef_eps": float(COEF_EPS),
        "env_overrides": {
            "SALUKI_ENET_PROJECT_ROOT": os.environ.get("SALUKI_ENET_PROJECT_ROOT"),
            "SALUKI_ENET_FEATURE_ROOT": os.environ.get("SALUKI_ENET_FEATURE_ROOT"),
            "SALUKI_ENET_NORMALIZED_PATH": os.environ.get("SALUKI_ENET_NORMALIZED_PATH"),
            "SALUKI_ENET_SPLIT_PATH": os.environ.get("SALUKI_ENET_SPLIT_PATH"),
            "SALUKI_ENET_GENETIC_BIOCHEMICAL_PATH": os.environ.get("SALUKI_ENET_GENETIC_BIOCHEMICAL_PATH"),
            "SALUKI_ENET_BIOCHEMICAL_FULL_PATH": os.environ.get("SALUKI_ENET_BIOCHEMICAL_FULL_PATH"),
            "SALUKI_ENET_OUT_BASE_DIR": os.environ.get("SALUKI_ENET_OUT_BASE_DIR"),
            "SALUKI_ENET_RUN_TAG": os.environ.get("SALUKI_ENET_RUN_TAG"),
            "SALUKI_ENET_GROUP": os.environ.get("SALUKI_ENET_GROUP"),
            "SALUKI_ENET_FOLDS": os.environ.get("SALUKI_ENET_FOLDS"),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_config_and_raise(message: str) -> None:
    print("[CONFIG]", flush=True)
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False), flush=True)
    raise FileNotFoundError(message)


def print_config() -> None:
    print("[CONFIG]", flush=True)
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False), flush=True)


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


def prefixed_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


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


def assert_no_forbidden_feature_columns(feature_specs: dict[str, dict[str, Any]]) -> None:
    forbidden = {"GENEID", "ENSID", "HALFLIFE"}
    bad: dict[str, list[str]] = {}
    for key, spec in feature_specs.items():
        cols = [col for col in spec["feature_names"] if col in forbidden]
        if cols:
            bad[key] = cols
    if bad:
        raise ValueError(f"Forbidden identifier/target columns in feature specs: {bad}")


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
    assert_no_forbidden_feature_columns(specs)
    return specs


def materialize_feature_group(feature_spec: dict[str, Any], source_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for component in feature_spec["components"]:
        source = str(component["source"])
        columns = list(component["columns"])
        frame = source_tables[source].loc[:, columns].apply(pd.to_numeric, errors="coerce")
        frames.append(frame)
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
    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]
    feature_names = np.asarray(X.columns.astype(str), dtype=object)
    X_train = X.loc[train_ids].to_numpy(dtype=np.float32, copy=True)
    X_val = X.loc[val_ids].to_numpy(dtype=np.float32, copy=True)
    X_test = X.loc[test_ids].to_numpy(dtype=np.float32, copy=True)

    n_features_before = int(X_train.shape[1])
    keep_finite = np.isfinite(X_train).any(axis=0)
    n_after_finite = int(keep_finite.sum())
    if n_after_finite <= 0:
        raise ValueError("All feature columns are non-finite in train.")
    X_train = X_train[:, keep_finite]
    X_val = X_val[:, keep_finite]
    X_test = X_test[:, keep_finite]
    names_after_finite = feature_names[keep_finite]

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
    keep_variable = np.isfinite(train_std) & (train_std > float(LOW_VAR_EPS))
    n_after_low_var = int(keep_variable.sum())
    if n_after_low_var <= 0:
        raise ValueError(f"All feature columns removed by LOW_VAR_EPS={LOW_VAR_EPS}.")
    X_train = X_train[:, keep_variable]
    X_val = X_val[:, keep_variable]
    X_test = X_test[:, keep_variable]
    kept_feature_names = names_after_finite[keep_variable].astype(str).tolist()

    scaler = StandardScaler(copy=False)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    if CLIP_Z is not None and float(CLIP_Z) > 0:
        clip_z = float(CLIP_Z)
        np.clip(X_train_s, -clip_z, clip_z, out=X_train_s)
        np.clip(X_val_s, -clip_z, clip_z, out=X_val_s)
        np.clip(X_test_s, -clip_z, clip_z, out=X_test_s)

    filter_stats = {
        "n_features_before": n_features_before,
        "n_features_after_all_nan_filter": n_after_finite,
        "n_features_after_low_var_filter": n_after_low_var,
        "n_features_after_filter": n_after_low_var,
        "n_removed_all_nan": int(n_features_before - n_after_finite),
        "n_removed_low_var": int(n_after_finite - n_after_low_var),
        "low_var_eps": float(LOW_VAR_EPS),
        "clip_z": float(CLIP_Z),
        "clip_z_enabled": bool(CLIP_Z is not None and float(CLIP_Z) > 0),
    }
    return {
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "kept_feature_names": kept_feature_names,
        "filter_stats": filter_stats,
    }


def fit_and_predict_elasticnet(
    X: pd.DataFrame,
    y: pd.Series,
    split_ids: dict[str, list[str]],
    fold: int,
    group_key: str,
) -> dict[str, Any]:
    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]
    preproc = fit_train_preprocessor(X, split_ids)
    X_train = preproc["X_train"]
    X_val = preproc["X_val"]
    X_test = preproc["X_test"]
    y_train = y.loc[train_ids].to_numpy(dtype=np.float64)
    y_val = y.loc[val_ids].to_numpy(dtype=np.float64)
    y_test = y.loc[test_ids].to_numpy(dtype=np.float64)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=0))
    if not np.isfinite(y_std) or y_std < TARGET_STD_EPS:
        y_std = 1.0
    y_train_s = (y_train - y_mean) / y_std

    best: dict[str, Any] | None = None
    best_val_r2 = -np.inf
    for alpha in ALPHAS:
        for l1_ratio in L1_RATIOS:
            print(
                f"[ENET] group={group_key} fold={fold} "
                f"alpha={float(alpha):.3e} l1_ratio={float(l1_ratio)}",
                flush=True,
            )
            model = ElasticNet(
                alpha=float(alpha),
                l1_ratio=float(l1_ratio),
                fit_intercept=True,
                max_iter=20000,
                tol=1e-4,
                selection="random",
                random_state=int(SEED + fold),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X_train, y_train_s)
            n_convergence_warnings = int(
                sum(issubclass(warning.category, ConvergenceWarning) for warning in caught)
            )
            n_iter = int(np.max(np.atleast_1d(getattr(model, "n_iter_", -1))))
            dual_gap = float(np.max(np.atleast_1d(getattr(model, "dual_gap_", np.nan))))
            converged = bool(n_convergence_warnings == 0)
            pred_val = model.predict(X_val) * y_std + y_mean
            val_metrics = regression_metrics(y_val, pred_val)
            if val_metrics["r2"] > best_val_r2:
                best_val_r2 = float(val_metrics["r2"])
                best = {
                    "model": model,
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "val_pred": pred_val,
                    "val_metrics": val_metrics,
                    "n_iter": n_iter,
                    "dual_gap": dual_gap,
                    "converged": converged,
                    "n_convergence_warnings": n_convergence_warnings,
                }

    if best is None:
        raise RuntimeError("ElasticNet validation tuning failed to select a model.")
    model = best["model"]
    pred_train = model.predict(X_train) * y_std + y_mean
    pred_test = model.predict(X_test) * y_std + y_mean
    coef_standardized_target = np.asarray(model.coef_, dtype=np.float64)
    coef_y_scale = coef_standardized_target * y_std
    nonzero_mask = np.abs(coef_standardized_target) > COEF_EPS

    return {
        "alpha": float(best["alpha"]),
        "l1_ratio": float(best["l1_ratio"]),
        "n_iter": int(best["n_iter"]),
        "dual_gap": float(best["dual_gap"]),
        "converged": bool(best["converged"]),
        "n_convergence_warnings": int(best["n_convergence_warnings"]),
        "target_scaler": {"mean": y_mean, "std": y_std},
        "filter_stats": preproc["filter_stats"],
        "kept_feature_names": preproc["kept_feature_names"],
        "n_nonzero_coef": int(nonzero_mask.sum()),
        "coef_standardized_target": coef_standardized_target,
        "coef_y_scale": coef_y_scale,
        "predictions": {
            "train": pd.DataFrame({"ENSID": train_ids, "y_true": y_train, "y_pred": pred_train}),
            "val": pd.DataFrame({"ENSID": val_ids, "y_true": y_val, "y_pred": best["val_pred"]}),
            "test": pd.DataFrame({"ENSID": test_ids, "y_true": y_test, "y_pred": pred_test}),
        },
        "metrics": {
            "train": regression_metrics(y_train, pred_train),
            "val": best["val_metrics"],
            "test": regression_metrics(y_test, pred_test),
        },
    }


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        col
        for col in fold_metrics.columns
        if col.startswith("train_") or col.startswith("val_") or col.startswith("test_")
    ]
    for group, sub in fold_metrics.groupby("feature_group", sort=False):
        label = str(sub["feature_label"].iloc[0])
        for col in metric_cols:
            split, metric = col.split("_", 1)
            values = pd.to_numeric(sub[col], errors="coerce")
            rows.append(
                {
                    "feature_group": group,
                    "feature_label": label,
                    "split": split,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "n_folds": int(values.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def feature_missing_stats(X: pd.DataFrame) -> dict[str, Any]:
    miss = X.isna().mean()
    return {
        "mean_missing_ratio": float(miss.mean()),
        "median_missing_ratio": float(miss.median()),
        "max_missing_ratio": float(miss.max()),
        "n_all_missing_cols": int((miss >= 1.0).sum()),
        "n_missing_gt_50pct": int((miss > 0.5).sum()),
        "n_missing_gt_90pct": int((miss > 0.9).sum()),
    }


def write_progress_checkpoint(
    fold_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    pred_val_rows: list[pd.DataFrame],
    pred_test_rows: list[pd.DataFrame],
    nonzero_rows: list[pd.DataFrame],
    top_rows: list[pd.DataFrame],
    manifest: dict[str, Any],
    run_config: dict[str, Any],
    alignment: dict[str, Any],
) -> None:
    fold_metrics = pd.DataFrame(fold_rows)
    summary = summarize_metrics(fold_metrics) if fold_rows else pd.DataFrame()
    selected = pd.DataFrame(selected_rows)
    pred_val = pd.concat(pred_val_rows, ignore_index=True) if pred_val_rows else pd.DataFrame()
    pred_test = pd.concat(pred_test_rows, ignore_index=True) if pred_test_rows else pd.DataFrame()
    nonzero = pd.concat(nonzero_rows, ignore_index=True) if nonzero_rows else pd.DataFrame()
    top = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()

    fold_metrics.to_csv(OUT_ROOT / "elasticnet_fold_metrics.csv", index=False)
    summary.to_csv(OUT_ROOT / "elasticnet_summary_metrics.csv", index=False)
    selected.to_csv(OUT_ROOT / "selected_params_per_fold.csv", index=False)
    pred_val.to_csv(OUT_ROOT / "pred_val.csv", index=False)
    pred_test.to_csv(OUT_ROOT / "pred_test.csv", index=False)
    nonzero.to_csv(OUT_ROOT / "nonzero_coefficients_per_fold.csv", index=False)
    top.to_csv(OUT_ROOT / "top_coefficients_by_group.csv", index=False)
    write_json(OUT_ROOT / "feature_group_manifest.json", manifest)
    write_json(
        OUT_ROOT / "run_summary.json",
        {
            "config": run_config,
            "alignment": alignment,
            "partial": True,
            "n_fold_metric_rows": int(len(fold_metrics)),
            "n_prediction_val_rows": int(len(pred_val)),
            "n_prediction_test_rows": int(len(pred_test)),
            "n_nonzero_coefficients": int(len(nonzero)),
            "checkpointed_at": now_iso(),
        },
    )
    print(
        f"[CHECKPOINT] wrote current ElasticNet progress: "
        f"fold_rows={len(fold_metrics)} pred_val_rows={len(pred_val)} pred_test_rows={len(pred_test)}",
        flush=True,
    )


def main() -> None:
    print_config()
    validate_config_paths()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_config = config_dict()
    write_json(OUT_ROOT / "run_config.json", run_config)

    normalized = load_normalized_table()
    split_df = load_split_table()
    genetic_biochemical = load_feature_table(GENETIC_BIOCHEMICAL_PATH)
    biochemical = load_feature_table(BIOCHEMICAL_FULL_PATH)
    sample_df, source_tables, alignment = align_tables(normalized, genetic_biochemical, biochemical)
    y = sample_df["HALFLIFE"].astype(float)

    feature_specs = build_feature_specs(list(genetic_biochemical.columns), list(biochemical.columns))
    unknown_groups = sorted(set(ACTIVE_GROUPS) - set(feature_specs))
    if unknown_groups:
        raise ValueError(f"Unknown ACTIVE_GROUPS: {unknown_groups}. Available={sorted(feature_specs)}")
    invalid_folds = sorted(set(int(fold) for fold in CV_FOLDS) - set(DEFAULT_CV_FOLDS))
    if invalid_folds:
        raise ValueError(f"Invalid CV_FOLDS: {invalid_folds}. Available={DEFAULT_CV_FOLDS}")

    manifest: dict[str, Any] = {
        "config": run_config,
        "alignment": alignment,
        "forbidden_feature_columns": ["GENEID", "ENSID", "HALFLIFE"],
        "preprocessing": {
            "train_only_all_nonfinite_filter": True,
            "train_only_median_impute": True,
            "train_only_low_variance_filter": True,
            "standard_scaler_fit_train_only": True,
            "clip_z": float(CLIP_Z),
            "target_standardized_fit_train_only": True,
        },
        "feature_groups": {},
    }

    fold_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    pred_val_rows: list[pd.DataFrame] = []
    pred_test_rows: list[pd.DataFrame] = []
    nonzero_rows: list[pd.DataFrame] = []
    top_rows: list[pd.DataFrame] = []

    for group_key in ACTIVE_GROUPS:
        spec = feature_specs[group_key]
        print(f"[GROUP] {group_key}: materializing {spec['n_features']} features", flush=True)
        X = materialize_feature_group(spec, source_tables)
        manifest["feature_groups"][group_key] = {
            "label": spec["label"],
            "parts": list(spec["parts"]),
            "n_features": int(spec["n_features"]),
            "components": [
                {"source": c["source"], "n_columns": len(c["columns"])}
                for c in spec["components"]
            ],
            "missing_stats": feature_missing_stats(X),
        }

        for fold in CV_FOLDS:
            print(f"[FOLD] group={group_key} fold={fold}", flush=True)
            split_ids = regression_split_from_fold(sample_df, split_df, int(fold))
            result = fit_and_predict_elasticnet(X, y, split_ids, int(fold), group_key)
            if not bool(result["converged"]):
                print(
                    "[WARN] selected ElasticNet did not converge: "
                    f"group={group_key} fold={fold} alpha={result['alpha']} "
                    f"l1_ratio={result['l1_ratio']} n_iter={result['n_iter']} "
                    f"dual_gap={result['dual_gap']} "
                    f"n_convergence_warnings={result['n_convergence_warnings']}",
                    flush=True,
                )
            filter_stats = result["filter_stats"]
            metrics = result["metrics"]
            row = {
                "feature_group": group_key,
                "feature_label": spec["label"],
                "cv_fold": int(fold),
                "n_features_before": int(filter_stats["n_features_before"]),
                "n_features_after_filter": int(filter_stats["n_features_after_filter"]),
                "selected_alpha": float(result["alpha"]),
                "selected_l1_ratio": float(result["l1_ratio"]),
                "n_nonzero_coef": int(result["n_nonzero_coef"]),
                "converged": bool(result["converged"]),
                **prefixed_metrics(metrics["train"], "train"),
                **prefixed_metrics(metrics["val"], "val"),
                **prefixed_metrics(metrics["test"], "test"),
            }
            fold_rows.append(row)
            selected_rows.append(
                {
                    "feature_group": group_key,
                    "feature_label": spec["label"],
                    "cv_fold": int(fold),
                    "selected_alpha": float(result["alpha"]),
                    "selected_l1_ratio": float(result["l1_ratio"]),
                    "n_nonzero_coef": int(result["n_nonzero_coef"]),
                    "n_iter": int(result["n_iter"]),
                    "dual_gap": float(result["dual_gap"]),
                    "converged": bool(result["converged"]),
                    "n_convergence_warnings": int(result["n_convergence_warnings"]),
                    **filter_stats,
                    "target_mean": float(result["target_scaler"]["mean"]),
                    "target_std": float(result["target_scaler"]["std"]),
                    "train_n": int(len(split_ids["train"])),
                    "val_n": int(len(split_ids["val"])),
                    "test_n": int(len(split_ids["test"])),
                }
            )

            for split_name, bucket in (("val", pred_val_rows), ("test", pred_test_rows)):
                pred = result["predictions"][split_name].copy()
                pred.insert(0, "feature_group", group_key)
                pred.insert(1, "feature_label", spec["label"])
                pred.insert(2, "cv_fold", int(fold))
                bucket.append(pred)

            kept_names = np.asarray(result["kept_feature_names"], dtype=object)
            coef_std = np.asarray(result["coef_standardized_target"], dtype=np.float64)
            coef_y = np.asarray(result["coef_y_scale"], dtype=np.float64)
            nz = np.where(np.abs(coef_std) > COEF_EPS)[0]
            coef_df = pd.DataFrame(
                {
                    "feature_group": group_key,
                    "feature_label": spec["label"],
                    "cv_fold": int(fold),
                    "feature": kept_names[nz],
                    "coef_standardized_target": coef_std[nz],
                    "coef_y_scale": coef_y[nz],
                    "abs_coef_y_scale": np.abs(coef_y[nz]),
                    "selected_alpha": float(result["alpha"]),
                    "selected_l1_ratio": float(result["l1_ratio"]),
                }
            ).sort_values("abs_coef_y_scale", ascending=False)
            nonzero_rows.append(coef_df)
            top_rows.append(coef_df.head(100).copy())
            write_progress_checkpoint(
                fold_rows,
                selected_rows,
                pred_val_rows,
                pred_test_rows,
                nonzero_rows,
                top_rows,
                manifest,
                run_config,
                alignment,
            )

        del X
        gc.collect()

    fold_metrics = pd.DataFrame(fold_rows)
    summary = summarize_metrics(fold_metrics)
    selected = pd.DataFrame(selected_rows)
    pred_val = pd.concat(pred_val_rows, ignore_index=True) if pred_val_rows else pd.DataFrame()
    pred_test = pd.concat(pred_test_rows, ignore_index=True) if pred_test_rows else pd.DataFrame()
    nonzero = pd.concat(nonzero_rows, ignore_index=True) if nonzero_rows else pd.DataFrame()
    top = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()

    fold_metrics.to_csv(OUT_ROOT / "elasticnet_fold_metrics.csv", index=False)
    summary.to_csv(OUT_ROOT / "elasticnet_summary_metrics.csv", index=False)
    selected.to_csv(OUT_ROOT / "selected_params_per_fold.csv", index=False)
    pred_val.to_csv(OUT_ROOT / "pred_val.csv", index=False)
    pred_test.to_csv(OUT_ROOT / "pred_test.csv", index=False)
    nonzero.to_csv(OUT_ROOT / "nonzero_coefficients_per_fold.csv", index=False)
    top.to_csv(OUT_ROOT / "top_coefficients_by_group.csv", index=False)
    write_json(OUT_ROOT / "feature_group_manifest.json", manifest)
    write_json(
        OUT_ROOT / "run_summary.json",
        {
            "config": run_config,
            "alignment": alignment,
            "n_fold_metric_rows": int(len(fold_metrics)),
            "n_prediction_val_rows": int(len(pred_val)),
            "n_prediction_test_rows": int(len(pred_test)),
            "n_nonzero_coefficients": int(len(nonzero)),
            "completed_at": now_iso(),
        },
    )
    print(f"[DONE] ElasticNet outputs written to: {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
