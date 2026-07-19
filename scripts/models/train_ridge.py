"""
14ridge_saluki_regulatory_features_noargs.py

Human five-fold Ridge comparison for Saluki sequence/genetic and
biochemical/regulatory features:

1. SalukiBasic8-Ridge
2. SalukiGenetic23005-Ridge
3. SalukiBiochemical371-Ridge
4. SalukiGenetic+Biochemical371-Ridge
5. SalukiAll23384-Ridge
6. SalukiBiochemicalFullNoBasic-Ridge
7. SalukiGenetic+BiochemicalFull-Ridge

The script follows the same outer-fold and inner-validation protocol used by
the neural runs and previous Ridge supplements. Ridge alpha is selected on the
fold-aligned validation split, and the selected train-only model is evaluated
directly on the held-out test split without train+val refit.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else Path(default)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


PROJECT_ROOT = env_path("SALUKI_PROJECT_ROOT", Path(__file__).resolve().parents[2])
RESULT_ROOT = env_path(
    "SALUKI_RIDGE_RESULT_ROOT",
    os.environ.get("SALUKI_RESULT_ROOT", PROJECT_ROOT / "results" / "runs"),
)
DEFAULT_PROCESSED_ROOT = env_path("SALUKI_RIDGE_PROCESSED_ROOT", os.environ.get("SALUKI_PROCESSED_ROOT", PROJECT_ROOT / "data" / "processed"))


@dataclass(frozen=True)
class Config:
    PROCESSED_ROOT: str = str(DEFAULT_PROCESSED_ROOT)
    SPECIES: str = "human"
    SEED: int = 42
    CV_K: int = 5
    CV_FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
    INNER_VAL_RATIO: float = 0.10
    STRATIFY_BINS: int = 20
    TARGET_STD_EPS: float = 1e-6
    EXPECTED_OVERLAP_N: int = 12981
    ALPHAS: tuple[float, ...] = tuple(float(x) for x in np.logspace(-4, 4, 41))
    FIT_INTERCEPT: bool = True
    RIDGE_SOLVER: str = "lsqr"
    RIDGE_TOL: float = 1e-4
    RIDGE_MAX_ITER: int = 10000
    RESULT_SLUG: str = env_str("SALUKI_RIDGE_RESULT_SLUG", "ridge_saluki_genetic_biochemical_human")


CFG = Config()


FEATURE_GROUPS: tuple[dict[str, Any], ...] = (
    {"key": "saluki_basic8", "label": "SalukiBasic8-Ridge", "parts": ("basic",)},
    {"key": "genetic_only", "label": "SalukiGenetic23005-Ridge", "parts": ("genetic",)},
    {"key": "biochemical_371", "label": "SalukiBiochemical371-Ridge", "parts": ("biochemical_371",)},
    {
        "key": "genetic_plus_biochemical_371",
        "label": "SalukiGenetic+Biochemical371-Ridge",
        "parts": ("genetic", "biochemical_371"),
    },
    {"key": "all_23384", "label": "SalukiAll23384-Ridge", "parts": ("basic", "genetic", "biochemical_371")},
    {
        "key": "biochemical_full_no_basic",
        "label": "SalukiBiochemicalFullNoBasic-Ridge",
        "parts": ("biochemical_full_no_basic",),
    },
    {
        "key": "genetic_plus_biochemical_full",
        "label": "SalukiGenetic+BiochemicalFull-Ridge",
        "parts": ("genetic", "biochemical_full_no_basic"),
    },
    {
        "key": "basic_plus_biochemical_full",
        "label": "SalukiBasic8+BiochemicalFull-Ridge",
        "parts": ("basic", "biochemical_full_no_basic"),
    },
)

PAIRED_DELTA_BASELINE_GROUP = "genetic_only"
PAIRED_DELTA_COMPARISON_GROUPS: tuple[str, ...] = (
    "biochemical_371",
    "genetic_plus_biochemical_371",
    "all_23384",
    "biochemical_full_no_basic",
    "genetic_plus_biochemical_full",
    "basic_plus_biochemical_full",
)
FEATURE_GROUP_BY_KEY = {str(spec["key"]): spec for spec in FEATURE_GROUPS}
SALUKI_RIDGE_GROUP_ENV = "SALUKI_RIDGE_GROUP"
SALUKI_RIDGE_FOLDS_ENV = "SALUKI_RIDGE_FOLDS"
SALUKI_RIDGE_LOW_VAR_EPS_ENV = "SALUKI_RIDGE_LOW_VAR_EPS"
SALUKI_RIDGE_CLIP_Z_ENV = "SALUKI_RIDGE_CLIP_Z"
SALUKI_RIDGE_NORMALIZED_PATH_ENV = "SALUKI_RIDGE_NORMALIZED_PATH"
SALUKI_RIDGE_SPLIT_PATH_ENV = "SALUKI_RIDGE_SPLIT_PATH"
SALUKI_RIDGE_FEATURE_ROOT_ENV = "SALUKI_RIDGE_FEATURE_ROOT"
SALUKI_RIDGE_GENETIC_BIOCHEMICAL_PATH_ENV = "SALUKI_RIDGE_GENETIC_BIOCHEMICAL_PATH"
SALUKI_RIDGE_BIOCHEMICAL_FULL_PATH_ENV = "SALUKI_RIDGE_BIOCHEMICAL_FULL_PATH"
DEFAULT_LOW_VAR_EPS = 1e-6
DEFAULT_CLIP_Z = 10.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_git_commit(project_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    value = str(proc.stdout).strip()
    return value or None


def collect_script_runtime_metadata() -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "absolute_script_path": str(script_path),
        "script_sha256": compute_file_sha256(script_path),
        "git_commit": resolve_git_commit(PROJECT_ROOT),
    }


def resolve_saluki_feature_root() -> Path:
    override = os.environ.get(SALUKI_RIDGE_FEATURE_ROOT_ENV, os.environ.get("SALUKI_FEATURE_ROOT", "")).strip()
    if override:
        path = Path(override)
        required = ("alldataGeneticBiochemical.tsv.gz", "alldataBiochemical.tsv.gz")
        if path.exists() and all((path / name).exists() for name in required):
            return path
        raise FileNotFoundError(f"{SALUKI_RIDGE_FEATURE_ROOT_ENV}/SALUKI_FEATURE_ROOT does not contain required TSVs: {path}")
    candidates = (
        PROJECT_ROOT / "data" / "saluki_features" / "human",
        PROJECT_ROOT / "data" / "datasets" / "datasets" / "human",
    )
    required = ("alldataGeneticBiochemical.tsv.gz", "alldataBiochemical.tsv.gz")
    for path in candidates:
        if path.exists() and all((path / name).exists() for name in required):
            return path
    raise FileNotFoundError(
        "Could not find Saluki human feature root containing required TSVs: "
        f"{[str(x) for x in candidates]}"
    )


def input_paths() -> dict[str, Path]:
    processed_root = Path(CFG.PROCESSED_ROOT)
    saluki_root = resolve_saluki_feature_root()
    paths = {
        "normalized": env_path(
            SALUKI_RIDGE_NORMALIZED_PATH_ENV,
            os.environ.get("SALUKI_NORMALIZED_PATH", processed_root / "human" / "normalized.csv.gz"),
        ),
        "cv_split": env_path(
            SALUKI_RIDGE_SPLIT_PATH_ENV,
            os.environ.get("SALUKI_SPLIT_PATH", processed_root / "human" / "splits" / f"cv_k{CFG.CV_K}_seed{CFG.SEED}.csv"),
        ),
        "genetic_biochemical": env_path(
            SALUKI_RIDGE_GENETIC_BIOCHEMICAL_PATH_ENV,
            os.environ.get("SALUKI_GENETIC_BIOCHEMICAL_PATH", saluki_root / "alldataGeneticBiochemical.tsv.gz"),
        ),
        "biochemical": env_path(
            SALUKI_RIDGE_BIOCHEMICAL_FULL_PATH_ENV,
            os.environ.get("SALUKI_BIOCHEMICAL_FULL_PATH", saluki_root / "alldataBiochemical.tsv.gz"),
        ),
    }
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")
    return paths


def parse_requested_folds() -> tuple[int, ...]:
    value = os.environ.get(SALUKI_RIDGE_FOLDS_ENV, "").strip()
    if not value:
        return CFG.CV_FOLDS
    folds: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            fold = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid {SALUKI_RIDGE_FOLDS_ENV} token {token!r}; expected comma-separated ints") from exc
        if fold not in CFG.CV_FOLDS:
            raise ValueError(f"Invalid fold {fold}; allowed folds are {list(CFG.CV_FOLDS)}")
        if fold not in folds:
            folds.append(fold)
    if not folds:
        raise ValueError(f"{SALUKI_RIDGE_FOLDS_ENV} was set but no valid folds were provided")
    return tuple(folds)


def parse_requested_group_keys() -> tuple[str, ...]:
    value = os.environ.get(SALUKI_RIDGE_GROUP_ENV, "").strip()
    if not value:
        return tuple(str(spec["key"]) for spec in FEATURE_GROUPS)
    keys = [token.strip() for token in value.split(",") if token.strip()]
    if not keys:
        raise ValueError(f"{SALUKI_RIDGE_GROUP_ENV} was set but no valid feature group was provided")
    unknown = [key for key in keys if key not in FEATURE_GROUP_BY_KEY]
    if unknown:
        raise ValueError(
            f"Unknown {SALUKI_RIDGE_GROUP_ENV} value(s): {unknown}. "
            f"Available groups: {sorted(FEATURE_GROUP_BY_KEY)}"
        )
    deduped: list[str] = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return tuple(deduped)


def parse_float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return float(default)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={value!r}; expected a float") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"Invalid {name}={value!r}; expected a finite float")
    return parsed


def parse_low_var_eps() -> float:
    eps = parse_float_env(SALUKI_RIDGE_LOW_VAR_EPS_ENV, DEFAULT_LOW_VAR_EPS)
    if eps < 0:
        raise ValueError(f"{SALUKI_RIDGE_LOW_VAR_EPS_ENV} must be >= 0, got {eps}")
    return eps


def parse_clip_z() -> float:
    return parse_float_env(SALUKI_RIDGE_CLIP_Z_ENV, DEFAULT_CLIP_Z)


def _make_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    uniq = np.unique(y[~np.isnan(y)])
    b = int(min(n_bins, max(2, len(uniq))))
    s = pd.Series(y)
    try:
        bins = pd.qcut(s, q=b, labels=False, duplicates="drop")
    except Exception:
        ranks = s.rank(method="average", na_option="keep")
        bins = pd.qcut(ranks, q=b, labels=False, duplicates="drop")
    arr = bins.to_numpy()
    return np.where(pd.isna(arr), -1, arr).astype(int)


def stratified_train_val(ids: list[str], y: np.ndarray, val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    ids_arr = np.asarray(ids, dtype=object)
    bins = _make_bins(y, CFG.STRATIFY_BINS)
    rng = np.random.default_rng(seed)
    tr: list[str] = []
    va: list[str] = []
    for bucket in [x for x in np.unique(bins) if x >= 0]:
        idx = np.where(bins == bucket)[0]
        block = ids_arr[idx].copy()
        rng.shuffle(block)
        n = len(block)
        n_va = int(math.floor(n * val_ratio))
        tr.extend(block[: n - n_va].tolist())
        va.extend(block[n - n_va :].tolist())
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom == 0:
        return 0.0
    return float((x * y).sum() / denom)


def spearmanr_np(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return pearsonr_np(rx, ry)


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": r2_np(y_true, y_pred),
        "pearson": pearsonr_np(y_true, y_pred),
        "spearman": spearmanr_np(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def prediction_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    abs_err = np.abs(y_true - y_pred)
    return {
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "pred_p001": float(np.quantile(y_pred, 0.001)),
        "pred_p999": float(np.quantile(y_pred, 0.999)),
        "abs_err_max": float(np.max(abs_err)),
        "abs_err_p99": float(np.quantile(abs_err, 0.99)),
    }


def load_human_processed_table(paths: dict[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(paths["normalized"], compression="gzip", usecols=["ENSID", "HALFLIFE"])
    need = ["ENSID", "HALFLIFE"]
    missing = [col for col in need if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in processed human table: {missing}")
    df = df.copy()
    df["ENSID"] = df["ENSID"].astype(str)
    df["HALFLIFE"] = pd.to_numeric(df["HALFLIFE"], errors="coerce")
    df = df.dropna(subset=["HALFLIFE"]).reset_index(drop=True)
    if df["ENSID"].duplicated().any():
        dup = df.loc[df["ENSID"].duplicated(), "ENSID"].head(10).tolist()
        raise ValueError(f"normalized.csv.gz contains duplicated ENSID values, examples={dup}")
    return df


def load_human_cv_assignments(paths: dict[str, Path]) -> pd.DataFrame:
    cv = pd.read_csv(paths["cv_split"])
    cv["ENSID"] = cv["ENSID"].astype(str)
    cv["fold"] = pd.to_numeric(cv["fold"], errors="coerce").astype("Int64")
    return cv.dropna(subset=["fold"]).astype({"fold": int})


def read_saluki_table_columns(path: Path) -> list[str]:
    header = pd.read_csv(path, sep="\t", compression="infer", nrows=0)
    required = {"GENEID", "HALFLIFE"}
    missing = sorted(required.difference(header.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return header.columns.astype(str).tolist()


def load_saluki_feature_table(path: Path, feature_cols: list[str] | None = None) -> tuple[pd.Series, pd.DataFrame]:
    table_cols = read_saluki_table_columns(path)
    forbidden_feature_cols = {"GENEID", "ENSID", "HALFLIFE"}
    if feature_cols is None:
        feature_cols = [col for col in table_cols if col not in forbidden_feature_cols]
    else:
        feature_cols = [str(col) for col in feature_cols]
    bad_requested = sorted(forbidden_feature_cols.intersection(feature_cols))
    if bad_requested:
        raise ValueError(f"Forbidden ID/target columns requested as features from {path}: {bad_requested}")
    missing_requested = sorted(set(feature_cols).difference(table_cols))
    if missing_requested:
        raise ValueError(f"{path} is missing requested feature columns, examples={missing_requested[:10]}")

    dtype_map: dict[str, str] = {col: "float32" for col in feature_cols}
    dtype_map["GENEID"] = "string"
    dtype_map["HALFLIFE"] = "float64"
    df = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        usecols=["GENEID", "HALFLIFE", *feature_cols],
        dtype=dtype_map,
        na_values=["NA", "NaN", ""],
        low_memory=False,
    )
    df["GENEID"] = df["GENEID"].astype(str)
    if df["GENEID"].duplicated().any():
        dup = df.loc[df["GENEID"].duplicated(), "GENEID"].head(10).tolist()
        raise ValueError(f"{path} contains duplicated GENEID values, examples={dup}")
    y = df.set_index("GENEID", drop=True)["HALFLIFE"].astype(np.float64)
    features = df.set_index("GENEID", drop=True)[feature_cols]
    bad_cols = sorted(forbidden_feature_cols.intersection(features.columns))
    if bad_cols:
        raise ValueError(f"Forbidden ID/target columns entered feature matrix: {bad_cols}")
    return y, features


def validate_saluki_alignment(
    normalized_df: pd.DataFrame,
    genetic_biochemical_y: pd.Series,
    biochemical_y: pd.Series,
) -> dict[str, Any]:
    norm_y = normalized_df.set_index("ENSID", drop=True)["HALFLIFE"].astype(np.float64)
    overlap = norm_y.index.intersection(genetic_biochemical_y.index).intersection(biochemical_y.index)
    overlap = pd.Index(overlap.astype(str), name="ENSID")
    if len(overlap) != int(CFG.EXPECTED_OVERLAP_N):
        raise ValueError(
            f"Expected overlap={CFG.EXPECTED_OVERLAP_N}, got {len(overlap)} "
            f"(normalized={len(norm_y)}, genetic_biochemical={len(genetic_biochemical_y)}, "
            f"biochemical={len(biochemical_y)})"
        )

    norm_aligned = norm_y.loc[overlap].to_numpy(dtype=np.float64)
    genetic_aligned = genetic_biochemical_y.loc[overlap].to_numpy(dtype=np.float64)
    biochemical_aligned = biochemical_y.loc[overlap].to_numpy(dtype=np.float64)
    genetic_exact = np.array_equal(norm_aligned, genetic_aligned)
    biochemical_exact = np.array_equal(norm_aligned, biochemical_aligned)
    genetic_max_abs = float(np.max(np.abs(norm_aligned - genetic_aligned)))
    biochemical_max_abs = float(np.max(np.abs(norm_aligned - biochemical_aligned)))
    genetic_close = bool(np.allclose(norm_aligned, genetic_aligned, rtol=0.0, atol=1e-8))
    biochemical_close = bool(np.allclose(norm_aligned, biochemical_aligned, rtol=0.0, atol=1e-8))
    if not genetic_close or not biochemical_close:
        raise ValueError(
            "HALFLIFE mismatch between normalized.csv.gz and Saluki TSVs: "
            f"genetic_biochemical_allclose={genetic_close} max_abs_diff={genetic_max_abs}; "
            f"biochemical_allclose={biochemical_close} max_abs_diff={biochemical_max_abs}"
        )
    return {
        "overlap_n": int(len(overlap)),
        "normalized_n": int(len(norm_y)),
        "genetic_biochemical_n": int(len(genetic_biochemical_y)),
        "biochemical_n": int(len(biochemical_y)),
        "halflife_exact_match_genetic_biochemical": bool(genetic_exact),
        "halflife_exact_match_biochemical": bool(biochemical_exact),
        "halflife_allclose_atol": 1e-8,
        "halflife_allclose_genetic_biochemical": genetic_close,
        "halflife_allclose_biochemical": biochemical_close,
        "halflife_max_abs_diff_genetic_biochemical": genetic_max_abs,
        "halflife_max_abs_diff_biochemical": biochemical_max_abs,
        "overlap_ids": overlap.tolist(),
    }


def regression_split_from_fold(df: pd.DataFrame, cv: pd.DataFrame, fold: int) -> dict[str, list[str]]:
    test_ids = cv.loc[cv["fold"] == int(fold), "ENSID"].astype(str).tolist()
    pool = cv.loc[cv["fold"] != int(fold), "ENSID"].astype(str).tolist()
    dfi = df.set_index("ENSID", drop=False)
    test_ids = [ensid for ensid in test_ids if ensid in dfi.index]
    pool = [ensid for ensid in pool if ensid in dfi.index]
    y_pool = dfi.loc[pool, "HALFLIFE"].to_numpy(np.float64)
    train_ids, val_ids = stratified_train_val(
        pool,
        y_pool,
        CFG.INNER_VAL_RATIO,
        seed=CFG.SEED * 100 + int(fold),
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def fit_train_imputer_and_scaler(
    X: pd.DataFrame,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> dict[str, Any]:
    low_var_eps = parse_low_var_eps()
    clip_z = parse_clip_z()
    X_train = X.loc[train_ids].to_numpy(dtype=np.float32, copy=True)
    X_val = X.loc[val_ids].to_numpy(dtype=np.float32, copy=True)
    X_test = X.loc[test_ids].to_numpy(dtype=np.float32, copy=True)

    n_features_before = int(X_train.shape[1])
    finite_train = np.isfinite(X_train)
    keep_finite = finite_train.any(axis=0)
    n_features_after_all_nan_filter = int(keep_finite.sum())
    if n_features_after_all_nan_filter <= 0:
        raise ValueError("All feature columns are non-finite in the train split after all-NA filter.")
    if n_features_after_all_nan_filter < n_features_before:
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
    keep_variable = np.isfinite(train_std) & (train_std > float(low_var_eps))
    n_features_after_low_var_filter = int(keep_variable.sum())
    if n_features_after_low_var_filter <= 0:
        raise ValueError(
            "All feature columns were removed by low-variance filtering "
            f"(LOW_VAR_EPS={low_var_eps})."
        )
    if n_features_after_low_var_filter < n_features_after_all_nan_filter:
        X_train = X_train[:, keep_variable]
        X_val = X_val[:, keep_variable]
        X_test = X_test[:, keep_variable]
        train_median = train_median[keep_variable]

    scaler = StandardScaler(copy=False)
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clip_enabled = bool(clip_z > 0)
    if clip_enabled:
        np.clip(X_train_s, -clip_z, clip_z, out=X_train_s)
        np.clip(X_val_s, -clip_z, clip_z, out=X_val_s)
        np.clip(X_test_s, -clip_z, clip_z, out=X_test_s)

    filter_stats = {
        "n_features_before": n_features_before,
        "n_features_after_all_nan_filter": n_features_after_all_nan_filter,
        "n_features_after_low_var_filter": n_features_after_low_var_filter,
        "n_removed_all_nan": int(n_features_before - n_features_after_all_nan_filter),
        "n_removed_low_var": int(n_features_after_all_nan_filter - n_features_after_low_var_filter),
        "low_var_eps": float(low_var_eps),
        "clip_z": float(clip_z),
        "clip_z_enabled": clip_enabled,
    }
    return {
        "X_train": X_train_s,
        "X_val": X_val_s,
        "X_test": X_test_s,
        "train_median": train_median,
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler": scaler,
        "filter_stats": filter_stats,
    }


def fit_and_predict_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    split_ids: dict[str, list[str]],
) -> dict[str, Any]:
    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]

    preproc = fit_train_imputer_and_scaler(
        X,
        train_ids,
        val_ids,
        test_ids,
    )
    X_train_s = preproc["X_train"]
    X_val_s = preproc["X_val"]
    X_test_s = preproc["X_test"]
    train_median = preproc["train_median"]
    x_scaler_mean = preproc["scaler_mean"]
    x_scaler = preproc["scaler"]
    filter_stats = preproc["filter_stats"]
    y_train = y.loc[train_ids].to_numpy(dtype=np.float64)
    y_val = y.loc[val_ids].to_numpy(dtype=np.float64)
    y_test = y.loc[test_ids].to_numpy(dtype=np.float64)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std(ddof=0))
    if not np.isfinite(y_std) or y_std < CFG.TARGET_STD_EPS:
        y_std = 1.0
    y_train_s = (y_train - y_mean) / y_std

    best_alpha = None
    best_val_metrics = None
    best_val_pred = None
    best_model = None
    best_val_r2 = -np.inf
    for alpha in CFG.ALPHAS:
        model = Ridge(
            alpha=float(alpha),
            fit_intercept=bool(CFG.FIT_INTERCEPT),
            solver=CFG.RIDGE_SOLVER,
            tol=float(CFG.RIDGE_TOL),
            max_iter=int(CFG.RIDGE_MAX_ITER),
        )
        model.fit(X_train_s, y_train_s)
        pred_val = model.predict(X_val_s) * y_std + y_mean
        metrics = regression_metrics(y_val, pred_val)
        if metrics["r2"] > best_val_r2:
            best_val_r2 = float(metrics["r2"])
            best_alpha = float(alpha)
            best_val_metrics = metrics
            best_val_pred = pred_val
            best_model = model

    if best_model is None or best_alpha is None or best_val_metrics is None or best_val_pred is None:
        raise RuntimeError("Validation tuning failed to select a Ridge model.")

    pred_train = best_model.predict(X_train_s) * y_std + y_mean
    pred_test = best_model.predict(X_test_s) * y_std + y_mean
    pred_diagnostics = {
        "train": prediction_diagnostics(y_train, pred_train),
        "val": prediction_diagnostics(y_val, best_val_pred),
        "test": prediction_diagnostics(y_test, pred_test),
    }

    return {
        "alpha": best_alpha,
        "target_scaler": {"mean": y_mean, "std": y_std},
        "filter_stats": filter_stats,
        "x_imputer_train_median_preview": [float(x) for x in train_median[:10]],
        "x_scaler_mean_preview": [float(x) for x in x_scaler_mean[:10]],
        "x_scaler_scale_preview": [float(x) for x in x_scaler.scale_.astype(np.float64)[:10]],
        "predictions": {
            "train": pd.DataFrame({"ENSID": train_ids, "y_true": y_train, "y_pred": pred_train}),
            "val": pd.DataFrame({"ENSID": val_ids, "y_true": y_val, "y_pred": best_val_pred}),
            "test": pd.DataFrame({"ENSID": test_ids, "y_true": y_test, "y_pred": pred_test}),
        },
        "metrics": {
            "train": regression_metrics(y_train, pred_train),
            "val": best_val_metrics,
            "test": regression_metrics(y_test, pred_test),
        },
        "prediction_diagnostics": pred_diagnostics,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_feature_names(batch_root: Path, group_key: str, feature_names: list[str]) -> str:
    path = batch_root / f"feature_names_{group_key}.txt"
    path.write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    return str(path)


def write_id_list(path: Path, ids: list[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def prefixed_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{name}": float(value) for name, value in metrics.items()}


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        col
        for col in fold_metrics.columns
        if any(col.startswith(f"{split}_") for split in ("train", "val", "test"))
    ]
    for group_key, sub in fold_metrics.groupby("feature_group", sort=False):
        label = str(sub["feature_label"].iloc[0])
        for col in metric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            split, metric = col.split("_", 1)
            rows.append(
                {
                    "feature_group": group_key,
                    "feature_label": label,
                    "split": split,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "n_folds": int(values.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def build_paired_delta_vs_genetic_only(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in fold_metrics.columns
        if any(col.startswith(f"{split}_") for split in ("train", "val", "test"))
    ]
    indexed = fold_metrics.set_index(["cv_fold", "feature_group"], drop=False)
    rows: list[dict[str, Any]] = []
    for fold in sorted(pd.to_numeric(fold_metrics["cv_fold"], errors="coerce").dropna().astype(int).unique()):
        baseline_key = (fold, PAIRED_DELTA_BASELINE_GROUP)
        if baseline_key not in indexed.index:
            continue
        baseline = indexed.loc[baseline_key]
        for group in PAIRED_DELTA_COMPARISON_GROUPS:
            compare_key = (fold, group)
            if compare_key not in indexed.index:
                continue
            compared = indexed.loc[compare_key]
            row: dict[str, Any] = {
                "cv_fold": int(fold),
                "baseline_group": PAIRED_DELTA_BASELINE_GROUP,
                "baseline_label": str(baseline["feature_label"]),
                "comparison_group": group,
                "comparison_label": str(compared["feature_label"]),
                "comparison": f"{group} - {PAIRED_DELTA_BASELINE_GROUP}",
                "baseline_n_features": int(baseline["n_features"]),
                "comparison_n_features": int(compared["n_features"]),
                "baseline_selected_alpha": float(baseline["selected_alpha"]),
                "comparison_selected_alpha": float(compared["selected_alpha"]),
            }
            for col in metric_cols:
                baseline_value = float(baseline[col])
                compared_value = float(compared[col])
                row[f"baseline_{col}"] = baseline_value
                row[f"comparison_{col}"] = compared_value
                row[f"delta_{col}"] = compared_value - baseline_value
            rows.append(row)
    base_cols = [
        "cv_fold",
        "baseline_group",
        "baseline_label",
        "comparison_group",
        "comparison_label",
        "comparison",
        "baseline_n_features",
        "comparison_n_features",
        "baseline_selected_alpha",
        "comparison_selected_alpha",
    ]
    metric_delta_cols = [
        name
        for col in metric_cols
        for name in (f"baseline_{col}", f"comparison_{col}", f"delta_{col}")
    ]
    return pd.DataFrame(rows, columns=[*base_cols, *metric_delta_cols])


def assert_no_forbidden_feature_columns(feature_specs: dict[str, dict[str, Any]]) -> None:
    forbidden = {"GENEID", "ENSID", "HALFLIFE"}
    for key, spec in feature_specs.items():
        bad = sorted(forbidden.intersection(set(spec["feature_names"])))
        if bad:
            raise ValueError(f"Feature group {key!r} contains forbidden columns: {bad}")


def feature_missing_stats_from_ratios(miss: pd.Series) -> dict[str, Any]:
    if miss.empty:
        return {
            "mean_missing_ratio": 0.0,
            "median_missing_ratio": 0.0,
            "max_missing_ratio": 0.0,
            "columns_all_missing_count": 0,
            "columns_missing_gt_50pct": 0,
            "columns_missing_gt_90pct": 0,
        }
    return {
        "mean_missing_ratio": float(miss.mean()),
        "median_missing_ratio": float(miss.median()),
        "max_missing_ratio": float(miss.max()),
        "columns_all_missing_count": int((miss >= 1.0).sum()),
        "columns_missing_gt_50pct": int((miss > 0.5).sum()),
        "columns_missing_gt_90pct": int((miss > 0.9).sum()),
    }


def feature_missing_stats_for_spec(
    feature_spec: dict[str, Any],
    missing_ratios_by_source: dict[str, pd.Series],
) -> dict[str, Any]:
    chunks: list[pd.Series] = []
    for component in feature_spec["components"]:
        source = str(component["source"])
        columns = list(component["columns"])
        if columns:
            chunks.append(missing_ratios_by_source[source].loc[columns])
    if not chunks:
        return feature_missing_stats_from_ratios(pd.Series(dtype=np.float64))
    return feature_missing_stats_from_ratios(pd.concat(chunks, axis=0))


def make_feature_spec(
    key: str,
    components: list[tuple[str, list[str]]],
) -> dict[str, Any]:
    group_meta = FEATURE_GROUP_BY_KEY[key]
    normalized_components: list[dict[str, Any]] = []
    feature_names: list[str] = []
    for source, columns in components:
        cols = [str(col) for col in columns]
        normalized_components.append({"source": source, "columns": tuple(cols)})
        feature_names.extend(cols)
    return {
        "key": key,
        "label": str(group_meta["label"]),
        "parts": tuple(group_meta["parts"]),
        "components": tuple(normalized_components),
        "feature_names": tuple(feature_names),
        "n_features": int(len(feature_names)),
    }


def build_feature_specs(
    genetic_biochemical_columns: list[str],
    biochemical_columns: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    forbidden = {"GENEID", "ENSID", "HALFLIFE"}
    genetic_biochemical_feature_cols = [col for col in genetic_biochemical_columns if col not in forbidden]
    biochemical_feature_cols = [col for col in biochemical_columns if col not in forbidden]
    basic_cols = [col for col in genetic_biochemical_feature_cols if str(col).startswith("BASIC.")]
    genetic_cols = [col for col in genetic_biochemical_feature_cols if str(col).startswith("GENETIC.")]
    biochemical_371_cols = [col for col in genetic_biochemical_feature_cols if str(col).startswith("BIOCHEMICAL.")]
    biochemical_full_prefixes = ("RIP.", "TE.", "M6ACLIP.", "ENCORI.", "eCLIP.", "PARCLIP.")
    biochemical_full_cols = [
        col
        for col in biochemical_feature_cols
        if any(str(col).startswith(prefix) for prefix in biochemical_full_prefixes)
    ]
    required_groups = {
        "basic": basic_cols,
        "genetic": genetic_cols,
        "biochemical_371": biochemical_371_cols,
        "biochemical_full_no_basic": biochemical_full_cols,
    }
    empty = {name: cols for name, cols in required_groups.items() if not cols}
    if empty:
        raise ValueError(f"Empty Saluki feature partitions: {list(empty)}")

    feature_specs = {
        "saluki_basic8": make_feature_spec(
            "saluki_basic8",
            [("genetic_biochemical", basic_cols)],
        ),
        "genetic_only": make_feature_spec(
            "genetic_only",
            [("genetic_biochemical", genetic_cols)],
        ),
        "biochemical_371": make_feature_spec(
            "biochemical_371",
            [("genetic_biochemical", biochemical_371_cols)],
        ),
        "genetic_plus_biochemical_371": make_feature_spec(
            "genetic_plus_biochemical_371",
            [("genetic_biochemical", [*genetic_cols, *biochemical_371_cols])],
        ),
        "all_23384": make_feature_spec(
            "all_23384",
            [("genetic_biochemical", [*basic_cols, *genetic_cols, *biochemical_371_cols])],
        ),
        "biochemical_full_no_basic": make_feature_spec(
            "biochemical_full_no_basic",
            [("biochemical", biochemical_full_cols)],
        ),
        "genetic_plus_biochemical_full": make_feature_spec(
            "genetic_plus_biochemical_full",
            [
                ("genetic_biochemical", genetic_cols),
                ("biochemical", biochemical_full_cols),
            ],
        ),
        "basic_plus_biochemical_full": make_feature_spec(
            "basic_plus_biochemical_full",
            [
                ("genetic_biochemical", basic_cols),
                ("biochemical", biochemical_full_cols),
            ],
        ),
    }
    assert_no_forbidden_feature_columns(feature_specs)
    return feature_specs, {
        "basic": basic_cols,
        "genetic": genetic_cols,
        "biochemical_371": biochemical_371_cols,
        "biochemical_full_no_basic": biochemical_full_cols,
    }


def collect_required_source_columns(feature_specs: list[dict[str, Any]]) -> dict[str, list[str]]:
    required: dict[str, list[str]] = {"genetic_biochemical": [], "biochemical": []}
    for spec in feature_specs:
        for component in spec["components"]:
            source = str(component["source"])
            for col in component["columns"]:
                if col not in required[source]:
                    required[source].append(str(col))
    return required


def materialize_feature_group(
    feature_spec: dict[str, Any],
    source_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for component in feature_spec["components"]:
        source = str(component["source"])
        columns = list(component["columns"])
        if columns:
            frames.append(source_tables[source].loc[:, columns])
    if not frames:
        raise ValueError(f"Feature spec {feature_spec['key']!r} has no materializable columns")
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, axis=1, copy=False)


def main() -> None:
    paths = input_paths()
    batch_root = RESULT_ROOT / CFG.RESULT_SLUG / f"{CFG.RESULT_SLUG}_{now_ts()}"
    ensure_dir(batch_root)

    script_metadata = collect_script_runtime_metadata()
    active_group_keys = parse_requested_group_keys()
    active_folds = parse_requested_folds()
    effective_low_var_eps = parse_low_var_eps()
    effective_clip_z = parse_clip_z()
    genetic_biochemical_columns = read_saluki_table_columns(paths["genetic_biochemical"])
    biochemical_columns = read_saluki_table_columns(paths["biochemical"])
    feature_specs, source_partitions = build_feature_specs(genetic_biochemical_columns, biochemical_columns)
    active_feature_specs = [feature_specs[key] for key in active_group_keys]
    required_source_columns = collect_required_source_columns(active_feature_specs)

    print(f"[Load] normalized={paths['normalized']}", flush=True)
    df = load_human_processed_table(paths)
    cv = load_human_cv_assignments(paths)
    print(
        f"[Config] active_groups={list(active_group_keys)} active_folds={list(active_folds)}",
        flush=True,
    )
    print(
        f"[Features] loading genetic/biochemical table: {paths['genetic_biochemical']} "
        f"n_feature_cols={len(required_source_columns['genetic_biochemical'])}",
        flush=True,
    )
    genetic_biochemical_y, genetic_biochemical_features = load_saluki_feature_table(
        paths["genetic_biochemical"],
        required_source_columns["genetic_biochemical"],
    )
    print(
        f"[Features] loading biochemical table: {paths['biochemical']} "
        f"n_feature_cols={len(required_source_columns['biochemical'])}",
        flush=True,
    )
    biochemical_y, biochemical_features = load_saluki_feature_table(
        paths["biochemical"],
        required_source_columns["biochemical"],
    )

    alignment = validate_saluki_alignment(df, genetic_biochemical_y, biochemical_y)
    ensids = pd.Index(alignment["overlap_ids"], name="ENSID")
    df = df.set_index("ENSID", drop=False).loc[ensids].reset_index(drop=True)
    target = df.set_index("ENSID", drop=True)["HALFLIFE"].astype(np.float64)
    genetic_biochemical_features = genetic_biochemical_features.reindex(ensids)
    biochemical_features = biochemical_features.reindex(ensids)
    source_tables = {
        "genetic_biochemical": genetic_biochemical_features,
        "biochemical": biochemical_features,
    }
    missing_ratios_by_source = {
        source: table.isna().mean(axis=0) for source, table in source_tables.items()
    }
    print(f"[Check] overlap={alignment['overlap_n']} HALFLIFE allclose OK", flush=True)

    feature_name_paths: dict[str, str] = {}
    manifest_groups: dict[str, Any] = {}
    for spec in active_feature_specs:
        key = str(spec["key"])
        feature_names = list(spec["feature_names"])
        feature_name_paths[key] = write_feature_names(batch_root, key, feature_names)
        manifest_groups[key] = {
            "label": spec["label"],
            "parts": list(spec["parts"]),
            "n_features": int(len(feature_names)),
            "feature_names_path": feature_name_paths[key],
            "feature_name_preview": feature_names[:20],
            "feature_name_tail": feature_names[-20:],
            "source_components": [
                {
                    "source": component["source"],
                    "n_features": len(component["columns"]),
                    "feature_name_preview": list(component["columns"])[:20],
                    "feature_name_tail": list(component["columns"])[-20:],
                }
                for component in spec["components"]
            ],
            "missing_stats": feature_missing_stats_for_spec(spec, missing_ratios_by_source),
        }

    write_json(
        batch_root / "feature_group_manifest.json",
        {
            "script_metadata": script_metadata,
            "config": asdict(CFG),
            "environment_overrides": {
                SALUKI_RIDGE_GROUP_ENV: os.environ.get(SALUKI_RIDGE_GROUP_ENV),
                SALUKI_RIDGE_FOLDS_ENV: os.environ.get(SALUKI_RIDGE_FOLDS_ENV),
                SALUKI_RIDGE_LOW_VAR_EPS_ENV: os.environ.get(SALUKI_RIDGE_LOW_VAR_EPS_ENV),
                SALUKI_RIDGE_CLIP_Z_ENV: os.environ.get(SALUKI_RIDGE_CLIP_Z_ENV),
            },
            "effective_preprocessing_values": {
                "low_var_eps": effective_low_var_eps,
                "clip_z": effective_clip_z,
                "clip_z_enabled": bool(effective_clip_z > 0),
            },
            "input_paths": {name: str(path) for name, path in paths.items()},
            "id_alignment": {
                "normalized": "ENSID",
                "alldataGeneticBiochemical": "GENEID -> ENSID",
                "alldataBiochemical": "GENEID -> ENSID",
                "overlap_n": alignment["overlap_n"],
                "expected_overlap_n": CFG.EXPECTED_OVERLAP_N,
                "halflife_exact_match_genetic_biochemical": alignment[
                    "halflife_exact_match_genetic_biochemical"
                ],
                "halflife_exact_match_biochemical": alignment["halflife_exact_match_biochemical"],
                "halflife_allclose_atol": alignment["halflife_allclose_atol"],
                "halflife_allclose_genetic_biochemical": alignment[
                    "halflife_allclose_genetic_biochemical"
                ],
                "halflife_allclose_biochemical": alignment["halflife_allclose_biochemical"],
                "halflife_max_abs_diff_genetic_biochemical": alignment[
                    "halflife_max_abs_diff_genetic_biochemical"
                ],
                "halflife_max_abs_diff_biochemical": alignment["halflife_max_abs_diff_biochemical"],
            },
            "available_feature_groups": list(feature_specs.keys()),
            "active_feature_groups": list(active_group_keys),
            "active_cv_folds": list(active_folds),
            "required_source_columns_loaded": {
                source: len(cols) for source, cols in required_source_columns.items()
            },
            "source_feature_partitions": {
                name: {
                    "n_features": len(cols),
                    "feature_name_preview": cols[:20],
                    "feature_name_tail": cols[-20:],
                }
                for name, cols in source_partitions.items()
            },
            "preprocessing": {
                "imputer": "per-fold train median; all-NaN train columns use 0.0",
                "feature_filter": (
                    "train-only: remove columns with no finite train values before median impute; "
                    "after median impute remove columns with train std <= LOW_VAR_EPS"
                ),
                "scaler": "StandardScaler fit on train only",
                "z_score_clipping": "after StandardScaler, clip transformed features to [-CLIP_Z, CLIP_Z] if CLIP_Z > 0",
                "default_low_var_eps": DEFAULT_LOW_VAR_EPS,
                "default_clip_z": DEFAULT_CLIP_Z,
                "target": "train-only standardization for Ridge fit; predictions inverse-transformed",
                "ridge_alpha_selection": "validation R2 within each outer fold",
                "refit_train_val": False,
                "forbidden_feature_columns": ["GENEID", "ENSID", "HALFLIFE"],
            },
            "feature_groups": manifest_groups,
        },
    )
    write_json(
        batch_root / "resolved_run_config.json",
        {
            "batch_mode": "cv_multi_fold",
            "experiment_slug": CFG.RESULT_SLUG,
            "batch_root": str(batch_root),
            "split_id_dir": str(batch_root / "splits"),
            "requested_cv_folds": list(active_folds),
            "configured_cv_folds": list(CFG.CV_FOLDS),
            "requested_feature_groups": list(active_group_keys),
            "effective_preprocessing_values": {
                "low_var_eps": effective_low_var_eps,
                "clip_z": effective_clip_z,
                "clip_z_enabled": bool(effective_clip_z > 0),
            },
            "config": asdict(CFG),
            "script_metadata": script_metadata,
        },
    )

    fold_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    pred_val_rows: list[pd.DataFrame] = []
    pred_test_rows: list[pd.DataFrame] = []
    split_dir = batch_root / "splits"
    ensure_dir(split_dir)

    for fold in active_folds:
        split_ids = regression_split_from_fold(df, cv, int(fold))
        for split_name in ("train", "val", "test"):
            write_id_list(split_dir / f"fold_{int(fold)}_{split_name}_ids.txt", split_ids[split_name])
        print(
            f"[Fold {fold}] n_train={len(split_ids['train'])} "
            f"n_val={len(split_ids['val'])} n_test={len(split_ids['test'])}",
            flush=True,
        )
        for spec in active_feature_specs:
            key = str(spec["key"])
            label = str(spec["label"])
            X = materialize_feature_group(spec, source_tables)
            try:
                print(f"[Fold {fold}] fitting {label} n_features={X.shape[1]}", flush=True)
                fit_payload = fit_and_predict_ridge(X, target, split_ids)
                metrics = fit_payload["metrics"]
                pred_diag = fit_payload["prediction_diagnostics"]
                filter_stats = fit_payload["filter_stats"]

                row = {
                    "feature_group": key,
                    "feature_label": label,
                    "cv_fold": int(fold),
                    "n_features": int(X.shape[1]),
                    "selected_alpha": float(fit_payload["alpha"]),
                    **filter_stats,
                    **prefixed_metrics(metrics["train"], "train"),
                    **prefixed_metrics(pred_diag["train"], "train"),
                    **prefixed_metrics(metrics["val"], "val"),
                    **prefixed_metrics(pred_diag["val"], "val"),
                    **prefixed_metrics(metrics["test"], "test"),
                    **prefixed_metrics(pred_diag["test"], "test"),
                }
                fold_rows.append(row)
                selected_rows.append(
                    {
                        "feature_group": key,
                        "feature_label": label,
                        "cv_fold": int(fold),
                        "selected_alpha": float(fit_payload["alpha"]),
                        "val_r2": float(metrics["val"]["r2"]),
                        "test_r2": float(metrics["test"]["r2"]),
                        "n_features": int(X.shape[1]),
                        **filter_stats,
                    }
                )
                for split_name, collector in (("val", pred_val_rows), ("test", pred_test_rows)):
                    pred_df = fit_payload["predictions"][split_name].copy()
                    pred_df.insert(0, "feature_group", key)
                    pred_df.insert(1, "feature_label", label)
                    pred_df.insert(2, "cv_fold", int(fold))
                    pred_df["selected_alpha"] = float(fit_payload["alpha"])
                    collector.append(pred_df)
                del fit_payload, metrics, pred_diag, filter_stats
            finally:
                del X
                gc.collect()

    fold_metrics = pd.DataFrame(fold_rows)
    selected_alpha = pd.DataFrame(selected_rows)
    pred_val = pd.concat(pred_val_rows, axis=0, ignore_index=True)
    pred_test = pd.concat(pred_test_rows, axis=0, ignore_index=True)
    summary = summarize_metrics(fold_metrics)
    paired_delta = build_paired_delta_vs_genetic_only(fold_metrics)

    fold_metrics.to_csv(batch_root / "fold_metrics.csv", index=False)
    summary.to_csv(batch_root / "summary_metrics.csv", index=False)
    paired_delta.to_csv(batch_root / "paired_delta_vs_genetic_only.csv", index=False)
    pred_val.to_csv(batch_root / "pred_val.csv", index=False)
    pred_test.to_csv(batch_root / "pred_test.csv", index=False)
    selected_alpha.to_csv(batch_root / "selected_alpha_per_fold.csv", index=False)
    write_json(
        batch_root / "run_summary.json",
        {
            "batch_root": str(batch_root),
            "active_feature_groups": list(active_group_keys),
            "active_cv_folds": list(active_folds),
            "mean_test_r2": {
                key: float(
                    fold_metrics.loc[fold_metrics["feature_group"] == key, "test_r2"].astype(float).mean()
                )
                for key in fold_metrics["feature_group"].unique()
            },
            "paired_delta_vs_genetic_only_path": str(batch_root / "paired_delta_vs_genetic_only.csv"),
            "script_metadata": script_metadata,
        },
    )
    print(f"[DONE] batch_root={batch_root}", flush=True)


if __name__ == "__main__":
    main()
