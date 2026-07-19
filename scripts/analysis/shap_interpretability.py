#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No-args XGB interpretation with TreeSHAP, permutation importance, and feature-family summaries.

This script retrains the selected Saluki XGB feature group with the same fold
protocol as the main XGB run, then computes XGBoost's built-in TreeSHAP
contributions on held-out test samples. It is intended for interpretation, not
for changing the benchmark protocol.

SHAP values here identify predictive associations, not causal effects.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBRegressor


# =============================================================================
# CONFIG: paths are resolved from environment first. Keep these aligned with
# the clean XGBoost benchmark run.
# =============================================================================
def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else Path(default)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


PROJECT_ROOT = env_path("SALUKI_PROJECT_ROOT", Path(__file__).resolve().parents[2])
RUNS_ROOT = env_path("SALUKI_RUNS_ROOT", PROJECT_ROOT / "results" / "runs")

NORMALIZED_PATH = env_path("SALUKI_SHAP_NORMALIZED_PATH", os.environ.get("SALUKI_NORMALIZED_PATH", PROJECT_ROOT / "data/processed/human/normalized.csv.gz"))
SPLIT_PATH = env_path("SALUKI_SHAP_SPLIT_PATH", os.environ.get("SALUKI_SPLIT_PATH", PROJECT_ROOT / "data/processed/human/splits/cv_k5_seed42.csv"))
FEATURE_ROOT = env_path("SALUKI_SHAP_FEATURE_ROOT", os.environ.get("SALUKI_FEATURE_ROOT", PROJECT_ROOT / "data/saluki_features/human"))
GENETIC_BIOCHEMICAL_PATH = env_path(
    "SALUKI_SHAP_GENETIC_BIOCHEMICAL_PATH",
    os.environ.get("SALUKI_GENETIC_BIOCHEMICAL_PATH", FEATURE_ROOT / "alldataGeneticBiochemical.tsv.gz"),
)
BIOCHEMICAL_FULL_PATH = env_path(
    "SALUKI_SHAP_BIOCHEMICAL_FULL_PATH",
    os.environ.get("SALUKI_BIOCHEMICAL_FULL_PATH", FEATURE_ROOT / "alldataBiochemical.tsv.gz"),
)

OUT_BASE_DIR = env_path("SALUKI_SHAP_OUT_BASE_DIR", RUNS_ROOT)
RUN_TAG = env_str("SALUKI_SHAP_RUN_TAG", "shap_interpretability")
OUT_DIR = OUT_BASE_DIR / RUN_TAG

FEATURE_GROUP = "basic_plus_biochemical_full"
CV_FOLDS = [0, 1, 2, 3, 4]
SEED = 42
INNER_VAL_RATIO = 0.10
STRATIFY_BINS = 20
EXPECTED_OVERLAP = 12981

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
}
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

MAX_SHAP_TEST_SAMPLES_PER_FOLD: int | None = None
PLOT_FEATURE_CANDIDATE_LIMIT_PER_FOLD = 100
TOP_N_FEATURES = 20
DEPENDENCE_TOP_N = 8

RUN_PERMUTATION_IMPORTANCE = False
PERMUTATION_IMPORTANCE_TOP_N = 80
PERMUTATION_REPEATS = 3

TARGET_FEATURES = [
    "RIP.YTHDF2.polyA.He",
    "BASIC.ORFEXONDENSITY",
    "eCLIP.LARP4.HepG2",
    "eCLIP.PABPC4.K562",
    "RIP.YTHDF3_polyA_He",
    "eCLIP.SUB1.HepG2",
    "RIP.HDAC1",
    "eCLIP.RPS3.HepG2",
    "eCLIP.DDX3X.HepG2",
    "eCLIP.YBX3.K562",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def config_dict() -> dict[str, Any]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "timestamp": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "normalized_path": str(NORMALIZED_PATH),
        "split_path": str(SPLIT_PATH),
        "feature_root": str(FEATURE_ROOT),
        "genetic_biochemical_path": str(GENETIC_BIOCHEMICAL_PATH),
        "biochemical_full_path": str(BIOCHEMICAL_FULL_PATH),
        "out_base_dir": str(OUT_BASE_DIR),
        "run_tag": RUN_TAG,
        "out_dir": str(OUT_DIR),
        "feature_group": FEATURE_GROUP,
        "cv_folds": list(CV_FOLDS),
        "seed": SEED,
        "inner_val_ratio": INNER_VAL_RATIO,
        "stratify_bins": STRATIFY_BINS,
        "expected_overlap": EXPECTED_OVERLAP,
        "xgb_params": dict(XGB_PARAMS, n_jobs=N_JOBS),
        "max_shap_test_samples_per_fold": MAX_SHAP_TEST_SAMPLES_PER_FOLD,
        "plot_feature_candidate_limit_per_fold": PLOT_FEATURE_CANDIDATE_LIMIT_PER_FOLD,
        "top_n_features": TOP_N_FEATURES,
        "dependence_top_n": DEPENDENCE_TOP_N,
        "run_permutation_importance": RUN_PERMUTATION_IMPORTANCE,
        "permutation_importance_top_n": PERMUTATION_IMPORTANCE_TOP_N,
        "permutation_repeats": PERMUTATION_REPEATS,
        "target_features": list(TARGET_FEATURES),
    }


def print_config_and_raise(message: str) -> None:
    print("[CONFIG]", flush=True)
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False), flush=True)
    raise FileNotFoundError(message)


def validate_config_paths() -> None:
    required = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "NORMALIZED_PATH": NORMALIZED_PATH,
        "SPLIT_PATH": SPLIT_PATH,
        "FEATURE_ROOT": FEATURE_ROOT,
        "GENETIC_BIOCHEMICAL_PATH": GENETIC_BIOCHEMICAL_PATH,
        "BIOCHEMICAL_FULL_PATH": BIOCHEMICAL_FULL_PATH,
    }
    missing = {name: str(path) for name, path in required.items() if not path.exists()}
    if missing:
        print_config_and_raise(f"Missing CONFIG path(s): {missing}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def bh_fdr(p_values: pd.Series | np.ndarray) -> np.ndarray:
    p = pd.to_numeric(pd.Series(p_values), errors="coerce").to_numpy(dtype=np.float64)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(p)
    if int(valid.sum()) == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    adjusted = ranked * m / np.arange(1, len(ranked) + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(pv)
    out[order] = adjusted
    q[valid] = out
    return q


def _make_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    uniq = np.unique(y[np.isfinite(y)])
    n = int(min(n_bins, max(2, len(uniq))))
    s = pd.Series(y)
    try:
        bins = pd.qcut(s, q=n, labels=False, duplicates="drop")
    except Exception:
        bins = pd.qcut(s.rank(method="average"), q=n, labels=False, duplicates="drop")
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


def finite_corr(x: np.ndarray, y: np.ndarray, ranked: bool = False) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if ranked:
        x = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def load_normalized() -> pd.DataFrame:
    df = pd.read_csv(NORMALIZED_PATH, compression="gzip", usecols=["ENSID", "HALFLIFE"])
    df["ENSID"] = df["ENSID"].astype(str)
    df["HALFLIFE"] = pd.to_numeric(df["HALFLIFE"], errors="coerce")
    df = df.dropna(subset=["HALFLIFE"]).drop_duplicates("ENSID")
    return df.reset_index(drop=True)


def load_split() -> pd.DataFrame:
    cv = pd.read_csv(SPLIT_PATH)
    if "fold" not in cv.columns and "cv_fold" in cv.columns:
        cv = cv.rename(columns={"cv_fold": "fold"})
    if "fold" not in cv.columns:
        raise ValueError(f"{SPLIT_PATH} must contain either 'fold' or 'cv_fold'")
    cv["ENSID"] = cv["ENSID"].astype(str)
    cv["fold"] = pd.to_numeric(cv["fold"], errors="coerce").astype("Int64")
    return cv.dropna(subset=["fold"]).astype({"fold": int})


def read_columns(path: Path) -> list[str]:
    return pd.read_csv(path, sep="\t", compression="infer", nrows=0).columns.astype(str).tolist()


def build_feature_specs(gb_cols: list[str], bio_cols: list[str]) -> dict[str, dict[str, Any]]:
    forbidden = {"GENEID", "ENSID", "HALFLIFE"}
    gb_features = [c for c in gb_cols if c not in forbidden]
    bio_features = [c for c in bio_cols if c not in forbidden]
    basic = [c for c in gb_features if c.startswith("BASIC.")]
    genetic = [c for c in gb_features if c.startswith("GENETIC.")]
    biochemical_371 = [c for c in gb_features if c.startswith("BIOCHEMICAL.")]
    bio_prefixes = ("RIP.", "TE.", "M6ACLIP.", "ENCORI.", "eCLIP.", "PARCLIP.")
    bio_full = [c for c in bio_features if any(c.startswith(p) for p in bio_prefixes)]
    specs = {
        "saluki_basic8": {"components": [("genetic_biochemical", basic)]},
        "genetic_only": {"components": [("genetic_biochemical", genetic)]},
        "biochemical_371": {"components": [("genetic_biochemical", biochemical_371)]},
        "biochemical_full_no_basic": {"components": [("biochemical", bio_full)]},
        "basic_plus_biochemical_full": {"components": [("genetic_biochemical", basic), ("biochemical", bio_full)]},
        "genetic_plus_biochemical_371": {"components": [("genetic_biochemical", [*genetic, *biochemical_371])]},
        "all_23384": {"components": [("genetic_biochemical", [*basic, *genetic, *biochemical_371])]},
        "genetic_plus_biochemical_full": {"components": [("genetic_biochemical", genetic), ("biochemical", bio_full)]},
    }
    for key, spec in specs.items():
        names = [col for _, cols in spec["components"] for col in cols]
        bad = sorted(set(names).intersection(forbidden))
        if bad:
            raise ValueError(f"Feature group {key} contains forbidden columns: {bad}")
        spec["feature_names"] = names
        spec["n_features"] = len(names)
    return specs


def load_feature_source(path: Path, columns: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    forbidden = {"GENEID", "ENSID", "HALFLIFE"}
    bad = sorted(set(columns).intersection(forbidden))
    if bad:
        raise ValueError(f"Forbidden columns requested as features: {bad}")
    dtype = {col: "float32" for col in columns}
    dtype["GENEID"] = "string"
    dtype["HALFLIFE"] = "float64"
    df = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        usecols=["GENEID", "HALFLIFE", *columns],
        dtype=dtype,
        na_values=["NA", "NaN", ""],
        low_memory=False,
    )
    df["GENEID"] = df["GENEID"].astype(str)
    if df["GENEID"].duplicated().any():
        raise ValueError(f"{path} contains duplicated GENEID values")
    indexed = df.set_index("GENEID", drop=True)
    return indexed["HALFLIFE"].astype(np.float64), indexed[columns]


def validate_alignment(norm: pd.DataFrame, gb_y: pd.Series, bio_y: pd.Series) -> pd.Index:
    norm_y = norm.set_index("ENSID")["HALFLIFE"].astype(np.float64)
    overlap = norm_y.index.intersection(gb_y.index).intersection(bio_y.index)
    overlap = pd.Index(overlap.astype(str), name="ENSID")
    if EXPECTED_OVERLAP is not None and len(overlap) != int(EXPECTED_OVERLAP):
        raise ValueError(f"Expected overlap={EXPECTED_OVERLAP}, got {len(overlap)}")
    y0 = norm_y.loc[overlap].to_numpy(np.float64)
    y1 = gb_y.loc[overlap].to_numpy(np.float64)
    y2 = bio_y.loc[overlap].to_numpy(np.float64)
    if not np.allclose(y0, y1, rtol=0.0, atol=1e-8):
        raise ValueError(f"HALFLIFE mismatch for genetic_biochemical, max_abs={float(np.max(np.abs(y0-y1)))}")
    if not np.allclose(y0, y2, rtol=0.0, atol=1e-8):
        raise ValueError(f"HALFLIFE mismatch for biochemical, max_abs={float(np.max(np.abs(y0-y2)))}")
    return overlap


def materialize_feature_group(spec: dict[str, Any], sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [sources[source].loc[:, cols] for source, cols in spec["components"] if cols]
    if not frames:
        raise ValueError("Feature spec has no columns")
    return pd.concat(frames, axis=1, copy=False) if len(frames) > 1 else frames[0]


def split_from_fold(df: pd.DataFrame, cv: pd.DataFrame, fold: int) -> dict[str, list[str]]:
    dfi = df.set_index("ENSID", drop=False)
    test_ids = [x for x in cv.loc[cv["fold"] == fold, "ENSID"].astype(str).tolist() if x in dfi.index]
    pool = [x for x in cv.loc[cv["fold"] != fold, "ENSID"].astype(str).tolist() if x in dfi.index]
    train_ids, val_ids = stratified_train_val(
        pool,
        dfi.loc[pool, "HALFLIFE"].to_numpy(np.float64),
        INNER_VAL_RATIO,
        seed=SEED * 100 + fold,
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def best_iteration_from_model(model: XGBRegressor) -> int:
    value = getattr(model, "best_iteration", None)
    if value is None:
        value = getattr(model, "best_iteration_", None)
    return int(value) if value is not None else int(XGB_PARAMS["n_estimators"]) - 1


def predict_best(model: XGBRegressor, X: np.ndarray) -> np.ndarray:
    best_iteration = best_iteration_from_model(model)
    try:
        return np.asarray(model.predict(X, iteration_range=(0, best_iteration + 1)), dtype=np.float64)
    except TypeError:
        return np.asarray(model.predict(X), dtype=np.float64)


def booster_contribs(model: XGBRegressor, X: np.ndarray) -> np.ndarray:
    booster = model.get_booster()
    dmat = DMatrix(X)
    best_iteration = best_iteration_from_model(model)
    try:
        contribs = booster.predict(dmat, pred_contribs=True, iteration_range=(0, best_iteration + 1))
    except TypeError:
        try:
            contribs = booster.predict(dmat, pred_contribs=True, ntree_limit=best_iteration + 1)
        except TypeError:
            contribs = booster.predict(dmat, pred_contribs=True)
    return np.asarray(contribs, dtype=np.float64)


def train_xgb(X: pd.DataFrame, y: pd.Series, split_ids: dict[str, list[str]], fold: int) -> XGBRegressor:
    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    X_train = X.loc[train_ids].to_numpy(dtype=np.float32, copy=True)
    X_val = X.loc[val_ids].to_numpy(dtype=np.float32, copy=True)
    y_train = y.loc[train_ids].to_numpy(np.float64)
    y_val = y.loc[val_ids].to_numpy(np.float64)
    model = XGBRegressor(**dict(XGB_PARAMS, n_jobs=N_JOBS, random_state=SEED + fold, verbosity=1))
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def feature_family(feature: str) -> str:
    if feature.startswith("BIOCHEMICAL."):
        rest = feature[len("BIOCHEMICAL.") :]
        return rest.split(".", 1)[0]
    return feature.split(".", 1)[0]


def feature_interpretation_stub(feature: str) -> str:
    family = feature_family(feature)
    if family == "BASIC":
        return "transcript architecture signal"
    if family in {"RIP", "eCLIP", "ENCORI", "PARCLIP"}:
        return "RNA-binding protein/regulatory occupancy signal"
    if family == "M6ACLIP":
        return "m6A-associated regulatory signal"
    if family == "TE":
        return "translation efficiency-associated signal"
    if family == "GENETIC":
        return "sequence/genetic feature signal"
    return f"{family} predictive feature"


def normalize_feature_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value).lower()


def resolve_target_features(feature_names: list[str], targets: list[str]) -> pd.DataFrame:
    norm_to_feature = {normalize_feature_token(name): name for name in feature_names}
    rows: list[dict[str, Any]] = []
    for target in targets:
        if target in feature_names:
            rows.append({"requested_feature": target, "resolved_feature": target, "match_type": "exact"})
            continue
        token = normalize_feature_token(target)
        if token in norm_to_feature:
            rows.append({"requested_feature": target, "resolved_feature": norm_to_feature[token], "match_type": "normalized"})
            continue
        matches = [name for name in feature_names if token and token in normalize_feature_token(name)]
        if not matches:
            short = target.split(".", 1)[-1]
            short_token = normalize_feature_token(short)
            matches = [name for name in feature_names if short_token and short_token in normalize_feature_token(name)]
        rows.append(
            {
                "requested_feature": target,
                "resolved_feature": matches[0] if matches else "",
                "match_type": "substring" if matches else "missing",
                "n_candidate_matches": len(matches),
                "candidate_examples": ";".join(matches[:5]),
            }
        )
    return pd.DataFrame(rows)


def shap_feature_summary(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    fold: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, feature in enumerate(feature_names):
        values = feature_values[:, idx].astype(np.float64)
        shap = shap_values[:, idx].astype(np.float64)
        finite = np.isfinite(values) & np.isfinite(shap)
        high_minus_low = float("nan")
        if int(finite.sum()) >= 8 and np.nanstd(values[finite]) > 0:
            lo = np.nanquantile(values[finite], 0.25)
            hi = np.nanquantile(values[finite], 0.75)
            low_shap = shap[finite & (values <= lo)]
            high_shap = shap[finite & (values >= hi)]
            if len(low_shap) and len(high_shap):
                high_minus_low = float(np.nanmedian(high_shap) - np.nanmedian(low_shap))
        rows.append(
            {
                "cv_fold": fold,
                "feature": feature,
                "family": feature_family(feature),
                "mean_abs_shap": float(np.nanmean(np.abs(shap))),
                "mean_shap": float(np.nanmean(shap)),
                "median_shap": float(np.nanmedian(shap)),
                "positive_shap_fraction": float(np.nanmean(shap > 0)),
                "feature_mean": float(np.nanmean(values)),
                "feature_missing_fraction": float(np.mean(~np.isfinite(values))),
                "spearman_feature_vs_shap": finite_corr(values, shap, ranked=True),
                "pearson_feature_vs_shap": finite_corr(values, shap, ranked=False),
                "high_minus_low_median_shap": high_minus_low,
            }
        )
    return pd.DataFrame(rows)


def aggregate_shap_summary(per_fold: pd.DataFrame, gain: pd.DataFrame) -> pd.DataFrame:
    agg = (
        per_fold.groupby(["feature", "family"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", "std"),
            mean_shap=("mean_shap", "mean"),
            positive_shap_fraction=("positive_shap_fraction", "mean"),
            spearman_feature_vs_shap=("spearman_feature_vs_shap", "mean"),
            high_minus_low_median_shap=("high_minus_low_median_shap", "mean"),
            n_folds=("cv_fold", "nunique"),
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    gain_agg = gain.groupby("feature", as_index=False).agg(sum_gain=("gain", "sum"), mean_gain=("gain", "mean"), sum_weight=("weight", "sum"))
    out = agg.merge(gain_agg, on="feature", how="left")
    out[["sum_gain", "mean_gain", "sum_weight"]] = out[["sum_gain", "mean_gain", "sum_weight"]].fillna(0.0)
    out.insert(0, "rank_mean_abs_shap", np.arange(1, len(out) + 1))
    return out


def collect_gain_importance(model: XGBRegressor, feature_names: list[str], fold: int) -> pd.DataFrame:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    rows = []
    for idx, feature in enumerate(feature_names):
        key = f"f{idx}"
        rows.append({"cv_fold": fold, "feature": feature, "family": feature_family(feature), "gain": float(gain.get(key, 0.0)), "weight": float(weight.get(key, 0.0))})
    return pd.DataFrame(rows)


def permutation_importance(
    model: XGBRegressor,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_names: list[str],
    candidate_features: list[str],
    fold: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED * 1000 + fold)
    base_pred = predict_best(model, X_eval)
    base_r2 = r2_np(y_eval, base_pred)
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    rows: list[dict[str, Any]] = []
    for feature in candidate_features:
        idx = name_to_idx.get(feature)
        if idx is None:
            continue
        drops = []
        for repeat in range(PERMUTATION_REPEATS):
            Xp = X_eval.copy()
            rng.shuffle(Xp[:, idx])
            pred = predict_best(model, Xp)
            score = r2_np(y_eval, pred)
            drops.append(base_r2 - score)
            rows.append(
                {
                    "cv_fold": fold,
                    "feature": feature,
                    "family": feature_family(feature),
                    "repeat": repeat,
                    "base_r2": base_r2,
                    "permuted_r2": score,
                    "r2_drop": base_r2 - score,
                }
            )
        print(f"[PERM] fold={fold} feature={feature} mean_r2_drop={np.mean(drops):.6f}", flush=True)
    return pd.DataFrame(rows)


def summarize_families(shap_top: pd.DataFrame, gain_all: pd.DataFrame) -> pd.DataFrame:
    top50 = set(shap_top.head(50)["feature"].astype(str))
    shap_family = shap_top.groupby("family", as_index=False).agg(
        n_features=("feature", "count"),
        sum_abs_shap=("mean_abs_shap", "sum"),
        mean_abs_shap=("mean_abs_shap", "mean"),
        top50_count=("feature", lambda x: int(sum(v in top50 for v in x))),
    )
    gain_family = gain_all.groupby("family", as_index=False).agg(sum_gain=("gain", "sum"), mean_gain=("gain", "mean"))
    out = shap_family.merge(gain_family, on="family", how="outer").fillna(0.0)
    top_features = (
        shap_top.sort_values("mean_abs_shap", ascending=False)
        .groupby("family")["feature"]
        .apply(lambda x: ";".join(x.head(10).tolist()))
        .reset_index(name="top_features")
    )
    out = out.merge(top_features, on="family", how="left").sort_values("sum_abs_shap", ascending=False)
    return out.reset_index(drop=True)


def direction_table(
    shap_top: pd.DataFrame,
    resolved_targets: pd.DataFrame,
    per_fold_shap: pd.DataFrame,
) -> pd.DataFrame:
    target_features = [x for x in resolved_targets["resolved_feature"].astype(str).tolist() if x]
    selected = []
    for feature in [*target_features, *shap_top.head(20)["feature"].astype(str).tolist()]:
        if feature not in selected:
            selected.append(feature)
    subset = shap_top.set_index("feature").reindex(selected).dropna(subset=["mean_abs_shap"]).reset_index()
    rows = []
    fold_direction = per_fold_shap.loc[
        per_fold_shap["feature"].isin(selected),
        ["feature", "cv_fold", "high_minus_low_median_shap"],
    ].copy()
    for row in subset.to_dict("records"):
        delta = float(row.get("high_minus_low_median_shap", np.nan))
        per_feature = fold_direction[fold_direction["feature"] == row["feature"]]
        n_pos = int((per_feature["high_minus_low_median_shap"] > 0).sum())
        n_neg = int((per_feature["high_minus_low_median_shap"] < 0).sum())
        n_valid = int(np.isfinite(per_feature["high_minus_low_median_shap"]).sum())
        max_same_direction = max(n_pos, n_neg)
        direction_consistency = float(max_same_direction / n_valid) if n_valid else float("nan")
        if max_same_direction < min(4, max(n_valid, 1)):
            direction = "direction not stable across folds"
        elif np.isfinite(delta):
            direction = "high value -> predicted longer half-life" if delta > 0 else "high value -> predicted shorter half-life"
        else:
            direction = "direction not stable/estimable"
        rows.append(
            {
                "feature": row["feature"],
                "family": row["family"],
                "model_effect_direction": direction,
                "mean_abs_shap": row["mean_abs_shap"],
                "mean_shap": row["mean_shap"],
                "spearman_feature_vs_shap": row["spearman_feature_vs_shap"],
                "high_minus_low_median_shap": delta,
                "n_positive_direction_folds": n_pos,
                "n_negative_direction_folds": n_neg,
                "n_direction_valid_folds": n_valid,
                "direction_consistency": direction_consistency,
                "biological_interpretation": feature_interpretation_stub(row["feature"]),
                "caution": "Predictive association only; not a causal effect.",
            }
        )
    return pd.DataFrame(rows)


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping plots: {exc}", flush=True)
        return None


def save_bar_plot(shap_top: pd.DataFrame, out_dir: Path) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    data = shap_top.head(TOP_N_FEATURES).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(data))))
    ax.barh(data["feature"], data["mean_abs_shap"], color="#4979a8")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("")
    ax.set_title("Top XGB features by mean |SHAP|")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"shap_top20_bar.{ext}", dpi=220)
    plt.close(fig)


def save_beeswarm_like(sample_rows: pd.DataFrame, shap_top: pd.DataFrame, out_dir: Path) -> None:
    plt = try_import_matplotlib()
    if plt is None or sample_rows.empty:
        return
    top = shap_top.head(TOP_N_FEATURES)["feature"].tolist()
    data = sample_rows[sample_rows["feature"].isin(top)].copy()
    if data.empty:
        return
    order = top[::-1]
    y_map = {feature: idx for idx, feature in enumerate(order)}
    rng = np.random.default_rng(SEED)
    data["y"] = data["feature"].map(y_map).astype(float) + rng.normal(0, 0.08, size=len(data))
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(order))))
    sc = ax.scatter(
        data["shap_value"],
        data["y"],
        c=data["feature_value_rank"],
        cmap="viridis",
        s=8,
        alpha=0.55,
        linewidths=0,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("SHAP value")
    ax.set_title("Top-feature SHAP distribution")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Within-feature value rank")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"shap_beeswarm_top20.{ext}", dpi=220)
    plt.close(fig)


def save_dependence_plots(sample_rows: pd.DataFrame, shap_top: pd.DataFrame, resolved_targets: pd.DataFrame, out_dir: Path) -> None:
    plt = try_import_matplotlib()
    if plt is None or sample_rows.empty:
        return
    target = [x for x in resolved_targets["resolved_feature"].astype(str).tolist() if x]
    features: list[str] = []
    for feature in [*target, *shap_top.head(DEPENDENCE_TOP_N)["feature"].astype(str).tolist()]:
        if feature not in features:
            features.append(feature)
    features = features[: max(DEPENDENCE_TOP_N, 1)]
    n = len(features)
    if n == 0:
        return
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.4 * rows), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        data = sample_rows[sample_rows["feature"] == feature]
        ax.scatter(data["feature_value"], data["shap_value"], s=8, alpha=0.45, color="#305f8d", linewidths=0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(feature)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("SHAP value")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"shap_dependence_top_features.{ext}", dpi=220)
    plt.close(fig)


def write_report(out_dir: Path, shap_top: pd.DataFrame, family: pd.DataFrame, direction: pd.DataFrame) -> None:
    lines = [
        "# XGB SHAP Interpretability Report",
        "",
        "SHAP values identify predictive associations, not causal effects.",
        "",
        "## Top Features",
        "",
    ]
    for row in shap_top.head(10).itertuples(index=False):
        lines.append(f"- `{row.feature}` ({row.family}): mean |SHAP| = {row.mean_abs_shap:.4f}")
    lines.extend(["", "## Feature Families", ""])
    for row in family.head(8).itertuples(index=False):
        lines.append(f"- `{row.family}`: sum |SHAP| = {row.sum_abs_shap:.4f}, top50_count = {int(row.top50_count)}")
    lines.extend(["", "## Direction Table Preview", ""])
    for row in direction.head(12).itertuples(index=False):
        lines.append(f"- `{row.feature}`: {row.model_effect_direction}")
    (out_dir / "xgb_shap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[CONFIG]", flush=True)
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False), flush=True)
    validate_config_paths()
    cfg = config_dict()
    write_json(OUT_DIR / "xgb_shap_run_config.json", cfg)
    write_json(OUT_DIR / "run_config.json", cfg)

    print("[LOAD] normalized/splits/header", flush=True)
    norm = load_normalized()
    cv = load_split()
    gb_cols = read_columns(GENETIC_BIOCHEMICAL_PATH)
    bio_cols = read_columns(BIOCHEMICAL_FULL_PATH)
    feature_specs = build_feature_specs(gb_cols, bio_cols)
    if FEATURE_GROUP not in feature_specs:
        raise ValueError(f"Unknown FEATURE_GROUP={FEATURE_GROUP}. Available={sorted(feature_specs)}")
    spec = feature_specs[FEATURE_GROUP]
    required: dict[str, list[str]] = {"genetic_biochemical": [], "biochemical": []}
    for source, cols in spec["components"]:
        required[source].extend(cols)

    print(f"[LOAD] materializing feature group={FEATURE_GROUP} n_features={spec['n_features']}", flush=True)
    gb_y, gb_features = load_feature_source(GENETIC_BIOCHEMICAL_PATH, required["genetic_biochemical"])
    bio_y, bio_features = load_feature_source(BIOCHEMICAL_FULL_PATH, required["biochemical"])
    overlap = validate_alignment(norm, gb_y, bio_y)
    y = norm.set_index("ENSID")["HALFLIFE"].loc[overlap].astype(np.float64)
    sources = {
        "genetic_biochemical": gb_features.loc[overlap],
        "biochemical": bio_features.loc[overlap],
    }
    X = materialize_feature_group(spec, sources)
    X = X.loc[overlap]
    feature_names = list(X.columns.astype(str))
    resolved_targets = resolve_target_features(feature_names, TARGET_FEATURES)
    resolved_targets.to_csv(OUT_DIR / "shap_target_feature_resolution.csv", index=False)

    dfi = norm[norm["ENSID"].isin(overlap)].reset_index(drop=True)
    per_fold_shap: list[pd.DataFrame] = []
    per_fold_gain: list[pd.DataFrame] = []
    plot_rows: list[pd.DataFrame] = []
    perm_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for fold in CV_FOLDS:
        print(f"[FOLD] {fold}", flush=True)
        split_ids = split_from_fold(dfi, cv, int(fold))
        model = train_xgb(X, y, split_ids, int(fold))
        gain_df = collect_gain_importance(model, feature_names, int(fold))
        per_fold_gain.append(gain_df)

        test_ids = split_ids["test"]
        if MAX_SHAP_TEST_SAMPLES_PER_FOLD is not None and len(test_ids) > int(MAX_SHAP_TEST_SAMPLES_PER_FOLD):
            rng = np.random.default_rng(SEED * 10 + int(fold))
            test_ids = rng.choice(np.asarray(test_ids, dtype=object), size=int(MAX_SHAP_TEST_SAMPLES_PER_FOLD), replace=False).tolist()
        X_test = X.loc[test_ids].to_numpy(dtype=np.float32, copy=True)
        y_test = y.loc[test_ids].to_numpy(np.float64)
        pred_test = predict_best(model, X_test)
        contribs = booster_contribs(model, X_test)
        shap_values = contribs[:, :-1]
        base_value = contribs[:, -1]
        fold_rows.append(
            {
                "cv_fold": fold,
                "n_train": len(split_ids["train"]),
                "n_val": len(split_ids["val"]),
                "n_test_shap": len(test_ids),
                "best_iteration": best_iteration_from_model(model),
                "test_r2_on_shap_subset": r2_np(y_test, pred_test),
                "mean_base_value": float(np.mean(base_value)),
                "max_abs_pred_minus_shap_sum": float(np.max(np.abs(pred_test - contribs.sum(axis=1)))),
            }
        )
        shap_df = shap_feature_summary(shap_values, X_test, feature_names, int(fold))
        per_fold_shap.append(shap_df)

        candidate_features = (
            shap_df.sort_values("mean_abs_shap", ascending=False)
            .head(PLOT_FEATURE_CANDIDATE_LIMIT_PER_FOLD)["feature"]
            .astype(str)
            .tolist()
        )
        for feat in resolved_targets["resolved_feature"].astype(str):
            if feat and feat not in candidate_features:
                candidate_features.append(feat)
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        fold_plot_rows = []
        for feature in candidate_features:
            idx = name_to_idx.get(feature)
            if idx is None:
                continue
            values = X_test[:, idx].astype(np.float64)
            ranks = pd.Series(values).rank(pct=True).fillna(0.5).to_numpy(np.float64)
            fold_plot_rows.append(
                pd.DataFrame(
                    {
                        "cv_fold": fold,
                        "sample_id": test_ids,
                        "feature": feature,
                        "family": feature_family(feature),
                        "feature_value": values,
                        "feature_value_rank": ranks,
                        "shap_value": shap_values[:, idx].astype(np.float64),
                    }
                )
            )
        if fold_plot_rows:
            plot_rows.append(pd.concat(fold_plot_rows, ignore_index=True))

        if RUN_PERMUTATION_IMPORTANCE:
            fold_top = shap_df.sort_values("mean_abs_shap", ascending=False).head(PERMUTATION_IMPORTANCE_TOP_N)["feature"].astype(str).tolist()
            for feat in resolved_targets["resolved_feature"].astype(str):
                if feat and feat not in fold_top:
                    fold_top.append(feat)
            perm_rows.append(permutation_importance(model, X_test, y_test, feature_names, fold_top, int(fold)))

        pd.concat(per_fold_shap, ignore_index=True).to_csv(
            OUT_DIR / "shap_per_fold_feature_summary.partial.csv",
            index=False,
        )
        pd.concat(per_fold_gain, ignore_index=True).to_csv(
            OUT_DIR / "xgb_gain_weight_importance_by_fold.partial.csv",
            index=False,
        )
        if RUN_PERMUTATION_IMPORTANCE and perm_rows:
            pd.concat(perm_rows, ignore_index=True).to_csv(
                OUT_DIR / "permutation_importance.partial.csv",
                index=False,
            )

        del model, X_test, y_test, pred_test, contribs, shap_values
        gc.collect()

    fold_diagnostics = pd.DataFrame(fold_rows)
    fold_diagnostics.to_csv(OUT_DIR / "shap_fold_diagnostics.csv", index=False)
    per_fold_shap_df = pd.concat(per_fold_shap, ignore_index=True)
    per_fold_shap_df.to_csv(OUT_DIR / "shap_per_fold_feature_summary.csv", index=False)
    gain_all = pd.concat(per_fold_gain, ignore_index=True)
    gain_all.to_csv(OUT_DIR / "xgb_gain_weight_importance_by_fold.csv", index=False)
    shap_top = aggregate_shap_summary(per_fold_shap_df, gain_all)
    shap_top.to_csv(OUT_DIR / "shap_top_features.csv", index=False)
    family = summarize_families(shap_top, gain_all)
    family.to_csv(OUT_DIR / "shap_family_summary.csv", index=False)
    direction = direction_table(shap_top, resolved_targets, per_fold_shap_df)
    direction.to_csv(OUT_DIR / "shap_direction_biological_interpretation.csv", index=False)

    if perm_rows:
        perm = pd.concat(perm_rows, ignore_index=True)
        perm.to_csv(OUT_DIR / "permutation_importance.csv", index=False)
        perm_summary = (
            perm.groupby(["feature", "family"], as_index=False)
            .agg(mean_r2_drop=("r2_drop", "mean"), std_r2_drop=("r2_drop", "std"), n=("r2_drop", "count"))
            .sort_values("mean_r2_drop", ascending=False)
        )
        perm_summary.to_csv(OUT_DIR / "permutation_importance_summary.csv", index=False)

    sample_plot = pd.concat(plot_rows, ignore_index=True) if plot_rows else pd.DataFrame()
    if not sample_plot.empty:
        sample_plot.to_csv(OUT_DIR / "shap_plot_sample_values.csv.gz", index=False, compression="gzip")
    save_bar_plot(shap_top, OUT_DIR)
    save_beeswarm_like(sample_plot, shap_top, OUT_DIR)
    save_dependence_plots(sample_plot, shap_top, resolved_targets, OUT_DIR)
    write_report(OUT_DIR, shap_top, family, direction)
    print(f"[DONE] outputs written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
