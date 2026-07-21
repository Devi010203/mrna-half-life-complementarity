#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No-args XGB feature-family ablation for Saluki regulatory features."""

from __future__ import annotations

import gc
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


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

NORMALIZED_PATH = env_path("SALUKI_ABLATION_NORMALIZED_PATH", os.environ.get("SALUKI_NORMALIZED_PATH", PROJECT_ROOT / "data/processed/human/normalized.csv.gz"))
SPLIT_PATH = env_path("SALUKI_ABLATION_SPLIT_PATH", os.environ.get("SALUKI_SPLIT_PATH", PROJECT_ROOT / "data/processed/human/splits/cv_k5_seed42.csv"))
FEATURE_ROOT = env_path("SALUKI_ABLATION_FEATURE_ROOT", os.environ.get("SALUKI_FEATURE_ROOT", PROJECT_ROOT / "data/saluki_features/human"))
GENETIC_BIOCHEMICAL_PATH = env_path(
    "SALUKI_ABLATION_GENETIC_BIOCHEMICAL_PATH",
    os.environ.get("SALUKI_GENETIC_BIOCHEMICAL_PATH", FEATURE_ROOT / "alldataGeneticBiochemical.tsv.gz"),
)
BIOCHEMICAL_FULL_PATH = env_path(
    "SALUKI_ABLATION_BIOCHEMICAL_FULL_PATH",
    os.environ.get("SALUKI_BIOCHEMICAL_FULL_PATH", FEATURE_ROOT / "alldataBiochemical.tsv.gz"),
)

OUT_BASE_DIR = env_path("SALUKI_ABLATION_OUT_BASE_DIR", RUNS_ROOT)
RUN_TAG = env_str("SALUKI_ABLATION_RUN_TAG", "feature_family_ablation")
OUT_DIR = OUT_BASE_DIR / RUN_TAG

CV_FOLDS = [0, 1, 2, 3, 4]
SEED = 42
INNER_VAL_RATIO = 0.10
STRATIFY_BINS = 20
EXPECTED_OVERLAP = 12981
ACTIVE_FEATURE_SETS = [
    "full_basic_plus_biochemical_full",
    "remove_basic",
    "remove_rip",
    "remove_encori",
    "remove_eclip",
    "remove_rip_encori",
    "only_basic",
    "only_rip",
    "only_encori",
    "only_eclip",
    "only_regulatory_no_basic",
]

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
        "cv_folds": list(CV_FOLDS),
        "seed": SEED,
        "inner_val_ratio": INNER_VAL_RATIO,
        "stratify_bins": STRATIFY_BINS,
        "expected_overlap": EXPECTED_OVERLAP,
        "active_feature_sets": list(ACTIVE_FEATURE_SETS),
        "xgb_params": dict(XGB_PARAMS, n_jobs=N_JOBS),
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
    adjusted = ranked * float(len(ranked)) / np.arange(1, len(ranked) + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(pv)
    out[order] = adjusted
    q[valid] = out
    return q


def finite_corr(x: np.ndarray, y: np.ndarray, ranked: bool = False) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if ranked:
        x = pd.Series(x).rank(method="average").to_numpy(np.float64)
        y = pd.Series(y).rank(method="average").to_numpy(np.float64)
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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": r2_np(y_true, y_pred),
        "pearson": finite_corr(y_true, y_pred),
        "spearman": finite_corr(y_true, y_pred, ranked=True),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def load_normalized() -> pd.DataFrame:
    df = pd.read_csv(NORMALIZED_PATH, compression="gzip", usecols=["ENSID", "HALFLIFE"])
    df["ENSID"] = df["ENSID"].astype(str)
    df["HALFLIFE"] = pd.to_numeric(df["HALFLIFE"], errors="coerce")
    df = df.dropna(subset=["HALFLIFE"]).drop_duplicates("ENSID").reset_index(drop=True)
    return df


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


def bio_family(col: str) -> str:
    return str(col).split(".", 1)[0]


def build_feature_sets(gb_cols: list[str], bio_cols: list[str]) -> dict[str, dict[str, Any]]:
    forbidden = {"GENEID", "ENSID", "HALFLIFE"}
    gb_features = [c for c in gb_cols if c not in forbidden]
    bio_features = [c for c in bio_cols if c not in forbidden]
    basic = [c for c in gb_features if c.startswith("BASIC.")]
    bio_prefixes = ("RIP.", "TE.", "M6ACLIP.", "ENCORI.", "eCLIP.", "PARCLIP.")
    bio_full = [c for c in bio_features if any(c.startswith(prefix) for prefix in bio_prefixes)]
    families = {family: [c for c in bio_full if c.startswith(f"{family}.")] for family in ["RIP", "TE", "M6ACLIP", "ENCORI", "eCLIP", "PARCLIP"]}

    def spec(key: str, components: list[tuple[str, list[str]]], description: str) -> dict[str, Any]:
        names = [col for _, cols in components for col in cols]
        bad = sorted(set(names).intersection(forbidden))
        if bad:
            raise ValueError(f"{key} contains forbidden columns: {bad}")
        return {"key": key, "description": description, "components": components, "feature_names": names, "n_features": len(names)}

    without = lambda fams: [c for c in bio_full if bio_family(c) not in set(fams)]
    sets = {
        "full_basic_plus_biochemical_full": spec("full_basic_plus_biochemical_full", [("genetic_biochemical", basic), ("biochemical", bio_full)], "BASIC plus full regulatory/biochemical annotations"),
        "remove_basic": spec("remove_basic", [("biochemical", bio_full)], "Full regulatory annotations without BASIC"),
        "remove_rip": spec("remove_rip", [("genetic_biochemical", basic), ("biochemical", without(["RIP"]))], "Full minus RIP"),
        "remove_encori": spec("remove_encori", [("genetic_biochemical", basic), ("biochemical", without(["ENCORI"]))], "Full minus ENCORI"),
        "remove_eclip": spec("remove_eclip", [("genetic_biochemical", basic), ("biochemical", without(["eCLIP"]))], "Full minus eCLIP"),
        "remove_rip_encori": spec("remove_rip_encori", [("genetic_biochemical", basic), ("biochemical", without(["RIP", "ENCORI"]))], "Full minus RIP and ENCORI"),
        "only_basic": spec("only_basic", [("genetic_biochemical", basic)], "Only BASIC transcript architecture features"),
        "only_rip": spec("only_rip", [("biochemical", families["RIP"])], "Only RIP annotations"),
        "only_encori": spec("only_encori", [("biochemical", families["ENCORI"])], "Only ENCORI annotations"),
        "only_eclip": spec("only_eclip", [("biochemical", families["eCLIP"])], "Only eCLIP annotations"),
        "only_regulatory_no_basic": spec("only_regulatory_no_basic", [("biochemical", bio_full)], "All regulatory annotations without BASIC"),
    }
    return sets


def collect_required_columns(feature_sets: list[dict[str, Any]]) -> dict[str, list[str]]:
    required = {"genetic_biochemical": [], "biochemical": []}
    for feature_set in feature_sets:
        for source, cols in feature_set["components"]:
            for col in cols:
                if col not in required[source]:
                    required[source].append(col)
    return required


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


def materialize_feature_set(feature_set: dict[str, Any], sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [sources[source].loc[:, cols] for source, cols in feature_set["components"] if cols]
    if not frames:
        raise ValueError(f"Feature set {feature_set['key']} has no columns")
    return pd.concat(frames, axis=1, copy=False) if len(frames) > 1 else frames[0]


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


def fit_eval_xgb(X: pd.DataFrame, y: pd.Series, split_ids: dict[str, list[str]], fold: int) -> dict[str, Any]:
    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]
    X_train = X.loc[train_ids].to_numpy(dtype=np.float32, copy=True)
    X_val = X.loc[val_ids].to_numpy(dtype=np.float32, copy=True)
    X_test = X.loc[test_ids].to_numpy(dtype=np.float32, copy=True)
    y_train = y.loc[train_ids].to_numpy(np.float64)
    y_val = y.loc[val_ids].to_numpy(np.float64)
    y_test = y.loc[test_ids].to_numpy(np.float64)
    model = XGBRegressor(**dict(XGB_PARAMS, n_jobs=N_JOBS, random_state=SEED + fold, verbosity=1))
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_train = predict_best(model, X_train)
    pred_val = predict_best(model, X_val)
    pred_test = predict_best(model, X_test)
    return {
        "best_iteration": best_iteration_from_model(model),
        "best_score": None if getattr(model, "best_score", None) is None else float(model.best_score),
        "metrics": {
            "train": regression_metrics(y_train, pred_train),
            "val": regression_metrics(y_val, pred_val),
            "test": regression_metrics(y_test, pred_test),
        },
        "predictions": {
            "val": pd.DataFrame({"ENSID": val_ids, "y_true": y_val, "y_pred": pred_val}),
            "test": pd.DataFrame({"ENSID": test_ids, "y_true": y_test, "y_pred": pred_test}),
        },
    }


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in fold_metrics.columns if col.startswith("test_") and pd.api.types.is_numeric_dtype(fold_metrics[col])]
    rows: list[dict[str, Any]] = []
    for feature_set, sub in fold_metrics.groupby("feature_set", sort=False):
        row: dict[str, Any] = {"feature_set": feature_set, "n_folds": int(sub["cv_fold"].nunique())}
        for col in metric_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=0)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_delta_vs_full(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    base = fold_metrics[fold_metrics["feature_set"] == "full_basic_plus_biochemical_full"].set_index("cv_fold")
    rows: list[dict[str, Any]] = []
    for _, row in fold_metrics.iterrows():
        if row["feature_set"] == "full_basic_plus_biochemical_full":
            continue
        fold = int(row["cv_fold"])
        if fold not in base.index:
            continue
        base_row = base.loc[fold]
        out = {
            "cv_fold": fold,
            "feature_set": row["feature_set"],
            "comparison": f"{row['feature_set']} - full_basic_plus_biochemical_full",
            "full_test_r2": float(base_row["test_r2"]),
            "comparison_test_r2": float(row["test_r2"]),
            "delta_test_r2": float(row["test_r2"] - base_row["test_r2"]),
            "full_n_features": int(base_row["n_features"]),
            "comparison_n_features": int(row["n_features"]),
        }
        rows.append(out)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, delta: pd.DataFrame) -> None:
    lines = [
        "# XGB Feature-family Ablation Report",
        "",
        "This analysis retrains XGB after removing or isolating feature families. Test folds are evaluated once.",
        "",
        "## Summary",
        "",
    ]
    for row in summary.sort_values("test_r2_mean", ascending=False).itertuples(index=False):
        lines.append(f"- `{row.feature_set}`: test R2 = {row.test_r2_mean:.4f} +/- {row.test_r2_std:.4f}")
    if not delta.empty:
        lines.extend(["", "## Delta vs Full", ""])
        dsum = delta.groupby("feature_set", as_index=False).agg(delta_test_r2_mean=("delta_test_r2", "mean"), delta_test_r2_std=("delta_test_r2", "std"))
        for row in dsum.sort_values("delta_test_r2_mean").itertuples(index=False):
            lines.append(f"- `{row.feature_set}`: delta test R2 = {row.delta_test_r2_mean:.4f} +/- {row.delta_test_r2_std:.4f}")
    (OUT_DIR / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_partial(
    fold_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    pred_val_rows: list[pd.DataFrame],
    pred_test_rows: list[pd.DataFrame],
) -> None:
    if fold_rows:
        pd.DataFrame(fold_rows).to_csv(OUT_DIR / "ablation_fold_metrics.partial.csv", index=False)
    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(OUT_DIR / "selected_best_iteration_per_fold.partial.csv", index=False)
    if pred_val_rows:
        pd.concat(pred_val_rows, ignore_index=True).to_csv(OUT_DIR / "pred_val.partial.csv", index=False)
    if pred_test_rows:
        pd.concat(pred_test_rows, ignore_index=True).to_csv(OUT_DIR / "pred_test.partial.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[CONFIG]", flush=True)
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False), flush=True)
    validate_config_paths()
    write_json(OUT_DIR / "run_config.json", config_dict())

    norm = load_normalized()
    cv = load_split()
    feature_sets = build_feature_sets(read_columns(GENETIC_BIOCHEMICAL_PATH), read_columns(BIOCHEMICAL_FULL_PATH))
    unknown = [key for key in ACTIVE_FEATURE_SETS if key not in feature_sets]
    if unknown:
        raise ValueError(f"Unknown ACTIVE_FEATURE_SETS: {unknown}. Available={sorted(feature_sets)}")
    active_specs = [feature_sets[key] for key in ACTIVE_FEATURE_SETS]
    required = collect_required_columns(active_specs)
    print(f"[LOAD] loading required columns: genetic_biochemical={len(required['genetic_biochemical'])} biochemical={len(required['biochemical'])}", flush=True)
    gb_y, gb_features = load_feature_source(GENETIC_BIOCHEMICAL_PATH, required["genetic_biochemical"])
    bio_y, bio_features = load_feature_source(BIOCHEMICAL_FULL_PATH, required["biochemical"])
    overlap = validate_alignment(norm, gb_y, bio_y)
    y = norm.set_index("ENSID")["HALFLIFE"].loc[overlap].astype(np.float64)
    dfi = norm[norm["ENSID"].isin(overlap)].reset_index(drop=True)
    sources = {"genetic_biochemical": gb_features.loc[overlap], "biochemical": bio_features.loc[overlap]}

    manifest = {
        key: {
            "description": value["description"],
            "n_features": value["n_features"],
            "components": [{"source": source, "n_columns": len(cols), "columns": cols} for source, cols in value["components"]],
        }
        for key, value in feature_sets.items()
    }
    write_json(OUT_DIR / "feature_set_manifest.json", manifest)

    fold_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    pred_val_rows: list[pd.DataFrame] = []
    pred_test_rows: list[pd.DataFrame] = []

    for key in ACTIVE_FEATURE_SETS:
        spec = feature_sets[key]
        print(f"[FEATURE_SET] {key}: n_features={spec['n_features']}", flush=True)
        X = materialize_feature_set(spec, sources).loc[overlap]
        for fold in CV_FOLDS:
            print(f"[FOLD] feature_set={key} fold={fold}", flush=True)
            split_ids = split_from_fold(dfi, cv, int(fold))
            result = fit_eval_xgb(X, y, split_ids, int(fold))
            row: dict[str, Any] = {
                "feature_set": key,
                "description": spec["description"],
                "cv_fold": int(fold),
                "n_features": int(spec["n_features"]),
                "n_train": len(split_ids["train"]),
                "n_val": len(split_ids["val"]),
                "n_test": len(split_ids["test"]),
                "best_iteration": int(result["best_iteration"]),
                "best_score": result["best_score"],
            }
            row.update(prefixed(result["metrics"]["train"], "train"))
            row.update(prefixed(result["metrics"]["val"], "val"))
            row.update(prefixed(result["metrics"]["test"], "test"))
            fold_rows.append(row)
            selected_rows.append(
                {
                    "feature_set": key,
                    "cv_fold": int(fold),
                    "best_iteration": int(result["best_iteration"]),
                    "best_score": result["best_score"],
                    "n_features": int(spec["n_features"]),
                }
            )
            for split_name, collector in (("val", pred_val_rows), ("test", pred_test_rows)):
                pred = result["predictions"][split_name].copy()
                pred.insert(0, "feature_set", key)
                pred.insert(1, "cv_fold", int(fold))
                pred.insert(2, "n_features", int(spec["n_features"]))
                collector.append(pred)
            write_partial(fold_rows, selected_rows, pred_val_rows, pred_test_rows)
            gc.collect()
        del X
        gc.collect()

    fold_metrics = pd.DataFrame(fold_rows)
    summary = summarize_fold_metrics(fold_metrics)
    delta = paired_delta_vs_full(fold_metrics)
    fold_metrics.to_csv(OUT_DIR / "ablation_fold_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "ablation_summary_metrics.csv", index=False)
    delta.to_csv(OUT_DIR / "paired_delta_vs_full.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT_DIR / "selected_best_iteration_per_fold.csv", index=False)
    pd.concat(pred_val_rows, ignore_index=True).to_csv(OUT_DIR / "pred_val.csv", index=False)
    pd.concat(pred_test_rows, ignore_index=True).to_csv(OUT_DIR / "pred_test.csv", index=False)
    write_report(summary, delta)
    print(f"[DONE] outputs written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
