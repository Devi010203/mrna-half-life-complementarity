#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No-args SHAP profile comparison for fusion-helped vs fusion-hurt samples."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG: paths are resolved from environment first. Point these to clean
# fusion-gain and SHAP runs.
# =============================================================================
def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else Path(default)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


PROJECT_ROOT = env_path("SALUKI_PROJECT_ROOT", Path(__file__).resolve().parents[2])
RUNS_ROOT = env_path("SALUKI_RUNS_ROOT", PROJECT_ROOT / "results" / "runs")

FUSION_PROFILE_ROOT = env_path(
    "SALUKI_HELPED_HURT_FUSION_PROFILE_ROOT",
    RUNS_ROOT / "fusion_gain",
)
FUSION_PROFILE_PATH = env_path("SALUKI_HELPED_HURT_FUSION_PROFILE_PATH", FUSION_PROFILE_ROOT / "fusion_gain_sample_profile.csv")

SHAP_RUN_ROOT = env_path(
    "SALUKI_HELPED_HURT_SHAP_RUN_ROOT",
    RUNS_ROOT / "shap_interpretability",
)
SHAP_PLOT_SAMPLE_PATH = SHAP_RUN_ROOT / "shap_plot_sample_values.csv.gz"
SHAP_TOP_FEATURES_PATH = SHAP_RUN_ROOT / "shap_top_features.csv"
SHAP_FAMILY_SUMMARY_PATH = SHAP_RUN_ROOT / "shap_family_summary.csv"

OUT_BASE_DIR = env_path("SALUKI_HELPED_HURT_OUT_BASE_DIR", RUNS_ROOT)
RUN_TAG = env_str("SALUKI_HELPED_HURT_RUN_TAG", "helped_hurt_shap")
OUT_DIR = OUT_BASE_DIR / RUN_TAG

PRIMARY_GROUP_COL = "fusion_gain_group_within_fold"
GLOBAL_GROUP_COL = "fusion_gain_group"
HELPED_LABEL = "fusion_helped_top20"
HURT_LABEL = "fusion_hurt_top20"
TOP_N_DRIVERS = 50


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def config_dict() -> dict[str, Any]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "timestamp": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "fusion_profile_root": str(FUSION_PROFILE_ROOT),
        "fusion_profile_path": str(FUSION_PROFILE_PATH),
        "shap_run_root": str(SHAP_RUN_ROOT),
        "shap_plot_sample_path": str(SHAP_PLOT_SAMPLE_PATH),
        "shap_top_features_path": str(SHAP_TOP_FEATURES_PATH),
        "shap_family_summary_path": str(SHAP_FAMILY_SUMMARY_PATH),
        "out_base_dir": str(OUT_BASE_DIR),
        "run_tag": RUN_TAG,
        "out_dir": str(OUT_DIR),
        "primary_group_col": PRIMARY_GROUP_COL,
        "global_group_col": GLOBAL_GROUP_COL,
        "helped_label": HELPED_LABEL,
        "hurt_label": HURT_LABEL,
        "top_n_drivers": TOP_N_DRIVERS,
    }


def print_config_and_raise(message: str) -> None:
    print("[CONFIG]", flush=True)
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False), flush=True)
    raise FileNotFoundError(message)


def validate_config_paths() -> None:
    required = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "FUSION_PROFILE_PATH": FUSION_PROFILE_PATH,
        "SHAP_PLOT_SAMPLE_PATH": SHAP_PLOT_SAMPLE_PATH,
        "SHAP_TOP_FEATURES_PATH": SHAP_TOP_FEATURES_PATH,
        "SHAP_FAMILY_SUMMARY_PATH": SHAP_FAMILY_SUMMARY_PATH,
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


def safe_mannwhitneyu(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    try:
        from scipy.stats import mannwhitneyu  # type: ignore

        return float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def cliffs_delta(x: np.ndarray, y: np.ndarray, max_pairs: int = 2_000_000) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if len(x) * len(y) > max_pairs:
        rng = np.random.default_rng(42)
        n = int(math.sqrt(max_pairs))
        x = rng.choice(x, size=n, replace=True)
        y = rng.choice(y, size=n, replace=True)
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


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


def compare_values(values: pd.Series, groups: pd.Series) -> dict[str, float | int]:
    helped = pd.to_numeric(values[groups == HELPED_LABEL], errors="coerce").to_numpy(np.float64)
    hurt = pd.to_numeric(values[groups == HURT_LABEL], errors="coerce").to_numpy(np.float64)
    return {
        "helped_mean": float(np.nanmean(helped)),
        "hurt_mean": float(np.nanmean(hurt)),
        "helped_median": float(np.nanmedian(helped)),
        "hurt_median": float(np.nanmedian(hurt)),
        "mean_difference_helped_minus_hurt": float(np.nanmean(helped) - np.nanmean(hurt)),
        "median_difference_helped_minus_hurt": float(np.nanmedian(helped) - np.nanmedian(hurt)),
        "mannwhitney_p": safe_mannwhitneyu(helped, hurt),
        "cliffs_delta_helped_vs_hurt": cliffs_delta(helped, hurt),
        "n_helped": int(np.isfinite(helped).sum()),
        "n_hurt": int(np.isfinite(hurt).sum()),
    }


def feature_stats(merged: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = merged[group_col].astype(str)
    for (feature, family), sub in merged.groupby(["feature", "family"], sort=False):
        row: dict[str, Any] = {"feature": feature, "family": family, "group_col": group_col}
        row.update(compare_values(sub["shap_value"], groups.loc[sub.index]))
        row["helped_mean_abs_shap"] = float(np.nanmean(np.abs(sub.loc[groups.loc[sub.index] == HELPED_LABEL, "shap_value"])))
        row["hurt_mean_abs_shap"] = float(np.nanmean(np.abs(sub.loc[groups.loc[sub.index] == HURT_LABEL, "shap_value"])))
        row["helped_positive_fraction"] = float(np.nanmean(sub.loc[groups.loc[sub.index] == HELPED_LABEL, "shap_value"] > 0))
        row["hurt_positive_fraction"] = float(np.nanmean(sub.loc[groups.loc[sub.index] == HURT_LABEL, "shap_value"] > 0))
        row["spearman_shap_vs_fusion_gain"] = finite_corr(sub["shap_value"].to_numpy(), sub["fusion_gain"].to_numpy(), ranked=True)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value_bh_fdr"] = bh_fdr(out["mannwhitney_p"])
    return out.sort_values("q_value_bh_fdr")


def family_stats(merged: pd.DataFrame, group_col: str) -> pd.DataFrame:
    sample_family = (
        merged.groupby(["sample_id", "cv_fold", "family"], as_index=False)
        .agg(
            family_sum_shap=("shap_value", "sum"),
            family_sum_abs_shap=("shap_value", lambda x: float(np.sum(np.abs(x)))),
            n_features_available=("feature", "nunique"),
            fusion_gain=( "fusion_gain", "first"),
            group=(group_col, "first"),
        )
        .rename(columns={"group": group_col})
    )
    rows: list[dict[str, Any]] = []
    groups = sample_family[group_col].astype(str)
    for family, sub in sample_family.groupby("family", sort=False):
        row: dict[str, Any] = {"family": family, "group_col": group_col}
        row.update(compare_values(sub["family_sum_shap"], groups.loc[sub.index]))
        row["helped_mean_abs_family_shap"] = float(np.nanmean(sub.loc[groups.loc[sub.index] == HELPED_LABEL, "family_sum_abs_shap"]))
        row["hurt_mean_abs_family_shap"] = float(np.nanmean(sub.loc[groups.loc[sub.index] == HURT_LABEL, "family_sum_abs_shap"]))
        row["spearman_family_shap_vs_fusion_gain"] = finite_corr(sub["family_sum_shap"].to_numpy(), sub["fusion_gain"].to_numpy(), ranked=True)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value_bh_fdr"] = bh_fdr(out["mannwhitney_p"])
    return out.sort_values("q_value_bh_fdr")


def helped_drivers(merged: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    helped = merged[merged[group_col].astype(str) == HELPED_LABEL]
    agg = (
        helped.groupby(["feature", "family"], as_index=False)
        .agg(
            helped_mean_shap=("shap_value", "mean"),
            helped_median_shap=("shap_value", "median"),
            helped_mean_abs_shap=("shap_value", lambda x: float(np.mean(np.abs(x)))),
            helped_positive_fraction=("shap_value", lambda x: float(np.mean(np.asarray(x) > 0))),
            n_helped=("shap_value", "count"),
        )
    )
    longer = agg[agg["helped_mean_shap"] > 0].sort_values("helped_mean_shap", ascending=False).head(TOP_N_DRIVERS)
    shorter = agg[agg["helped_mean_shap"] < 0].sort_values("helped_mean_shap", ascending=True).head(TOP_N_DRIVERS)
    return longer, shorter


def try_plot(feature_within: pd.DataFrame, family_within: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping optional plots: {exc}", flush=True)
        return
    for name, data, label_col in (
        ("shap_feature_helped_vs_hurt_top_differences", feature_within, "feature"),
        ("shap_family_helped_vs_hurt_differences", family_within, "family"),
    ):
        tmp = data.copy()
        tmp["abs_delta"] = tmp["mean_difference_helped_minus_hurt"].abs()
        tmp = tmp.sort_values("abs_delta", ascending=False).head(25).iloc[::-1]
        if tmp.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, max(4, 0.34 * len(tmp))))
        ax.barh(tmp[label_col], tmp["mean_difference_helped_minus_hurt"], color="#8a6fb0")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean SHAP difference: helped - hurt")
        ax.set_title("SHAP profile difference for fusion-helped vs fusion-hurt samples")
        for ext in ("png", "pdf"):
            fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=240, bbox_inches="tight")
        plt.close(fig)


def write_report(feature_within: pd.DataFrame, family_within: pd.DataFrame, longer: pd.DataFrame, shorter: pd.DataFrame) -> None:
    lines = [
        "# SHAP Profile: Fusion Helped vs Hurt",
        "",
        "Primary grouping uses within-fold top/bottom 20% fusion gain. Per-sample SHAP rows are the candidate features saved by script 20, not all 559 features.",
        "",
        "## Feature-level Differences",
        "",
    ]
    for row in feature_within.sort_values("q_value_bh_fdr").head(12).itertuples(index=False):
        lines.append(
            f"- `{row.feature}` ({row.family}): helped-hurt mean SHAP = "
            f"{row.mean_difference_helped_minus_hurt:.4f}, q = {row.q_value_bh_fdr:.3g}"
        )
    lines.extend(["", "## Family-level Differences", ""])
    for row in family_within.sort_values("q_value_bh_fdr").head(8).itertuples(index=False):
        lines.append(
            f"- `{row.family}`: helped-hurt mean family SHAP = "
            f"{row.mean_difference_helped_minus_hurt:.4f}, q = {row.q_value_bh_fdr:.3g}"
        )
    lines.extend(["", "## Helped Samples: Longer-prediction Drivers", ""])
    for row in longer.head(10).itertuples(index=False):
        lines.append(f"- `{row.feature}`: mean SHAP in helped = {row.helped_mean_shap:.4f}")
    lines.extend(["", "## Helped Samples: Shorter-prediction Drivers", ""])
    for row in shorter.head(10).itertuples(index=False):
        lines.append(f"- `{row.feature}`: mean SHAP in helped = {row.helped_mean_shap:.4f}")
    (OUT_DIR / "shap_profile_helped_vs_hurt_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[CONFIG]", flush=True)
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False), flush=True)
    validate_config_paths()
    write_json(OUT_DIR / "run_config.json", config_dict())

    profile_cols = ["sample_id", "cv_fold", "fusion_gain", GLOBAL_GROUP_COL, PRIMARY_GROUP_COL, "y_true", "xgb_pred", "fusion_pred"]
    profile = pd.read_csv(FUSION_PROFILE_PATH, usecols=lambda c: c in profile_cols)
    shap = pd.read_csv(SHAP_PLOT_SAMPLE_PATH)
    merged = shap.merge(profile, on=["sample_id", "cv_fold"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping rows between SHAP sample values and fusion profile")

    shap_top = pd.read_csv(SHAP_TOP_FEATURES_PATH)
    feature_rank = shap_top[["feature", "rank_mean_abs_shap", "mean_abs_shap"]].copy()
    merged = merged.merge(feature_rank, on="feature", how="left")

    feature_within = feature_stats(merged, PRIMARY_GROUP_COL)
    feature_global = feature_stats(merged, GLOBAL_GROUP_COL)
    family_within = family_stats(merged, PRIMARY_GROUP_COL)
    family_global = family_stats(merged, GLOBAL_GROUP_COL)
    longer, shorter = helped_drivers(merged, PRIMARY_GROUP_COL)

    feature_within.to_csv(OUT_DIR / "shap_feature_helped_vs_hurt_within_fold.csv", index=False)
    feature_global.to_csv(OUT_DIR / "shap_feature_helped_vs_hurt_global.csv", index=False)
    family_within.to_csv(OUT_DIR / "shap_family_helped_vs_hurt_within_fold.csv", index=False)
    family_global.to_csv(OUT_DIR / "shap_family_helped_vs_hurt_global.csv", index=False)
    longer.to_csv(OUT_DIR / "shap_longer_prediction_drivers_in_helped.csv", index=False)
    shorter.to_csv(OUT_DIR / "shap_shorter_prediction_drivers_in_helped.csv", index=False)
    try_plot(feature_within, family_within)
    write_report(feature_within, family_within, longer, shorter)
    print(f"[DONE] outputs written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
