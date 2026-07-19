#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No-args fusion case-study table for top helped and hurt transcripts."""

from __future__ import annotations

import json
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
    "SALUKI_CASE_FUSION_PROFILE_ROOT",
    RUNS_ROOT / "fusion_gain",
)
FUSION_PROFILE_PATH = env_path("SALUKI_CASE_FUSION_PROFILE_PATH", FUSION_PROFILE_ROOT / "fusion_gain_sample_profile.csv")

SHAP_RUN_ROOT = env_path(
    "SALUKI_CASE_SHAP_RUN_ROOT",
    RUNS_ROOT / "shap_interpretability",
)
SHAP_PLOT_SAMPLE_PATH = SHAP_RUN_ROOT / "shap_plot_sample_values.csv.gz"

BASIC_FEATURE_PATH = env_path(
    "SALUKI_CASE_BASIC_FEATURE_PATH",
    os.environ.get(
        "SALUKI_GENETIC_BIOCHEMICAL_PATH",
        PROJECT_ROOT / "data" / "saluki_features" / "human" / "alldataGeneticBiochemical.tsv.gz",
    ),
)

OUT_BASE_DIR = env_path("SALUKI_CASE_OUT_BASE_DIR", RUNS_ROOT)
RUN_TAG = env_str("SALUKI_CASE_RUN_TAG", "case_studies")
OUT_DIR = OUT_BASE_DIR / RUN_TAG

TOP_N = 30
BASIC_FEATURES = [
    "BASIC.UTR5LEN",
    "BASIC.CDSLEN",
    "BASIC.INTRONLEN",
    "BASIC.UTR3LEN",
    "BASIC.UTR5GC",
    "BASIC.CDSGC",
    "BASIC.UTR3GC",
    "BASIC.ORFEXONDENSITY",
]
TOP_SHAP_PER_SAMPLE = 8


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
        "basic_feature_path": str(BASIC_FEATURE_PATH),
        "out_base_dir": str(OUT_BASE_DIR),
        "run_tag": RUN_TAG,
        "out_dir": str(OUT_DIR),
        "top_n": TOP_N,
        "basic_features": list(BASIC_FEATURES),
        "top_shap_per_sample": TOP_SHAP_PER_SAMPLE,
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
        "BASIC_FEATURE_PATH": BASIC_FEATURE_PATH,
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


def load_basic_features(ids: list[str]) -> pd.DataFrame:
    df = pd.read_csv(
        BASIC_FEATURE_PATH,
        sep="\t",
        compression="infer",
        usecols=["GENEID", *BASIC_FEATURES],
        dtype={"GENEID": "string"},
        low_memory=False,
    )
    df["sample_id"] = df["GENEID"].astype(str)
    df = df.drop(columns=["GENEID"])
    return df[df["sample_id"].isin(ids)].copy()


def format_feature_values(rows: pd.DataFrame, sign: str) -> str:
    if sign == "positive":
        data = rows[rows["shap_value"] > 0].sort_values("shap_value", ascending=False).head(TOP_SHAP_PER_SAMPLE)
    else:
        data = rows[rows["shap_value"] < 0].sort_values("shap_value", ascending=True).head(TOP_SHAP_PER_SAMPLE)
    return "; ".join(f"{r.feature}:{r.shap_value:.4g}" for r in data.itertuples(index=False))


def format_family_summary(rows: pd.DataFrame) -> str:
    fam = rows.groupby("family", as_index=False).agg(sum_shap=("shap_value", "sum"), sum_abs_shap=("shap_value", lambda x: float(np.sum(np.abs(x)))))
    fam = fam.sort_values("sum_abs_shap", ascending=False).head(TOP_SHAP_PER_SAMPLE)
    return "; ".join(f"{r.family}:sum={r.sum_shap:.4g},abs={r.sum_abs_shap:.4g}" for r in fam.itertuples(index=False))


def build_shap_summary(shap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (sample_id, cv_fold), sub in shap.groupby(["sample_id", "cv_fold"], sort=False):
        rows.append(
            {
                "sample_id": sample_id,
                "cv_fold": int(cv_fold),
                "top_positive_shap_features": format_feature_values(sub, "positive"),
                "top_negative_shap_features": format_feature_values(sub, "negative"),
                "shap_family_summary_available_features": format_family_summary(sub),
                "n_available_shap_features": int(sub["feature"].nunique()),
                "sum_available_shap": float(sub["shap_value"].sum()),
                "sum_abs_available_shap": float(np.abs(sub["shap_value"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def select_cases(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    helped = profile.sort_values("fusion_gain", ascending=False).head(TOP_N).copy()
    hurt = profile.sort_values("fusion_gain", ascending=True).head(TOP_N).copy()
    helped.insert(0, "case_group", "top_helped")
    hurt.insert(0, "case_group", "top_hurt")
    return helped, hurt


def write_report(helped: pd.DataFrame, hurt: pd.DataFrame) -> None:
    lines = [
        "# Fusion Case-study Table Report",
        "",
        "Case studies are selected by `fusion_gain = abs_error_xgb - abs_error_fusion`.",
        "Per-sample SHAP summaries use the candidate SHAP rows saved by script 20 and do not cover all 559 features.",
        "",
        "## Top Helped",
        "",
    ]
    for row in helped.head(10).itertuples(index=False):
        lines.append(f"- `{row.sample_id}` fold={row.cv_fold}: fusion_gain={row.fusion_gain:.4f}, y={row.y_true:.4f}, xgb={row.xgb_pred:.4f}, fusion={row.fusion_pred:.4f}")
    lines.extend(["", "## Top Hurt", ""])
    for row in hurt.head(10).itertuples(index=False):
        lines.append(f"- `{row.sample_id}` fold={row.cv_fold}: fusion_gain={row.fusion_gain:.4f}, y={row.y_true:.4f}, xgb={row.xgb_pred:.4f}, fusion={row.fusion_pred:.4f}")
    (OUT_DIR / "fusion_case_study_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[CONFIG]", flush=True)
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False), flush=True)
    validate_config_paths()
    write_json(OUT_DIR / "run_config.json", config_dict())

    profile = pd.read_csv(FUSION_PROFILE_PATH)
    required = {"sample_id", "cv_fold", "y_true", "xgb_pred", "mrnalm_pred", "fusion_pred", "fusion_gain"}
    missing = sorted(required.difference(profile.columns))
    if missing:
        raise ValueError(f"Fusion profile missing required columns: {missing}")
    if not set(BASIC_FEATURES).issubset(profile.columns):
        print("[INFO] BASIC columns not complete in fusion profile; loading BASIC table", flush=True)
        profile = profile.merge(load_basic_features(profile["sample_id"].astype(str).tolist()), on="sample_id", how="left")
    helped, hurt = select_cases(profile)
    selected = pd.concat([helped, hurt], ignore_index=True)

    shap = pd.read_csv(SHAP_PLOT_SAMPLE_PATH)
    shap = shap.merge(selected[["sample_id", "cv_fold"]], on=["sample_id", "cv_fold"], how="inner")
    shap_summary = build_shap_summary(shap)
    combined = selected.merge(shap_summary, on=["sample_id", "cv_fold"], how="left")

    output_cols = [
        "case_group",
        "sample_id",
        "cv_fold",
        "y_true",
        "xgb_pred",
        "mrnalm_pred",
        "fusion_pred",
        "xgb_residual",
        "fusion_residual",
        "abs_err_xgb",
        "abs_err_fusion",
        "fusion_gain",
        "fusion_gain_group_within_fold",
        *BASIC_FEATURES,
        "top_positive_shap_features",
        "top_negative_shap_features",
        "shap_family_summary_available_features",
        "n_available_shap_features",
        "sum_available_shap",
        "sum_abs_available_shap",
    ]
    output_cols = [col for col in output_cols if col in combined.columns]
    combined = combined.loc[:, output_cols]
    combined[combined["case_group"] == "top_helped"].to_csv(OUT_DIR / "fusion_case_study_top_helped.csv", index=False)
    combined[combined["case_group"] == "top_hurt"].to_csv(OUT_DIR / "fusion_case_study_top_hurt.csv", index=False)
    combined.to_csv(OUT_DIR / "fusion_case_study_combined.csv", index=False)
    write_report(helped, hurt)
    print(f"[DONE] outputs written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
