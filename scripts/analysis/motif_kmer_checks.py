#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""No-args k-mer/motif enrichment for fusion-helped vs fusion-hurt transcripts."""

from __future__ import annotations

import itertools
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG: paths are resolved from environment first. Point these to the clean
# late-fusion run and canonical normalized table.
# =============================================================================
def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else Path(default)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


PROJECT_ROOT = env_path("SALUKI_PROJECT_ROOT", Path(__file__).resolve().parents[2])
RUNS_ROOT = env_path("SALUKI_RUNS_ROOT", PROJECT_ROOT / "results" / "runs")

NORMALIZED_PATH = env_path(
    "SALUKI_MOTIF_NORMALIZED_PATH",
    os.environ.get("SALUKI_NORMALIZED_PATH", PROJECT_ROOT / "data" / "processed" / "human" / "normalized.csv.gz"),
)
LATE_FUSION_RUN_ROOT = env_path(
    "SALUKI_MOTIF_LATE_FUSION_RUN_ROOT",
    RUNS_ROOT / "late_fusion",
)
LATE_FUSION_PRED_TEST_PATH = env_path(
    "SALUKI_MOTIF_LATE_FUSION_PRED_TEST_PATH",
    LATE_FUSION_RUN_ROOT / "late_fusion_pred_test.csv",
)

OUT_BASE_DIR = env_path("SALUKI_MOTIF_OUT_BASE_DIR", RUNS_ROOT)
RUN_TAG = env_str("SALUKI_MOTIF_RUN_TAG", "motif_kmer_checks")
OUT_DIR = OUT_BASE_DIR / RUN_TAG

XGB_METHOD = "xgb_only"
FUSION_METHOD = "linear_fusion"
TOP_FRACTION = 0.20
REGIONS = ["5UTR", "ORF", "3UTR"]
KMER_KS = [3, 4]

MOTIF_PATTERNS = {
    "AU_rich_ATTTA": "ATTTA",
    "U_rich_TTTTT": "TTTTT",
    "ARE_WWATTTAWW": r"[AT][AT]ATTTA[AT][AT]",
    "DRACH_like": r"[AGT][AG]AC[ACT]",
    "PUM_like_TGTANATA": r"TGTA[ACGT]ATA",
    "polyA_signal_AATAAA": "AATAAA",
    "CPE_TTTTAT": "TTTTAT",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def config_dict() -> dict[str, Any]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "timestamp": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "normalized_path": str(NORMALIZED_PATH),
        "late_fusion_run_root": str(LATE_FUSION_RUN_ROOT),
        "late_fusion_pred_test_path": str(LATE_FUSION_PRED_TEST_PATH),
        "out_base_dir": str(OUT_BASE_DIR),
        "run_tag": RUN_TAG,
        "out_dir": str(OUT_DIR),
        "xgb_method": XGB_METHOD,
        "fusion_method": FUSION_METHOD,
        "top_fraction": TOP_FRACTION,
        "regions": list(REGIONS),
        "kmer_ks": list(KMER_KS),
        "motif_patterns": dict(MOTIF_PATTERNS),
    }


def print_config_and_raise(message: str) -> None:
    print("[CONFIG]", flush=True)
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False), flush=True)
    raise FileNotFoundError(message)


def validate_config_paths() -> None:
    required = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "NORMALIZED_PATH": NORMALIZED_PATH,
        "LATE_FUSION_PRED_TEST_PATH": LATE_FUSION_PRED_TEST_PATH,
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


def clean_seq(value: object) -> str:
    seq = re.sub(r"\s+", "", str(value or "").upper())
    seq = seq.replace("U", "T")
    return "".join(ch for ch in seq if ch in {"A", "C", "G", "T"})


def load_fusion_gain() -> pd.DataFrame:
    df = pd.read_csv(LATE_FUSION_PRED_TEST_PATH)
    required = {"sample_id", "cv_fold", "method", "y_true", "y_pred", "xgb_pred"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"late fusion prediction file missing required columns: {missing}")
    xgb = df[df["method"].astype(str) == XGB_METHOD][["sample_id", "cv_fold", "y_true", "xgb_pred"]].copy()
    fusion = (
        df[df["method"].astype(str) == FUSION_METHOD][["sample_id", "cv_fold", "y_pred"]]
        .rename(columns={"y_pred": "fusion_pred"})
        .copy()
    )
    out = xgb.merge(fusion, on=["sample_id", "cv_fold"], how="inner")
    out["xgb_residual"] = out["y_true"] - out["xgb_pred"]
    out["fusion_residual"] = out["y_true"] - out["fusion_pred"]
    out["abs_err_xgb"] = np.abs(out["xgb_residual"])
    out["abs_err_fusion"] = np.abs(out["fusion_residual"])
    out["fusion_gain"] = out["abs_err_xgb"] - out["abs_err_fusion"]
    lo = out["fusion_gain"].quantile(TOP_FRACTION)
    hi = out["fusion_gain"].quantile(1.0 - TOP_FRACTION)
    out["fusion_gain_group"] = "neutral_middle60"
    out.loc[out["fusion_gain"] >= hi, "fusion_gain_group"] = "fusion_helped_top20"
    out.loc[out["fusion_gain"] <= lo, "fusion_gain_group"] = "fusion_hurt_top20"
    labels = []
    for _, sub in out.groupby("cv_fold", sort=False):
        f_lo = sub["fusion_gain"].quantile(TOP_FRACTION)
        f_hi = sub["fusion_gain"].quantile(1.0 - TOP_FRACTION)
        fold_label = pd.Series("neutral_middle60", index=sub.index)
        fold_label.loc[sub["fusion_gain"] >= f_hi] = "fusion_helped_top20"
        fold_label.loc[sub["fusion_gain"] <= f_lo] = "fusion_hurt_top20"
        labels.append(fold_label)
    out["fusion_gain_group_within_fold"] = pd.concat(labels).sort_index()
    return out


def load_sequences(ids: list[str]) -> pd.DataFrame:
    cols = ["ENSID", "HALFLIFE", *REGIONS]
    df = pd.read_csv(NORMALIZED_PATH, compression="gzip", usecols=cols)
    df["sample_id"] = df["ENSID"].astype(str)
    for region in REGIONS:
        df[region] = df[region].map(clean_seq)
    return df[df["sample_id"].isin(ids)].drop(columns=["ENSID"]).copy()


def kmer_counter(seq: str, k: int) -> tuple[Counter[str], int]:
    denom = max(len(seq) - k + 1, 0)
    if denom <= 0:
        return Counter(), 0
    return Counter(seq[i : i + k] for i in range(denom)), denom


def motif_frequency(seq: str, pattern: str) -> float:
    if not seq:
        return float("nan")
    try:
        count = len(re.findall(f"(?=({pattern}))", seq))
    except re.error:
        count = seq.count(pattern)
    return float(count) / float(max(len(seq), 1))


def gc_fraction(seq: str) -> float:
    if not seq:
        return float("nan")
    return float(seq.count("G") + seq.count("C")) / float(len(seq))


def try_mannwhitneyu(x: np.ndarray, y: np.ndarray) -> float:
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


def compare_feature(values: pd.Series, groups: pd.Series) -> dict[str, Any]:
    helped = pd.to_numeric(values[groups == "fusion_helped_top20"], errors="coerce").to_numpy(np.float64)
    hurt = pd.to_numeric(values[groups == "fusion_hurt_top20"], errors="coerce").to_numpy(np.float64)
    return {
        "helped_mean": float(np.nanmean(helped)),
        "hurt_mean": float(np.nanmean(hurt)),
        "helped_median": float(np.nanmedian(helped)),
        "hurt_median": float(np.nanmedian(hurt)),
        "mean_difference_helped_minus_hurt": float(np.nanmean(helped) - np.nanmean(hurt)),
        "median_difference_helped_minus_hurt": float(np.nanmedian(helped) - np.nanmedian(hurt)),
        "mannwhitney_p": try_mannwhitneyu(helped, hurt),
        "cliffs_delta_helped_vs_hurt": cliffs_delta(helped, hurt),
        "n_helped": int(np.isfinite(helped).sum()),
        "n_hurt": int(np.isfinite(hurt).sum()),
    }


def build_enrichment(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        raise ValueError(f"Missing group_col={group_col!r}")
    rows: list[dict[str, Any]] = []
    groups = df[group_col].astype(str)
    for region in REGIONS:
        print(f"[REGION] {region}", flush=True)
        seqs = df[region].astype(str)
        for k in KMER_KS:
            counters = []
            denoms = []
            for seq in seqs:
                counter, denom = kmer_counter(seq, int(k))
                counters.append(counter)
                denoms.append(denom)
            denoms_arr = np.asarray(denoms, dtype=np.float64)
            for tup in itertools.product("ACGT", repeat=k):
                kmer = "".join(tup)
                values = pd.Series(
                    [
                        (float(counter.get(kmer, 0)) / float(denom)) if denom > 0 else np.nan
                        for counter, denom in zip(counters, denoms_arr)
                    ],
                    index=df.index,
                )
                row = {
                    "group_col": group_col,
                    "region": region,
                    "feature_type": "kmer",
                    "feature": kmer,
                    "k": k,
                }
                row.update(compare_feature(values, groups))
                rows.append(row)
        for name, pattern in MOTIF_PATTERNS.items():
            values = seqs.map(lambda s, pat=pattern: motif_frequency(s, pat))
            row = {
                "group_col": group_col,
                "region": region,
                "feature_type": "motif",
                "feature": name,
                "k": np.nan,
                "pattern": pattern,
            }
            row.update(compare_feature(values, groups))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value_bh_fdr"] = bh_fdr(out["mannwhitney_p"])
    return out


def region_length_gc_differences(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = df[group_col].astype(str)
    for region in REGIONS:
        seqs = df[region].astype(str)
        for metric_name, values in (
            ("length", seqs.map(len)),
            ("gc_fraction", seqs.map(gc_fraction)),
        ):
            row = {
                "group_col": group_col,
                "region": region,
                "metric": metric_name,
            }
            row.update(compare_feature(values, groups))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value_bh_fdr"] = bh_fdr(out["mannwhitney_p"])
    return out


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping plots: {exc}", flush=True)
        return None


def save_region_plot(enrich: pd.DataFrame, region: str) -> None:
    plt = try_import_matplotlib()
    if plt is None:
        return
    data = enrich[enrich["region"] == region].copy()
    data["abs_delta"] = data["median_difference_helped_minus_hurt"].abs()
    data = data.sort_values("abs_delta", ascending=False).head(25).iloc[::-1]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(data))))
    labels = data["feature_type"].astype(str) + ":" + data["feature"].astype(str)
    ax.barh(labels, data["median_difference_helped_minus_hurt"], color="#8a6fb0")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Median frequency difference: helped - hurt")
    ax.set_title(f"Top motif/k-mer enrichment in {region}")
    fig.tight_layout()
    safe_region = region.lower().replace("5", "5").replace("3", "3")
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"top_motif_enrichment_{safe_region}.{ext}", dpi=220)
    plt.close(fig)


def write_report(enrich: pd.DataFrame, sample_df: pd.DataFrame, confound: pd.DataFrame) -> None:
    lines = [
        "# Fusion Gain Motif/K-mer Enrichment Report",
        "",
        "This is an exploratory sequence-context analysis. Enrichment patterns are predictive associations, not causal motifs.",
        "",
        "Primary enrichment output uses within-fold helped/hurt grouping.",
        "",
        f"- Helped top fraction: {TOP_FRACTION:.2f}",
        f"- Within-fold helped samples: {(sample_df['fusion_gain_group_within_fold'] == 'fusion_helped_top20').sum()}",
        f"- Within-fold hurt samples: {(sample_df['fusion_gain_group_within_fold'] == 'fusion_hurt_top20').sum()}",
        "",
        "## Top Enrichments",
        "",
    ]
    tmp = enrich.copy()
    tmp["abs_delta"] = tmp["median_difference_helped_minus_hurt"].abs()
    for row in tmp.sort_values("abs_delta", ascending=False).head(20).itertuples(index=False):
        lines.append(
            f"- `{row.region}` `{row.feature_type}:{row.feature}`: median helped-hurt = "
            f"{row.median_difference_helped_minus_hurt:.6g}, Cliff's delta = {row.cliffs_delta_helped_vs_hurt:.4f}, "
            f"p = {row.mannwhitney_p:.3g}, q = {row.q_value_bh_fdr:.3g}"
        )
    lines.extend(["", "## Region Length/GC Confounding Check", ""])
    for row in confound.itertuples(index=False):
        lines.append(
            f"- `{row.region}` `{row.metric}`: median helped-hurt = "
            f"{row.median_difference_helped_minus_hurt:.6g}, q = {row.q_value_bh_fdr:.3g}"
        )
    (OUT_DIR / "motif_enrichment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[CONFIG]", flush=True)
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False), flush=True)
    validate_config_paths()
    cfg = config_dict()
    write_json(OUT_DIR / "motif_enrichment_run_config.json", cfg)
    write_json(OUT_DIR / "run_config.json", cfg)

    gain = load_fusion_gain()
    seq = load_sequences(gain["sample_id"].astype(str).tolist())
    df = gain.merge(seq, on="sample_id", how="inner")
    df.to_csv(OUT_DIR / "fusion_gain_sequence_groups.csv", index=False)

    enrich_global = build_enrichment(df, "fusion_gain_group")
    enrich_within = build_enrichment(df, "fusion_gain_group_within_fold")
    enrich_global = enrich_global.sort_values("median_difference_helped_minus_hurt", ascending=False)
    enrich_within = enrich_within.sort_values("median_difference_helped_minus_hurt", ascending=False)
    enrich_global.to_csv(OUT_DIR / "motif_enrichment_fusion_helped_vs_hurt_global.csv", index=False)
    enrich_within.to_csv(OUT_DIR / "motif_enrichment_fusion_helped_vs_hurt.csv", index=False)
    enrich_within.to_csv(OUT_DIR / "motif_enrichment_fusion_helped_vs_hurt_within_fold.csv", index=False)
    confound_global = region_length_gc_differences(df, "fusion_gain_group")
    confound_within = region_length_gc_differences(df, "fusion_gain_group_within_fold")
    confound_global.to_csv(OUT_DIR / "region_length_gc_differences_global.csv", index=False)
    confound_within.to_csv(OUT_DIR / "region_length_gc_differences_within_fold.csv", index=False)
    for region in REGIONS:
        save_region_plot(enrich_within, region)
    write_report(enrich_within, df, confound_within)
    print(f"[DONE] outputs written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
