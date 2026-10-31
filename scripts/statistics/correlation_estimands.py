"""Original fixed-prediction correlation/paired-bootstrap estimands. No model fitting."""

from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, pearsonr

N_BOOTSTRAP=5000
BOOTSTRAP_SEED=42
STRATA=("ORTHOLOG_REPRESENTED","NO_CURRENT_1TO1_ORTHOLOG")
MODEL_COLUMNS={"mrnalm":"pred_mrnalm_ensemble","common_compact_xgb":"pred_common_compact_xgb_ensemble","common_fusion":"pred_common_fusion_ensemble"}
CONTRASTS={"FUSION_MINUS_LM":("common_fusion","mrnalm","fusion_minus_lm"),"FUSION_MINUS_XGB":("common_fusion","common_compact_xgb","fusion_minus_xgb"),"XGB_MINUS_LM":("common_compact_xgb","mrnalm","xgb_minus_lm")}

def correlation(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size != second.size or first.size < 3 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise RuntimeError("Correlation input integrity failed")
    spearman = float(np.corrcoef(rankdata(first), rankdata(second))[0, 1])
    pearson = float(np.corrcoef(first, second)[0, 1])
    return spearman, pearson

def complementarity_row(domain: str, target: np.ndarray, lm: np.ndarray, xgb: np.ndarray) -> dict[str, Any]:
    prediction_spearman, prediction_pearson = correlation(lm, xgb)
    residual_lm = np.asarray(target, dtype=float) - np.asarray(lm, dtype=float)
    residual_xgb = np.asarray(target, dtype=float) - np.asarray(xgb, dtype=float)
    residual_spearman, residual_pearson = correlation(residual_lm, residual_xgb)
    xgb_residual_vs_lm_spearman, xgb_residual_vs_lm_pearson = correlation(residual_xgb, lm)
    lm_residual_vs_xgb_spearman, lm_residual_vs_xgb_pearson = correlation(residual_lm, xgb)
    return {
        "domain": domain,
        "n": len(target),
        "prediction_lm_vs_xgb_spearman": prediction_spearman,
        "prediction_lm_vs_xgb_pearson": prediction_pearson,
        "residual_lm_vs_xgb_spearman": residual_spearman,
        "residual_lm_vs_xgb_pearson": residual_pearson,
        "xgb_residual_vs_lm_prediction_spearman": xgb_residual_vs_lm_spearman,
        "xgb_residual_vs_lm_prediction_pearson": xgb_residual_vs_lm_pearson,
        "lm_residual_vs_xgb_prediction_spearman": lm_residual_vs_xgb_spearman,
        "lm_residual_vs_xgb_prediction_pearson": lm_residual_vs_xgb_pearson,
        "used_to_modify_fusion": False,
    }

def bootstrap_mouse(target: np.ndarray, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(target)
    rows: list[dict[str, Any]] = []
    for iteration in range(N_BOOTSTRAP):
        index = rng.integers(0, n, size=n)
        values: dict[str, float] = {}
        for model, prediction in predictions.items():
            rho, pearson = correlation(prediction[index], target[index])
            values[f"spearman_{model}"] = rho
            values[f"pearson_{model}"] = pearson
        rows.append(
            {
                "bootstrap_index": iteration,
                "n": n,
                "seed": BOOTSTRAP_SEED,
                **values,
                "delta_spearman_xgb_minus_lm": values["spearman_common_compact_xgb"] - values["spearman_mrnalm"],
                "delta_spearman_fusion_minus_xgb": values["spearman_common_fusion"] - values["spearman_common_compact_xgb"],
                "delta_spearman_fusion_minus_lm": values["spearman_common_fusion"] - values["spearman_mrnalm"],
                "delta_pearson_xgb_minus_lm": values["pearson_common_compact_xgb"] - values["pearson_mrnalm"],
                "delta_pearson_fusion_minus_xgb": values["pearson_common_fusion"] - values["pearson_common_compact_xgb"],
                "delta_pearson_fusion_minus_lm": values["pearson_common_fusion"] - values["pearson_mrnalm"],
            }
        )
        if (iteration + 1) % 1000 == 0:
            print(f"[M3 paired bootstrap] {iteration + 1}/{N_BOOTSTRAP}", flush=True)
    return pd.DataFrame(rows)

def bootstrap_strata(data: pd.DataFrame) -> pd.DataFrame:
    seed_sequence = np.random.SeedSequence(BOOTSTRAP_SEED)
    child_sequences = seed_sequence.spawn(len(STRATA))
    rows: list[dict[str, Any]] = []
    for stream_index, (stratum, child) in enumerate(zip(STRATA, child_sequences)):
        subset = data.loc[data["ortholog_status"].eq(stratum)].reset_index(drop=True)
        target = subset["mouse_HALFLIFE"].to_numpy(dtype=float)
        predictions = {model: subset[column].to_numpy(dtype=float) for model, column in MODEL_COLUMNS.items()}
        rng = np.random.default_rng(child)
        n = len(subset)
        for iteration in range(N_BOOTSTRAP):
            indices = rng.integers(0, n, size=n)
            values: dict[str, float] = {}
            for model, prediction in predictions.items():
                spearman, pearson = correlation(prediction[indices], target[indices])
                values[f"spearman_{model}"] = spearman
                values[f"pearson_{model}"] = pearson
            for _contrast, (model_a, model_b, suffix) in CONTRASTS.items():
                for metric in ("spearman", "pearson"):
                    values[f"delta_{metric}_{suffix}"] = values[f"{metric}_{model_a}"] - values[f"{metric}_{model_b}"]
            rows.append(
                {
                    "ortholog_status": stratum,
                    "bootstrap_index": iteration,
                    "n": n,
                    "root_seed": BOOTSTRAP_SEED,
                    "rng_stream_index": stream_index,
                    "rng_spawn_key": ".".join(map(str, child.spawn_key)),
                    **values,
                }
            )
            if (iteration + 1) % 1000 == 0:
                print(f"[M4 paired bootstrap {stratum}] {iteration + 1}/{N_BOOTSTRAP}", flush=True)
    return pd.DataFrame(rows)

def paired_correlations(target: np.ndarray, predictions: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, float]:
    sampled_target = target[indices]
    target_rank = rankdata(sampled_target)
    target_centered = sampled_target - sampled_target.mean()
    target_rank_centered = target_rank - target_rank.mean()
    target_norm = float(np.sqrt(np.sum(target_centered**2)))
    target_rank_norm = float(np.sqrt(np.sum(target_rank_centered**2)))
    values: dict[str, float] = {}
    for model, prediction in predictions.items():
        sampled = prediction[indices]
        centered = sampled - sampled.mean()
        ranked = rankdata(sampled)
        rank_centered = ranked - ranked.mean()
        values[f"pearson_{model}"] = float(np.dot(centered, target_centered) / (np.sqrt(np.sum(centered**2)) * target_norm))
        values[f"spearman_{model}"] = float(
            np.dot(rank_centered, target_rank_centered) / (np.sqrt(np.sum(rank_centered**2)) * target_rank_norm)
        )
    for _contrast, (model_a, model_b, suffix) in CONTRASTS.items():
        for metric in ("spearman", "pearson"):
            values[f"delta_{metric}_{suffix}"] = values[f"{metric}_{model_a}"] - values[f"{metric}_{model_b}"]
    return values

def bootstrap_experiment(task: tuple[str, int, int, np.ndarray, dict[str, np.ndarray]]) -> pd.DataFrame:
    experiment_id, stable_index, seed, target, predictions = task
    rng = np.random.default_rng(seed)
    n = len(target)
    rows: list[dict[str, Any]] = []
    for iteration in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)
        rows.append(
            {
                "experiment_id": experiment_id,
                "stable_experiment_index": stable_index,
                "bootstrap_index": iteration,
                "seed": seed,
                "n": n,
                **paired_correlations(target, predictions, indices),
            }
        )
    return pd.DataFrame(rows)

def finite_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if np.ptp(xx) == 0 or np.ptp(yy) == 0:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(xx, yy).statistic)
    if method == "pearson":
        return float(pearsonr(xx, yy).statistic)
    raise ValueError(method)

def fast_spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(np.dot(rx, rx) * np.dot(ry, ry)))
    if denom == 0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)

def bootstrap_context(
    frame: pd.DataFrame, seed: int, n_boot: int
) -> pd.DataFrame:
    target = frame["context_target"].to_numpy(float)
    lm = frame["mrna_lm_oof"].to_numpy(float)
    xgb = frame["xgb_oof"].to_numpy(float)
    fusion = frame["fusion_oof"].to_numpy(float)
    n = len(frame)
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(n_boot):
        take = rng.integers(0, n, size=n)
        y = target[take]
        lm_rho = fast_spearman(y, lm[take])
        xgb_rho = fast_spearman(y, xgb[take])
        fusion_rho = fast_spearman(y, fusion[take])
        rows.append(
            {
                "bootstrap_iteration": iteration + 1,
                "lm_spearman": lm_rho,
                "xgb_spearman": xgb_rho,
                "fusion_spearman": fusion_rho,
                "fusion_minus_lm": fusion_rho - lm_rho,
                "fusion_minus_xgb": fusion_rho - xgb_rho,
                "xgb_minus_lm": xgb_rho - lm_rho,
            }
        )
    return pd.DataFrame(rows)
