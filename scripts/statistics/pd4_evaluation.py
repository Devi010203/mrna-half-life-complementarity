"""Report-only post-freeze statistics. Frozen helper is the estimand authority."""
import hashlib
import numpy as np
import pandas as pd
SEEDS = [42, 2026, 3407]
ARM_NAMES = ["A_LABEL_ONLY", "B_BIOCHEM_KD", "C_COMPACT_KD"]
def require(condition, message):
    if not condition: raise ValueError(message)
from pd4_estimands import score, matched_estimands, paired_gene_bootstrap, ci_summary

METRICS = ["r2", "spearman", "pearson", "rmse"]
DELTAS = ["B-A", "B-C", "C-A"]
SCOPES = ["seed_42", "seed_2026", "seed_3407", "equal_seed_prediction_mean"]


def secondary(y, predictions):
    views = [predictions[:, i, :] for i in range(3)] + [predictions.mean(axis=1)]
    return np.array([[list(score(y, view[:, a]).values()) for a in range(3)] for view in views])


def deltas(values):
    return np.stack((values[..., 1, :] - values[..., 0, :], values[..., 1, :] - values[..., 2, :],
                     values[..., 2, :] - values[..., 0, :]), axis=-2)


def calibration(y, pred):
    require(np.isfinite(y).all() and np.isfinite(pred).all() and np.std(y) > 0 and np.std(pred) > 0, "Invalid or constant predictions")
    ratio = float(np.std(pred) / np.std(y))
    require(1e-6 <= ratio <= 1e6, "Numerical SD-ratio pathology")
    slope = float(np.sum((pred - pred.mean()) * (y - y.mean())) / np.sum((pred - pred.mean()) ** 2))
    return {"slope": slope, "intercept": float(y.mean() - slope * pred.mean()),
            "prediction_sd_ratio": ratio, "mean_error": float(np.mean(pred - y)),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2)))}


def decisions(point, primary_ci, calibration_table):
    checks = {}
    for d, name, arm in [(0, "BIOCHEM_KD", ARM_NAMES[1]), (2, "COMMON_TEACHER", ARM_NAMES[2])]:
        fold = point["fold_seed_mean_delta"][:, d]
        seed = point["seed_fold_mean_delta"][:, d]
        candidate = calibration_table.loc[calibration_table.arm.eq(arm)]
        bad_cal = int((candidate.groupby("fold").slope.mean() <= 0).sum()) >= 4 or int((candidate.groupby("seed").slope.mean() <= 0).sum()) >= 2
        reversal = any(int((fold[:, m] < 0).sum()) >= 4 and int((seed[:, m] < 0).sum()) >= 2 for m in [1, 2])
        c = {"effect_ge_0_010": bool(point["overall_matched_delta"][d, 0] >= .010),
             "positive_folds_ge_4": bool((fold[:, 0] > 0).sum() >= 4),
             "positive_seeds_ge_2": bool((seed[:, 0] > 0).sum() >= 2),
             "primary_CI_lower_gt_0": bool(primary_ci["ci_low"][d, 0] > 0),
             "no_systematic_correlation_reversal": not reversal, "no_calibration_pathology": not bad_cal,
             "no_numerical_pathology": True}
        checks[name] = {"GO": all(c.values()), "checks": c, "positive_folds": int((fold[:, 0] > 0).sum()),
                        "positive_seeds": int((seed[:, 0] > 0).sum()),
                        "STRONG": all(c.values()) and bool(point["overall_matched_delta"][d, 0] >= .020)}
    ba, bc, ca = point["overall_matched_delta"][:, 0]
    bg, cg = checks["BIOCHEM_KD"]["GO"], checks["COMMON_TEACHER"]["GO"]
    specific = bg and bc >= .005 and primary_ci["ci_low"][1, 0] > 0
    if specific:
        terminal = "BIOCHEMICAL_PRIVILEGED_DISTILLATION_SUPPORTED"
    elif bg:
        terminal = "GENERIC_TEACHER_DISTILLATION_SUPPORTED_BUT_NOT_BIOCHEMICAL_SPECIFIC"
    elif cg:
        terminal = "COMMON_TEACHER_DISTILLATION_SUPPORTED"
    elif ba < .010 and ca < .010 and any(v > 0 and primary_ci["ci_low"][i, 0] > 0 for i, v in [(0, ba), (2, ca)]):
        terminal = "DISTILLATION_EFFECT_PRACTICALLY_SMALL"
    else:
        terminal = "DISTILLATION_NOT_SUPPORTED"
    return {"terminal_interpretation": terminal, "candidate_checks": checks,
            "STOP_FOR_SMALL_EFFECT": bool(ba < .010),
            "biochemical_specific_supported": bool(specific),
            "BC_magnitude": "strong" if bc >= .010 else "modest" if bc >= .005 else "too_small" if bc > 0 else "no_superiority",
            "BC_CI_lower_positive": bool(primary_ci["ci_low"][1, 0] > 0),
            "READY_FOR_SECONDARY_EXTERNAL_EVALUATION": "YES" if bg or cg else "NO",
            "recommended_route": [name for name, v in checks.items() if v["GO"]],
            "recommendation_requires_full_integrity_PASS": True,
            "external_execution_authorized": False, "stop_after_PD4": True}


def bootstrap(y, predictions, folds, n=5000):
    primary = np.empty((n, 3, 4))
    pooled = np.empty((n, 4, 3, 4))
    digest = hashlib.sha256()
    for i, index, value in paired_gene_bootstrap(y, predictions, folds, n_resamples=n, seed=42):
        primary[i] = value
        pooled[i] = secondary(y[index], predictions[index])
        digest.update(np.asarray(index, dtype="<i8").tobytes())
        if (i + 1) % 1000 == 0:
            print(f"paired bootstrap {i+1}/{n}; no selection or fitting", flush=True)
    require(np.isfinite(primary).all() and np.isfinite(pooled).all(), "Degenerate bootstrap; do not replace draws")
    return primary, pooled, digest.hexdigest()
