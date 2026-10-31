"""Frozen estimands. No filesystem reads, model fitting, or execution at import."""
import numpy as np
from scipy.stats import rankdata


def score(y, pred):
    y, pred = np.asarray(y, np.float64), np.asarray(pred, np.float64)
    if len(y) < 2 or y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all() or np.std(y) == 0 or np.std(pred) == 0:
        raise ValueError("Invalid/nonfinite/constant metric arrays")
    return {"r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
            "spearman": float(np.corrcoef(rankdata(y), rankdata(pred))[0, 1]),
            "pearson": float(np.corrcoef(y, pred)[0, 1]), "rmse": float(np.sqrt(np.mean((y - pred) ** 2)))}


def matched_estimands(y, prediction, folds):
    """prediction axes: gene, seed([42,2026,3407]), arm([A,B,C])."""
    y, prediction, folds = np.asarray(y), np.asarray(prediction), np.asarray(folds)
    if prediction.shape != (len(y), 3, 3) or folds.shape != y.shape or set(folds) != set(range(5)):
        raise ValueError("Wrong fixed paired matrix")
    names = ("r2", "spearman", "pearson", "rmse")
    scores = np.empty((5, 3, 3, 4))
    for fold in range(5):
        mask = folds == fold
        for seed in range(3):
            for arm in range(3):
                values = score(y[mask], prediction[mask, seed, arm])
                scores[fold, seed, arm] = [values[n] for n in names]
    delta = np.stack([scores[:, :, 1] - scores[:, :, 0], scores[:, :, 1] - scores[:, :, 2], scores[:, :, 2] - scores[:, :, 0]], axis=2)
    return {"metric_names": names, "delta_names": ("B-A", "B-C", "C-A"), "fold_seed_arm_scores": scores,
            "fold_seed_deltas": delta, "overall_matched_delta": delta.mean(axis=(0, 1)),
            "fold_seed_mean_delta": delta.mean(axis=1), "seed_fold_mean_delta": delta.mean(axis=0),
            "fold_population_SD": delta.mean(axis=1).std(axis=0, ddof=0),
            "seed_population_SD": delta.mean(axis=0).std(axis=0, ddof=0)}


def paired_gene_bootstrap(y, prediction, folds, n_resamples=5000, seed=42):
    """One global paired gene resample shared by all 9 predictions. No seed resampling.

    Formal calls use 5000/42. Small synthetic PRE tests may request fewer draws.
    A degenerate resample fails rather than silently changes sampling rules.
    """
    y, prediction, folds = np.asarray(y), np.asarray(prediction), np.asarray(folds)
    rng = np.random.default_rng(seed)
    for iteration in range(n_resamples):
        index = rng.integers(0, len(y), len(y))
        point = matched_estimands(y[index], prediction[index], folds[index])
        yield iteration, index, point["overall_matched_delta"]


def ci_summary(samples):
    samples = np.asarray(samples)
    if not np.isfinite(samples).all():
        raise ValueError("Nonfinite bootstrap")
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    return {"ci_low": low, "ci_high": high, "fraction_delta_gt_zero": (samples > 0).mean(axis=0), "is_p_value": False}
