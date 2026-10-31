#!/usr/bin/env python3
"""Post hoc fixed-prediction statistics; no training, inference or fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO
CORE = ['LM', 'Bio-XGB', 'Bio-Fusion', 'Common-XGB', 'Common-Fusion']
HUMAN = ['BASIC', 'LM', 'Ridge', 'ElasticNet', 'Bio-XGB', 'Bio-Fusion', 'Common-XGB', 'Common-Fusion']
CONTRASTS = [('Bio-XGB', 'LM'), ('Bio-Fusion', 'LM'), ('Bio-Fusion', 'Bio-XGB'), ('Common-XGB', 'LM'), ('Common-Fusion', 'LM'), ('Common-Fusion', 'Common-XGB'), ('Bio-XGB', 'Common-XGB'), ('Bio-Fusion', 'Common-Fusion')]
SEED, NBOOT = 42, 5000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_freeze(root):
    freeze = json.loads((root / 'data/input_manifest.json').read_text())
    for rel, digest in freeze['input_hashes'].items():
        if sha(root / rel) != digest:
            raise ValueError('Frozen input changed: ' + rel)


def correlation_columns(y, p):
    yc, pc = y - np.mean(y), p - np.mean(p, axis=0)
    return np.sum(yc[:, None] * pc, axis=0) / np.sqrt(np.sum(yc**2) * np.sum(pc**2, axis=0))


def metrics(y, p, include_r2=True):
    rho = correlation_columns(rankdata(y, method='average'), rankdata(p, method='average', axis=0))
    pearson = correlation_columns(y, p)
    values = [rho, pearson]
    if include_r2:
        values.append(1 - np.sum((y[:, None] - p)**2, axis=0) / np.sum((y-y.mean())**2))
    return np.stack(values, axis=1)


def evaluate(d, models, include_r2, out, name):
    names = ['spearman', 'pearson'] + (['r2'] if include_r2 else [])
    y, p = d.target.to_numpy(), d[models].to_numpy()
    assert d.ENSID.is_unique and np.isfinite(y).all() and np.isfinite(p).all()
    point = metrics(y, p, include_r2)
    rng = np.random.default_rng(SEED)
    draws = np.empty((NBOOT, len(models), len(names)))
    t = time.monotonic()
    for b in range(NBOOT):
        ix = rng.integers(0, len(d), size=len(d))
        draws[b] = metrics(y[ix], p[ix], include_r2)
        if (b+1) % 1000 == 0:
            print(f'{name}: {b+1}/{NBOOT} paired resamples; {time.monotonic()-t:.1f}s', flush=True)
    assert np.isfinite(draws).all()
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0, method='linear')
    rows = []
    for i, model in enumerate(models):
        for j, metric in enumerate(names):
            rows.append(dict(population=name, model=model, metric=metric, n=len(d), estimate=point[i,j], ci95_low=lo[i,j], ci95_high=hi[i,j], resamples=NBOOT, seed=SEED))
    pd.DataFrame(rows).to_csv(out / f'{name}_metrics.csv', index=False)
    delta_rows = []
    for a, b in CONTRASTS:
        ia, ib = models.index(a), models.index(b)
        delta = draws[:, ia, :] - draws[:, ib, :]
        dlo, dhi = np.quantile(delta, [0.025, 0.975], axis=0, method='linear')
        for j, metric in enumerate(names):
            delta_rows.append(dict(population=name, first=a, second=b, metric=metric, n=len(d), estimate=point[ia,j]-point[ib,j], ci95_low=dlo[j], ci95_high=dhi[j], resamples=NBOOT, seed=SEED))
    pd.DataFrame(delta_rows).to_csv(out / f'{name}_paired_deltas.csv', index=False)
    np.savez_compressed(out / f'{name}_bootstrap_draws.npz', values=draws, models=np.array(models), metrics=np.array(names), seed=SEED, resamples=NBOOT)


def fold_summary(d, out):
    rows = []
    for fold, sub in d.groupby('fold', sort=True):
        values = metrics(sub.target.to_numpy(), sub[HUMAN].to_numpy())
        for i, model in enumerate(HUMAN):
            for j, metric in enumerate(['spearman', 'pearson', 'r2']):
                rows.append(dict(fold=fold, model=model, metric=metric, n=len(sub), estimate=values[i,j]))
    rows = pd.DataFrame(rows)
    rows.to_csv(out / 'human_fold_metrics.csv', index=False)
    rows.groupby(['model', 'metric'], sort=False).estimate.agg(mean='mean', population_sd=lambda x: x.std(ddof=0), n_folds='count').reset_index().to_csv(out / 'human_five_fold_summary.csv', index=False)


def coverage(human, d, out):
    counts = d.groupby(['group', 'coverage_status'], dropna=False).size().rename('n').reset_index()
    counts.to_csv(out / 'coverage_status_counts.csv', index=False)
    cols = [x for x in d.select_dtypes(include='number').columns if x != 'fold']
    rows, differences, performance = [], [], []
    for col in cols:
        group_values = {}
        for group, sub in d.groupby('group', sort=True):
            x = sub[col].dropna().to_numpy()
            group_values[group] = x
            rows.append(dict(group=group, variable=col, n_total=len(sub), n_observed=len(x), n_missing=len(sub)-len(x), mean=np.mean(x) if len(x) else np.nan, sd=np.std(x, ddof=1) if len(x)>1 else np.nan, median=np.median(x) if len(x) else np.nan, q1=np.quantile(x,.25) if len(x) else np.nan, q3=np.quantile(x,.75) if len(x) else np.nan, minimum=np.min(x) if len(x) else np.nan, maximum=np.max(x) if len(x) else np.nan))
        a, b = group_values['matched'], group_values['unmatched']
        denom = np.sqrt((np.var(a,ddof=1) + np.var(b,ddof=1))/2)
        differences.append(dict(variable=col, matched_n_observed=len(a), unmatched_n_observed=len(b), mean_difference=np.mean(a)-np.mean(b), standardized_mean_difference=(np.mean(a)-np.mean(b))/denom if denom>0 else np.nan, definition='matched_minus_unmatched; observed_values_only'))
    for group, sub in d.groupby('group', sort=True):
        h = human.set_index('ENSID').loc[sub.ENSID]
        v = metrics(h.target.to_numpy(), h[CORE].to_numpy())
        for i, model in enumerate(CORE):
            for j, metric in enumerate(['spearman', 'pearson', 'r2']):
                performance.append(dict(group=group, model=model, metric=metric, n=len(h), estimate=v[i,j], estimand='internal_human_OOF; descriptive; not_external_performance'))
    pd.DataFrame(rows).to_csv(out / 'coverage_descriptive_statistics.csv', index=False)
    pd.DataFrame(differences).to_csv(out / 'coverage_standardized_differences.csv', index=False)
    pd.DataFrame(performance).to_csv(out / 'coverage_internal_oof_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve() if args.output else REPO / 'outputs/statistics_1'
    if not out.is_relative_to(REPO/'outputs') or out==REPO/'outputs':
        raise ValueError('Output must be a new directory inside outputs/')
    if out.exists():
        raise FileExistsError('Choose a new output directory: ' + str(out))
    check_freeze(root)
    out.mkdir(parents=True)
    h = pd.read_csv(root / 'data/human/oof.csv', float_precision='round_trip')
    e = pd.read_csv(root / 'data/external/matched.csv', float_precision='round_trip')
    c = pd.read_csv(root / 'data/human/coverage_features.csv', float_precision='round_trip')
    evaluate(e, CORE, False, out, 'external_human')
    evaluate(h, HUMAN, True, out, 'human_pooled_oof')
    fold_summary(h, out)
    coverage(h, c, out)
    check_freeze(root)
    info = dict(status='COMPLETE', training=False, inference=False, fitting=False, draft_changes=False, seed=SEED, resamples=NBOOT, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, analysis_script_sha256=sha(Path(__file__)), plan_sha256=sha(root/'docs/statistical_protocol.md'))
    (out / 'RUN_INFO.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
