#!/usr/bin/env python3
"""Independent point metrics, resample spot checks, CI/contrast and freeze QA."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO


def independent(y, p):
    return {'spearman': float(spearmanr(y,p).statistic), 'pearson': float(pearsonr(y,p).statistic), 'r2': float(1 - np.dot(y-p,y-p)/np.dot(y-y.mean(),y-y.mean()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report=(args.report or REPO/'outputs/statistical_acceptance_1.json').resolve()
    if not report.is_relative_to(REPO/'outputs') or report.exists():
        raise ValueError('Choose a new report file inside outputs/')
    root = args.root.resolve()
    out = args.results.resolve() if args.results else REPO/'results/unified'
    checks = []
    def check(name, actual, expected, atol=1e-12):
        a,b = np.asarray(actual),np.asarray(expected)
        ok = np.allclose(a,b,atol=atol,rtol=0,equal_nan=False)
        delta = float(np.max(np.abs(a-b)))
        checks.append(dict(check=name, passed=bool(ok), max_abs_difference=delta, tolerance=atol))
        if not ok: raise AssertionError(checks[-1])
    freeze=json.loads((root/'data/input_manifest.json').read_text())
    for rel,digest in freeze['input_hashes'].items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
    tables = {}
    for population, filename, n in [('external_human','external_matched.csv',11381),('human_pooled_oof','human_oof.csv',12981)]:
        d=pd.read_csv(root/({'external_matched.csv':'data/external/matched.csv','human_oof.csv':'data/human/oof.csv'}[filename]),float_precision='round_trip')
        tables[population]=d
        assert len(d)==d.ENSID.nunique()==n
        assert set(d.fold)==set(range(5))
        tab=pd.read_csv(out/f'{population}_metrics.csv',float_precision='round_trip')
        delta=pd.read_csv(out/f'{population}_paired_deltas.csv',float_precision='round_trip')
        saved=np.load(out/f'{population}_bootstrap_draws.npz',allow_pickle=False)
        models=saved['models'].tolist(); names=saved['metrics'].tolist();draws=saved['values']
        assert draws.shape==(5000,len(models),len(names)) and np.isfinite(draws).all()
        assert saved['seed']==42
        y=d.target.to_numpy()
        points={m:independent(y,d[m].to_numpy()) for m in models}
        for row in tab.itertuples():
            i,j=models.index(row.model),names.index(row.metric)
            check(f'{population}/{row.model}/{row.metric}/point',row.estimate,points[row.model][row.metric])
            ci=np.quantile(draws[:,i,j],[.025,.975],method='linear')
            check(f'{population}/{row.model}/{row.metric}/CI',[row.ci95_low,row.ci95_high],ci)
        for row in delta.itertuples():
            ia,ib,j=models.index(row.first),models.index(row.second),names.index(row.metric)
            check(f'{population}/{row.first}-{row.second}/{row.metric}/delta',row.estimate,points[row.first][row.metric]-points[row.second][row.metric])
            ci=np.quantile(draws[:,ia,j]-draws[:,ib,j],[.025,.975],method='linear')
            check(f'{population}/{row.first}-{row.second}/{row.metric}/pairedCI',[row.ci95_low,row.ci95_high],ci)
        rng=np.random.default_rng(42)
        selected=set(range(20))|set(range(4980,5000))
        maximum=0.
        for b in range(5000):
            ix=rng.integers(0,n,size=n)
            if b not in selected: continue
            for i,m in enumerate(models):
                vals=independent(y[ix],d[m].to_numpy()[ix])
                expected=np.array([vals[name] for name in names])
                maximum=max(maximum,float(np.max(np.abs(draws[b,i]-expected))))
        check(population+'/40_bootstrap_draws_independent_scipy_max_error',maximum,0.)
        if population=='external_human':
            old=pd.read_csv(root/'results/external/original_primary_bootstrap.csv',float_precision='round_trip')
            for m,col in [('LM','rho_lm'),('Bio-XGB','rho_xgb'),('Bio-Fusion','rho_fusion')]:
                check('Stage14_all_5000_draws_'+m,draws[:,models.index(m),names.index('spearman')],old[col].to_numpy())
    h=tables['human_pooled_oof'].set_index('ENSID');e=tables['external_human']
    assert set(e.ENSID).issubset(h.index)
    check('external_human_fold_alignment',e.fold,h.loc[e.ENSID,'fold'],0)
    check('external_original_target_alignment',e.original_target,h.loc[e.ENSID,'target'])
    for col in ['LM','Bio-XGB','Bio-Fusion','Common-XGB','Common-Fusion']:
        check('external_prediction_alignment_'+col,e[col],h.loc[e.ENSID,col])
    folds=pd.read_csv(out/'human_fold_metrics.csv')
    for row in folds.itertuples():
        d=h[h.fold.eq(row.fold)]
        check(f'fold{row.fold}/{row.model}/{row.metric}',row.estimate,independent(d.target.to_numpy(),d[row.model].to_numpy())[row.metric])
    summary=pd.read_csv(out/'human_five_fold_summary.csv')
    for row in summary.itertuples():
        vals=folds.loc[folds.model.eq(row.model)&folds.metric.eq(row.metric),'estimate'].to_numpy()
        check('five_fold_mean/'+row.model+'/'+row.metric,row.mean,vals.mean())
        check('five_fold_population_SD/'+row.model+'/'+row.metric,row.population_sd,vals.std(ddof=0))
    for name,ref,group in [('BASIC','xgb_summary.csv','saluki_basic8'),('Bio-XGB','xgb_summary.csv','basic_plus_biochemical_full'),('Ridge','ridge_summary.csv','basic_plus_biochemical_full')]:
        old=pd.read_csv(root/({'xgb_summary.csv':'results/human/xgb_summary_metrics.csv','ridge_summary.csv':'results/human/ridge_summary_metrics.csv'}[ref]))
        old=old[old.feature_group.eq(group)&old.split.eq('test')]
        for row in old.itertuples():
            if row.metric not in {'r2','spearman','pearson'}:continue
            new=summary.loc[summary.model.eq(name)&summary.metric.eq(row.metric)].iloc[0]
            # Original y_true was rounded in some files. Require agreement at 1e-7.
            check('stored_fold_mean/'+name+'/'+row.metric,new['mean'],row.mean,1e-7)
            check('stored_fold_SD/'+name+'/'+row.metric,new.population_sd,row.std,1e-7)
    old=pd.read_csv(root/'results/mouse/M3_HUMAN_OOF_MODEL_METRICS.csv')
    current=pd.read_csv(out/'human_pooled_oof_metrics.csv')
    for key,model in [('mrnalm','LM'),('common_compact_xgb','Common-XGB'),('common_fusion','Common-Fusion')]:
        row=old[old.human_fold.astype(str).eq('POOLED')&old.model.eq(key)].iloc[0]
        for metric in ['r2','pearson','spearman']:
            check('stored_common_pooled/'+model+'/'+metric,current.loc[current.model.eq(model)&current.metric.eq(metric),'estimate'].iloc[0],row[metric])
    cov=pd.read_csv(root/'data/human/coverage_features.csv',float_precision='round_trip')
    assert cov.ENSID.tolist()==h.index.tolist()
    assert cov.group.value_counts().to_dict()=={'matched':11381,'unmatched':1600}
    assert set(cov.loc[cov.group.eq('matched'),'ENSID'])==set(e.ENSID)
    cp=pd.read_csv(out/'coverage_internal_oof_metrics.csv')
    for row in cp.itertuples():
        sub=h.loc[cov.loc[cov.group.eq(row.group),'ENSID']]
        check('coverage/'+row.group+'/'+row.model+'/'+row.metric,row.estimate,independent(sub.target.to_numpy(),sub[row.model].to_numpy())[row.metric])
    result=dict(status='PASS',independent_metric_implementation='scipy.stats.spearmanr/pearsonr; separate R2 dot-product formula',bootstrap_spot_checks_per_population=40,stage14_bootstrap_draws_checked=5000,input_hashes_checked=len(freeze['input_hashes']),checks=checks,number_of_checks=len(checks),limitations='Independent numerical implementation, not an independent human review. No model retraining uncertainty or exhaustive source-assay proof.')
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__=='__main__':main()
