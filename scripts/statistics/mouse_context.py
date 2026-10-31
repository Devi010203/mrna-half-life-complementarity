"""Saved-prediction M3–M5 and context evaluation; original bootstrap functions."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from release_paths import ROOT,new_output,check_inputs
from checks import Checks
from correlation_estimands import (MODEL_COLUMNS,STRATA,CONTRASTS,correlation,
    complementarity_row,bootstrap_mouse,bootstrap_strata,bootstrap_experiment,bootstrap_context)

def read(relative):return pd.read_csv(ROOT/relative,float_precision='round_trip')

def model_checks(checks,reference,frame,boot,prefix):
    """Recalculate both correlations, model CIs and all pairwise differences."""
    points={m:correlation(frame[c].to_numpy(),frame.mouse_HALFLIFE.to_numpy()) for m,c in MODEL_COLUMNS.items()}
    for row in reference.itertuples():
        checks.add(prefix+'/'+row.model+'/n','COUNT',len(frame),row.n,tolerance=0)
        for j,(metric,column) in enumerate([('spearman','spearman_rho'),('pearson','pearson_r')]):
            checks.add(prefix+'/'+row.model+'/'+metric,'POINT_RECALCULATED',points[row.model][j],getattr(row,column),len(frame))
            if boot is not None:
                series=boot[f'{metric}_{row.model}']
                for suffix,value in [('ci_lower',series.quantile(.025)),('ci_upper',series.quantile(.975)),('bootstrap_median',series.median())]:
                    checks.add(prefix+'/'+row.model+'/'+metric+'_'+suffix,'CI_RECALCULATED',value,getattr(row,metric+'_'+suffix),len(frame))
    return points

def delta_checks(checks,reference,points,boot,prefix):
    for row in reference.itertuples():
        for j,metric in enumerate(['spearman','pearson']):
            estimate=points[row.model_a][j]-points[row.model_b][j]
            checks.add(prefix+'/'+row.contrast+'/'+metric,'POINT_RECALCULATED',estimate,getattr(row,'delta_'+metric+'_point'),row.n)
            if boot is not None:
                values=boot[f'{metric}_{row.model_a}']-boot[f'{metric}_{row.model_b}']
                for suffix,value in [('ci_lower',values.quantile(.025)),('ci_upper',values.quantile(.975)),('bootstrap_median',values.median())]:
                    checks.add(prefix+'/'+row.contrast+'/'+metric+'_'+suffix,'CI_RECALCULATED',value,getattr(row,'delta_'+metric+'_'+suffix),row.n)
                checks.add(prefix+'/'+row.contrast+'/'+metric+'/positive_fraction','CI_RECALCULATED',(values>0).mean(),getattr(row,'fraction_delta_'+metric+'_gt_zero'),row.n,tolerance=0)

def mouse_primary(out,do_ci):
    c=Checks();m=read('data/mouse/predictions.csv')
    assert len(m)==m.mouse_ENSID.nunique()==13744
    assert m.ortholog_status.value_counts().to_dict()=={STRATA[0]:10200,STRATA[1]:3544}
    ids=read('data/mouse/ortholog_manifest.csv')
    assert ids.mouse_ENSID.equals(m.mouse_ENSID) and ids.ortholog_status.equals(m.ortholog_status)
    boot=bootstrap_mouse(m.mouse_HALFLIFE.to_numpy(),{k:m[v].to_numpy() for k,v in MODEL_COLUMNS.items()}) if do_ci else None
    points=model_checks(c,read('results/mouse/M3_ALL_MOUSE_MODEL_METRICS.csv'),m,boot,'ALL_MOUSE')
    delta_checks(c,read('results/mouse/M3_ALL_MOUSE_PAIRED_MODEL_DELTAS.csv'),points,boot,'ALL_MOUSE')
    strata=bootstrap_strata(m) if do_ci else None
    metrics=read('results/mouse/M4_STRATUM_MODEL_METRICS.csv');deltas=read('results/mouse/M4_STRATUM_PAIRED_MODEL_DELTAS.csv')
    stratum_points={}
    for scope in STRATA:
        frame=m[m.ortholog_status.eq(scope)]
        b=None if strata is None else strata[strata.ortholog_status.eq(scope)]
        pt=model_checks(c,metrics[metrics.scope.eq(scope)],frame,b,scope)
        delta_checks(c,deltas[deltas.scope.eq(scope)],pt,b,scope);stratum_points[scope]=pt
    for row in read('results/mouse/M4_FUSION_GAIN_HETEROGENEITY.csv').itertuples():
        a,b,suffix=CONTRASTS[row.contrast];j=['spearman','pearson'].index(row.metric)
        point=stratum_points[STRATA[0]][a][j]-stratum_points[STRATA[0]][b][j]-stratum_points[STRATA[1]][a][j]+stratum_points[STRATA[1]][b][j]
        c.add('heterogeneity/'+row.contrast+'/'+row.metric,'POINT_RECALCULATED',point,row.heterogeneity_point)
        if strata is not None:
            values=[strata.loc[strata.ortholog_status.eq(s),f'delta_{row.metric}_{suffix}'].to_numpy() for s in STRATA]
            dist=values[0]-values[1]
            for name,value in [('ci_lower',np.quantile(dist,.025)),('ci_upper',np.quantile(dist,.975)),('bootstrap_median',np.median(dist))]:
                c.add('heterogeneity/'+row.contrast+'/'+row.metric+'/'+name,'CI_RECALCULATED',value,getattr(row,'heterogeneity_'+name))
    h=read('data/human/oof.csv')
    ref=read('results/mouse/M3_HUMAN_MOUSE_LM_XGB_COMPLEMENTARITY.csv')
    for scope,y,lm,xgb in [('HUMAN_POOLED_OOF',h.target,h.LM,h['Common-XGB']),('ALL_MOUSE',m.mouse_HALFLIFE,m.pred_mrnalm_ensemble,m.pred_common_compact_xgb_ensemble)]:
        vals=complementarity_row(scope,y.to_numpy(),lm.to_numpy(),xgb.to_numpy())
        row=ref.loc[ref.domain.eq(scope)]
        if row.empty and scope=='HUMAN_POOLED_OOF':row=ref.loc[~ref.domain.eq('ALL_MOUSE')]
        assert len(row)==1
        for key,value in vals.items():
            if key in row.columns and isinstance(value,(int,float)) and not isinstance(value,bool):
                c.add('complementarity/'+scope+'/'+key,'POINT_RECALCULATED',value,row.iloc[0][key])
    if do_ci:
        boot.to_csv(out/'mouse_bootstrap.csv.gz',index=False)
        strata.to_csv(out/'ortholog_bootstrap.csv.gz',index=False)
    return c.save(out,'mouse primary, ortholog, heterogeneity and complementarity',bootstrap_recomputed=do_ci,bootstrap_resamples=5000 if do_ci else 0,root_seed=42)

def mouse_measurements(out,do_ci,workers):
    c=Checks();m=read('data/mouse/predictions.csv');manifest=read('data/mouse/measurement_manifest.csv').sort_values('stable_experiment_index')
    raw=pd.read_csv(ROOT/'data/mouse/measurements.txt.gz',sep='\t',float_precision='round_trip').set_index('ensembl_gene_id')
    assert manifest.stable_experiment_index.tolist()==list(range(27))
    assert set(manifest.orientation_multiplier)=={1} and not raw.index.has_duplicates
    raw=raw.loc[m.mouse_ENSID];tasks=[];points={};rows=[]
    for r in manifest.itertuples():
        mask=raw[r.measurement_column].notna().to_numpy()
        y=raw.loc[mask,r.measurement_column].to_numpy()*r.orientation_multiplier
        assert len(y)==r.n_model_matched and np.isfinite(y).all()
        pred={k:m.loc[mask,col].to_numpy() for k,col in MODEL_COLUMNS.items()}
        points[r.experiment_id]={k:correlation(v,y) for k,v in pred.items()}
        for model,value in points[r.experiment_id].items():rows.append(dict(experiment_id=r.experiment_id,stable_experiment_index=r.stable_experiment_index,n=len(y),model=model,spearman_rho=value[0],pearson_r=value[1]))
        tasks.append((r.experiment_id,int(r.stable_experiment_index),int(r.bootstrap_seed),y,pred))
    boot_by={}
    if do_ci:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i,b in enumerate(ex.map(bootstrap_experiment,tasks),1):
                boot_by[b.experiment_id.iloc[0]]=b
                print(f'mouse measurement bootstrap {i}/27 complete',flush=True)
    ref=read('results/mouse/M5_PER_EXPERIMENT_MODEL_METRICS.csv')
    for r in ref.itertuples():
        n=len(tasks[r.stable_experiment_index][3]);c.add(r.experiment_id+'/n','COUNT',n,r.n,tolerance=0)
        for j,metric in enumerate(['spearman','pearson']):
            c.add(r.experiment_id+'/'+r.model+'/'+metric,'POINT_RECALCULATED',points[r.experiment_id][r.model][j],getattr(r,'spearman_rho' if j==0 else 'pearson_r'),n)
            if do_ci:
                v=boot_by[r.experiment_id][f'{metric}_{r.model}']
                for suffix,value in [('ci_lower',v.quantile(.025)),('ci_upper',v.quantile(.975)),('bootstrap_median',v.median())]:
                    c.add(r.experiment_id+'/'+r.model+'/'+metric+'_'+suffix,'CI_RECALCULATED',value,getattr(r,metric+'_'+suffix),n)
    delta_ref=read('results/mouse/M5_PER_EXPERIMENT_MODEL_DELTAS.csv')
    for name,frame in delta_ref.groupby('experiment_id',sort=False):delta_checks(c,frame,points[name],boot_by.get(name),name)
    pd.DataFrame(rows).to_csv(out/'measurement_points.csv',index=False)
    if do_ci:pd.concat(boot_by.values(),ignore_index=True).to_csv(out/'measurement_bootstrap.csv.gz',index=False)
    return c.save(out,'all27 mouse measurements',bootstrap_recomputed=do_ci,bootstrap_resamples=5000 if do_ci else 0,seeds='42 + frozen experiment index',measurement_count=27,negative_cases_retained=True)

def contexts(out,do_ci,workers):
    c=Checks();p=read('data/external/contexts.csv.gz');reference=read('results/external/per_context_metrics.csv')
    summaries=read('results/external/per_context_bootstrap_summary.csv')
    assert len(reference)==16 and reference.context_index.tolist()==list(range(16))
    cols={'lm':'mrna_lm_oof','xgb':'xgb_oof','fusion':'fusion_oof'}
    jobs=[(p[p.context_index.eq(i)].copy(),42+i,5000) for i in range(16)]
    boot=[]
    if do_ci:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i,b in enumerate(ex.map(context_task,jobs)):
                boot.append(b);print(f'human context bootstrap {i+1}/16 complete',flush=True)
    rows=[]
    for r in reference.itertuples():
        d=jobs[r.context_index][0];assert len(d)==r.n_model_matched and d.ENSID.is_unique
        vals={}
        for model,col in cols.items():
            scores=correlation(d[col].to_numpy(),d.context_target.to_numpy())
            for j,metric in enumerate(['spearman','pearson']):
                vals[model+'_'+metric]=scores[j]
                c.add(f'context{r.context_index}/{model}/{metric}','POINT_RECALCULATED',scores[j],getattr(r,model+'_'+metric),len(d))
        for a,b in [('fusion','lm'),('fusion','xgb'),('xgb','lm')]:
            name=a+'_minus_'+b;vals[name]=vals[a+'_spearman']-vals[b+'_spearman']
            c.add(f'context{r.context_index}/{name}','POINT_RECALCULATED',vals[name],getattr(r,name),len(d))
        rows.append(dict(context_index=r.context_index,n=len(d),**vals))
    if do_ci:
        for r in summaries.itertuples():
            v=boot[r.context_index][r.quantity]
            for name,value in [('bootstrap_median',v.median()),('ci95_low',v.quantile(.025)),('ci95_high',v.quantile(.975)),('fraction_greater_than_zero',(v>0).mean())]:
                c.add(f'context{r.context_index}/{r.quantity}/{name}','CI_RECALCULATED',value,getattr(r,name))
        pd.concat([b.assign(context_index=i) for i,b in enumerate(boot)]).to_csv(out/'context_bootstrap.csv.gz',index=False)
    pd.DataFrame(rows).to_csv(out/'context_points.csv',index=False)
    return c.save(out,'all16 human contexts',bootstrap_recomputed=do_ci,bootstrap_resamples=5000 if do_ci else 0,seeds='42 + frozen context index')

def context_task(args):return bootstrap_context(*args)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',choices=['mouse','measurements','contexts'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--points-only',action='store_true',help='Explicitly omit bootstrap; report cannot claim CI recalculation.')
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();assert 1<=a.workers<=4
    check_inputs();out=new_output(a.output)
    if a.scope=='mouse':mouse_primary(out,not a.points_only)
    elif a.scope=='measurements':mouse_measurements(out,not a.points_only,a.workers)
    else:contexts(out,not a.points_only,a.workers)
    check_inputs()

if __name__=='__main__':main()
