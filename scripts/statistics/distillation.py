"""Recompute the frozen PD4 matched estimands from the canonical long table."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from release_paths import ROOT,new_output,check_inputs
from checks import Checks
from pd4_estimands import matched_estimands,ci_summary
from pd4_evaluation import SEEDS,ARM_NAMES,METRICS,DELTAS,SCOPES,calibration,secondary,deltas,decisions,bootstrap

def read(name):return pd.read_csv(ROOT/name,float_precision='round_trip')

def paired_arrays():
    d=read('data/distillation/predictions.csv');h=read('data/human/oof.csv')
    assert len(d)==116829 and not d.duplicated(['ENSID','seed','arm']).any()
    assert d.groupby('ENSID').size().eq(9).all()
    assert set(d.seed)==set(SEEDS) and set(d.arm)==set(ARM_NAMES)
    genes=h.ENSID.tolist();pred=np.empty((len(genes),3,3))
    y=folds=None
    for s,seed in enumerate(SEEDS):
        for a,arm in enumerate(ARM_NAMES):
            f=d[d.seed.eq(seed)&d.arm.eq(arm)].set_index('ENSID').loc[genes]
            assert len(f)==12981 and f.index.is_unique
            if y is None:y=f.y_true.to_numpy();folds=f.fold.to_numpy()
            else:assert np.array_equal(f.y_true.to_numpy(),y) and np.array_equal(f.fold.to_numpy(),folds)
            pred[:,s,a]=f.prediction
    assert np.array_equal(folds,h.fold) and np.isfinite(pred).all()
    return y,pred,folds

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True)
    ap.add_argument('--points-only',action='store_true');args=ap.parse_args()
    check_inputs();out=new_output(args.output);c=Checks()
    y,pred,folds=paired_arrays();point=matched_estimands(y,pred,folds)
    scores=[];cal=[]
    for f in range(5):
        mask=folds==f
        for s,seed in enumerate(SEEDS):
            for a,arm in enumerate(ARM_NAMES):
                keys=dict(fold=f,seed=seed,arm=arm,n=int(mask.sum()))
                scores.append(dict(keys,**dict(zip(METRICS,point['fold_seed_arm_scores'][f,s,a]))))
                cal.append(dict(keys,**calibration(y[mask],pred[mask,s,a])))
    score_frame=pd.DataFrame(scores);cal_frame=pd.DataFrame(cal)
    ref=read('results/distillation/fold_seed_arm_metrics.csv').set_index(['fold','seed','arm'])
    for row in score_frame.itertuples():
        old=ref.loc[(row.fold,row.seed,row.arm)]
        for metric in METRICS:c.add(f'{row.fold}/{row.seed}/{row.arm}/{metric}','POINT_RECALCULATED',getattr(row,metric),old[metric],row.n)
        c.add(f'{row.fold}/{row.seed}/{row.arm}/n','COUNT',row.n,old.n,tolerance=0)
    score_frame.to_csv(out/'fold_seed_arm_metrics.csv',index=False)
    cal_frame.to_csv(out/'calibration_diagnostics.csv',index=False)
    primary=read('results/distillation/matched_delta_summary.csv')
    for row in primary.itertuples():
        d,m=DELTAS.index(row.delta),METRICS.index(row.metric)
        for name,value in [('estimate',point['overall_matched_delta'][d,m]),('fold_population_SD',point['fold_population_SD'][d,m]),('seed_population_SD',point['seed_population_SD'][d,m])]:
            c.add(row.delta+'/'+row.metric+'/'+name,'POINT_RECALCULATED',value,getattr(row,name),12981)
    sec=secondary(y,pred);delta_sec=deltas(sec)
    for row in read('results/distillation/secondary_oof_metrics.csv').itertuples():
        c.add(row.scope+'/'+row.arm+'/'+row.metric,'POINT_RECALCULATED',sec[SCOPES.index(row.scope),ARM_NAMES.index(row.arm),METRICS.index(row.metric)],row.estimate,12981)
    for row in read('results/distillation/secondary_oof_paired_deltas.csv').itertuples():
        c.add(row.scope+'/'+row.delta+'/'+row.metric,'POINT_RECALCULATED',delta_sec[SCOPES.index(row.scope),DELTAS.index(row.delta),METRICS.index(row.metric)],row.estimate,12981)
    if not args.points_only:
        primary_boot,secondary_boot,index_hash=bootstrap(y,pred,folds,n=5000)
        expected=json.loads((ROOT/'data/distillation/pairing.json').read_text())
        assert index_hash==expected['paired_index_stream_sha256'],'Bootstrap index order changed'
        p_ci=ci_summary(primary_boot);s_ci=ci_summary(secondary_boot);d_ci=ci_summary(deltas(secondary_boot))
        for row in primary.itertuples():
            d,m=DELTAS.index(row.delta),METRICS.index(row.metric)
            for name in ['ci_low','ci_high','fraction_delta_gt_zero']:
                c.add(row.delta+'/'+row.metric+'/'+name,'CI_RECALCULATED',p_ci[name][d,m],getattr(row,name),12981)
        for row in read('results/distillation/secondary_oof_metrics.csv').itertuples():
            ix=(SCOPES.index(row.scope),ARM_NAMES.index(row.arm),METRICS.index(row.metric))
            for name in ['ci_low','ci_high']:c.add(row.scope+'/'+row.arm+'/'+row.metric+'/'+name,'CI_RECALCULATED',s_ci[name][ix],getattr(row,name),12981)
        for row in read('results/distillation/secondary_oof_paired_deltas.csv').itertuples():
            ix=(SCOPES.index(row.scope),DELTAS.index(row.delta),METRICS.index(row.metric))
            for name in ['ci_low','ci_high','fraction_delta_gt_zero']:c.add(row.scope+'/'+row.delta+'/'+row.metric+'/'+name,'CI_RECALCULATED',d_ci[name][ix],getattr(row,name),12981)
        # Full distribution comparison is stronger than comparing only CI endpoints.
        with np.load(ROOT/'results/distillation/bootstrap_draws.npz',allow_pickle=False) as old:
            c.add('primary_bootstrap_array_max_error','CI_RECALCULATED',np.max(np.abs(primary_boot-old['primary_delta'])),0,tolerance=1e-10)
            c.add('secondary_bootstrap_array_max_error','CI_RECALCULATED',np.max(np.abs(secondary_boot-old['secondary_scores'])),0,tolerance=1e-10)
        decision=decisions(point,p_ci,cal_frame)
        old=json.loads((ROOT/'results/distillation/registered_decision.json').read_text())
        for k,v in decision.items():assert v==old[k],('decision mismatch',k)
        (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
        np.savez_compressed(out/'bootstrap_draws.npz',primary_delta=primary_boot,secondary_scores=secondary_boot)
        result_rows=[]
        for d,name in enumerate(DELTAS):
            for m,metric in enumerate(METRICS):
                result_rows.append(dict(delta=name,metric=metric,estimate=point['overall_matched_delta'][d,m],ci_low=p_ci['ci_low'][d,m],ci_high=p_ci['ci_high'][d,m]))
        pd.DataFrame(result_rows).to_csv(out/'matched_delta_summary.csv',index=False)
    check_inputs()
    c.save(out,'PD4 matched primary and secondary estimands',n_genes=12981,runs=45,rows=116829,
           bootstrap_recomputed=not args.points_only,bootstrap_resamples=0 if args.points_only else 5000,seed=42,
           gene_order='Canonical human ENSID order, identical to original paired_arrays',practical_threshold=0.010,
           biochemical_specific_threshold=0.005,training=False,inference=False)

if __name__=='__main__':main()
