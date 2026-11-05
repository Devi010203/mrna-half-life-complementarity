#!/usr/bin/env python3
"""Reproduce the frozen GSE336417 evaluation and independently check every draw."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from inputs import ROOT,output_path,verify_inputs

def source_check():
    source=ROOT/'data/organoid/sources'
    geo=pd.read_csv(source/'GSE336417_SLAMseq_halflife.csv.gz',keep_default_na=False)
    sheet=pd.read_excel(source/'41586_2026_10890_MOESM7_ESM.xlsx',sheet_name='Supplementary Table 5e',header=1,keep_default_na=False)
    assert len(geo)==len(sheet)==5888
    assert geo.gene_name.is_unique and sheet.gene_name.is_unique
    assert geo.gene_name.ne('').all() and geo.gene_name.eq(geo.gene_name.str.strip()).all()
    assert set(geo.gene_name)==set(sheet.gene_name)
    sheet=sheet.set_index('gene_name').loc[geo.gene_name]
    assert np.allclose(geo.hl_exp_ctrl,sheet.hl_exp_ctrl,rtol=1e-9,atol=1e-9)
    assert np.allclose(geo.r2_exp_ctrl,sheet.r2_exp_ctrl,rtol=0,atol=5.1e-10)
    qc=np.isfinite(geo.hl_exp_ctrl)&(geo.hl_exp_ctrl>0)&np.isfinite(geo.r2_exp_ctrl)&geo.r2_exp_ctrl.between(0,1)&(geo.r2_exp_ctrl>.6)
    assert qc.sum()==5369
    assert np.array_equal(geo.r2_exp_ctrl>.6,sheet.r2_exp_ctrl>.6)
    return {'published_n':5888,'control_qc_n':5369,'identical_control_QC_membership':True,
            'half_life_max_abs_difference_hours':float(np.max(abs(geo.hl_exp_ctrl.to_numpy()-sheet.hl_exp_ctrl.to_numpy()))),
            'fit_max_abs_difference':float(np.max(abs(geo.r2_exp_ctrl.to_numpy()-sheet.r2_exp_ctrl.to_numpy()))),
            'peak_calling_and_decay_fitting_repeated':False}

def compare_reference(out):
    reference=ROOT/'results/organoid'; errors={}
    for ref in sorted(reference.glob('*.csv')):
        a=pd.read_csv(ref,float_precision='round_trip')
        b=pd.read_csv(out/ref.name,float_precision='round_trip')
        pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=0,atol=5e-12)
    for name in ['primary','three_targets']:
        a=np.load(reference/f'{name}_bootstrap.npz',allow_pickle=False)
        b=np.load(out/f'{name}_bootstrap.npz',allow_pickle=False)
        assert set(a.files)==set(b.files)
        errors[name]={}
        for key in a.files:
            assert a[key].shape==b[key].shape
            error=float(np.max(abs(a[key]-b[key])))
            assert error<5e-12,(name,key,error)
            errors[name][key]=error
    old=json.loads((reference/'RUN_COMPLETE.json').read_text())
    new=json.loads((out/'RUN_COMPLETE.json').read_text())
    for key in ['primary_bootstrap','three_target_bootstrap']:
        assert old[key]==new[key]
    assert new['fold_positive']==5 and new['fold_negative']==0
    primary=pd.read_csv(out/'Primary_deltas.csv').iloc[0]
    assert [f'{primary[c]:.6f}' for c in ['estimate','ci_low','ci_high']]==['0.017859','0.012561','0.023250']
    return {'all_reference_CSVs_checked':len(list(reference.glob('*.csv'))),
            'bootstrap_reference_max_absolute_errors':errors,'identical_index_stream_hashes':True,
            'primary_n':4890,'three_target_n':4801,'positive_folds':5,
            'primary_delta':primary['estimate'],'primary_ci':[primary.ci_low,primary.ci_high]}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default=None)
    args=ap.parse_args()
    out=output_path(args.output or 'outputs/organoid_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    start=time.monotonic(); n=verify_inputs(); source=source_check()
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    for script,option in [('run_evaluation.py','--output'),('independent_recheck.py','--results')]:
        subprocess.run([sys.executable,'-B',str(Path(__file__).parent/script),option,str(out)],env=env,check=True)
    comparison=compare_reference(out)
    independent=json.loads((out/'independent_checks/INDEPENDENT_STATISTICAL_RECHECK.json').read_text())
    assert independent['status']=='PASS'
    verify_inputs()
    report={'status':'PASS','elapsed_seconds':time.monotonic()-start,'input_hashes_verified':n,
            'source_correspondence':source,'reference_comparison':comparison,
            'independent_all_draws_recalculated':10000,'training_inference_or_fitting':False,
            'implementation_scope':'Independent implementation, same pinned environment; not biological replication.'}
    (out/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
