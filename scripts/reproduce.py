#!/usr/bin/env python3
"""CPU-only reproduction of saved-prediction statistics, main tables and figures."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from release_paths import ROOT,check_inputs,new_output,sha

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',help='New directory below outputs/; default uses a unique UTC timestamp.')
    ap.add_argument('--workers',type=int,default=2,choices=range(1,5),help='CPU workers for contexts/measurements; BLAS threads fixed to one.')
    ap.add_argument('--core-only',action='store_true',help='Recompute main CIs, but only points for27 measurements/16 contexts; explicitly report their CIs as saved LEVEL1.')
    a=ap.parse_args();start=time.monotonic()
    name=a.output or 'outputs/reproduction_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=new_output(name);logdir=out/'logs';logdir.mkdir()
    frozen=check_inputs()
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env['MPLCONFIGDIR']=str(out/'cache/matplotlib');env['XDG_CACHE_HOME']=str(out/'cache')
    journal=[]
    def run(label,arguments):
        command=[sys.executable,*arguments];begin=time.monotonic()
        print(f'[{label}] started',flush=True)
        with (logdir/(label+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        entry=dict(step=label,command=['python',*arguments],exit_code=result.returncode,seconds=time.monotonic()-begin,log='logs/'+label+'.log')
        journal.append(entry)
        (out/'EXECUTION.json').write_text(json.dumps(journal,indent=2)+'\n')
        if result.returncode:
            print((logdir/(label+'.log')).read_text()[-5000:])
            raise RuntimeError(f'{label} failed; see {out.relative_to(ROOT)}/logs/{label}.log')
        print(f'[{label}] PASS ({entry["seconds"]:.1f}s)',flush=True)
    rel=out.relative_to(ROOT).as_posix()
    try:
        run('input_structure',['tests/test_release.py'])
        run('human_external',['scripts/statistics/human_external.py','--output',rel+'/unified'])
        run('unified_independent',['scripts/statistics/verify_unified.py','--results',rel+'/unified','--report',rel+'/UNIFIED_INDEPENDENT.json'])
        run('unified_reference',['scripts/check_results.py','--run',rel])
        run('mouse_ortholog',['scripts/statistics/mouse_context.py','--scope','mouse','--output',rel+'/mouse'])
        for scope in ['measurements','contexts']:
            run(scope,['scripts/statistics/mouse_context.py','--scope',scope,'--output',rel+'/'+scope,'--workers',str(a.workers)]+(['--points-only'] if a.core_only else []))
        run('distillation',['scripts/statistics/distillation.py','--output',rel+'/distillation'])
        run('organoid',['scripts/organoid/reproduce.py','--output',rel+'/organoid'])
        run('main_tables',['scripts/main_tables.py','--run',rel])
        run('supplementary',['scripts/tables.py','--output',rel+'/supplementary'])
        run('figures',['scripts/figures.py','--output-dir',rel+'/figures'])
        check_inputs()
        result=dict(status='PASS',seconds=time.monotonic()-start,frozen_inputs_checked=frozen,steps=journal,
             training=False,inference=False,model_fitting=False,external_upload=False,
             point_recalculation=['Tables1–3','human/external Bio and Common','mouse primary and ortholog','27 measurements','16 contexts','PD4 primary and secondary','GSE336417 primary, three targets, folds and coverage'],
             bootstrap_recalculation=['human/external','mouse primary','ortholog heterogeneity','PD4 primary and secondary','GSE336417 primary and three targets, independently verified']+([] if a.core_only else ['27 measurements','16 contexts']),
             stored_only=['Full24-system TableS1 beyond included OOF controls','full ablation/permutation runs','fusion coefficient refitting','Common-Full sensitivity','curated provenance S12','External-PC1 R construction (separate command)','balanced/no-QN sensitivity statistics','ortholog context/target-concordance diagnostics']+(['27-measurement/16-context CIs'] if a.core_only else []),
             scientific_script_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'scripts').rglob('*.py'))})
        (out/'REPRODUCTION_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PASS: '+rel+'/REPRODUCTION_REPORT.json',flush=True)
    except Exception as exc:
        (out/'FAILED.json').write_text(json.dumps({'status':'FAIL','error':str(exc),'steps':journal},indent=2)+'\n')
        raise

if __name__=='__main__':main()
