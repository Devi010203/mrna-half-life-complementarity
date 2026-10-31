"""Compare recalculated unified outputs, approved figure data and ESM sources."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from release_paths import ROOT,check_inputs

def check_unified(out):
    checks=[]
    for source in sorted((ROOT/'results/unified').glob('*.csv')):
        got=pd.read_csv(out/source.name,float_precision='round_trip')
        old=pd.read_csv(source,float_precision='round_trip')
        assert got.shape==old.shape and list(got)==list(old),source.name
        for col in old:
            if pd.api.types.is_numeric_dtype(old[col]):
                a,b=got[col].to_numpy(float),old[col].to_numpy(float)
                assert np.allclose(a,b,rtol=0,atol=1e-12,equal_nan=True),(source.name,col)
                error=float(np.nanmax(np.abs(a-b))) if np.isfinite(a-b).any() else 0.
            else:
                assert got[col].fillna('').equals(old[col].fillna('')),(source.name,col)
                error=0.
            checks.append(dict(file=source.name,column=col,max_abs_difference=error))
    for source in sorted((ROOT/'results/unified').glob('*.npz')):
        with np.load(source,allow_pickle=False) as old,np.load(out/source.name,allow_pickle=False) as got:
            assert old.files==got.files
            for key in old.files:
                if np.issubdtype(old[key].dtype,np.number):
                    assert np.allclose(got[key],old[key],atol=1e-12,rtol=0),(source.name,key)
                    error=float(np.max(np.abs(got[key]-old[key])))
                else:assert np.array_equal(got[key],old[key]);error=0.
                checks.append(dict(file=source.name,column=key,max_abs_difference=error))
    return dict(status='PASS',checks=checks,comparisons=len(checks),maximum_absolute_difference=max(r['max_abs_difference'] for r in checks),mode='recalculated outputs compared to frozen references; distributions included')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True);a=ap.parse_args()
    run=(ROOT/a.run).resolve();report=check_unified(run/'unified')
    (run/'UNIFIED_REFERENCE_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    check_inputs();print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
