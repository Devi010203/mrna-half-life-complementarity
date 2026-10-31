"""Explicit separation of recalculated points, recalculated CIs and saved evidence."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

class Checks:
    def __init__(self):self.rows=[]
    def add(self,label,kind,actual,expected,n=None,tolerance=1e-10):
        actual,expected=float(actual),float(expected)
        error=abs(actual-expected)
        ok=bool(np.isfinite(actual) and np.isfinite(expected) and error<=tolerance)
        self.rows.append(dict(check=label,kind=kind,n=n,recomputed=actual,stored=expected,
                              absolute_difference=error,tolerance=tolerance,passed=ok))
        if not ok:raise AssertionError(self.rows[-1])
    def save(self,out,scope,**extra):
        pd.DataFrame(self.rows).to_csv(Path(out)/'consistency_checks.csv',index=False)
        result=dict(status='PASS',scope=scope,checks=len(self.rows),
            point_checks=sum(r['kind']=='POINT_RECALCULATED' for r in self.rows),
            ci_checks=sum(r['kind']=='CI_RECALCULATED' for r in self.rows),
            saved_reference_checks=sum(r['kind']=='SAVED_REFERENCE_VERIFIED' for r in self.rows),
            maximum_absolute_difference=max((r['absolute_difference'] for r in self.rows),default=0.),**extra)
        (Path(out)/'REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return result
