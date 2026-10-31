"""Generate main Tables 1–3 from newly calculated statistics, then verify cells."""
import argparse
import json
from pathlib import Path
import pandas as pd
from release_paths import ROOT,new_output

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True);a=ap.parse_args()
    run=(ROOT/a.run).resolve();out=new_output(run/'tables')
    fold=pd.read_csv(run/'unified/human_five_fold_summary.csv').set_index(['model','metric'])
    ext=pd.read_csv(run/'unified/external_human_metrics.csv').set_index(['model','metric'])
    pooled=pd.read_csv(run/'unified/human_pooled_oof_metrics.csv').set_index(['model','metric'])
    mouse=pd.read_csv(run/'mouse/consistency_checks.csv').set_index('check')
    labels={'BASIC':'BASIC XGB','Common-XGB':'Common-Compact XGB'}
    t1=[];t2=[];t3a=[];t3b=[]
    for model in ['BASIC','LM','Bio-XGB','Bio-Fusion']:
        r=fold.loc[(model,'r2')]
        t1.append(dict(model=labels.get(model,model),n=12981,r2_mean=r['mean'],r2_population_SD=r.population_sd,
                       mean_spearman=fold.loc[(model,'spearman'),'mean'],mean_pearson=fold.loc[(model,'pearson'),'mean']))
    for model in ['LM','Bio-XGB','Bio-Fusion','Common-XGB','Common-Fusion']:
        s=ext.loc[(model,'spearman')];p=ext.loc[(model,'pearson')]
        t2.append(dict(model=labels.get(model,model),n=int(s.n),spearman=s.estimate,ci_low=s.ci95_low,ci_high=s.ci95_high,pearson=p.estimate))
    for model in ['LM','Common-XGB','Common-Fusion']:
        s=pooled.loc[(model,'spearman')];p=pooled.loc[(model,'pearson')];r=pooled.loc[(model,'r2')]
        t3a.append(dict(model=labels.get(model,model),n=int(s.n),spearman=s.estimate,ci_low=s.ci95_low,ci_high=s.ci95_high,pearson=p.estimate,r2=r.estimate))
    for model,label in [('mrnalm','LM'),('common_compact_xgb','Common-Compact XGB'),('common_fusion','Common-Fusion')]:
        def value(key):return mouse.loc['ALL_MOUSE/'+model+'/'+key,'recomputed']
        t3b.append(dict(model=label,n=13744,spearman=value('spearman'),ci_low=value('spearman_ci_lower'),ci_high=value('spearman_ci_upper'),
                       pearson=value('pearson'),pearson_ci_low=value('pearson_ci_lower'),pearson_ci_high=value('pearson_ci_upper')))
    tables={'Table1':t1,'Table2':t2,'Table3A':t3a,'Table3B':t3b}
    saved=json.loads((ROOT/'data/sources/main_table_cells.json').read_text())
    checks=[];markdown=['# Reproduced main tables','', 'Numerical values below are from this run, not copied from the manuscript.','']
    for name,rows in tables.items():
        d=pd.DataFrame(rows);d.to_csv(out/(name+'.csv'),index=False)
        markdown += ['## '+name,'','| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']
        for row in rows:
            markdown.append('| '+' | '.join(str(v) if isinstance(v,(str,int)) else f'{v:.9f}' for v in row.values())+' |')
            precision=4 if name=='Table1' else 6
            values=[v for k,v in row.items() if k not in {'model','n'}]
            displayed=[f'{v:.{precision}f}' for v in values]
            assert displayed==saved[name][row['model']],(name,row['model'],displayed,saved[name][row['model']])
            checks.append({'table':name,'model':row['model'],'numeric_cells':len(displayed),'matches_manuscript':True})
        markdown.append('')
    (out/'MAIN_TABLES.md').write_text('\n'.join(markdown)+'\n')
    report={'status':'PASS','tables':['Table1','Table2','Table3A','Table3B'],'rows':len(checks),'numeric_cells':sum(r['numeric_cells'] for r in checks),'checks':checks,'source':'newly recalculated statistics'}
    (out/'MAIN_TABLE_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
