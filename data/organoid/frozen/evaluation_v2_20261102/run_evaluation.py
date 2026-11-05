"""Frozen fixed-OOF evaluation; no training, inference, calibration or selection."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr, pearsonr

ROOT=Path(__file__).resolve().parent
V1=ROOT.parent / "preflight_20261102"
EV=ROOT.parents[1]
PUBLIC=EV / "release_preparation/github_migration_20261031/execution/working_repository"
MODELS=["LM","Bio-XGB","Bio-Fusion","Common-XGB","Common-Fusion"]
TARGETS=["Saluki","External-PC1","Organoid"]
CONTRASTS=[("Bio-Fusion","Bio-XGB","PRIMARY"),
           ("Bio-Fusion","LM","SECONDARY"),
           ("Common-Fusion","Common-XGB","SECONDARY"),
           ("Common-Fusion","LM","SECONDARY"),
           ("Bio-XGB","LM","SECONDARY_V2")]
B=5000


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def verify_inputs():
    record=json.loads((ROOT / "PROTOCOL_V2_FREEZE.json").read_text())
    for row in record["files"]:
        assert digest(ROOT.parent / row["path"])==row["sha256"]
    subprocess.run([sys.executable,"-I","-B",str(V1 / "verify_and_freeze.py")],check=True)


def correlation(a):
    centered=a-a.mean(axis=0)
    ss=np.einsum("ij,ij->j",centered,centered)
    if np.any(ss<=0) or not np.isfinite(ss).all():
        raise ValueError("Undefined correlation: constant or nonfinite sampled column; no draw replacement allowed")
    cross=np.einsum("ij,ik->jk",centered,centered)
    return cross/np.sqrt(np.outer(ss,ss))


def matrices(raw,linear):
    return correlation(rankdata(raw,axis=0,method="average")),correlation(linear)


def self_test():
    a=np.array([[1,3,1],[1,2,4],[3,2,2],[7,1,3],[5,9,8]],dtype=float)
    for ix in [np.arange(5),np.array([0,0,2,4,2])]:
        x=a[ix]
        sr,pr=matrices(x,x)
        for i,j in combinations(range(3),2):
            assert abs(sr[i,j]-spearmanr(x[:,i],x[:,j]).statistic)<1e-12
            assert abs(pr[i,j]-pearsonr(x[:,i],x[:,j]).statistic)<1e-12
    try:
        correlation(np.ones((4,2)))
    except ValueError:
        pass
    else:
        raise AssertionError("Constant-draw test failed")
    print("PASS: synthetic ties, duplicated draws, Pearson/Spearman and undefined-draw tests")


def bootstrap(raw,linear,name,out,log):
    n,k=raw.shape
    assert np.isfinite(raw).all() and np.isfinite(linear).all()
    observed_s,observed_p=matrices(raw,linear)
    draws_s=np.empty((B,k,k)); draws_p=np.empty_like(draws_s)
    rng=np.random.default_rng(42)
    stream=sha256()
    for b in range(B):
        idx=rng.integers(0,n,size=n)
        stream.update(idx.astype("<i8",copy=False).tobytes())
        try:
            draws_s[b],draws_p[b]=matrices(raw[idx],linear[idx])
        except ValueError:
            np.save(out / f"{name}_undefined_draw_indices.npy",idx,allow_pickle=False)
            raise
        if (b+1)%1000==0:
            log(f"{name}: {b+1}/{B} paired bootstrap draws")
    assert np.isfinite(draws_s).all() and np.isfinite(draws_p).all()
    np.savez_compressed(out / f"{name}_bootstrap.npz",spearman=draws_s,pearson=draws_p,
                        point_spearman=observed_s,point_pearson=observed_p)
    meta=dict(n=n,replicates=B,seed=42,index_stream_sha256=stream.hexdigest(),
              index_encoding="concatenated little-endian signed int64 arrays in draw order",
              undefined_draws=0,discarded_draws=0,replaced_draws=0)
    return observed_s,observed_p,draws_s,draws_p,meta


def interval(a):
    lo,hi=np.quantile(a,[.025,.975],method="linear")
    return dict(ci_low=float(lo),ci_high=float(hi))


def prepare():
    h=pd.read_csv(PUBLIC / "data/human/oof.csv",float_precision="round_trip")
    e=pd.read_csv(PUBLIC / "data/external/matched.csv",float_precision="round_trip")
    f=pd.read_csv(PUBLIC / "data/human/coverage_features.csv",float_precision="round_trip")
    p=pd.read_csv(V1 / "audit/primary_gene_manifest.csv",float_precision="round_trip")
    t=pd.read_csv(V1 / "audit/three_target_gene_manifest.csv",float_precision="round_trip")
    assert h.ENSID.is_unique and e.ENSID.is_unique and f.ENSID.is_unique
    assert len(h)==12981 and len(e)==11381 and len(p)==4890 and len(t)==4801
    joined=p[["ENSID","hl_exp_ctrl","r2_exp_ctrl"]].merge(h,on="ENSID",validate="one_to_one",sort=False)
    assert joined.ENSID.tolist()==p.ENSID.tolist()==sorted(p.ENSID)
    assert joined.fold.astype(int).tolist()==p.fold.astype(int).tolist()
    assert joined.gene_symbol.tolist()==p.gene_name.tolist()
    assert np.isfinite(joined[MODELS+["target","hl_exp_ctrl","r2_exp_ctrl"]]).all().all()
    assert joined.hl_exp_ctrl.gt(0).all() and joined.r2_exp_ctrl.gt(.6).all()
    assert joined.r2_exp_ctrl.le(1).all() and joined.hl_exp_ctrl.gt(12).sum()==251
    joined=joined.rename(columns={"target":"Saluki","hl_exp_ctrl":"Organoid"})
    joined["log2_half_life"]=np.log2(joined.Organoid)
    joined["half_life_gt_12h"]=joined.Organoid.gt(12)
    joined=joined[["ENSID","gene_symbol","fold","Saluki","Organoid","log2_half_life","r2_exp_ctrl","half_life_gt_12h"]+MODELS]
    common=joined.set_index("ENSID").loc[t.ENSID].reset_index()
    ep=e.set_index("ENSID").loc[t.ENSID]
    assert np.allclose(common.Saluki,ep.original_target,rtol=0,atol=2e-12)
    for model in MODELS:
        assert np.allclose(common[model],ep[model],rtol=0,atol=2e-12),model
    assert common.fold.tolist()==ep.fold.tolist()
    common["External-PC1"]=ep.target.to_numpy()
    assert np.isfinite(common[TARGETS]).all().all()
    hf=h.merge(f.drop(columns=["gene_symbol","fold","target"]),on="ENSID",validate="one_to_one")
    fp=f.set_index("ENSID").loc[h.ENSID]
    assert np.allclose(h.target,fp.target,rtol=0,atol=2e-12)
    assert np.array_equal(h.fold,fp.fold)
    g=pd.read_csv(V1 / "sources/GSE336417_SLAMseq_halflife.csv.gz")
    all_symbols=set(g.gene_name)
    ids=set(joined.ENSID); cids=set(common.ENSID)
    hf["coverage_group"]=np.select([hf.ENSID.isin(ids),hf.gene_symbol.isin(all_symbols)],
                                   ["PRIMARY_INCLUDED","PUBLISHED_QC_FAIL"],default="NO_EXACT_PUBLISHED_MATCH")
    hf["has_external_pc1"]=hf.ENSID.isin(e.ENSID)
    groups={"ALL_HUMAN":np.ones(len(hf),dtype=bool),
            "PRIMARY_INCLUDED":hf.ENSID.isin(ids),
            "PUBLISHED_QC_FAIL":hf.coverage_group.eq("PUBLISHED_QC_FAIL"),
            "NO_EXACT_PUBLISHED_MATCH":hf.coverage_group.eq("NO_EXACT_PUBLISHED_MATCH"),
            "NOT_PRIMARY":~hf.ENSID.isin(ids),
            "THREE_TARGETS":hf.ENSID.isin(cids),
            "PRIMARY_NO_EXTERNAL_PC1":hf.ENSID.isin(ids-cids)}
    assert [int(v.sum()) for v in groups.values()]==[12981,4890,425,7666,8091,4801,89]
    return joined,common,hf,groups


def main():
    verify_inputs(); self_test()
    out=ROOT / "results"
    out.mkdir(exist_ok=False)
    start=datetime.now(timezone.utc).isoformat(); tic=time.monotonic()
    def log(message):
        line=f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(line,flush=True)
        with (out / "run.log").open("a") as f:
            f.write(line+"\n")
    code_paths=[Path(__file__),ROOT / "independent_recheck.py"]
    code_hashes={p.name:digest(p) for p in code_paths}
    (out / "RUN_START.json").write_text(json.dumps(dict(started_utc=start,
        protocol_freeze_sha256=digest(ROOT / "PROTOCOL_V2_FREEZE.json"),code_sha256=code_hashes,
        software=dict(Python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__)),indent=2)+"\n")
    primary,common,human,groups=prepare()
    log("All preflight hashes, joined IDs, saved predictions and targets verified")
    primary.to_csv(out / "primary_evaluation_data.csv",index=False)
    common.to_csv(out / "three_target_evaluation_data.csv",index=False)
    human[["ENSID","gene_symbol","fold","coverage_group","has_external_pc1"]].to_csv(out / "coverage_groups.csv",index=False)
    raw=primary[MODELS+["Organoid"]].to_numpy(float)
    linear=raw.copy(); linear[:,-1]=np.log2(linear[:,-1])
    s,p,bs,bp,meta1=bootstrap(raw,linear,"primary",out,log)
    tables={}
    records=[]
    for metric,point,draws in [("Spearman",s,bs),("Pearson_log2_half_life",p,bp)]:
        for i,model in enumerate(MODELS):
            records.append(dict(n=len(primary),model=model,metric=metric,estimate=point[i,5],**interval(draws[:,i,5])))
    tables["Primary_models"]=pd.DataFrame(records)
    records=[]
    for a,b,role in CONTRASTS:
        i,j=MODELS.index(a),MODELS.index(b)
        records.append(dict(n=len(primary),contrast=f"{a} minus {b}",role=role,metric="Delta_Spearman",
            estimate=s[i,5]-s[j,5],**interval(bs[:,i,5]-bs[:,j,5])))
    tables["Primary_deltas"]=pd.DataFrame(records)
    folds=[]
    for fold,df in primary.groupby("fold",sort=True):
        sm,_=matrices(df[["Bio-XGB","Bio-Fusion","Organoid"]].to_numpy(),df[["Bio-XGB","Bio-Fusion","Organoid"]].to_numpy())
        delta=sm[1,2]-sm[0,2]
        folds.append(dict(fold=int(fold),n=len(df),Bio_XGB_Spearman=sm[0,2],Bio_Fusion_Spearman=sm[1,2],
                          Fusion_minus_XGB=delta,direction="positive" if delta>0 else "negative" if delta<0 else "zero"))
    assert len(folds)==5 and sum(r["n"] for r in folds)==4890
    tables["Fold_diagnostics"]=pd.DataFrame(folds)
    raw=common[MODELS+TARGETS].to_numpy(float)
    linear=raw.copy();linear[:,-1]=np.log2(linear[:,-1])
    s,p,bs,bp,meta2=bootstrap(raw,linear,"three_targets",out,log)
    records=[]
    for j,target in enumerate(TARGETS,5):
        for i,model in enumerate(MODELS):
            records.append(dict(n=len(common),target=target,model=model,metric="Spearman",estimate=s[i,j],**interval(bs[:,i,j])))
    tables["Three_target_models"]=pd.DataFrame(records)
    records=[]
    for metric,point,draws in [("Spearman",s,bs),("Pearson_organoid_log2",p,bp)]:
        for i,j in combinations(range(5,8),2):
            records.append(dict(n=len(common),target_a=TARGETS[i-5],target_b=TARGETS[j-5],metric=metric,
                                estimate=point[i,j],**interval(draws[:,i,j])))
    tables["Target_concordance"]=pd.DataFrame(records)
    records=[]
    for j,target in enumerate(TARGETS,5):
        records.append(dict(n=len(common),target=target,contrast="Bio-Fusion minus Bio-XGB",metric="Delta_Spearman",
                            estimate=s[2,j]-s[1,j],**interval(bs[:,2,j]-bs[:,1,j])))
    tables["Three_target_deltas"]=pd.DataFrame(records)
    records=[]
    for j in [5,6]:
        records.append(dict(n=len(common),comparison=f"Organoid gain minus {TARGETS[j-5]} gain",
                            metric="Difference_of_paired_Spearman_gains",
                            estimate=(s[2,7]-s[1,7])-(s[2,j]-s[1,j]),
                            **interval((bs[:,2,7]-bs[:,1,7])-(bs[:,2,j]-bs[:,1,j]))))
    tables["Gain_shift_descriptive"]=pd.DataFrame(records)
    feature_cols=["target","TOTAL_LEN_SEQ","UTR5LEN","CDSLEN","UTR3LEN","UTR5GC","CDSGC","UTR3GC",
                  "n_measurements","n_datasets","n_cell_lines"]
    summaries=[]; coverage=[]; original=[]
    for name,mask in groups.items():
        df=human.loc[mask]
        for col in feature_cols:
            v=df[col].to_numpy(float); v=v[np.isfinite(v)]
            q=np.quantile(v,[.25,.5,.75],method="linear") if len(v) else np.array([np.nan]*3)
            summaries.append(dict(group=name,n=len(df),feature=col,n_valid=len(v),n_missing=len(df)-len(v),
                                  q25=q[0],median=q[1],q75=q[2],iqr=q[2]-q[0]))
        coverage.append(dict(group=name,n=len(df),external_pc1_available=int(df.has_external_pc1.sum()),
                             external_pc1_fraction=float(df.has_external_pc1.mean())))
        sm,_=matrices(df[MODELS+["target"]].to_numpy(),df[MODELS+["target"]].to_numpy())
        for i,model in enumerate(MODELS):
            original.append(dict(group=name,n=len(df),target="original_Saluki_only",model=model,Spearman=sm[i,5]))
    tables["Coverage_features"]=pd.DataFrame(summaries)
    tables["Coverage_counts"]=pd.DataFrame(coverage)
    tables["Coverage_original_OOF"]=pd.DataFrame(original)
    tables["Interpretation_notes"]=pd.DataFrame([
        ("Sole primary","Bio-Fusion minus Bio-XGB Spearman on 4,890 genes"),
        ("Fit quality","R2_exp >0.6 meets authors' control fit criterion, not proof of precise half-life"),
        ("Long estimates","All 251 primary estimates >12 h retained; no clipping or sensitivity selection"),
        ("Bootstrap","5,000 paired gene draws; seed 42; average ranks recalculated; linear percentile CI"),
        ("Uncertainty","Conditional on fixed models and targets; not training, fit or inter-patient uncertainty"),
        ("Secondary","Descriptive; no multiplicity-adjusted confirmatory claims or model reweighting"),
        ("Coverage","Read depth/expression in this SLAM-seq experiment unavailable; old external coverage only"),
        ("Scope","One MSK107Li shCtrl PDO context; not unseen genes or multiple patients")],columns=["item","note"])
    for name,table in tables.items():
        table.to_csv(out / f"{name}.csv",index=False)
    with pd.ExcelWriter(out / "GSE336417_FORMAL_RESULTS.xlsx",engine="openpyxl") as writer:
        for name,table in tables.items():
            table.to_excel(writer,sheet_name=name,index=False)
        for name,table in [("Primary_gene_data",primary),("Three_target_gene_data",common)]:
            table.to_excel(writer,sheet_name=name,index=False)
        for ws in writer.book:
            ws.freeze_panes="A2";ws.auto_filter.ref=ws.dimensions
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width=min(55,max(15,len(str(col[0].value))+2))
    verify_inputs()
    assert all(digest(ROOT / name)==h for name,h in code_hashes.items())
    summary=dict(status="COMPLETED_PENDING_INDEPENDENT_RECHECK",started_utc=start,
                  completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-tic,
                  primary_bootstrap=meta1,three_target_bootstrap=meta2,
                  primary_columns=MODELS+["Organoid"],three_target_columns=MODELS+TARGETS,
                  n_long_half_life_retained=251,fold_positive=sum(r["direction"]=="positive" for r in folds),
                  fold_negative=sum(r["direction"]=="negative" for r in folds),
                  no_training_inference_or_fitting=True)
    (out / "RUN_COMPLETE.json").write_text(json.dumps(summary,indent=2)+"\n")
    log("Evaluation completed; independent recheck required")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
