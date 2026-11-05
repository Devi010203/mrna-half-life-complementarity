"""Independent source reconstruction and full bootstrap/statistical recheck.

Does not import run_evaluation.py. Uses csv/math input handling, pandas ranking
and correlation, and explicitly interpolated sorted bootstrap percentiles.
Recalculates every bootstrap draw for both frozen cohorts, not a spot check.
"""
import argparse
import csv
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import openpyxl
import pandas as pd

ROOT=Path(__file__).resolve().parent
PUBLIC=ROOT.parents[1]
FROZEN=PUBLIC / "data/organoid/frozen"
V1=FROZEN / "preflight_20261102"
SOURCE=PUBLIC / "data/organoid/sources"
from inputs import verify_inputs
MODELS=["LM","Bio-XGB","Bio-Fusion","Common-XGB","Common-Fusion"]
TARGETS=["Saluki","External-PC1","Organoid"]
PAIRS=[("Bio-Fusion","Bio-XGB"),("Bio-Fusion","LM"),("Common-Fusion","Common-XGB"),
       ("Common-Fusion","LM"),("Bio-XGB","LM")]
TOL=5e-12


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def numbers_close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=TOL),(a,b)


def corr(a,rank=False):
    frame=pd.DataFrame(a)
    if rank:
        frame=frame.rank(axis=0,method="average")
    result=frame.corr(method="pearson").to_numpy()
    assert np.isfinite(result).all(),"Undefined independent statistic"
    return result


def qlinear(x,p):
    s=np.sort(np.asarray(x)); position=(len(s)-1)*p
    lower=math.floor(position);upper=math.ceil(position)
    return float(s[lower]+(s[upper]-s[lower])*(position-lower))


def check_row(row,point,draws=None):
    numbers_close(row["estimate"],point)
    if draws is not None:
        numbers_close(row["ci_low"],qlinear(draws,.025))
        numbers_close(row["ci_high"],qlinear(draws,.975))


def main(OUT):
    start=datetime.now(timezone.utc).isoformat()
    audit=OUT / "independent_checks"
    audit.mkdir(exist_ok=False)
    def log(message):
        text=f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(text,flush=True)
        with (audit / "recheck.log").open("a") as f:
            f.write(text+"\n")
    verify_inputs()
    frozen=json.loads((FROZEN / "evaluation_v2_20261102/PROTOCOL_V2_FREEZE.json").read_text())
    for r in frozen["files"]:
        assert digest(FROZEN / r["path"])==r["sha256"]
    execution=json.loads((OUT / "RUN_START.json").read_text())
    assert frozen["frozen_utc"]<execution["started_utc"]<start
    assert execution["protocol_freeze_sha256"]==digest(FROZEN / "evaluation_v2_20261102/PROTOCOL_V2_FREEZE.json")
    for name,hsh in execution["code_sha256"].items():
        assert digest(ROOT / name)==hsh
    saved_complete=json.loads((OUT / "RUN_COMPLETE.json").read_text())
    human=read(PUBLIC / "data/human/oof.csv")
    external=read(PUBLIC / "data/external/matched.csv")
    by_id={r["ENSID"]:r for r in human};by_symbol={r["gene_symbol"]:r for r in human}
    ext={r["ENSID"]:r for r in external}
    assert len(by_id)==len(by_symbol)==12981 and len(ext)==11381
    with gzip.open(SOURCE / "GSE336417_SLAMseq_halflife.csv.gz","rt",encoding="utf-8-sig",newline="") as f:
        geo=list(csv.DictReader(f))
    control={}; all_symbols={r["gene_name"] for r in geo}
    for r in geo:
        hl,fit=float(r["hl_exp_ctrl"]),float(r["r2_exp_ctrl"])
        if math.isfinite(hl) and hl>0 and math.isfinite(fit) and .6<fit<=1 and r["gene_name"] in by_symbol:
            h=by_symbol[r["gene_name"]]
            assert all(math.isfinite(float(h[m])) for m in MODELS)
            control[h["ENSID"]]=(hl,fit)
    ids=sorted(control);shared=sorted(set(ids)&set(ext))
    assert len(ids)==4890 and len(shared)==4801
    assert sum(control[i][0]>12 for i in ids)==251
    assert ids==[r["ENSID"] for r in read(V1 / "audit/primary_gene_manifest.csv")]
    assert shared==[r["ENSID"] for r in read(V1 / "audit/three_target_gene_manifest.csv")]
    for filename,expected_ids in [("primary_evaluation_data.csv",ids),("three_target_evaluation_data.csv",shared)]:
        saved=read(OUT / filename)
        assert [r["ENSID"] for r in saved]==expected_ids
        for r,i in zip(saved,expected_ids):
            h=by_id[i]
            assert int(r["fold"])==int(h["fold"]) and r["gene_symbol"]==h["gene_symbol"]
            for m in MODELS:
                numbers_close(r[m],h[m])
            numbers_close(r["Saluki"],h["target"])
            numbers_close(r["Organoid"],control[i][0]);numbers_close(r["r2_exp_ctrl"],control[i][1])
            numbers_close(r["log2_half_life"],math.log2(control[i][0]))
            assert (r["half_life_gt_12h"]=="True")==bool(control[i][0]>12)
            if "External-PC1" in r:
                numbers_close(r["External-PC1"],ext[i]["target"])
                numbers_close(h["target"],ext[i]["original_target"])
                for m in MODELS:
                    numbers_close(h[m],ext[i][m])
    log("Independent source reconstruction, IDs/order, original values and 251 long estimates PASS")
    store={}; max_errors={}
    for name,ii in [("primary",ids),("three_targets",shared)]:
        x=np.array([[float(by_id[i][m]) for m in MODELS]+([control[i][0]] if name=="primary" else
                    [float(by_id[i]["target"]),float(ext[i]["target"]),control[i][0]]) for i in ii])
        linear=x.copy();linear[:,-1]=np.log2(linear[:,-1])
        s,p=corr(x,True),corr(linear,False)
        ref=np.load(OUT / f"{name}_bootstrap.npz",allow_pickle=False)
        assert np.max(abs(s-ref["point_spearman"]))<TOL
        assert np.max(abs(p-ref["point_pearson"]))<TOL
        n,k=x.shape; sb=np.empty((5000,k,k));pb=np.empty_like(sb)
        rng=np.random.default_rng(42); stream=sha256()
        for j in range(5000):
            draw=rng.integers(n,size=n)
            stream.update(draw.astype("<i8",copy=False).tobytes())
            sb[j]=corr(x[draw],True);pb[j]=corr(linear[draw],False)
            if (j+1)%1000==0:
                log(f"{name}: independently recalculated {j+1}/5000 draws")
        reference_key="primary_bootstrap" if name=="primary" else "three_target_bootstrap"
        assert stream.hexdigest()==saved_complete[reference_key]["index_stream_sha256"]
        err_s=float(np.max(abs(sb-ref["spearman"])))
        err_p=float(np.max(abs(pb-ref["pearson"])))
        assert err_s<TOL and err_p<TOL
        max_errors[name]=dict(spearman_max_abs=err_s,pearson_max_abs=err_p,all_5000_draws_verified=True,
                             index_stream_sha256=stream.hexdigest())
        store[name]=(s,p,sb,pb)
    s,p,sb,pb=store["primary"]
    for r in read(OUT / "Primary_models.csv"):
        i=MODELS.index(r["model"])
        point,draws=(s,sb) if r["metric"]=="Spearman" else (p,pb)
        check_row(r,point[i,5],draws[:,i,5])
    assert len(read(OUT / "Primary_models.csv"))==10
    for r,(a,b) in zip(read(OUT / "Primary_deltas.csv"),PAIRS):
        assert r["contrast"]==f"{a} minus {b}"
        i,j=MODELS.index(a),MODELS.index(b)
        check_row(r,s[i,5]-s[j,5],sb[:,i,5]-sb[:,j,5])
    assert len(read(OUT / "Primary_deltas.csv"))==5
    for r in read(OUT / "Fold_diagnostics.csv"):
        ii=[i for i in ids if int(by_id[i]["fold"])==int(r["fold"])]
        a=np.array([[float(by_id[i]["Bio-XGB"]),float(by_id[i]["Bio-Fusion"]),control[i][0]] for i in ii])
        mat=corr(a,True);delta=mat[1,2]-mat[0,2]
        assert len(ii)==int(r["n"])
        numbers_close(r["Bio_XGB_Spearman"],mat[0,2]);numbers_close(r["Bio_Fusion_Spearman"],mat[1,2])
        numbers_close(r["Fusion_minus_XGB"],delta)
        assert r["direction"]==("positive" if delta>0 else "negative" if delta<0 else "zero")
    s,p,sb,pb=store["three_targets"]
    for r in read(OUT / "Three_target_models.csv"):
        i,j=MODELS.index(r["model"]),5+TARGETS.index(r["target"])
        check_row(r,s[i,j],sb[:,i,j])
    for r in read(OUT / "Target_concordance.csv"):
        i,j=5+TARGETS.index(r["target_a"]),5+TARGETS.index(r["target_b"])
        point,draws=(s,sb) if r["metric"]=="Spearman" else (p,pb)
        check_row(r,point[i,j],draws[:,i,j])
    for r in read(OUT / "Three_target_deltas.csv"):
        j=5+TARGETS.index(r["target"])
        check_row(r,s[2,j]-s[1,j],sb[:,2,j]-sb[:,1,j])
    for r,j in zip(read(OUT / "Gain_shift_descriptive.csv"),[5,6]):
        check_row(r,(s[2,7]-s[1,7])-(s[2,j]-s[1,j]),(sb[:,2,7]-sb[:,1,7])-(sb[:,2,j]-sb[:,1,j]))
    # Coverage and all descriptive point estimates are independently rebuilt.
    selected=set(ids); triples=set(shared)
    groups={"ALL_HUMAN":set(by_id),"PRIMARY_INCLUDED":selected,
            "PUBLISHED_QC_FAIL":{i for i,h in by_id.items() if h["gene_symbol"] in all_symbols and i not in selected},
            "NO_EXACT_PUBLISHED_MATCH":{i for i,h in by_id.items() if h["gene_symbol"] not in all_symbols},
            "NOT_PRIMARY":set(by_id)-selected,"THREE_TARGETS":triples,"PRIMARY_NO_EXTERNAL_PC1":selected-triples}
    feats={r["ENSID"]:r for r in read(PUBLIC / "data/human/coverage_features.csv")}
    for r in read(OUT / "Coverage_features.csv"):
        ii=groups[r["group"]];v=[]
        for i in ii:
            val=feats[i][r["feature"]]
            if val and math.isfinite(float(val)):
                v.append(float(val))
        assert len(ii)==int(r["n"]) and len(v)==int(r["n_valid"]) and len(ii)-len(v)==int(r["n_missing"])
        if v:
            q1,q2,q3=[qlinear(v,q) for q in [.25,.5,.75]]
            for col,val in [("q25",q1),("median",q2),("q75",q3),("iqr",q3-q1)]:
                numbers_close(r[col],val)
        else:
            assert all(r[col]=="" for col in ["q25","median","q75","iqr"])
    for r in read(OUT / "Coverage_counts.csv"):
        ii=groups[r["group"]];available=len(ii&set(ext))
        assert int(r["n"])==len(ii) and int(r["external_pc1_available"])==available
        numbers_close(r["external_pc1_fraction"],available/len(ii))
    group_matrices={}
    for name,ii in groups.items():
        a=np.array([[float(by_id[i][m]) for m in MODELS+["target"]] for i in sorted(ii)])
        group_matrices[name]=corr(a,True)
    for r in read(OUT / "Coverage_original_OOF.csv"):
        numbers_close(r["Spearman"],group_matrices[r["group"]][MODELS.index(r["model"]),5])
    log("Every reported point estimate, percentile CI, fold and coverage summary PASS")
    # Re-open the workbook and compare every cell to its CSV source.
    wb=openpyxl.load_workbook(OUT / "GSE336417_FORMAL_RESULTS.xlsx",read_only=True,data_only=True)
    for ws in wb:
        filename={"Primary_gene_data":"primary_evaluation_data.csv","Three_target_gene_data":"three_target_evaluation_data.csv"}.get(ws.title,ws.title+".csv")
        with (OUT / filename).open(newline="") as f:
            csvrows=list(csv.reader(f))
        xlrows=list(ws.values)
        assert len(csvrows)==len(xlrows)
        for csvrow,xlrow in zip(csvrows,xlrows):
            assert len(csvrow)==len(xlrow)
            for a,b in zip(csvrow,xlrow):
                if b is None:
                    assert a==""
                elif isinstance(b,bool):
                    assert a==str(b)
                elif isinstance(b,(int,float)):
                    assert math.isclose(float(a),b,rel_tol=3e-15,abs_tol=TOL)
                else:
                    assert a==b
    sheet_count=len(wb.sheetnames);wb.close()
    verify_inputs()
    result=dict(status="PASS",started_utc=start,completed_utc=datetime.now(timezone.utc).isoformat(),
        scope="Independent implementation in same Python environment; not independent biological replication",
        all_bootstrap_draws_recomputed=10000,max_absolute_tolerance=TOL,bootstrap_comparison=max_errors,
        checks={"protocol_frozen_before_first_performance":True,"v1_and_inputs_unchanged":True,
                "source_reconstruction_ids_order_targets_predictions":True,"all_251_long_estimates_retained":True,
                "all_paired_CIs_independently_interpolated":True,"fold_and_coverage_independently_recomputed":True,
                "xlsx_all_cells_match_csv":True,"xlsx_sheets":sheet_count,
                "training_inference_refitting_performed":False})
    (audit / "INDEPENDENT_STATISTICAL_RECHECK.json").write_text(json.dumps(result,indent=2)+"\n")
    log("INDEPENDENT STATISTICAL RECHECK PASS")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--results",required=True)
    args=parser.parse_args()
    from inputs import output_path
    main(output_path(args.results,existing=True))
