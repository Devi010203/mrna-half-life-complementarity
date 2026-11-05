#!/usr/bin/env python3
"""Approved classical Figures 1–4 and matched-distillation S1 and external scatter S2; saved values only.

Default outputs: outputs/figures_1/ (PNG/SVG/PDF plus QA). No training,
inference, SHAP estimation, performance calculation or bootstrap. The supplied
figures/ directory is protected from overwriting.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path

from release_paths import ROOT
def external_source(relative):
    return ROOT / relative
OUT = ROOT / "outputs/figures_1"
WORK = OUT / "qa"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/.cache"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

C = ["#427BA2", "#CE9657", "#846BA6"]
MARKERS = ["o", "s", "D"]
MODELS = ["mrnalm", "common_compact_xgb", "common_fusion"]
NAMES = ["LM", "Common-XGB", "Common-Fusion"]
BIO_NAMES = ["LM", "Bio-XGB", "Bio-Fusion"]
DPI = 600
SOURCES = json.loads((ROOT / "data/figure_sources.json").read_text())

D, RAW = {}, {}
QA = {"status":"APPROVED_FIGURES_WITH_REVISED_EXTERNAL_FIGURE2_AND_S2",
      "new_performance_metrics":False,"training_or_inference":False,"new_shap_or_bootstrap":False,
      "source_tables":[],"source_cells":[],"scatters":[],"beeswarm":{},"figures":[],
      "style_assumption":"Figures 1/3/4/S1 retain their approved layouts; revised Figure 2 and relocated scatter S2 use frozen references."}
TABLE_ROWS = []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def previous_snapshot():
    return {rel:sha(external_source(rel)) for rel,_ in SOURCES.values()}

def load():
    for key, (rel, digest) in SOURCES.items():
        p = external_source(rel)
        assert sha(p) == digest, ("modified source", key)
        if p.suffix in {".csv", ".gz"}:
            d = pd.read_csv(p, float_precision="round_trip")
            # Canonical tables are viewed in memory; no duplicate prediction files.
            if key == "human":
                d=d.rename(columns={"fold":"cv_fold","target":"saluki_target","LM":"mrna_lm_oof","Bio-XGB":"xgb_oof","Bio-Fusion":"fusion_oof"})
            elif key == "external":
                d=d.rename(columns={"fold":"cv_fold","target":"external_PC1","original_target":"saluki_target","LM":"mrna_lm_oof","Bio-XGB":"xgb_oof","Bio-Fusion":"fusion_oof"})
            elif key == "common_human":
                d=d.rename(columns={"ENSID":"human_ENSID","fold":"human_fold","target":"human_HALFLIFE","LM":"pred_mrnalm","Common-XGB":"pred_common_compact_xgb","Common-Fusion":"pred_common_fusion"})
            D[key] = d
            QA["source_tables"].append({"key": key, "path_relative_to_repository": rel, "sha256": digest, "rows": len(d), "columns": d.columns.tolist()})
            if key not in {"human", "external", "common_human", "mouse", "shap", "strata_ids"}:
                with p.open(newline="", encoding="utf-8-sig") as handle:
                    RAW[key] = list(csv.DictReader(handle))
    for key, ident, n, columns in [
        ("human", "ENSID", 12981, ["saluki_target", "mrna_lm_oof", "xgb_oof", "fusion_oof"]),
        ("external", "ENSID", 11381, ["external_PC1", "saluki_target", "mrna_lm_oof", "xgb_oof", "fusion_oof"]),
        ("common_human", "human_ENSID", 12981, ["human_HALFLIFE", "pred_mrnalm", "pred_common_compact_xgb", "pred_common_fusion"]),
        ("mouse", "mouse_ENSID", 13744, ["mouse_HALFLIFE", "pred_mrnalm_ensemble", "pred_common_compact_xgb_ensemble", "pred_common_fusion_ensemble"]),
    ]:
        d = D[key]
        assert len(d) == d[ident].nunique() == n
        assert np.isfinite(d[columns].to_numpy()).all()
    assert D["mouse"].mouse_ENSID.tolist() == D["strata_ids"].mouse_ENSID.tolist()
    assert D["mouse"].ortholog_status.equals(D["strata_ids"].ortholog_status)
    assert D["mouse"].ortholog_status.value_counts().to_dict() == {"ORTHOLOG_REPRESENTED": 10200, "NO_CURRENT_1TO1_ORTHOLOG": 3544}
    assert len(D["contexts"]) == 16 and len(D["measurement_metrics"]) == 81
    assert not D["measurement_metrics"].duplicated(["experiment_id", "model"]).any()
    assert D["contexts"].context_index.tolist() == list(range(16))
    assert D["measurement_order"].stable_experiment_index.tolist() == list(range(27))
    h = D["human"].set_index("ENSID")
    s = D["shap"]
    ids = s[["sample_id", "cv_fold"]].drop_duplicates()
    assert len(ids) == 12981 and set(ids.sample_id) == set(h.index)
    assert np.array_equal(ids.cv_fold.to_numpy(), h.loc[ids.sample_id, "cv_fold"].to_numpy())
    assert not s.duplicated(["sample_id", "feature"]).any()
    runs = D["pd4_runs"]
    assert len(runs) == 45 and not runs.duplicated(["fold", "seed", "arm"]).any()
    assert set(runs.seed) == {42, 2026, 3407} and set(runs.fold) == set(range(5))
    assert set(runs.groupby("arm").size()) == {15}
    assert set(runs.arm) == {"A_LABEL_ONLY", "B_BIOCHEM_KD", "C_COMPACT_KD"}
    assert np.isfinite(runs.r2).all()


def saved(key, column, **where):
    rows = [(i+2, row) for i, row in enumerate(RAW[key]) if all(row[k] == str(v) for k, v in where.items())]
    assert len(rows) == 1, (key, where)
    line, row = rows[0]
    QA["source_cells"].append({"source": key, "csv_row": line, "column": column, "original_string": row[column], "selector": where})
    return float(row[column])


def style():
    plt.rcParams.update({"font.family":"Arial","font.size":8.5,"axes.labelsize":8.5,
        "xtick.labelsize":8,"ytick.labelsize":8,"axes.linewidth":0.8,"axes.grid":False,
        "text.color":"black","axes.labelcolor":"black","axes.edgecolor":"black",
        "axes.unicode_minus":True,"legend.frameon":False,"legend.fontsize":7.5,
        "figure.facecolor":"white","savefig.facecolor":"white","text.usetex":False,
        "svg.fonttype":"none", "pdf.fonttype":42, "ps.fonttype":42,
        "svg.hashsalt":"approved-classical-manuscript-20261009-v2"})

def canvas(width,height):
    return plt.figure(figsize=(width/25.4,height/25.4),dpi=150,facecolor="white")

def axes_mm(fig,box):
    w,h=fig.get_size_inches()*25.4
    x,y,ww,hh=box
    return fig.add_axes([x/w,y/h,ww/w,hh/h])

def head(ax,letter,text="",color="black",pad=5):
    ax.annotate(letter,(0,1),xycoords="axes fraction",xytext=(0,pad),textcoords="offset points",
                fontsize=10,weight="bold",ha="left",va="bottom",clip_on=False)
    if text:
        ax.annotate(text,(0,1),xycoords="axes fraction",xytext=(12,pad+0.2),textcoords="offset points",
                    fontsize=8.5,ha="left",va="bottom",color=color,clip_on=False)

def boxed(ax, xticks, xminor, yticks=None, yminor=None):
    for edge in ["left","right","top","bottom"]:
        ax.spines[edge].set_visible(True)
        ax.spines[edge].set_color("black")
        ax.spines[edge].set_linewidth(.8)
    ax.set_xticks(xticks); ax.set_xticks(xminor,minor=True)
    if yticks is not None: ax.set_yticks(yticks)
    if yminor is not None: ax.set_yticks(yminor,minor=True)
    ax.tick_params(axis="both",which="major",direction="in",length=2,width=.65,pad=4.5,
                   top=True,right=True,labeltop=False,labelright=False,labelsize=8)
    ax.tick_params(axis="both",which="minor",direction="in",length=1,width=.45,
                   top=True,right=True,labeltop=False,labelright=False,labelleft=False,labelbottom=False)
    assert all(ax.spines[e].get_visible() for e in ["left","right","top","bottom"])

def native_ticks(limits):
    lo,hi=limits
    major=np.arange(math.ceil(lo/10)*10,hi+1e-9,10)
    minor=np.arange(math.ceil(lo/5)*5,hi+1e-9,5)
    return major,[v for v in minor if not np.any(np.isclose(v,major))]

def scatter(fig,box,source,xcol,ycol,ident,color,panel,limits,stats,*,subset=None,
            xlabel="",ylabel="",diagonal=False):
    frame=D[source] if subset is None else D[source].loc[subset]
    x,y=frame[xcol].to_numpy(),frame[ycol].to_numpy()
    low,high=limits
    assert np.isfinite(x).all() and np.isfinite(y).all()
    outside=(x<low)|(x>high)|(y<low)|(y>high)
    assert not outside.any(),(panel,"out-of-range genes")
    ax=axes_mm(fig,box)
    if diagonal:
        ax.plot(limits,limits,color="#888888",lw=.65,linestyle=(0,(4,3)),zorder=1)
    # Fixed physical area scaling, independent of data density and performance.
    marker_area=1.45*(min(box[2],box[3])/66.0)**2
    marks=ax.scatter(x,y,s=marker_area,c=color,marker="o",alpha=.26,
                     linewidths=0,edgecolors="none",zorder=2)
    assert np.array_equal(np.asarray(marks.get_offsets()),np.column_stack([x,y]))
    assert marks.get_array() is None and len(marks.get_offsets())==len(frame)
    ax.set_xlim(*limits);ax.set_ylim(*limits);ax.set_aspect("equal",adjustable="box")
    major,minor=native_ticks(limits)
    boxed(ax,major,minor,major,minor)
    ax.set_xlabel(xlabel,labelpad=4.5)
    ax.set_ylabel(ylabel,labelpad=4.5)
    ax.text(.052,.946,stats,transform=ax.transAxes,fontsize=8,
            ha="left",va="top",linespacing=1.4,color="black")
    assert not any(t.get_text() in {"Five-fold means","y = x"} or t.get_text().startswith("n =") for t in ax.texts)
    assert len(ax.lines)==int(diagonal)
    QA["scatters"].append({"panel":panel,"source":source,"x_column":xcol,"y_column":ycol,
        "n":len(frame),"id_order_sha256":hashlib.sha256("\n".join(frame[ident]).encode()).hexdigest(),
        "out_of_range":int(outside.sum()),"dropped":0,"sampled":False,"jitter":False,
        "offsets_equal_source":True,"limits":list(limits),"identity_line":diagonal,
        "identity_label":False,"n_label":False,"five_fold_mean_label":False,
        "marker_area_pt2":marker_area,"alpha":.26,"hexbin":False,"density_contours":False,
        "colormap":False,"colorbar":False,"full_frame":True,"inward_major_and_minor_ticks":True})
    return ax

def label_axis(fig,box,n):
    ax=axes_mm(fig,box);ax.set_xlim(0,1);ax.set_ylim(n-.5,-.5)
    ax.set_xticks([]);ax.set_yticks([]);ax.axis("off")
    return ax

def label(ax,text,row,*,right=False,size=8):
    return ax.text(.98 if right else 0,row,text,ha="right" if right else "left",
                   va="center",fontsize=size,color="black")

def colhead(ax,text,right=False):
    ax.annotate(text,(1 if right else 0,1),xycoords="axes fraction",xytext=(0,5),textcoords="offset points",
                fontsize=8,ha="right" if right else "left",va="bottom",clip_on=False)

def dot_axis(fig,box,n,xlimits,majors):
    ax=axes_mm(fig,box);ax.set_xlim(*xlimits);ax.set_ylim(n-.5,-.5)
    step=majors[1]-majors[0]
    minors=[(a+b)/2 for a,b in zip(majors[:-1],majors[1:])]
    boxed(ax,majors,minors,range(n),[])
    ax.set_yticklabels([])
    ax.tick_params(axis="y",length=1.5,width=.45)
    ax.set_xlabel("Spearman ρ",labelpad=4)
    return ax

def paired(ax,values,row):
    assert all(ax.get_xlim()[0]<v<ax.get_xlim()[1] for v in values)
    yy=[row-.14,row,row+.14]
    ax.plot(values,yy,color="#B9BEC3",lw=.65,zorder=1)
    for i,v in enumerate(values):
        ax.plot(v,yy[i],marker=MARKERS[i],ms=3.5,mfc=C[i],mec="white",mew=.3,ls="none",zorder=3)

def legend(ax,names):
    ax.legend(handles=[Line2D([],[],marker=MARKERS[i],ls="none",color=C[i],ms=3.8,label=name)
                       for i,name in enumerate(names)],
              loc="lower right",bbox_to_anchor=(1,1.009),borderaxespad=0,ncol=3,
              handlelength=.7,handletextpad=.3,columnspacing=.85,fontsize=7.5)

def guides(axes,n):
    for ax in axes:
        for row in range(n):
            ax.axhline(row+.5,color="#EAEDF0",lw=.35,zorder=0)

def finish(fig,index,panels):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    visible=[];outside=[];small=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():continue
        bb=t.get_window_extent(renderer)
        if bb.x0 < -.5 or bb.y0 < -.5 or bb.x1 > fig.bbox.width+.5 or bb.y1 > fig.bbox.height+.5:
            outside.append(t.get_text())
        if t.get_fontsize()<7.5:small.append(t.get_text())
        visible.append((t,bb))
    collisions=[]
    for i,(t,bb) in enumerate(visible):
        for u,uu in visible[i+1:]:
            if bb.overlaps(uu):collisions.append([t.get_text(),u.get_text()])
    assert not outside,(index,"text outside",outside)
    assert not small,(index,"small fonts",small)
    assert not collisions,(index,"text collisions",collisions)
    for row in TABLE_ROWS:
        if row[0].figure is fig:
            boxes=[t.get_window_extent(renderer) for t in row]
            assert all(a.x1+1<b.x0 for a,b in zip(boxes,boxes[1:])),(index,"table label collision")
    exports=[]
    for extension in ["png", "svg", "pdf"]:
        path=OUT/f"Figure_{index}.{extension}"
        metadata={"Title":f"Figure {index} — classical manuscript data figure"}
        if extension=="png":
            metadata.update({"Description":"Frozen predictions and saved metrics only; all genes retained.",
                             "NewPerformanceMetrics":"false","SourceHashChecks":"passed"})
        fig.savefig(path,dpi=DPI,facecolor="white",metadata=metadata)
        exports.append({"file":path.name,"sha256":sha(path),"bytes":path.stat().st_size})
    fig.savefig(WORK/f"Figure_{index}_print_size_check.png",dpi=150,facecolor="white")
    QA["figures"].append({"file":f"Figure_{index}","exports":exports,"panels":panels,"mm":list(np.round(fig.get_size_inches()*25.4,3)),
                          "dpi":DPI,"outside_canvas":outside,"text_collisions":collisions,
                          "min_font_pt":min(t.get_fontsize() for t,_ in visible),
                          "rasterized_collections":sum(c.get_rasterized() for ax in fig.axes for c in ax.collections)})
    plt.close(fig)

def figure1():
    fig=canvas(183,184)
    for i,col in enumerate(["mrna_lm_oof","xgb_oof","fusion_oof"]):
        src="bio_summary" if i==1 else "fusion_summary"
        where={"feature_group":"basic_plus_biochemical_full"} if i==1 else {"method":["mrnalm_only","","linear_fusion"][i]}
        r2=saved(src,"mean",split="test",metric="r2",**where)
        rho=saved(src,"mean",split="test",metric="spearman",**where)
        ax=scatter(fig,[18+56*i,122,48,48],"human","saluki_target",col,"ENSID",C[i],
                   f"1{chr(97+i)}",(-20,20),f"R² = {r2:.4f}\nSpearman ρ = {rho:.4f}",
                   xlabel="Observed stability (PC1)",ylabel="OOF prediction" if i==0 else "",diagonal=True)
        head(ax,chr(97+i),BIO_NAMES[i],C[i])
    ranking=D["shap_rank"].sort_values("rank_mean_abs_shap").head(20)
    features=ranking.feature.tolist()
    sample=D["shap"].loc[D["shap"].feature.isin(features)]
    assert len(sample)==259620 and (sample.groupby("feature").sample_id.nunique()==12981).all()
    xmin=math.floor(sample.shap_value.min())-.3
    xmax=math.ceil(sample.shap_value.max())+.3
    ax=axes_mm(fig,[66,14,111,86])
    ax.set_xlim(xmin,xmax);ax.set_ylim(19.6,-.6)
    major=np.arange(math.ceil(xmin/2)*2,xmax,2)
    minor=np.arange(math.ceil(xmin),xmax,1)
    minor=[v for v in minor if v not in major]
    boxed(ax,major,minor,range(20),[])
    ax.set_yticklabels(features,fontsize=7.5)
    ax.tick_params(axis="y",length=1.5,width=.45,pad=4)
    ax.set_xlabel("SHAP value (predicted-target units)",labelpad=4.5)
    ax.axvline(0,color="#B7BDC3",lw=.6,zorder=0)
    for j,feature in enumerate(features):
        row=sample.loc[sample.feature.eq(feature)]
        xx=row.shap_value.to_numpy()
        bins=np.floor((xx-xmin)/(xmax-xmin)*160).astype(int)
        counts=np.bincount(bins,minlength=161)
        offset=np.zeros(len(xx))
        rng=np.random.default_rng(9100+j)
        for bin_id in np.unique(bins):
            ids=rng.permutation(np.flatnonzero(bins==bin_id))
            offset[ids]=(np.arange(len(ids))-(len(ids)-1)/2)*(.64/max(counts.max(),1))
        marks=ax.scatter(xx,j+offset,s=.8,c=C[1],alpha=.30,linewidths=0,edgecolors="none",zorder=2)
        assert np.array_equal(np.asarray(marks.get_offsets())[:,0],xx)
        assert len(marks.get_offsets())==12981
    title_ax=label_axis(fig,[6,14,45,86],20)
    head(title_ax,"d","Feature attributions (TreeSHAP)",pad=7)
    QA["beeswarm"]={"source":"shap","ranking_source":"shap_rank","features_in_saved_order":features,
        "n_genes_per_feature":12981,"n_points":259620,"subsampled":False,"shap_recomputed":False,
        "feature_rank_recomputed":False,"color":"single Bio-XGB orange; no feature-value encoding",
        "horizontal_offsets_equal_source":True,
        "display_packing":"160 fixed SHAP bins; symmetric offsets; display seeds 9100..9119"}
    finish(fig,1,["a","b","c","d"])

def figure2():
    from figures_external import render
    QA["external_figure2"] = render(ROOT, OUT, "2")
    style()


def figure_s2():
    from figures_external import render
    QA["external_figure_s2"] = render(ROOT, OUT, "s2")
    style()


def figure3():
    fig=canvas(183,152)
    for row in range(2):
        source="common_human" if row==0 else "mouse"
        xcol="human_HALFLIFE" if row==0 else "mouse_HALFLIFE"
        cols=["pred_mrnalm","pred_common_compact_xgb","pred_common_fusion"] if row==0 else ["pred_mrnalm_ensemble","pred_common_compact_xgb_ensemble","pred_common_fusion_ensemble"]
        for i,col in enumerate(cols):
            rho=saved("human_common_metrics","spearman",human_fold="POOLED",split="pooled_outer_test_oof",model=MODELS[i]) if row==0 else saved("mouse_metrics","spearman_rho",model=MODELS[i])
            letter=chr(97+row*3+i)
            ax=scatter(fig,[18+56*i,91 if row==0 else 17,48,48],source,xcol,col,
                       "human_ENSID" if row==0 else "mouse_ENSID",C[i],f"3{letter}",
                       (-20,20) if row==0 else (-16,16),f"Spearman ρ = {rho:.4f}",
                       xlabel="Observed human target" if row==0 else "Observed mouse target",
                       ylabel=("Human OOF prediction" if row==0 else "Mouse zero-shot prediction") if i==0 else "",
                       diagonal=row==0)
            head(ax,letter,NAMES[i],C[i])
    finish(fig,3,["a","b","c","d","e","f"])

def figure4():
    fig=canvas(183,244)
    for i,scope in enumerate(["ORTHOLOG_REPRESENTED","NO_CURRENT_1TO1_ORTHOLOG"]):
        subset=D["mouse"].ortholog_status.eq(scope)
        rho=saved("strata_metrics","spearman_rho",scope=scope,model="common_fusion")
        ax=scatter(fig,[18+91*i,170,62,62],"mouse","mouse_HALFLIFE","pred_common_fusion_ensemble",
                   "mouse_ENSID",C[2],f"4{chr(97+i)}",(-16,16),f"Spearman ρ = {rho:.4f}",
                   subset=subset,xlabel="Observed mouse target",ylabel="Common-Fusion prediction")
        head(ax,chr(97+i),"Represented" if i==0 else "Not represented")
    study=label_axis(fig,[6,14,25,130],27)
    assay=label_axis(fig,[33,14,13,130],27)
    cell=label_axis(fig,[49,14,38,130],27)
    ns=label_axis(fig,[89,14,14,130],27)
    dots=dot_axis(fig,[109,14,68,130],27,(.10,.82),[.2,.4,.6,.8])
    head(study,"c","Study");colhead(assay,"Assay");colhead(cell,"Cell type");colhead(ns,"n",True)
    legend(dots,NAMES);guides([study,assay,cell,ns,dots],27)
    order=D["measurement_order"]
    counts=Counter(zip(order.source_dataset,order.assay_type,order.cell_type))
    for row,r in order.iterrows():
        st,ct=r.source_dataset,r.cell_type;suffix=r.experiment_id.split("_")[-1]
        if suffix.isdigit() and counts[(st,r.assay_type,ct)]>1:st+=f" [{suffix}]"
        elif not suffix.isdigit():
            ct+=", "+suffix.replace("."," ").replace("LIFneg","LIF−").replace("LIFpos","LIF+").replace("RApos","RA+")
        texts=[label(study,st,row,size=7.5),label(assay,r.assay_type,row,size=7.5),
               label(cell,ct,row,size=7.5),label(ns,f"{int(r.n_model_matched):,}",row,right=True,size=7.5)]
        TABLE_ROWS.append(texts)
        values=[saved("measurement_metrics","spearman_rho",experiment_id=r.experiment_id,model=m) for m in MODELS]
        paired(dots,values,row)
    QA["measurement_ids_in_order"]=order.experiment_id.tolist()
    QA["negative_cases_retained"]=["Simon_4sU_MEF_1","Simon_4sU_MEF_2","Koszinowski_4sU_3T3_1"]
    assert set(QA["negative_cases_retained"]).issubset(set(order.experiment_id))
    assert int(order.loc[order.experiment_id.eq("Koszinowski_4sU_3T3_1"),"n_model_matched"].iloc[0])==376
    finish(fig,4,["a","b","c"])

def figure_s1():
    fig=canvas(183,100)
    ax=axes_mm(fig,[38,22,54,62])
    arms=["A_LABEL_ONLY","B_BIOCHEM_KD","C_COMPACT_KD"]
    labels=["A  Labels only","B  Biochemical KD","C  Compact KD"]
    colors=["#777F87",C[1],C[0]]
    ax.set_xlim(.395,.480);ax.set_ylim(2.55,-.55)
    boxed(ax,[.40,.42,.44,.46,.48],[.41,.43,.45,.47],range(3),[])
    ax.set_yticklabels(labels,fontsize=8)
    ax.set_xlabel("Outer-test R²",labelpad=5)
    head(ax,"a","Matched student runs")
    means=[]
    for row,arm in enumerate(arms):
        data=D["pd4_runs"].loc[D["pd4_runs"].arm.eq(arm)].sort_values(["fold","seed"])
        yy=row+np.linspace(-.21,.21,15)
        values=[saved("pd4_runs","r2",fold=int(r.fold),seed=int(r.seed),arm=arm) for _,r in data.iterrows()]
        assert all(ax.get_xlim()[0] < x < ax.get_xlim()[1] for x in values)
        ax.scatter(values,yy,s=11,marker="o",facecolors="none",edgecolors=colors[row],linewidths=.65,zorder=2)
        mean=saved("pd4_means","r2",arm=arm);means.append(mean)
        ax.plot(mean,row,marker="D",ms=5,mfc=colors[row],mec="black",mew=.4,zorder=3)
        ax.text(.478,row+.35,f"mean {mean:.4f}",fontsize=7.5,ha="right",va="center")
    bx=axes_mm(fig,[126,22,51,62])
    bx.set_xlim(-.008,.014);bx.set_ylim(2.55,-.55)
    boxed(bx,[-.005,0,.005,.010],[-.0025,.0025,.0075,.0125],range(3),[])
    bx.set_xticklabels(["−0.005","0","0.005","0.010"],fontsize=7.5)
    bx.set_yticklabels(["B−A","C−A","B−C"],fontsize=8)
    bx.set_xlabel("Matched ΔR²",labelpad=5)
    head(bx,"b","Primary paired differences")
    bx.axvline(0,color="#888888",lw=.7,zorder=0)
    bx.axvline(.010,color="#888888",lw=.7,ls=(0,(3,2)),zorder=0)
    contrasts=[]
    for row,delta in enumerate(["B-A","C-A","B-C"]):
        est=saved("pd4_deltas","estimate",delta=delta,metric="r2")
        low=saved("pd4_deltas","ci_low",delta=delta,metric="r2")
        high=saved("pd4_deltas","ci_high",delta=delta,metric="r2")
        assert bx.get_xlim()[0] < low <= est <= high < bx.get_xlim()[1]
        color=colors[1] if row!=1 else colors[2]
        bx.errorbar(est,row,xerr=[[est-low],[high-est]],fmt="D" if row==2 else "o",
            ms=4.4,color=color,mfc="white" if row==2 else color,mec=color,mew=.9,
            elinewidth=1,capsize=2,capthick=.8,zorder=3)
        bx.text(.0135,row+.35,f"{est:+.4f}",fontsize=7.5,ha="right",va="center")
        contrasts.append({"delta":delta,"estimate":est,"ci_low":low,"ci_high":high})
    fig.text(126/183,10/100,"Dashed: +0.010 practical threshold (B−A, C−A)",fontsize=7.5,ha="center")
    fig.text(38/183,3/100,"Small-effect STOP; no external student evaluation",fontsize=8,ha="left")
    QA["supplement_s1"]={"all_run_points":45,"points_per_arm":15,"arm_means_from_saved_source":means,
        "primary_contrasts":contrasts,"new_metrics_or_ci":False,"primary_not_ensemble_ci":True,
        "practical_threshold":.010,"threshold_scope":["B-A","C-A"],
        "stop_conclusion_retained":True,"author_visual_review":"new_figure_for_author_inspection"}
    finish(fig,"S1",["a","b"])


def overview():
    panels=[]
    for i in range(1,5):
        with Image.open(OUT/f"Figure_{i}.png") as im:
            panel=im.convert("RGB")
            panels.append(panel.resize((1100,round(panel.height*1100/panel.width)),Image.Resampling.LANCZOS))
    h1=max(p.height for p in panels[:2]);h2=max(p.height for p in panels[2:])
    sheet=Image.new("RGB",(2272,h1+h2+144),"white")
    draw=ImageDraw.Draw(sheet)
    font_path=font_manager.findfont(font_manager.FontProperties(family="Arial",weight="bold"))
    font=ImageFont.truetype(font_path,24)
    for i,im in enumerate(panels):
        x=24+(i%2)*1124;y=48+(h1+72 if i//2 else 0)
        draw.text((x+4,y-34),f"Figure {i+1}",font=font,fill="black")
        sheet.paste(im,(x,y))
    sheet.save(OUT/"Overview.png")
    QA["overview"]={"file":"Overview.png","sha256":sha(OUT/"Overview.png")}

def main():
    global OUT, WORK
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only",action="store_true",help="Check frozen inputs without rendering.")
    parser.add_argument("--output-dir",default="outputs/figures_1",help="New directory under outputs/.")
    parser.add_argument("--figures",choices=["all","main","s1","s2","supplementary"],default="all")
    args=parser.parse_args()
    OUT=(ROOT/args.output_dir).resolve()
    if not __debug__:raise RuntimeError('Do not disable verification with python -O')
    if not OUT.is_relative_to(ROOT/'outputs') or OUT==ROOT/'outputs':
        raise ValueError('Figure outputs must be inside outputs/')
    WORK=OUT/"qa"
    load()
    if args.validate_only:
        print(json.dumps({"status":"PASS","sources":len(SOURCES),"human":12981,"external":11381,
            "mouse":13744,"pd4_runs":45,"plots_or_metrics_computed":False}))
        return
    indices=([1,2,3,4] if args.figures in {"all","main"} else [])+(["S1"] if args.figures in {"all","s1","supplementary"} else [])+(["S2"] if args.figures in {"all","s2","supplementary"} else [])
    assert all(not (OUT/f"Figure_{i}.{ext}").exists() for i in indices for ext in ["png","svg","pdf"]), "Existing exports are protected; choose a new output directory."
    WORK.mkdir(parents=True,exist_ok=True)
    before=previous_snapshot()
    style()
    funcs=([figure1,figure2,figure3,figure4] if args.figures in {"all","main"} else [])+([figure_s1] if args.figures in {"all","s1","supplementary"} else [])+([figure_s2] if args.figures in {"all","s2","supplementary"} else [])
    for func in funcs:
        func();print(func.__name__+" PNG/SVG/PDF complete",flush=True)
    if args.figures in {"all","main"}:overview()
    for path,digest in before.items():assert sha(external_source(path))==digest,("prior file changed",path)
    for key,(rel,digest) in SOURCES.items():assert sha(external_source(rel))==digest,("source changed",key)
    QA["formal_source_count"]=len(SOURCES)
    QA["protected_previous_file_count"]=len(before)
    QA["runtime"]={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__}
    QA["script_sha256"]=sha(Path(__file__))
    QA["formal_export_formats"]=["png","svg","pdf"]
    external_checks=[QA[k] for k in ["external_figure2","external_figure_s2"] if k in QA]
    QA["total_scatter_panels"]=len(QA["scatters"])+sum(len(q["scatters"]) for q in external_checks)
    QA["total_source_cell_checks"]=len(QA["source_cells"])+sum(len(q["cells"]) for q in external_checks)
    QA["manual_visual_review"]="pending"
    (WORK/f"QA_FIGURES_{args.figures}.json").write_text(json.dumps(QA,indent=2)+"\n")
    manifest={"status":QA["status"],"sources":{k:{"path_relative_to_repository":p,"sha256":s,
        "unchanged_after_plotting":True} for k,(p,s) in SOURCES.items()},
        "protected_previous_files":before,"no_source_data_copied_or_modified":True}
    (WORK/f"SOURCE_MANIFEST_{args.figures}.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"PASS: {QA['total_scatter_panels']} all-gene scatters; {QA['total_source_cell_checks']} frozen statistic cells; no performance recomputation.")

if __name__=="__main__":
    main()
