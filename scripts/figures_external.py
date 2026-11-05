#!/usr/bin/env python3
"""Frozen external-human Figure 2 and supplementary S2; no metric calculation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/.cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from PIL import Image

SOURCES = {
    "external": ("data/external/matched.csv", "e5e5bc36ec09750ded055f5d3272422cba442e8e7d919149dc051f9019bd1794"),
    "target_agreement": ("results/external/current_scope_target_replication_metrics.csv", "18f2a82f8a76aa3ed9d6b1d2e4e671733e31b60824d4cec1af86d40311e5656d"),
    "external_metrics": ("results/external/external_primary_metrics.csv", "f0d10bcb47ceefd806c9b7dc2da7a29acf1e4c17641f90936466a3c04bd09d1d"),
    "contexts": ("results/external/per_context_metrics.csv", "a351682ddba45ef2fe5a43e1ddecd88067c0d858d92a3e322ca46e37d96edc86"),
    "models": ("results/organoid/Three_target_models.csv", "5142b5e6ea131b8ae9303b9479dcbfa2da5934be6599387789c63a2e7762d55f"),
    "deltas": ("results/organoid/Three_target_deltas.csv", "9c96ab26c6bb71df75dee2262956d7030f115fefe9357a1cbd62c6208d12b619"),
}
MODELS = ["LM", "Bio-XGB", "Bio-Fusion", "Common-XGB", "Common-Fusion"]
TARGETS = ["Saluki", "External-PC1", "Organoid"]
COLORS = {"LM": "#427BA2", "Bio-XGB": "#CE9657", "Bio-Fusion": "#846BA6",
          "Common-XGB": "#A45D20", "Common-Fusion": "#5F397E"}
MARKERS = {"LM": "o", "Bio-XGB": "s", "Bio-Fusion": "D", "Common-XGB": "s", "Common-Fusion": "D"}
QA = {"status": "FROZEN_REFERENCE_DISPLAY", "metric_calculation": False, "bootstrap": False,
      "training_or_inference": False, "sources": [], "cells": [], "scatters": [],
      "artists": [], "exports": [], "layouts": []}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(repo):
    manifest = {r["path"]: r for r in csv.DictReader((repo / "docs/file_manifest.csv").open())}
    data = {}
    for key, (rel, digest) in SOURCES.items():
        assert sha(repo / rel) == digest == manifest[rel]["sha256"], (key, "hash mismatch")
        if key == "external":
            data[key] = pd.read_csv(repo / rel, float_precision="round_trip")
            assert len(data[key]) == data[key].ENSID.nunique() == 11381
        else:
            with (repo / rel).open(newline="") as f:
                data[key] = list(csv.DictReader(f))
        QA["sources"].append({"key": key, "path": rel, "sha256": digest, "rows": len(data[key])})
    assert [int(r["context_index"]) for r in data["contexts"]] == list(range(16))
    assert len(data["models"]) == 15 and len(data["deltas"]) == 3
    assert all(r["n"] == "4801" for k in ["models", "deltas"] for r in data[k])
    return data


def saved(data, key, column, panel, **select):
    rows = [(i + 2, r) for i, r in enumerate(data[key]) if all(r[k] == str(v) for k, v in select.items())]
    assert len(rows) == 1, (key, select)
    i, row = rows[0]
    QA["cells"].append({"panel": panel, "source": SOURCES[key][0], "csv_row": i,
                        "column": column, "stored_string": row[column], "selector": select})
    return float(row[column])


def triple(data, key, panel, **select):
    assert saved(data, key, "n", panel, **select) == 4801
    vals = [saved(data, key, c, panel, **select) for c in ["estimate", "ci_low", "ci_high"]]
    assert vals[1] <= vals[0] <= vals[2]
    return vals


def style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((s for s in ["Arial", "Liberation Sans", "DejaVu Sans"] if s in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "font.size": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": .8,
        "axes.edgecolor": "black", "text.color": "black", "axes.labelcolor": "black",
        "figure.facecolor": "white", "savefig.facecolor": "white", "axes.grid": False,
        "legend.frameon": False, "legend.fontsize": 7.5, "svg.fonttype": "none",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "human-external-restructure-20261105"})
    QA["font"] = font


def axis(fig, rect):
    w, h = fig.get_size_inches() * 25.4
    x, y, ww, hh = rect
    return fig.add_axes([x / w, y / h, ww / w, hh / h])


def box(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(.8)
    ax.tick_params(which="major", direction="in", length=2.2, width=.65, pad=4.5, top=True, right=True)
    ax.tick_params(which="minor", direction="in", length=1.1, width=.45, top=True, right=True)


def head(ax, letter, title, subtitle=None):
    pad = 15 if subtitle else 6
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-2, pad), textcoords="offset points",
                ha="left", va="bottom", fontsize=10, weight="bold")
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(12, pad), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5)
    if subtitle:
        ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7.5)


def marker(model, size=4.4):
    return dict(marker=MARKERS[model], markersize=size,
        markerfacecolor="white" if model.startswith("Common") else COLORS[model],
        markeredgecolor=COLORS[model], markeredgewidth=.9, linestyle="none", zorder=4)


def range_point(ax, pos, v, model, orientation, panel):
    est, lo, hi = v
    c = COLORS[model]
    if orientation == "vertical":
        line, = ax.plot([pos, pos], [lo, hi], color=c, linewidth=.9, zorder=2)
        for end in [lo, hi]:
            ax.plot([pos - .020, pos + .020], [end, end], color=c, linewidth=.75)
        dot, = ax.plot(pos, est, **marker(model))
        assert np.array_equal(line.get_ydata(), [lo, hi]) and dot.get_ydata()[0] == est
        assert ax.get_ylim()[0] <= lo <= hi <= ax.get_ylim()[1]
    else:
        line, = ax.plot([lo, hi], [pos, pos], color=c, linewidth=1, zorder=2)
        for end in [lo, hi]:
            ax.plot([end, end], [pos - .055, pos + .055], color=c, linewidth=.8)
        dot, = ax.plot(est, pos, **marker(model, 4.6))
        assert np.array_equal(line.get_xdata(), [lo, hi]) and dot.get_xdata()[0] == est
        assert ax.get_xlim()[0] <= lo <= hi <= ax.get_xlim()[1]
    QA["artists"].append({"panel": panel, "model": model, "estimate": est, "ci": [lo, hi],
                           "source_equal_artist": True, "no_interval_clipping": True})


def scatter(fig, rect, data, ycol, color, panel, ylabel, stat):
    ax = axis(fig, rect)
    d = data["external"]
    x, y = d.target.to_numpy(), d[ycol].to_numpy()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    outside = (x < -21) | (x > 21) | (y < -21) | (y > 21)
    assert not outside.any()
    area = 1.45 * (min(rect[2:]) / 66.0) ** 2
    marks = ax.scatter(x, y, s=area, color=color, alpha=.26, linewidths=0, edgecolors="none")
    assert np.array_equal(np.asarray(marks.get_offsets()), np.column_stack([x, y]))
    ax.set_xlim(-21, 21); ax.set_ylim(-21, 21); ax.set_aspect("equal")
    ax.set_xticks([-20, -10, 0, 10, 20]); ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_xticks([-15, -5, 5, 15], minor=True); ax.set_yticks([-15, -5, 5, 15], minor=True)
    box(ax)
    ax.set_xlabel("External-PC1", labelpad=4.5)
    ax.set_ylabel(ylabel, labelpad=4.5)
    ax.text(.052, .946, f"Spearman ρ = {stat:.4f}", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    QA["scatters"].append({"panel": panel, "x": "target", "y": ycol, "n": len(d),
        "ids_sha256": hashlib.sha256("\n".join(d.ENSID).encode()).hexdigest(),
        "axis_limits": [-21, 21], "points_outside": int(outside.sum()), "dropped": 0,
        "source_equal_artist": True, "alpha": .26, "marker_area_pt2": area,
        "identity_line": False, "hexbin": False, "performance_recalculated": False})
    return ax


def target_models(fig, data):
    ax = axis(fig, [99, 160, 78, 60])
    ax.set_xlim(-.34, 2.34); ax.set_ylim(.38, .90)
    ax.set_xticks([0, 1, 2], ["Original\nSaluki", "External-PC1", "Organoid\nSLAM-seq"])
    ax.set_yticks([.4, .5, .6, .7, .8, .9]); ax.set_yticks([.45, .55, .65, .75, .85], minor=True)
    ax.set_ylabel("Spearman ρ", labelpad=5)
    box(ax); ax.tick_params(axis="x", length=0)
    head(ax, "b", "Three-target comparison", "Same 4,801 genes")
    for model, shift in zip(MODELS, [-.14, -.07, 0, .07, .14]):
        vals = [triple(data, "models", "2b", model=model, target=t, metric="Spearman") for t in TARGETS]
        xx = np.arange(3) + shift
        for x, v in zip(xx, vals):
            range_point(ax, x, v, model, "vertical", "2b")
    points = [ln for ln in ax.lines if len(ln.get_xdata()) == 1]
    bars = [ln for ln in ax.lines if len(ln.get_xdata()) == 2 and
            ln.get_xdata()[0] == ln.get_xdata()[1] and ln.get_ydata()[0] != ln.get_ydata()[1]]
    caps = [ln for ln in ax.lines if len(ln.get_xdata()) == 2 and
            ln.get_ydata()[0] == ln.get_ydata()[1] and ln.get_xdata()[0] != ln.get_xdata()[1]]
    assert len(ax.lines) == 60 and len(points) == 15 and len(bars) == 15 and len(caps) == 30
    QA["panel_2b"] = {"points": 15, "CI_bars": 15, "CI_caps": 30,
                      "between_target_lines": 0, "all_point_and_CI_values_unchanged": True}
    order = ["LM", None, "Bio-XGB", "Common-XGB", "Bio-Fusion", "Common-Fusion"]
    handles = [Line2D([], [], color=COLORS[m], label=m, **marker(m, 3.5)) if m else
               Line2D([], [], linestyle="none", label=" ") for m in order]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.50, .998), ncol=3,
              handlelength=.8, handletextpad=.35, columnspacing=.7, borderaxespad=.2, labelspacing=.65)


def target_gains(fig, data):
    labels = axis(fig, [6, 113, 40, 27]); labels.axis("off")
    labels.set_xlim(0, 1); labels.set_ylim(-.55, 2.55)
    head(labels, "c", "Paired fusion gain · same 4,801 genes")
    ax = axis(fig, [49, 113, 128, 27])
    ax.set_xlim(-.002, .026); ax.set_ylim(-.55, 2.55)
    ax.set_xticks([0, .01, .02]); ax.set_xticks([.005, .015, .025], minor=True)
    ax.set_yticks([]); box(ax)
    ax.axvline(0, color="#777777", lw=.75, zorder=0)
    ax.set_xlabel("ΔSpearman (Bio-Fusion − Bio-XGB)", labelpad=4)
    for y, target, label in zip([2, 1, 0], TARGETS, ["Original Saluki", "External-PC1", "Organoid SLAM-seq"]):
        labels.text(0, y, label, va="center", fontsize=8.5)
        vals = triple(data, "deltas", "2c", target=target, contrast="Bio-Fusion minus Bio-XGB", metric="Delta_Spearman")
        range_point(ax, y, vals, "Bio-Fusion", "horizontal", "2c")
        ax.annotate(f"+{vals[0]:.5f}", (vals[0], y), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5)


def contexts(fig, data):
    labs = axis(fig, [6, 15, 58, 75]); ns = axis(fig, [65, 15, 13, 75])
    dots = axis(fig, [88, 15, 89, 75])
    for ax in [labs, ns, dots]: ax.set_ylim(15.5, -.5)
    for ax in [labs, ns]: ax.set_xlim(0, 1); ax.axis("off")
    vals = [[float(r[c]) for c in ["lm_spearman", "xgb_spearman", "fusion_spearman"]] for r in data["contexts"]]
    lo, hi = math.floor(np.min(vals) * 10) / 10 - .01, math.ceil(np.max(vals) * 10) / 10 + .01
    ticks = np.arange(math.ceil(lo * 10) / 10, hi, .1)
    dots.set_xlim(lo, hi); dots.set_xticks(ticks); dots.set_xticks((ticks[:-1] + ticks[1:]) / 2, minor=True)
    dots.set_yticks([]); box(dots); dots.set_xlabel("Spearman ρ", labelpad=4)
    head(labs, "d", "Dataset–cell-line context")
    ns.annotate("n", (1, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points", ha="right", va="bottom", fontsize=8)
    dots.legend(handles=[Line2D([], [], color=COLORS[m], label=m, **marker(m, 3.5)) for m in MODELS[:3]],
                loc="lower right", bbox_to_anchor=(1, 1.02), ncol=3, borderaxespad=0,
                handlelength=.7, handletextpad=.35, columnspacing=1)
    uncertain = []
    for y, r in enumerate(data["contexts"]):
        for ax in [labs, ns, dots]: ax.axhline(y + .5, color="#E8EBEE", lw=.4, zorder=0)
        scope = r["context_id"]
        n = int(saved(data, "contexts", "n_model_matched", "2d", context_id=scope))
        low = saved(data, "contexts", "fusion_minus_xgb_ci95_low", "2d", context_id=scope)
        high = saved(data, "contexts", "fusion_minus_xgb_ci95_high", "2d", context_id=scope)
        is_uncertain = low <= 0 <= high
        if is_uncertain: uncertain.append(scope)
        p = r["dataset"].split("_")
        label = f"{p[0]} {p[2]} · {r['cell_line']}" + (" †" if is_uncertain else "")
        labs.text(0, y, label, va="center", fontsize=8)
        ns.text(1, y, f"{n:,}", ha="right", va="center", fontsize=8)
        values = [saved(data, "contexts", c, "2d", context_id=scope) for c in ["lm_spearman", "xgb_spearman", "fusion_spearman"]]
        yy = [y - .14, y, y + .14]
        dots.plot(values, yy, color="#B9BEC3", linewidth=.65, zorder=1)
        for value, yp, model in zip(values, yy, MODELS[:3]):
            dot, = dots.plot(value, yp, **marker(model, 3.5))
            assert dot.get_xdata()[0] == value and lo < value < hi
            QA["artists"].append({"panel": "2d", "context": scope, "model": model,
                                   "estimate": value, "source_equal_artist": True})
    assert uncertain == ["Ietswaart_etal_2024_K562__K562", "Muhar_etal_2018_many__K562"], uncertain
    QA["contexts_in_order"] = [r["context_id"] for r in data["contexts"]]
    QA["contexts_with_paired_CI_including_zero"] = uncertain
    fig.text(6 / 183, 1.6 / 232, "† Bio-Fusion − Bio-XGB paired CI includes zero (Table S8).", fontsize=7.5)


def export(fig, name, out, size):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip(): continue
        bb = t.get_window_extent(renderer)
        assert bb.x0 >= -1 and bb.y0 >= -1 and bb.x1 <= fig.bbox.width + 1 and bb.y1 <= fig.bbox.height + 1, (name, t.get_text(), bb)
        assert t.get_fontsize() >= 7.5
    for ext in ["png", "pdf", "svg"]:
        dest = out / f"{name}.{ext}"
        fig.savefig(dest, dpi=600)
        if ext == "svg":
            assert ET.parse(dest).getroot().findall(".//{http://www.w3.org/2000/svg}text")
        elif ext == "pdf":
            assert b"/FontFile2" in dest.read_bytes()
        else:
            with Image.open(dest) as im:
                assert np.allclose(im.size, np.array(size) * 600 / 25.4, atol=1)
                assert np.allclose(im.info["dpi"], [600, 600], atol=.01)
                QA["raster"] = {"size_pixels": list(im.size), "dpi_metadata": list(im.info["dpi"]),
                                "native_export_dpi": 600, "upscaled": False}
        QA["exports"].append({"file": dest.name, "size_mm": size, "sha256": sha(dest)})
    fig.savefig(out / f"{name}_review_150dpi.png", dpi=150)
    QA["layouts"].append({"figure": name, "text_in_canvas": True, "minimum_font_pt": 7.5})
    plt.close(fig)


def main_figure(data, out):
    fig = plt.figure(figsize=(183 / 25.4, 232 / 25.4), dpi=150)
    rho = saved(data, "target_agreement", "spearman", "2a")
    assert saved(data, "target_agreement", "n_genes", "2a") == 11381
    ax = scatter(fig, [16, 160, 60, 60], data, "original_target", "#777F87", "2a", "Original human target", rho)
    head(ax, "a", "Target agreement", "11,381 matched genes")
    target_models(fig, data); target_gains(fig, data); contexts(fig, data)
    export(fig, "Figure_2", out, [183, 232])


def check_workbooks(repo, data):
    book = load_workbook(repo / "supplementary/ESM_2_Supplementary_Tables_S1-S14.xlsx", read_only=True, data_only=True)
    model_rows = {r[0]: r[1:4] for r in book["S14a"].values if r[0] in MODELS}
    delta_rows = list(book["S14c"].values)[1:4]
    fmt = lambda r: f"{float(r['estimate']):.6f} [{float(r['ci_low']):.6f}, {float(r['ci_high']):.6f}]"
    for r in data["models"]: assert model_rows[r["model"]][TARGETS.index(r["target"]) ] == fmt(r)
    for target, wr in zip(TARGETS, delta_rows):
        row = next(r for r in data["deltas"] if r["target"] == target)
        assert wr[0].startswith(target + ":") and wr[1] == 4801 and wr[2] == fmt(row)
    book.close()
    book = load_workbook(repo / "results/organoid/GSE336417_FORMAL_RESULTS.xlsx", read_only=True, data_only=True)
    for key, sheet in [("models", "Three_target_models"), ("deltas", "Three_target_deltas")]:
        rows = list(book[sheet].values)
        assert len(rows) == len(data[key]) + 1
        for original, wr in zip(data[key], rows[1:]):
            for col, val in zip(rows[0], wr):
                if col in ["estimate", "ci_low", "ci_high"]: assert abs(float(original[col]) - val) < 1e-15
                else: assert original[col] == str(val)
    book.close()
    QA["workbook_checks"] = {"S14a": "PASS", "S14c_three_target_rows": "PASS", "formal_workbook_18_rows": "PASS"}


def supplementary(data, out):
    fig = plt.figure(figsize=(183 / 25.4, 76 / 25.4), dpi=150)
    fig.text(15 / 183, 70 / 76, "External-PC1 · 11,381 matched genes", fontsize=8.5)
    for x, model, original_name, panel in zip([15, 76, 137], MODELS[:3], ["mRNA-LM", "XGBoost", "Fusion"], ["a", "b", "c"]):
        rho = saved(data, "external_metrics", "spearman", f"S2{panel}", model=original_name)
        assert saved(data, "external_metrics", "n", f"S2{panel}", model=original_name) == 11381
        ax = scatter(fig, [x, 15, 44, 44], data, model, COLORS[model], f"S2{panel}", f"{model} OOF prediction", rho)
        head(ax, panel, model)
    export(fig, "Figure_S2", out, [183, 76])


_EMPTY_QA = json.dumps(QA)


def render(root, out, selection="2"):
    """Render only requested assets, with all outputs confined to outputs/."""
    global QA
    QA = json.loads(_EMPTY_QA)
    root, out = Path(root).resolve(), Path(out).resolve()
    if not out.is_relative_to(root / "outputs") or out == root / "outputs":
        raise ValueError("Figure output must be inside repository outputs/")
    names = ["Figure_2"] if selection == "2" else ["Figure_S2"]
    if selection not in {"2", "s2"}:
        raise ValueError(selection)
    if any((out / (name + "." + ext)).exists() for name in names for ext in ["png", "pdf", "svg"]):
        raise FileExistsError("Choose a new output directory; existing figures are protected")
    out.mkdir(parents=True, exist_ok=True)
    before = {rel: sha(root / rel) for rel, _ in SOURCES.values()}
    data = load(root); style()
    if selection == "2":
        main_figure(data, out)
        assert len(QA["scatters"]) == 1 and len(QA["artists"]) == 66
    else:
        supplementary(data, out)
        assert len(QA["scatters"]) == 3
    check_workbooks(root, data)
    assert before == {rel: sha(root / rel) for rel, _ in SOURCES.values()}
    qa_dir = out / "qa"; qa_dir.mkdir(exist_ok=True)
    tag = names[0]
    with (qa_dir / (tag + "_SOURCE_CELLS.csv")).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["panel", "source", "csv_row", "column", "stored_string", "selector"])
        w.writeheader()
        for row in QA["cells"]:
            w.writerow({**row, "selector": json.dumps(row["selector"], sort_keys=True)})
    QA["source_hashes_unchanged"] = True
    QA["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__}
    QA["status"] = "PASS_SAVED_DATA_RENDER_ONLY"
    QA["manual_visual_review"] = "pending_for_this_render"
    (qa_dir / (tag + "_QA.json")).write_text(json.dumps(QA, indent=2, ensure_ascii=False) + "\n")
    return QA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/external_figures")
    parser.add_argument("--figure", choices=["2", "s2", "all"], default="all")
    args = parser.parse_args()
    for selection in (["2", "s2"] if args.figure == "all" else [args.figure]):
        result = render(ROOT, ROOT / args.output_dir, selection)
        print(result["status"], selection)


if __name__ == "__main__":
    main()

