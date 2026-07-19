from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_utils import FIGURE_DIR, PROJECT_ROOT, save_figure, save_plotted_data, set_paper_style


OUTPUT_STEM = "fig1_workflow_schematic"


def draw_box(ax, xy, width, height, label, facecolor="#F7F7F7"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=0.9,
        edgecolor="#333333",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=7.5,
        linespacing=1.2,
    )


def draw_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color="#333333",
            shrinkA=1,
            shrinkB=1,
        )
    )


def main():
    set_paper_style()

    nodes = pd.DataFrame(
        [
            ("saluki", 0.03, 0.73, "Saluki human\nhalf-life data"),
            ("target", 0.03, 0.40, "Normalized\nHALFLIFE target"),
            ("features", 0.29, 0.73, "Saluki-derived\nfeature groups"),
            ("tabular", 0.53, 0.73, "Full tabular benchmark\nN = 12,981"),
            ("sequence", 0.29, 0.40, "mRNA sequence\nonly"),
            ("mrnalm", 0.53, 0.40, "mRNA-LM OOF\npredictions"),
            ("fusion", 0.76, 0.58, "Matched XGB/mRNA-LM\nfusion N = 12,981"),
            ("interpret", 0.76, 0.22, "SHAP, ablation,\nfusion-gain profiling"),
        ],
        columns=["node", "x", "y", "label"],
    )
    edges = pd.DataFrame(
        [
            ("saluki", "features", "right", "left"),
            ("saluki", "target", "bottom", "top"),
            ("target", "sequence", "right", "left"),
            ("features", "tabular", "right", "left"),
            ("sequence", "mrnalm", "right", "left"),
            ("tabular", "fusion", "right", "left"),
            ("mrnalm", "fusion", "right", "left"),
            ("fusion", "interpret", "bottom", "top"),
        ],
        columns=["source", "target", "source_side", "target_side"],
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_w, box_h = 0.18, 0.17
    for _, row in nodes.iterrows():
        color = "#FFFFFF"
        if row["node"] in {"tabular", "fusion"}:
            color = "#F2F2F2"
        if row["node"] == "mrnalm":
            color = "#F6F6F6"
        draw_box(ax, (row["x"], row["y"]), box_w, box_h, row["label"], color)

    anchors = {}
    for _, row in nodes.iterrows():
        x, y = row["x"], row["y"]
        anchors[row["node"]] = {
            "left": (x, y + box_h / 2),
            "right": (x + box_w, y + box_h / 2),
            "top": (x + box_w / 2, y + box_h),
            "bottom": (x + box_w / 2, y),
        }
    for _, row in edges.iterrows():
        draw_arrow(
            ax,
            anchors[row["source"]][row["source_side"]],
            anchors[row["target"]][row["target_side"]],
        )

    ax.text(
        0.02,
        0.055,
        "Clean current scope: full tabular benchmark N = 12,981; matched XGB/mRNA-LM fusion scope N = 12,981.",
        fontsize=7,
        ha="left",
        va="bottom",
        color="#333333",
    )
    ax.text(
        0.02,
        0.02,
        "Full benchmark and fusion scopes are tracked explicitly; all reported predictions are held out by fold.",
        fontsize=7,
        ha="left",
        va="bottom",
        color="#333333",
    )

    plotted = pd.concat(
        [
            nodes.assign(record_type="node"),
            edges.assign(record_type="edge"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_plotted_data(plotted, OUTPUT_STEM)
    save_figure(fig, OUTPUT_STEM)
    print(FIGURE_DIR / f"{OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
