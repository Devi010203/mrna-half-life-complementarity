from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import FIGURE_DIR, FROZEN_ROOT, MODEL_COLORS, save_figure, save_plotted_data, set_paper_style


CASE_STUDIES = FROZEN_ROOT / "case_studies" / "combined.csv"
OUTPUT_STEM = "fig_s2_case_study_predictions"


def plot_cases(ax, df, title):
    selected = df.copy()
    x_lookup = {sample: i for i, sample in enumerate(selected["sample_id"])}
    series = [
        ("True", "y_true", "#222222", "o"),
        ("XGB", "xgb_pred", MODEL_COLORS["XGBoost"], "s"),
        ("mRNA-LM", "mrnalm_pred", MODEL_COLORS["mRNA-LM"], "^"),
        ("Fusion", "fusion_pred", MODEL_COLORS["Fusion"], "D"),
    ]
    offsets = {"True": -0.24, "XGB": -0.08, "mRNA-LM": 0.08, "Fusion": 0.24}
    for label, col, color, marker in series:
        ax.scatter(
            [x_lookup[s] + offsets[label] for s in selected["sample_id"]],
            selected[col],
            s=20,
            color=color,
            marker=marker,
            label=label,
            zorder=3,
        )
    for _, row in selected.iterrows():
        x = x_lookup[row["sample_id"]]
        ax.plot(
            [x - 0.08, x + 0.24],
            [row["xgb_pred"], row["fusion_pred"]],
            color="#BDBDBD",
            linewidth=0.8,
            zorder=1,
        )
    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels(selected["sample_id"], rotation=45, ha="right")
    ax.set_ylabel("HALFLIFE / prediction")
    ax.set_title(title)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)


def main():
    set_paper_style()
    df = pd.read_csv(CASE_STUDIES)
    helped = df[df["case_group"] == "top_helped"].sort_values("fusion_gain", ascending=False).head(6)
    hurt = df[df["case_group"] == "top_hurt"].sort_values("fusion_gain", ascending=True).head(6)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    plot_cases(axes[0], helped, "A  Top fusion-helped cases")
    plot_cases(axes[1], hurt, "B  Top fusion-hurt cases")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout(w_pad=1.2)

    plotted = pd.concat(
        [helped.assign(panel="A_helped"), hurt.assign(panel="B_hurt")],
        ignore_index=True,
        sort=False,
    )
    save_plotted_data(plotted, OUTPUT_STEM)
    save_figure(fig, OUTPUT_STEM)
    print(FIGURE_DIR / f"{OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
