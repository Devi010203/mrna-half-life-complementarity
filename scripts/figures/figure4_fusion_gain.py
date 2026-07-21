from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import FIGURE_DIR, FROZEN_ROOT, GROUP_COLORS, save_figure, save_plotted_data, set_paper_style


PROFILE = FROZEN_ROOT / "fusion_gain" / "sample_profile.csv"
OUTPUT_STEM = "fig4_fusion_gain_profile"

GROUP_ORDER = ["fusion_helped_top20", "neutral_middle60", "fusion_hurt_top20"]
GROUP_LABELS = {
    "fusion_helped_top20": "Helped\nTop 20%",
    "neutral_middle60": "Neutral\nMiddle 60%",
    "fusion_hurt_top20": "Hurt\nBottom 20%",
}


def grouped_boxplot(ax, df, value_col, ylabel, title):
    data = [df.loc[df["fusion_gain_group_within_fold"] == g, value_col].dropna() for g in GROUP_ORDER]
    box = ax.boxplot(
        data,
        labels=[GROUP_LABELS[g] for g in GROUP_ORDER],
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, group in zip(box["boxes"], GROUP_ORDER):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_alpha(1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)


def main():
    set_paper_style()
    df = pd.read_csv(PROFILE)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
    grouped_boxplot(axes[0], df, "HALFLIFE", "half-life", "A  Target distribution")
    grouped_boxplot(axes[1], df, "xgb_residual", "XGB residual\n(y true - y pred)", "B  XGB residuals")

    positions = [0.85, 1.15, 1.85, 2.15]
    data = [
        df.loc[df["fusion_gain_group_within_fold"] == "fusion_helped_top20", "abs_err_xgb"].dropna(),
        df.loc[df["fusion_gain_group_within_fold"] == "fusion_helped_top20", "abs_err_fusion"].dropna(),
        df.loc[df["fusion_gain_group_within_fold"] == "fusion_hurt_top20", "abs_err_xgb"].dropna(),
        df.loc[df["fusion_gain_group_within_fold"] == "fusion_hurt_top20", "abs_err_fusion"].dropna(),
    ]
    box = axes[2].boxplot(
        data,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], ["#E15759", "#B07AA1", "#E15759", "#B07AA1"]):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
    axes[2].set_xticks([1.0, 2.0])
    axes[2].set_xticklabels(["Helped", "Hurt"])
    axes[2].set_ylabel("Absolute error")
    axes[2].set_title("C  XGB vs fusion error")
    axes[2].legend(
        handles=[
            plt.Line2D([0], [0], marker="s", linestyle="", color="#E15759", label="XGB"),
            plt.Line2D([0], [0], marker="s", linestyle="", color="#B07AA1", label="Fusion"),
        ],
        frameon=False,
        loc="upper right",
    )
    axes[2].grid(axis="y", color="#E6E6E6", linewidth=0.6)

    fig.tight_layout(w_pad=1.2)
    save_plotted_data(df, OUTPUT_STEM)
    save_figure(fig, OUTPUT_STEM)
    print(FIGURE_DIR / f"{OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
