from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import FIGURE_DIR, FROZEN_ROOT, MODEL_COLORS, save_figure, save_plotted_data, set_paper_style


PERM_SUMMARY = FROZEN_ROOT / "permutation" / "summary_metrics.csv"
PERM_FOLDS = FROZEN_ROOT / "permutation" / "fold_metrics.csv"
OUTPUT_STEM = "fig_s3_permutation_control"


def main():
    set_paper_style()
    summary = pd.read_csv(PERM_SUMMARY)
    folds = pd.read_csv(PERM_FOLDS)
    r2 = summary[summary["metric"] == "r2"].copy()
    pearson = summary[summary["metric"] == "pearson"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7))
    models = ["xgb", "ridge"]
    colors = [MODEL_COLORS["XGBoost"], MODEL_COLORS["Ridge"]]

    axes[0].bar(
        [m.upper() if m == "xgb" else "Ridge" for m in r2["model"]],
        r2["mean"],
        yerr=r2["std"],
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
        capsize=2.5,
    )
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("Test $R^2$")
    axes[0].set_title("A  Permuted labels")
    axes[0].grid(axis="y", color="#E6E6E6", linewidth=0.6)

    for i, model in enumerate(models):
        sub = folds[folds["model"] == model]
        axes[1].scatter(
            [i] * len(sub),
            sub["test_pearson"],
            s=16,
            color=colors[i],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.3,
        )
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["XGB", "Ridge"])
    axes[1].set_ylabel("Test Pearson")
    axes[1].set_title("B  Fold/seed correlations")
    axes[1].grid(axis="y", color="#E6E6E6", linewidth=0.6)

    fig.tight_layout(w_pad=1.2)
    plotted = pd.concat(
        [
            summary.assign(panel="A_summary"),
            folds.assign(panel="B_fold_seed"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_plotted_data(plotted, OUTPUT_STEM)
    save_figure(fig, OUTPUT_STEM)
    print(FIGURE_DIR / f"{OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
