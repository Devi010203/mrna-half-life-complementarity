from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import (
    FIGURE_DIR,
    MODEL_COLORS,
    FROZEN_ROOT,
    metric_rows,
    save_figure,
    save_plotted_data,
    set_paper_style,
    short_feature_label,
)


RIDGE_SUMMARY = FROZEN_ROOT / "ridge" / "summary_metrics.csv"
ELASTICNET_SUMMARY = FROZEN_ROOT / "elasticnet" / "summary_metrics.csv"
XGB_SUMMARY = FROZEN_ROOT / "xgboost" / "summary_metrics.csv"
OUTPUT_STEM = "fig2_feature_group_performance"

FEATURE_ORDER = [
    "saluki_basic8",
    "genetic_only",
    "biochemical_371",
    "genetic_plus_biochemical_371",
    "all_23384",
    "biochemical_full_no_basic",
    "basic_plus_biochemical_full",
    "genetic_plus_biochemical_full",
]
HIGHLIGHT = {
    "basic_plus_biochemical_full",
    "biochemical_full_no_basic",
    "genetic_only",
}


def main():
    set_paper_style()
    data = pd.concat(
        [
            metric_rows(RIDGE_SUMMARY, "Ridge"),
            metric_rows(ELASTICNET_SUMMARY, "ElasticNet"),
            metric_rows(XGB_SUMMARY, "XGBoost"),
        ],
        ignore_index=True,
    )
    data["feature_label"] = data["feature_group"].map(short_feature_label)
    order = [g for g in FEATURE_ORDER if g in set(data["feature_group"])]
    y_lookup = {group: idx for idx, group in enumerate(order)}
    offsets = {"Ridge": -0.22, "ElasticNet": 0.0, "XGBoost": 0.22}
    markers = {"Ridge": "o", "ElasticNet": "s", "XGBoost": "^"}

    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    for model in ["Ridge", "ElasticNet", "XGBoost"]:
        sub = data[data["model"] == model].copy()
        sub["y"] = sub["feature_group"].map(y_lookup) + offsets[model]
        ax.errorbar(
            sub["mean"],
            sub["y"],
            xerr=sub["std"],
            fmt=markers[model],
            markersize=4.2,
            linewidth=0.9,
            elinewidth=0.9,
            capsize=2.2,
            color=MODEL_COLORS[model],
            label=model,
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short_feature_label(g) for g in order])
    for tick, group in zip(ax.get_yticklabels(), order):
        if group in HIGHLIGHT:
            tick.set_fontweight("bold")
    ax.set_xlabel("Test $R^2$ (mean ± SD across folds)")
    ax.set_ylabel("Feature group")
    ax.set_xlim(0.20, 0.72)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Feature-group performance across tabular model classes")

    plotted = data.sort_values(["feature_group", "model"])
    save_plotted_data(plotted, OUTPUT_STEM)
    save_figure(fig, OUTPUT_STEM)
    print(FIGURE_DIR / f"{OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
