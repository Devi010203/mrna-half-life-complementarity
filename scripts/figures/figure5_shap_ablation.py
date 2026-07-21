from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import FIGURE_DIR, FROZEN_ROOT, save_figure, save_plotted_data, set_paper_style


SHAP_FAMILY = FROZEN_ROOT / "shap" / "family_summary.csv"
SHAP_TOP = FROZEN_ROOT / "shap" / "top_features.csv"
ABLATION_SUMMARY = FROZEN_ROOT / "ablation" / "summary_metrics.csv"
OUTPUT_STEM = "fig5_shap_and_family_ablation"

FAMILY_DISPLAY_ORDER = [
    "BASIC",
    "RIP",
    "ENCORI",
    "eCLIP",
    "M6ACLIP",
    "PARCLIP",
    "TE",
]

ABLATION_LABELS = {
    "remove_basic": "Remove BASIC",
    "remove_rip": "Remove RIP",
    "remove_encori": "Remove ENCORI",
    "remove_eclip": "Remove eCLIP",
    "remove_rip_encori": "Remove RIP + ENCORI",
    "only_basic": "BASIC only",
    "only_rip": "RIP only",
    "only_encori": "ENCORI only",
    "only_eclip": "eCLIP only",
    "only_regulatory_no_basic": "Full regulatory only",
}


def main():
    set_paper_style()
    family = pd.read_csv(SHAP_FAMILY)
    family_rank = {name: rank for rank, name in enumerate(FAMILY_DISPLAY_ORDER)}
    family["_display_rank"] = family["family"].map(family_rank).fillna(-1).astype(int)
    # Horizontal bars are drawn from bottom to top, so reverse the manuscript order.
    family = family.sort_values("_display_rank", ascending=False)
    family["mean_abs_shap_per_feature"] = family["sum_abs_shap"].astype(float) / family["n_features"].astype(float)
    family["top50_fraction"] = family["top50_count"].astype(float) / family["n_features"].astype(float)
    top = pd.read_csv(SHAP_TOP).head(15).sort_values("mean_abs_shap", ascending=True)
    ablation = pd.read_csv(ABLATION_SUMMARY)
    full_r2 = float(
        ablation.loc[
            ablation["feature_set"] == "full_basic_plus_biochemical_full",
            "test_r2_mean",
        ].iloc[0]
    )
    ablation = ablation[ablation["feature_set"] != "full_basic_plus_biochemical_full"].copy()
    ablation["delta_r2_vs_full"] = ablation["test_r2_mean"].astype(float) - full_r2
    ablation["feature_set_label"] = ablation["feature_set"].map(ABLATION_LABELS).fillna(
        ablation["feature_set"].str.replace("_", " ", regex=False)
    )
    ablation = ablation.sort_values("delta_r2_vs_full", ascending=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.0))
    ax_sum, ax_norm, ax_top, ax_ablation = axes.ravel()

    y_family = np.arange(len(family))
    ax_sum.barh(y_family, family["sum_abs_shap"], color="#6B6B6B", zorder=2)
    ax_sum.set_yticks(y_family)
    ax_sum.set_yticklabels(family["family"])
    ax_sum.set_xlabel("Family-summed |SHAP|")
    ax_sum.set_title("A  Family-summed SHAP")
    ax_sum.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)

    ax_norm.barh(y_family, family["mean_abs_shap_per_feature"], color="#59A14F", zorder=2)
    ax_norm.set_yticks(y_family)
    ax_norm.set_yticklabels([])
    ax_norm.set_xlabel("Mean |SHAP| per feature")
    ax_norm.set_title("B  Feature-count normalized")
    ax_norm.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)
    x_max = float(family["mean_abs_shap_per_feature"].max())
    ax_norm.set_xlim(0, x_max * 1.35)
    for y, row in zip(y_family, family.itertuples(index=False)):
        label = f"{int(row.top50_count)}/{int(row.n_features)}"
        ax_norm.text(
            float(row.mean_abs_shap_per_feature) + x_max * 0.03,
            y,
            label,
            va="center",
            ha="left",
            fontsize=6.5,
            color="black",
        )
    ax_norm.text(
        0.98,
        0.46,
        "labels: top-50 features / family size",
        transform=ax_norm.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="black",
    )

    ax_top.barh(top["feature"], top["mean_abs_shap"], color="#808080", zorder=2)
    ax_top.set_xlabel("Mean |SHAP|")
    ax_top.set_title("C  Top features")
    ax_top.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)

    colors = ["#E15759" if v < 0 else "#59A14F" for v in ablation["delta_r2_vs_full"]]
    ax_ablation.barh(ablation["feature_set_label"], ablation["delta_r2_vs_full"], color=colors, zorder=2)
    ax_ablation.axvline(0, color="#333333", linewidth=0.8)
    ax_ablation.set_xlabel("$\\Delta R^2$ relative to the full XGBoost model")
    ax_ablation.set_title("D  Family ablation")
    ax_ablation.grid(axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)

    fig.tight_layout(w_pad=1.2, h_pad=1.35)
    plotted = pd.concat(
        [
            family.drop(columns="_display_rank").assign(panel="A_family_summed_shap"),
            family.drop(columns="_display_rank").assign(panel="B_family_normalized_shap"),
            top.assign(panel="C_top_features"),
            ablation.assign(panel="D_family_ablation"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_plotted_data(plotted, OUTPUT_STEM)
    save_figure(fig, OUTPUT_STEM)
    print(FIGURE_DIR / f"{OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
