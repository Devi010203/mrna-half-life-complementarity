from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import FIGURE_DIR, FROZEN_ROOT, save_figure, save_plotted_data, set_paper_style


SHAP_FAMILY_DIFF = FROZEN_ROOT / "helped_hurt_shap" / "family.csv"
SHAP_FEATURE_DIFF = FROZEN_ROOT / "helped_hurt_shap" / "feature.csv"
OUTPUT_STEM = "fig_s4_helped_hurt_shap"


def main():
    set_paper_style()
    family = pd.read_csv(SHAP_FAMILY_DIFF).sort_values("mean_difference_helped_minus_hurt")
    feature = pd.read_csv(SHAP_FEATURE_DIFF)
    feature["abs_diff"] = feature["mean_difference_helped_minus_hurt"].abs()
    top = feature.sort_values("abs_diff", ascending=False).head(15).sort_values("mean_difference_helped_minus_hurt")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), gridspec_kw={"width_ratios": [1, 1.35]})

    fam_colors = ["#E15759" if v < 0 else "#4C78A8" for v in family["mean_difference_helped_minus_hurt"]]
    axes[0].barh(family["family"], family["mean_difference_helped_minus_hurt"], color=fam_colors)
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_xlabel("Helped - hurt mean SHAP")
    axes[0].set_title("A  Family-level difference")
    axes[0].grid(axis="x", color="#E6E6E6", linewidth=0.6)

    feat_colors = ["#E15759" if v < 0 else "#4C78A8" for v in top["mean_difference_helped_minus_hurt"]]
    axes[1].barh(top["feature"], top["mean_difference_helped_minus_hurt"], color=feat_colors)
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_xlabel("Helped - hurt mean SHAP")
    axes[1].set_title("B  Top feature differences")
    axes[1].grid(axis="x", color="#E6E6E6", linewidth=0.6)

    fig.tight_layout(w_pad=1.2)
    plotted = pd.concat(
        [
            family.assign(panel="A_family_difference"),
            top.assign(panel="B_feature_difference"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_plotted_data(plotted, OUTPUT_STEM)
    save_figure(fig, OUTPUT_STEM)
    print(FIGURE_DIR / f"{OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
