from math import log10

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import FIGURE_DIR, FROZEN_ROOT, save_figure, save_plotted_data, set_paper_style


MOTIF_ENRICHMENT = FROZEN_ROOT / "motif" / "motif_enrichment.csv"
REGION_CHECKS = FROZEN_ROOT / "motif" / "region_checks.csv"
OUTPUT_STEM = "fig_s1_motif_kmer_region_checks"


def neg_log10(value):
    return -log10(max(float(value), 1e-300))


def main():
    set_paper_style()
    df = pd.read_csv(MOTIF_ENRICHMENT)
    df["neg_log10_q"] = df["q_value_bh_fdr"].map(neg_log10)
    df["feature_label"] = df["region"].astype(str) + " " + df["feature"].astype(str)
    top = df.sort_values("q_value_bh_fdr", ascending=True).head(30).copy()
    top = top.sort_values("neg_log10_q", ascending=True)

    region = pd.read_csv(REGION_CHECKS)
    region["feature_label"] = region["region"].astype(str) + " " + region["metric"].astype(str)
    region["neg_log10_q"] = region["q_value_bh_fdr"].map(neg_log10)
    region = region.sort_values("neg_log10_q", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw={"width_ratios": [1.55, 1.0]})
    colors = top["feature_type"].map({"kmer": "#6B6B6B", "motif": "#4C78A8"}).fillna("#808080")
    axes[0].barh(top["feature_label"], top["neg_log10_q"], color=colors)
    axes[0].axvline(-log10(0.05), color="#B00020", linestyle="--", linewidth=0.9, label="FDR 0.05")
    axes[0].set_xlabel("-log10(FDR q-value)")
    axes[0].set_ylabel("Low-order feature")
    axes[0].set_title("A  Motif/k-mer checks")
    axes[0].grid(axis="x", color="#E6E6E6", linewidth=0.6)
    axes[0].legend(frameon=False, loc="lower right")

    region_colors = region["metric"].map({"length": "#4C78A8", "gc": "#59A14F"}).fillna("#808080")
    axes[1].barh(region["feature_label"], region["neg_log10_q"], color=region_colors)
    axes[1].axvline(-log10(0.05), color="#B00020", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("-log10(FDR q-value)")
    axes[1].set_title("B  Region length/GC")
    axes[1].grid(axis="x", color="#E6E6E6", linewidth=0.6)

    fig.tight_layout(w_pad=1.2)
    plotted = pd.concat(
        [
            top.assign(panel="A_motif_kmer"),
            region.assign(panel="B_region_length_gc"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_plotted_data(plotted, OUTPUT_STEM)
    save_figure(fig, OUTPUT_STEM)
    print(FIGURE_DIR / f"{OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
