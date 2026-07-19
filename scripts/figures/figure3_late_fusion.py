from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import (
    FIGURE_DIR,
    MODEL_COLORS,
    FROZEN_ROOT,
    clean_model_label,
    save_figure,
    save_plotted_data,
    set_paper_style,
)


MATCHED_SUMMARY = FROZEN_ROOT / "fusion" / "model_summary.csv"
PER_FOLD = FROZEN_ROOT / "fusion" / "per_fold_r2_coefficients.csv"
PREDICTION_PAIRS = FROZEN_ROOT / "fusion" / "oof_prediction_pairs.csv"
OUTPUT_STEM = "fig3_late_fusion"

MODEL_ORDER = [
    "xgb_basic_plus_biochemical_full",
    "mrnalm",
    "ridge_basic_plus_biochemical_full",
    "elasticnet_basic_plus_biochemical_full",
    "late_fusion_linear",
]


def fold_mean_corr(df: pd.DataFrame, x_col: str, y_col: str, method: str) -> float:
    values = []
    for _, sub in df.groupby("cv_fold", sort=True):
        values.append(sub[x_col].corr(sub[y_col], method=method))
    return float(pd.Series(values, dtype="float64").mean())


def main():
    set_paper_style()
    summary = pd.read_csv(MATCHED_SUMMARY)
    panel_a = summary[summary["metric"] == "r2"].copy()
    panel_a["model_label_short"] = panel_a["model"].map(clean_model_label)
    panel_a["order"] = panel_a["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    panel_a = panel_a.dropna(subset=["order"]).sort_values("order")
    panel_a["mean"] = panel_a["mean"].astype(float)
    panel_a["std"] = panel_a["std"].astype(float)

    fold = pd.read_csv(PER_FOLD)
    pairs = pd.read_csv(PREDICTION_PAIRS)
    pairs["xgb_residual"] = pairs["y_true"] - pairs["xgb_pred"]
    pairs["mrnalm_residual"] = pairs["y_true"] - pairs["mrnalm_pred"]

    fig = plt.figure(figsize=(7.8, 5.7))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.12], hspace=0.58, wspace=0.82)
    ax_model = fig.add_subplot(gs[0, 0:2])
    ax_delta = fig.add_subplot(gs[0, 2:4])
    ax_coef = fig.add_subplot(gs[0, 4:6])
    ax_pred = fig.add_subplot(gs[1, 0:3])
    ax_resid = fig.add_subplot(gs[1, 3:6])

    colors = [
        MODEL_COLORS["XGBoost"],
        MODEL_COLORS["mRNA-LM"],
        MODEL_COLORS["Ridge"],
        MODEL_COLORS["ElasticNet"],
        MODEL_COLORS["Fusion"],
    ]
    model_positions = list(range(len(panel_a)))
    ax_model.bar(
        model_positions,
        panel_a["mean"],
        yerr=panel_a["std"],
        color=colors[: len(panel_a)],
        edgecolor="#333333",
        linewidth=0.5,
        capsize=2.5,
    )
    ax_model.set_ylabel("Held-out test $R^2$")
    ax_model.set_ylim(0.35, 0.76)
    ax_model.set_title("A  Model comparison")
    ax_model.set_xticks(model_positions)
    ax_model.set_xticklabels(
        panel_a["model_label_short"],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax_model.tick_params(axis="x", pad=3)
    ax_model.grid(axis="y", color="#E6E6E6", linewidth=0.6)

    ax_delta.bar(
        fold["fold"].astype(str),
        fold["delta_r2_vs_xgb"],
        color=MODEL_COLORS["Fusion"],
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax_delta.axhline(0, color="#333333", linewidth=0.8)
    ax_delta.set_xlabel("Fold")
    ax_delta.set_ylabel("$\\Delta R^2$ vs XGB")
    ax_delta.set_title("B  Per-fold fusion gain")
    ax_delta.grid(axis="y", color="#E6E6E6", linewidth=0.6)

    ax_coef.plot(
        fold["fold"],
        fold["coef_xgb"],
        marker="o",
        color=MODEL_COLORS["XGBoost"],
        linewidth=1.1,
        label="$\\beta_{XGB}$",
    )
    ax_coef.plot(
        fold["fold"],
        fold["coef_mrnalm"],
        marker="s",
        color=MODEL_COLORS["mRNA-LM"],
        linewidth=1.1,
        label="$\\beta_{mRNA-LM}$",
    )
    ax_coef.set_xlabel("Fold")
    ax_coef.set_ylabel("Fusion coefficient")
    ax_coef.set_title("C  Linear fusion weights")
    ax_coef.set_xticks(fold["fold"])
    ax_coef.legend(frameon=False)
    ax_coef.grid(axis="y", color="#E6E6E6", linewidth=0.6)

    scatter_kwargs = dict(s=4, alpha=0.22, color="#4C78A8", linewidths=0, rasterized=True, zorder=2)
    ax_pred.scatter(pairs["xgb_pred"], pairs["mrnalm_pred"], **scatter_kwargs)
    ax_pred.set_xlabel("XGBoost prediction")
    ax_pred.set_ylabel("mRNA-LM prediction")
    ax_pred.set_title("D  Prediction correlation")
    pred_pearson = fold_mean_corr(pairs, "xgb_pred", "mrnalm_pred", "pearson")
    pred_spearman = fold_mean_corr(pairs, "xgb_pred", "mrnalm_pred", "spearman")
    ax_pred.text(
        0.04,
        0.96,
        f"Fold-mean r = {pred_pearson:.3f}\nFold-mean $\\rho$ = {pred_spearman:.3f}",
        transform=ax_pred.transAxes,
        va="top",
        ha="left",
        fontsize=7,
    )

    ax_resid.scatter(pairs["xgb_residual"], pairs["mrnalm_residual"], **scatter_kwargs)
    ax_resid.axhline(0, color="#666666", linewidth=0.7, zorder=1)
    ax_resid.axvline(0, color="#666666", linewidth=0.7, zorder=1)
    ax_resid.set_xlabel("XGBoost residual")
    ax_resid.set_ylabel("mRNA-LM residual")
    ax_resid.set_title("E  Residual correlation")
    resid_pearson = fold_mean_corr(pairs, "xgb_residual", "mrnalm_residual", "pearson")
    resid_spearman = fold_mean_corr(pairs, "xgb_residual", "mrnalm_residual", "spearman")
    ax_resid.text(
        0.04,
        0.96,
        f"Fold-mean r = {resid_pearson:.3f}\nFold-mean $\\rho$ = {resid_spearman:.3f}",
        transform=ax_resid.transAxes,
        va="top",
        ha="left",
        fontsize=7,
    )

    for ax in (ax_pred, ax_resid):
        ax.grid(True, color="#E6E6E6", linewidth=0.6, zorder=0)

    plotted = pd.concat(
        [
            panel_a.assign(panel="A_model_comparison"),
            fold.assign(panel="B_C_per_fold"),
            pairs.assign(panel="D_E_prediction_residual_scatter"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_plotted_data(plotted, OUTPUT_STEM)
    save_figure(fig, OUTPUT_STEM)
    print(FIGURE_DIR / f"{OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
