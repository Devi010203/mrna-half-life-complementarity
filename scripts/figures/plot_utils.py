from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "figures"
FROZEN_ROOT = PROJECT_ROOT / "results" / "frozen"

MODEL_COLORS = {
    "Ridge": "#4C78A8",
    "ElasticNet": "#59A14F",
    "XGBoost": "#E15759",
    "mRNA-LM": "#7F7F7F",
    "Fusion": "#B07AA1",
}

GROUP_COLORS = {
    "fusion_helped_top20": "#4C78A8",
    "neutral_middle60": "#9C9C9C",
    "fusion_hurt_top20": "#E15759",
}


def set_paper_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig, stem: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(
            FIGURE_DIR / f"{stem}.{ext}",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def save_plotted_data(df: pd.DataFrame, stem: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIGURE_DIR / f"{stem}_plotted_data.csv", index=False)


def metric_rows(
    csv_path: Path,
    model_name: str,
    split: str = "test",
    metric: str = "r2",
    group_col: str = "feature_group",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = df[(df["split"] == split) & (df["metric"] == metric)].copy()
    out["model"] = model_name
    out = out[[group_col, "model", "mean", "std"]].rename(columns={group_col: "feature_group"})
    out["mean"] = out["mean"].astype(float)
    out["std"] = out["std"].astype(float)
    return out


def short_feature_label(value: str) -> str:
    labels = {
        "saluki_basic8": "Basic architecture",
        "genetic_only": "Sequence-statistical",
        "biochemical_371": "Compact biochemical",
        "genetic_plus_biochemical_371": "Compact bio. + seq. stat.",
        "all_23384": "All tabular features",
        "biochemical_full_no_basic": "Full regulatory",
        "basic_plus_biochemical_full": "Full regulatory + basic",
        "genetic_plus_biochemical_full": "Full regulatory + seq. stat.",
    }
    return labels.get(value, value.replace("_", " "))


def clean_model_label(value: str) -> str:
    labels = {
        "xgb_basic_plus_biochemical_full": "XGB only",
        "mrnalm": "mRNA-LM only",
        "late_fusion_linear": "Linear fusion",
        "ridge_basic_plus_biochemical_full": "Ridge",
        "elasticnet_basic_plus_biochemical_full": "ElasticNet",
    }
    return labels.get(value, value.replace("_", " "))
