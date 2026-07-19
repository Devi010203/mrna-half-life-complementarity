import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results" / "frozen"


def summary_value(path: Path, **filters) -> tuple[float, float]:
    frame = pd.read_csv(path)
    for column, value in filters.items():
        frame = frame[frame[column] == value]
    if len(frame) != 1:
        raise AssertionError(f"Expected one row in {path}, found {len(frame)}")
    return float(frame.iloc[0]["mean"]), float(frame.iloc[0]["std"])


class FrozenResultTests(unittest.TestCase):
    def test_primary_tabular_results(self):
        mean, sd = summary_value(
            FROZEN / "xgboost" / "summary_metrics.csv",
            feature_group="basic_plus_biochemical_full",
            split="test",
            metric="r2",
        )
        self.assertAlmostEqual(mean, 0.676616082817748, places=12)
        self.assertAlmostEqual(sd, 0.0056853992476995, places=12)

        mean, sd = summary_value(
            FROZEN / "elasticnet" / "summary_metrics.csv",
            feature_group="genetic_plus_biochemical_full",
            split="test",
            metric="r2",
        )
        self.assertAlmostEqual(mean, 0.6373993363479694, places=12)
        self.assertAlmostEqual(sd, 0.0141454707070446, places=12)

    def test_sequence_and_fusion_results(self):
        summary = pd.read_csv(FROZEN / "fusion" / "model_summary.csv")
        r2 = summary[summary["metric"] == "r2"].set_index("model")

        self.assertAlmostEqual(r2.loc["mrnalm", "mean"], 0.4436297484408937, places=12)
        self.assertAlmostEqual(r2.loc["mrnalm", "std"], 0.0112215951784612, places=12)
        self.assertAlmostEqual(r2.loc["late_fusion_linear", "mean"], 0.7007589021237977, places=12)
        self.assertAlmostEqual(r2.loc["late_fusion_linear", "std"], 0.0079631207967958, places=12)
        self.assertEqual(int(r2.loc["late_fusion_linear", "n_total"]), 12981)

        fold = pd.read_csv(FROZEN / "fusion" / "per_fold_r2_coefficients.csv").sort_values("fold")
        expected = [0.0281, 0.0220, 0.0220, 0.0217, 0.0269]
        for observed, target in zip(fold["delta_r2_vs_xgb"], expected):
            self.assertAlmostEqual(observed, target, delta=5e-5)

    def test_permutation_controls(self):
        summary = pd.read_csv(FROZEN / "permutation" / "summary_metrics.csv")
        r2 = summary[summary["metric"] == "r2"].set_index("model")
        self.assertAlmostEqual(r2.loc["xgb", "mean"], -0.0029516353305744, places=12)
        self.assertAlmostEqual(r2.loc["ridge", "mean"], -0.0095290695877908, places=12)

    def test_split_size(self):
        split = pd.read_csv(ROOT / "data" / "processed" / "human" / "splits" / "cv_k5_seed42.csv")
        self.assertEqual(len(split), 12981)
        self.assertTrue(split["ENSID"].is_unique)
        self.assertEqual(split.groupby("fold").size().sort_index().tolist(), [2600, 2600, 2600, 2600, 2581])


if __name__ == "__main__":
    unittest.main()
