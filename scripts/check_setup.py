#!/usr/bin/env python
"""Check release assets and report optional inputs needed for full retraining."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_RELEASE_FILES = [
    ROOT / "data/processed/human/splits/cv_k5_seed42.csv",
    ROOT / "results/frozen/ridge/fold_metrics.csv",
    ROOT / "results/frozen/xgboost/fold_metrics.csv",
    ROOT / "results/frozen/elasticnet/fold_metrics.csv",
    ROOT / "results/frozen/fusion/model_summary.csv",
    ROOT / "results/frozen/fusion/oof_prediction_pairs.csv",
    ROOT / "results/frozen/shap/family_summary.csv",
    ROOT / "results/frozen/ablation/summary_metrics.csv",
    ROOT / "results/frozen/fusion_gain/sample_profile.csv",
    ROOT / "results/frozen/motif/motif_enrichment.csv",
    ROOT / "results/frozen/permutation/summary_metrics.csv",
]

OPTIONAL_FULL_RUN_INPUTS = [
    ROOT / "data/processed/human/normalized.csv.gz",
    ROOT / "data/saluki_features/human/alldataGeneticBiochemical.tsv.gz",
    ROOT / "data/saluki_features/human/alldataBiochemical.tsv.gz",
    ROOT / "models/mrna_5utr_model_p2_cp85600_best",
    ROOT / "models/codonbert",
    ROOT / "models/mrna_3utr_model_p2_cp99900_best",
]


def main() -> int:
    missing_release = [path for path in REQUIRED_RELEASE_FILES if not path.exists()]
    missing_full = [path for path in OPTIONAL_FULL_RUN_INPUTS if not path.exists()]

    print(f"Repository root: {ROOT}")
    print(f"Frozen release inputs: {len(REQUIRED_RELEASE_FILES) - len(missing_release)}/{len(REQUIRED_RELEASE_FILES)} present")
    if missing_release:
        print("ERROR: required release files are missing:")
        for path in missing_release:
            print(f"  - {path.relative_to(ROOT)}")
    else:
        print("OK: figure reproduction inputs are complete.")

    if missing_full:
        print("Optional full-training inputs not installed:")
        for path in missing_full:
            print(f"  - {path.relative_to(ROOT)}")
    else:
        print("OK: processed data, features, and checkpoints are present for full runs.")

    return 1 if missing_release else 0


if __name__ == "__main__":
    raise SystemExit(main())

