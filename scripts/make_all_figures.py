#!/usr/bin/env python
"""Regenerate all active publication figures from frozen result files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPTS = [
    "figure1_workflow.py",
    "figure2_feature_groups.py",
    "figure3_late_fusion.py",
    "figure4_fusion_gain.py",
    "figure5_shap_ablation.py",
    "supp_figure1_motif_kmer.py",
    "supp_figure2_case_studies.py",
    "supp_figure3_permutation.py",
    "supp_figure4_helped_hurt_shap.py",
]


def main() -> int:
    script_dir = ROOT / "scripts" / "figures"
    for name in FIGURE_SCRIPTS:
        path = script_dir / name
        print(f"[figure] {name}", flush=True)
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)
    print(f"Figures written to {ROOT / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

