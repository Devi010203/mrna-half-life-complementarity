#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect per-shard ElasticNet SLURM-array outputs into merged CSV files."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG: paths are resolved from environment first. The SLURM collector sets
# SALUKI_ENET_COLLECT_ROOT and SALUKI_ENET_COLLECT_RUN_TAG explicitly.
# =============================================================================
def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else Path(default)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


PROJECT_ROOT = env_path("SALUKI_PROJECT_ROOT", Path(__file__).resolve().parents[2])
RESULT_ROOT = env_path("SALUKI_RESULT_ROOT", PROJECT_ROOT / "results" / "runs")
OUT_BASE_DIR = env_path(
    "SALUKI_ENET_COLLECT_ROOT",
    RESULT_ROOT / "elasticnet_saluki_regulatory_features_human",
)
RUN_TAG = env_str("SALUKI_ENET_COLLECT_RUN_TAG", "elasticnet_saluki_regulatory_features_human_clean_codex1_full8_array")
ARRAY_ROOT = OUT_BASE_DIR / RUN_TAG
SHARDS_ROOT = ARRAY_ROOT / "shards"
EXPECTED_GROUPS = [
    "saluki_basic8",
    "genetic_only",
    "biochemical_371",
    "biochemical_full_no_basic",
    "basic_plus_biochemical_full",
    "genetic_plus_biochemical_371",
    "all_23384",
    "genetic_plus_biochemical_full",
]
EXPECTED_FOLDS = [0, 1, 2, 3, 4]
REQUIRE_ALL_SHARDS = True


CSV_OUTPUTS = {
    "elasticnet_fold_metrics.csv": "elasticnet_fold_metrics_merged.csv",
    "selected_params_per_fold.csv": "selected_params_per_fold_merged.csv",
    "pred_val.csv": "pred_val_merged.csv",
    "pred_test.csv": "pred_test_merged.csv",
    "nonzero_coefficients_per_fold.csv": "nonzero_coefficients_per_fold_merged.csv",
    "top_coefficients_by_group.csv": "top_coefficients_by_group_merged.csv",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def config_dict() -> dict[str, Any]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "timestamp": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "result_root": str(RESULT_ROOT),
        "out_base_dir": str(OUT_BASE_DIR),
        "run_tag": RUN_TAG,
        "array_root": str(ARRAY_ROOT),
        "shards_root": str(SHARDS_ROOT),
        "expected_groups": list(EXPECTED_GROUPS),
        "expected_folds": list(EXPECTED_FOLDS),
        "require_all_shards": bool(REQUIRE_ALL_SHARDS),
        "csv_outputs": dict(CSV_OUTPUTS),
        "env_overrides": {
            "SALUKI_PROJECT_ROOT": os.environ.get("SALUKI_PROJECT_ROOT"),
            "SALUKI_RESULT_ROOT": os.environ.get("SALUKI_RESULT_ROOT"),
            "SALUKI_ENET_COLLECT_ROOT": os.environ.get("SALUKI_ENET_COLLECT_ROOT"),
            "SALUKI_ENET_COLLECT_RUN_TAG": os.environ.get("SALUKI_ENET_COLLECT_RUN_TAG"),
        },
    }


def print_config() -> None:
    print("[CONFIG]", flush=True)
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False), flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def expected_shard_dirs() -> list[Path]:
    return [SHARDS_ROOT / group / f"fold_{fold}" for group in EXPECTED_GROUPS for fold in EXPECTED_FOLDS]


def validate_shards() -> list[Path]:
    if not SHARDS_ROOT.exists():
        raise FileNotFoundError(f"Missing SHARDS_ROOT: {SHARDS_ROOT}")
    expected = expected_shard_dirs()
    missing_dirs = [str(path) for path in expected if not path.exists()]
    if missing_dirs and REQUIRE_ALL_SHARDS:
        raise FileNotFoundError(f"Missing expected shard directories: {missing_dirs}")
    shard_dirs = [path for path in expected if path.exists()]
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories found under expected layout: {SHARDS_ROOT}")

    missing_files: dict[str, list[str]] = {}
    for shard_dir in shard_dirs:
        for filename in CSV_OUTPUTS:
            if not (shard_dir / filename).exists():
                missing_files.setdefault(filename, []).append(str(shard_dir / filename))
    if missing_files and REQUIRE_ALL_SHARDS:
        raise FileNotFoundError(f"Missing expected shard CSV files: {missing_files}")
    return shard_dirs


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        col
        for col in fold_metrics.columns
        if col.startswith("train_") or col.startswith("val_") or col.startswith("test_")
    ]
    for group, sub in fold_metrics.groupby("feature_group", sort=False):
        label = str(sub["feature_label"].iloc[0]) if "feature_label" in sub.columns else group
        for col in metric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            split, metric = col.split("_", 1)
            rows.append(
                {
                    "feature_group": group,
                    "feature_label": label,
                    "split": split,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "n_folds": int(values.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def collect_csv(shard_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for shard_dir in shard_dirs:
        path = shard_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.insert(0, "source_shard", str(shard_dir))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if {"feature_group", "cv_fold"}.issubset(merged.columns):
        sort_cols = ["feature_group", "cv_fold"]
        if "ENSID" in merged.columns:
            sort_cols.append("ENSID")
        elif "feature" in merged.columns:
            sort_cols.append("feature")
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
    return merged


def main() -> None:
    print_config()
    shard_dirs = validate_shards()
    ARRAY_ROOT.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "config": config_dict(),
        "n_shards_found": int(len(shard_dirs)),
        "shards": [str(path) for path in shard_dirs],
        "outputs": {},
        "completed_at": now_iso(),
    }

    merged_by_input: dict[str, pd.DataFrame] = {}
    for input_name, output_name in CSV_OUTPUTS.items():
        merged = collect_csv(shard_dirs, input_name)
        output_path = ARRAY_ROOT / output_name
        merged.to_csv(output_path, index=False)
        merged_by_input[input_name] = merged
        manifest["outputs"][output_name] = {
            "path": str(output_path),
            "n_rows": int(len(merged)),
            "n_columns": int(len(merged.columns)),
        }
        print(f"[COLLECT] {input_name} -> {output_path} rows={len(merged)}", flush=True)

    fold_metrics = merged_by_input.get("elasticnet_fold_metrics.csv", pd.DataFrame())
    if not fold_metrics.empty:
        summary = summarize_metrics(fold_metrics)
        summary_path = ARRAY_ROOT / "elasticnet_summary_metrics_merged.csv"
        summary.to_csv(summary_path, index=False)
        manifest["outputs"]["elasticnet_summary_metrics_merged.csv"] = {
            "path": str(summary_path),
            "n_rows": int(len(summary)),
            "n_columns": int(len(summary.columns)),
        }
        expected_n = len(EXPECTED_GROUPS) * len(EXPECTED_FOLDS)
        observed_n = int(len(fold_metrics[["feature_group", "cv_fold"]].drop_duplicates()))
        manifest["fold_coverage"] = {
            "expected_group_fold_count": int(expected_n),
            "observed_group_fold_count": observed_n,
            "complete": bool(observed_n == expected_n),
        }
        if REQUIRE_ALL_SHARDS and observed_n != expected_n:
            raise RuntimeError(f"Incomplete fold coverage: observed={observed_n}, expected={expected_n}")
    else:
        manifest["fold_coverage"] = {
            "expected_group_fold_count": int(len(EXPECTED_GROUPS) * len(EXPECTED_FOLDS)),
            "observed_group_fold_count": 0,
            "complete": False,
        }

    write_json(ARRAY_ROOT / "elasticnet_array_collect_manifest.json", manifest)
    print(f"[DONE] collected ElasticNet array outputs into: {ARRAY_ROOT}", flush=True)


if __name__ == "__main__":
    main()
