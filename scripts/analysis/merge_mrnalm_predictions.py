#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Merge per-fold mRNA-LM prediction CSV files into batch-level files."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# CONFIG: paths are resolved from environment first. Defaults point to the
# local clean human mRNA-LM anchor used for the current manuscript analyses.
# =============================================================================
def env_path(name: str, default: Path | str) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value) if value else Path(default)


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


PROJECT_ROOT = env_path("SALUKI_PROJECT_ROOT", Path(__file__).resolve().parents[2])
RUNS_ROOT = env_path("SALUKI_RUNS_ROOT", PROJECT_ROOT / "results" / "runs")
MRNALM_BATCH_ROOT = env_path(
    "SALUKI_MRNALM_MERGE_BATCH_ROOT",
    RUNS_ROOT / "mrnalm_agghead",
)
SUBRUN_NAME = env_str("SALUKI_MRNALM_MERGE_SUBRUN_NAME", "mrnalm_triplet_human_matched_lora")
MODEL_NAME = env_str("SALUKI_MRNALM_MERGE_MODEL_NAME", "mrnalm_best")
CV_FOLDS = [0, 1, 2, 3, 4]

VAL_FILENAME = "pred_val.csv"
TEST_FILENAME = "pred_test.csv"
OUT_VAL_PATH = MRNALM_BATCH_ROOT / "pred_val.csv"
OUT_TEST_PATH = MRNALM_BATCH_ROOT / "pred_test.csv"
OUT_CONFIG_PATH = MRNALM_BATCH_ROOT / "merge_mrnalm_fold_predictions_config.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def config_dict() -> dict[str, Any]:
    return {
        "script_path": str(Path(__file__).resolve()),
        "timestamp": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "mrnalm_batch_root": str(MRNALM_BATCH_ROOT),
        "subrun_name": SUBRUN_NAME,
        "model_name": MODEL_NAME,
        "cv_folds": list(CV_FOLDS),
        "val_filename": VAL_FILENAME,
        "test_filename": TEST_FILENAME,
        "out_val_path": str(OUT_VAL_PATH),
        "out_test_path": str(OUT_TEST_PATH),
        "out_config_path": str(OUT_CONFIG_PATH),
    }


def print_config() -> None:
    print("[CONFIG]")
    print(json.dumps(config_dict(), indent=2, ensure_ascii=False))


def fold_prediction_path(fold: int, filename: str) -> Path:
    return MRNALM_BATCH_ROOT / f"cv_fold_{int(fold):02d}" / SUBRUN_NAME / filename


def validate_inputs() -> None:
    if not MRNALM_BATCH_ROOT.exists():
        print_config()
        raise FileNotFoundError(f"Missing MRNALM_BATCH_ROOT: {MRNALM_BATCH_ROOT}")
    missing: list[str] = []
    for fold in CV_FOLDS:
        for filename in (VAL_FILENAME, TEST_FILENAME):
            path = fold_prediction_path(int(fold), filename)
            if not path.exists():
                missing.append(str(path))
    if missing:
        print_config()
        raise FileNotFoundError(f"Missing fold prediction file(s): {missing}")


def normalize_prediction_frame(path: Path, fold: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ENSID", "y_true", "y_pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df.loc[:, ["ENSID", "y_true", "y_pred"]].copy()
    out.insert(1, "cv_fold", int(fold))
    out.insert(2, "model_name", MODEL_NAME)
    out["ENSID"] = out["ENSID"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise")
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="raise")
    duplicated = out.duplicated(["ENSID", "cv_fold"], keep=False)
    if duplicated.any():
        examples = out.loc[duplicated, ["ENSID", "cv_fold"]].head(10).to_dict("records")
        raise ValueError(f"{path} has duplicate ENSID/cv_fold rows: {examples}")
    return out


def merge_split(filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in CV_FOLDS:
        path = fold_prediction_path(int(fold), filename)
        frames.append(normalize_prediction_frame(path, int(fold)))
    merged = pd.concat(frames, ignore_index=True)
    duplicated = merged.duplicated(["ENSID", "cv_fold"], keep=False)
    if duplicated.any():
        examples = merged.loc[duplicated, ["ENSID", "cv_fold"]].head(10).to_dict("records")
        raise ValueError(f"Merged {filename} has duplicate ENSID/cv_fold rows: {examples}")
    return merged


def main() -> None:
    print_config()
    validate_inputs()
    val = merge_split(VAL_FILENAME)
    test = merge_split(TEST_FILENAME)
    OUT_VAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    val.to_csv(OUT_VAL_PATH, index=False)
    test.to_csv(OUT_TEST_PATH, index=False)
    OUT_CONFIG_PATH.write_text(json.dumps(config_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] wrote {OUT_VAL_PATH} rows={len(val)}")
    print(f"[DONE] wrote {OUT_TEST_PATH} rows={len(test)}")


if __name__ == "__main__":
    main()
