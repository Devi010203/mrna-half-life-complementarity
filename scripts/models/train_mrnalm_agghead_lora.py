"""
12train_mrnalm_triplet_agghead_matched_lora_noargs.py

Matched-LoRA AggHead supplement run for the SeqHead paper.

This wrapper reuses the finalized AggHead training pipeline and only changes the
LoRA settings to match the finalized SeqHead family:
- LORA_R = 16
- LORA_ALPHA = 64
- LORA_DROPOUT = 0.0
- LORA_TARGET_MODULES = ("query", "key", "value")

All other human five-fold AggHead protocol choices are kept aligned to the
finalized `2585` run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "models" / "mrnalm_agghead_base.py"
LOGICAL_NAME = "matched_lora_human_agghead"
EXPERIMENT_SLUG = "mrnalm_triplet_human_matched_lora"
RUN_MANIFEST_NAME = "supplement_run_manifest.json"


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "mrnalm_triplet_agghead_base",
        BASE_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load base AggHead script from {BASE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_matched_cfg(module):
    return module.validate_run_cfg(
        replace(
            module.cfg,
            OUT_ROOT="",
            SPECIES=("human",),
            SPLIT_PROTOCOL="cv",
            CV_K=5,
            CV_FOLD="all",
            SEED=42,
            INNER_VAL_RATIO=0.10,
            STRATIFY_BINS=20,
            CHUNK_LEN_UTR5=512,
            CHUNK_LEN_CDS=1024,
            CHUNK_LEN_UTR3=1024,
            CHUNK_STRIDE_UTR5=512,
            CHUNK_STRIDE_CDS=1024,
            CHUNK_STRIDE_UTR3=512,
            REGION_AGG_MODE="mean_max",
            HEAD_DIM=256,
            HEAD_DROPOUT=0.3223,
            EPOCHS=16,
            BATCH_SIZE=8,
            GRAD_ACCUM_STEPS=2,
            LR=1.5935e-05,
            WEIGHT_DECAY=0.002159,
            WARMUP_RATIO=0.05,
            PATIENCE=5,
            LOSS_KIND="huber",
            HUBER_BETA=1.2783,
            USE_AMP=True,
            USE_LORA=True,
            UNFREEZE_BACKBONE=False,
            USE_INPUT_REQUIRE_GRADS_HOOK=True,
            LORA_R=16,
            LORA_ALPHA=64,
            LORA_DROPOUT=0.0,
            LORA_TARGET_MODULES=("query", "key", "value"),
            DEVICE=os.environ.get("SALUKI_DEVICE", "cuda"),
        )
    )


def write_run_manifest(batch_root: Path, module, run_cfg) -> None:
    payload = {
        "logical_name": LOGICAL_NAME,
        "experiment_slug": EXPERIMENT_SLUG,
        "batch_root": str(batch_root),
        "out_root": str(module.resolve_out_root(run_cfg)),
        "script_path": str(Path(__file__).resolve()),
        "base_script_path": str(BASE_SCRIPT_PATH),
    }
    (batch_root / RUN_MANIFEST_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    os.environ.setdefault("SALUKI_DEVICE", "cuda")
    module = load_base_module()
    module.resolve_experiment_slug = lambda _run_cfg: EXPERIMENT_SLUG
    module.cfg = build_matched_cfg(module)
    module.main()

    resolved_out_root = module.resolve_out_root(module.cfg)
    batch_roots = sorted(resolved_out_root.glob(f"{module.resolve_experiment_slug(module.cfg)}_*"))
    if batch_roots:
        latest_batch_root = batch_roots[-1]
        write_run_manifest(latest_batch_root, module, module.cfg)


if __name__ == "__main__":
    main()
