"""
8train_mrnalm_triplet_seqhead_noargs.py

mRNA-LM triplet training with a transcript chunk-sequence head for transcript half-life regression.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForMaskedLM, PreTrainedTokenizerFast

try:
    import optuna
except Exception as e:
    optuna = None
    OPTUNA_IMPORT_ERROR = e
else:
    OPTUNA_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if os.name == "nt":
    _project_root_str = str(PROJECT_ROOT)
    if _project_root_str.startswith("\\\\?\\"):
        PROJECT_ROOT = Path(_project_root_str[4:])
RESULT_ROOT = PROJECT_ROOT / "results" / "runs"
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
PROCESSED_TAG = "codex_1"
DEFAULT_UTR5_MODEL_DIR = PROJECT_ROOT / "models" / "mrna_5utr_model_p2_cp85600_best"
DEFAULT_CDS_MODEL_DIR = PROJECT_ROOT / "models" / "codonbert"
DEFAULT_UTR3_MODEL_DIR = PROJECT_ROOT / "models" / "mrna_3utr_model_p2_cp99900_best"


@dataclass(frozen=True)
class Config:
    PROCESSED_ROOT: str = str(DEFAULT_PROCESSED_ROOT / PROCESSED_TAG)
    OUT_ROOT: str = ""

    SPECIES: str | Tuple[str, ...] = "human"

    # split
    SPLIT_PROTOCOL: str = "cv"
    SEED: int = 42
    CV_K: int = 5
    CV_FOLD: int | Tuple[int, ...] | str = "all"
    INNER_VAL_RATIO: float = 0.10
    TRAIN_RATIO: float = 0.80
    VAL_RATIO: float = 0.10
    TEST_RATIO: float = 0.10
    STRATIFY_BINS: int = 20

    # model paths
    UTR5_MODEL_DIR: str = str(DEFAULT_UTR5_MODEL_DIR)
    CDS_MODEL_DIR: str = str(DEFAULT_CDS_MODEL_DIR)
    UTR3_MODEL_DIR: str = str(DEFAULT_UTR3_MODEL_DIR)

    # tokenization / chunking
    CHUNK_LEN_UTR5: int = 512
    CHUNK_LEN_CDS: int = 1024
    CHUNK_LEN_UTR3: int = 1024
    CHUNK_STRIDE_UTR5: int = 256
    CHUNK_STRIDE_CDS: int = 512
    CHUNK_STRIDE_UTR3: int = 512
    MAP_T_TO_U_FOR_MODEL: bool = True
    TRIM_CDS_TO_MULTIPLE_OF_3: bool = True

    # route-B sequence head
    HEAD_ARCH: str = "chunk_seq_saluki_v1"

    SEQ_HEAD_USE_CONV: bool = True
    SEQ_HEAD_CONV_KERNEL_SIZE_1: int = 3
    SEQ_HEAD_CONV_KERNEL_SIZE_2: int = 5
    SEQ_HEAD_CONV_DROPOUT: float = 0.10

    SEQ_HEAD_GRU_LAYERS: int = 2
    SEQ_HEAD_GRU_DROPOUT: float = 0.10

    SEQ_HEAD_ATTN_HIDDEN: int = 256
    SEQ_HEAD_INPUT_DROPOUT: float = 0.10
    SEQ_HEAD_META_DROPOUT: float = 0.05

    SEQ_HEAD_LOCAL_POS_MAX: int = 1024
    SEQ_HEAD_GLOBAL_POS_MAX: int = 4096

    HEAD_DIM: int = 512
    HEAD_DROPOUT: float = 0.35
    MAX_CHUNKS_PER_FORWARD: int = 64

    # target
    STANDARDIZE_TARGET: bool = True
    TARGET_STD_EPS: float = 1e-6

    # optimization
    DEVICE: str = os.environ.get("SALUKI_DEVICE", "cuda:0")
    EPOCHS: int = 16
    BATCH_SIZE: int = 8
    GRAD_ACCUM_STEPS: int = 2
    LR: float = 8.0e-5
    WEIGHT_DECAY: float = 2.0e-6
    WARMUP_RATIO: float = 0.05
    GRAD_CLIP_NORM: float = 1.0
    USE_AMP: bool = True
    USE_LORA: bool = True
    UNFREEZE_BACKBONE: bool = False
    USE_GRAD_CHECKPOINTING: bool = True
    GRAD_CHECKPOINT_USE_REENTRANT: bool = False
    USE_INPUT_REQUIRE_GRADS_HOOK: bool = False
    RESUME_TRAINING_IF_AVAILABLE: bool = True
    PATIENCE: int = 5
    EARLY_STOP_METRIC: str = "r2"

    # loss
    LOSS_KIND: str = "huber"
    HUBER_BETA: float = 1.25
    CORR_LOSS_WEIGHT: float = 0.01

    # head
    USE_CLIP: bool = False

    # LoRA
    LORA_R: int = 16
    LORA_ALPHA: int = 64
    LORA_DROPOUT: float = 0.05
    LORA_TARGET_MODULES: Tuple[str, ...] = ("query", "key", "value")
    LORA_HEAD_LR_RATIO: float = 1.0
    LORA_BODY_LR_RATIO: float = 0.5

    # Optuna
    USE_OPTUNA: bool = False
    OPTUNA_N_TRIALS: int = 36
    OPTUNA_TOP_K: int = 5
    OPTUNA_STARTUP_TRIALS: int = 8
    OPTUNA_PRUNER_WARMUP_EPOCHS: int = 3
    # Legacy field kept for backward compatibility; Optuna storage now always lives under session_root/optuna.
    OPTUNA_STORAGE: str = ""
    OPTUNA_RESUME_IF_EXISTS: bool = True
    OPTUNA_TIMEOUT_SEC: int = 0
    OPTUNA_OBJECTIVE_METRIC: str = "r2"

    NUM_WORKERS_TRAIN: int = 2
    NUM_WORKERS_EVAL: int = 2
    PIN_MEMORY_TRAIN: bool = True
    PIN_MEMORY_EVAL: bool = True
    PERSISTENT_WORKERS_TRAIN: bool = True
    PERSISTENT_WORKERS_EVAL: bool = False


cfg = Config()


def parse_requested_cuda_index(device: str) -> int | None:
    value = str(device).strip().lower()
    if value == "cpu":
        return None
    if value == "cuda":
        return 0
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1]
        if not suffix.isdigit():
            raise ValueError(f"Invalid CUDA device string: {device!r}")
        return int(suffix)
    raise ValueError(f"DEVICE must be 'cpu', 'cuda', or 'cuda:N', got {device!r}")


def enforce_single_visible_cuda_device(device: str) -> None:
    if os.environ.get("SALUKI_SKIP_CUDA_MASK") == "1":
        return
    idx = parse_requested_cuda_index(device)
    if idx is None:
        return
    desired = str(idx)
    marker_key = "SALUKI_CUDA_MASK_APPLIED"
    if os.environ.get(marker_key) == desired and os.environ.get("CUDA_VISIBLE_DEVICES") == desired:
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = desired
    os.environ[marker_key] = desired
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


def resolve_runtime_device(device: str) -> str:
    idx = parse_requested_cuda_index(device)
    if idx is None:
        return "cpu"
    return "cuda:0"


def normalize_species(species: str) -> str:
    value = str(species).strip().lower()
    if value not in {"human", "mouse"}:
        raise ValueError(f"SPECIES must be 'human' or 'mouse', got {species!r}")
    return value


def normalize_species_config(species: str | Tuple[str, ...] | List[str]) -> Tuple[str, ...]:
    if isinstance(species, str):
        raw_items = [species]
    elif isinstance(species, (tuple, list)):
        raw_items = list(species)
    else:
        raise ValueError(f"SPECIES must be a string or a tuple/list of strings, got {type(species).__name__}")

    if not raw_items:
        raise ValueError("SPECIES must contain at least one species.")

    normalized: List[str] = []
    seen = set()
    for item in raw_items:
        value = normalize_species(str(item))
        if value in seen:
            continue
        normalized.append(value)
        seen.add(value)

    if not normalized:
        raise ValueError("SPECIES must contain at least one valid species.")
    if len(normalized) > 2:
        raise ValueError(f"SPECIES supports at most two species, got {normalized!r}")
    return tuple(normalized)


def resolve_species_tag(species: str | Tuple[str, ...] | List[str]) -> str:
    return "_".join(normalize_species_config(species))


def normalize_metric_name(metric_name: str) -> str:
    value = str(metric_name).strip().lower()
    valid = {"pearson", "spearman", "r2", "mae", "rmse"}
    if value not in valid:
        raise ValueError(f"metric must be one of {sorted(valid)}, got {metric_name!r}")
    return value


def metric_should_maximize(metric_name: str) -> bool:
    return str(metric_name).strip().lower() in {"r2", "pearson", "spearman"}


def resolve_tuning_mode(run_cfg: Config) -> str:
    if bool(run_cfg.USE_LORA):
        return "lora"
    if bool(run_cfg.UNFREEZE_BACKBONE):
        return "full_finetune"
    return "frozen_backbone"


def resolve_cv_fold_values(run_cfg: Config) -> Tuple[int, ...]:
    if str(run_cfg.SPLIT_PROTOCOL).strip().lower() != "cv":
        return tuple()

    raw = run_cfg.CV_FOLD
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text == "all":
            folds = tuple(range(int(run_cfg.CV_K)))
        elif "," in text:
            folds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
        elif text:
            folds = (int(text),)
        else:
            raise ValueError("CV_FOLD cannot be an empty string.")
    elif isinstance(raw, int):
        folds = (int(raw),)
    elif isinstance(raw, (tuple, list)):
        folds = tuple(int(x) for x in raw)
    else:
        raise ValueError(f"CV_FOLD must be an int, tuple/list of ints, or 'all', got {type(raw).__name__}")

    if not folds:
        raise ValueError("CV_FOLD must contain at least one fold when SPLIT_PROTOCOL='cv'.")

    cv_k = int(run_cfg.CV_K)
    if cv_k <= 1:
        raise ValueError("CV_K must be > 1 when SPLIT_PROTOCOL='cv'.")

    normalized: List[int] = []
    seen = set()
    for fold in folds:
        value = int(fold)
        if value < 0 or value >= cv_k:
            raise ValueError(f"CV_FOLD values must be in [0, {cv_k - 1}], got {value}")
        if value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return tuple(normalized)


def validate_run_cfg(run_cfg: Config) -> Config:
    species = normalize_species_config(run_cfg.SPECIES)
    split_protocol = str(run_cfg.SPLIT_PROTOCOL).strip().lower()
    if split_protocol not in {"cv", "holdout"}:
        raise ValueError(f"SPLIT_PROTOCOL must be 'cv' or 'holdout', got {run_cfg.SPLIT_PROTOCOL!r}")
    cv_folds = resolve_cv_fold_values(replace(run_cfg, SPLIT_PROTOCOL=split_protocol)) if split_protocol == "cv" else tuple()

    early_metric = normalize_metric_name(run_cfg.EARLY_STOP_METRIC)
    optuna_metric = normalize_metric_name(run_cfg.OPTUNA_OBJECTIVE_METRIC)
    if early_metric != optuna_metric:
        raise ValueError(
            "mRNA-LM training keeps EARLY_STOP_METRIC aligned with OPTUNA_OBJECTIVE_METRIC; "
            f"got {run_cfg.EARLY_STOP_METRIC!r} vs {run_cfg.OPTUNA_OBJECTIVE_METRIC!r}"
        )

    loss_kind = str(run_cfg.LOSS_KIND).strip().lower()
    if loss_kind not in {"mse", "mse_corr", "huber"}:
        raise ValueError(f"LOSS_KIND must be one of ['huber', 'mse', 'mse_corr'], got {run_cfg.LOSS_KIND!r}")
    if bool(run_cfg.USE_LORA) and bool(run_cfg.UNFREEZE_BACKBONE):
        raise ValueError("USE_LORA=True and UNFREEZE_BACKBONE=True are mutually exclusive.")
    if int(run_cfg.CHUNK_LEN_UTR5) <= 2 or int(run_cfg.CHUNK_LEN_CDS) <= 2 or int(run_cfg.CHUNK_LEN_UTR3) <= 2:
        raise ValueError("CHUNK_LEN_UTR5 / CHUNK_LEN_CDS / CHUNK_LEN_UTR3 must all be > 2.")
    if int(run_cfg.CHUNK_STRIDE_UTR5) <= 0 or int(run_cfg.CHUNK_STRIDE_CDS) <= 0 or int(run_cfg.CHUNK_STRIDE_UTR3) <= 0:
        raise ValueError("CHUNK_STRIDE_UTR5 / CHUNK_STRIDE_CDS / CHUNK_STRIDE_UTR3 must all be > 0.")
    if int(run_cfg.EPOCHS) <= 0 or int(run_cfg.BATCH_SIZE) <= 0 or int(run_cfg.GRAD_ACCUM_STEPS) <= 0:
        raise ValueError("EPOCHS / BATCH_SIZE / GRAD_ACCUM_STEPS must all be > 0.")
    if int(run_cfg.HEAD_DIM) <= 0:
        raise ValueError("HEAD_DIM must be > 0.")
    if str(run_cfg.HEAD_ARCH).strip().lower() != "chunk_seq_saluki_v1":
        raise ValueError("HEAD_ARCH must be 'chunk_seq_saluki_v1'.")
    if int(run_cfg.SEQ_HEAD_GRU_LAYERS) <= 0:
        raise ValueError("SEQ_HEAD_GRU_LAYERS must be > 0.")
    if int(run_cfg.SEQ_HEAD_ATTN_HIDDEN) <= 0:
        raise ValueError("SEQ_HEAD_ATTN_HIDDEN must be > 0.")
    if int(run_cfg.SEQ_HEAD_LOCAL_POS_MAX) <= 0 or int(run_cfg.SEQ_HEAD_GLOBAL_POS_MAX) <= 0:
        raise ValueError("SEQ_HEAD_LOCAL_POS_MAX / SEQ_HEAD_GLOBAL_POS_MAX must both be > 0.")
    if int(run_cfg.MAX_CHUNKS_PER_FORWARD) <= 0:
        raise ValueError("MAX_CHUNKS_PER_FORWARD must be > 0.")
    if float(run_cfg.WARMUP_RATIO) < 0.0 or float(run_cfg.WARMUP_RATIO) >= 1.0:
        raise ValueError("WARMUP_RATIO must be in [0, 1).")
    if float(run_cfg.SEQ_HEAD_CONV_DROPOUT) < 0.0 or float(run_cfg.SEQ_HEAD_CONV_DROPOUT) >= 1.0:
        raise ValueError("SEQ_HEAD_CONV_DROPOUT must be in [0, 1).")
    if float(run_cfg.SEQ_HEAD_GRU_DROPOUT) < 0.0 or float(run_cfg.SEQ_HEAD_GRU_DROPOUT) >= 1.0:
        raise ValueError("SEQ_HEAD_GRU_DROPOUT must be in [0, 1).")
    if float(run_cfg.SEQ_HEAD_INPUT_DROPOUT) < 0.0 or float(run_cfg.SEQ_HEAD_INPUT_DROPOUT) >= 1.0:
        raise ValueError("SEQ_HEAD_INPUT_DROPOUT must be in [0, 1).")
    if float(run_cfg.SEQ_HEAD_META_DROPOUT) < 0.0 or float(run_cfg.SEQ_HEAD_META_DROPOUT) >= 1.0:
        raise ValueError("SEQ_HEAD_META_DROPOUT must be in [0, 1).")
    if bool(run_cfg.USE_CLIP):
        raise ValueError("Route-B sequence head does not support USE_CLIP=True; keep it False.")

    target_modules = tuple(str(x).strip() for x in run_cfg.LORA_TARGET_MODULES if str(x).strip())
    return replace(
        run_cfg,
        SPECIES=species,
        SPLIT_PROTOCOL=split_protocol,
        CV_FOLD=(cv_folds[0] if len(cv_folds) == 1 else cv_folds) if split_protocol == "cv" else run_cfg.CV_FOLD,
        EARLY_STOP_METRIC=early_metric,
        OPTUNA_OBJECTIVE_METRIC=optuna_metric,
        LOSS_KIND=loss_kind,
        HEAD_ARCH=str(run_cfg.HEAD_ARCH).strip().lower(),
        LORA_TARGET_MODULES=target_modules,
    )


cfg = validate_run_cfg(cfg)
enforce_single_visible_cuda_device(cfg.DEVICE)


def resolve_experiment_slug(run_cfg: Config) -> str:
    return f"mrnalm_triplet_seqhead_{resolve_species_tag(run_cfg.SPECIES)}_{resolve_tuning_mode(run_cfg)}"


def resolve_default_result_stem(run_cfg: Config) -> str:
    return f"mrnalm_triplet_{resolve_species_tag(run_cfg.SPECIES)}"


def resolve_out_root(run_cfg: Config) -> Path:
    raw = str(run_cfg.OUT_ROOT).strip()
    if raw:
        return Path(raw)
    return RESULT_ROOT / resolve_default_result_stem(run_cfg)


def resolve_session_root(run_cfg: Config) -> Path:
    return resolve_out_root(run_cfg) / f"{resolve_experiment_slug(run_cfg)}_{now_ts()}"


def resolve_optuna_root(session_root: Path) -> Path:
    return session_root / "optuna"


def resolve_optuna_runtime_paths(session_root: Path, run_cfg: Config) -> Dict[str, Path | str]:
    optuna_dir = resolve_optuna_root(session_root)
    study_name = resolve_optuna_study_name(run_cfg)
    db_path = optuna_dir / f"{study_name}.db"
    return {
        "optuna_dir": optuna_dir,
        "trials_dir": optuna_dir / "trials",
        "db_path": db_path,
        "storage_url": f"sqlite:///{db_path.as_posix()}",
        "study_name": study_name,
    }


def resolve_optuna_study_name(run_cfg: Config) -> str:
    return resolve_experiment_slug(run_cfg)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_git_commit(project_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    value = str(proc.stdout).strip()
    return value or None


def collect_script_runtime_metadata() -> Dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "absolute_script_path": str(script_path),
        "script_sha256": compute_file_sha256(script_path),
        "git_commit": resolve_git_commit(PROJECT_ROOT),
    }


def summarize_numeric_series(values: List[int]) -> Dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "mean": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "max": int(arr.max()),
    }


def count_trainable_parameters(model: nn.Module) -> Dict[str, float | int]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    ratio = float(trainable / total) if total > 0 else 0.0
    return {"total_params": total, "trainable_params": trainable, "trainable_ratio": ratio}


def classify_trainable_param_group(param_name: str) -> str:
    name = str(param_name)
    if "lora_A" in name or "lora_B" in name:
        return "lora_ab"
    if name.startswith(
        (
            "region_embed.",
            "local_pos_embed.",
            "global_pos_embed.",
            "scalar_meta_proj.",
            "seq_input_norm.",
            "pre_seq_conv.",
            "chunk_seq_encoder.",
            "chunk_attn.",
            "final_dense.",
            "layer_norm.",
            "decoder.",
        )
    ):
        return "head"
    if name.startswith(("utr5.", "cds.", "utr3.")):
        return "backbone"
    return "other_requires_grad"


def collect_trainable_parameter_details(model: nn.Module) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    summary = {
        "total_trainable_params": 0,
        "n_trainable_tensors": 0,
        "lora_ab_params": 0,
        "head_params": 0,
        "backbone_params": 0,
        "other_requires_grad_params": 0,
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        numel = int(param.numel())
        group = classify_trainable_param_group(name)
        rows.append({"name": str(name), "shape": list(param.shape), "numel": numel, "group": group})
        summary["total_trainable_params"] += numel
        summary["n_trainable_tensors"] += 1
        key = f"{group}_params"
        if key in summary:
            summary[key] += numel
    return rows, summary


def collect_first_step_grad_norms(model: nn.Module, scale_factor: float = 1.0) -> Dict[str, Any]:
    safe_scale = float(scale_factor) if np.isfinite(float(scale_factor)) and float(scale_factor) > 0 else 1.0
    by_param: List[Dict[str, Any]] = []
    by_group: Dict[str, List[float]] = {}
    no_grad_param_names: List[str] = []
    n_trainable_tensors_total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n_trainable_tensors_total += 1
        if param.grad is None:
            no_grad_param_names.append(str(name))
            continue
        grad_norm = float(param.grad.detach().float().norm().item() / safe_scale)
        group = classify_trainable_param_group(name)
        by_param.append(
            {"name": str(name), "shape": list(param.shape), "numel": int(param.numel()), "group": group, "grad_norm": grad_norm}
        )
        by_group.setdefault(group, []).append(grad_norm)
    return {
        "scale_factor": safe_scale,
        "n_trainable_tensors_total": n_trainable_tensors_total,
        "n_tensors_with_grad": len(by_param),
        "n_tensors_without_grad": len(no_grad_param_names),
        "no_grad_param_names": no_grad_param_names,
        "by_group_summary": {
            key: {
                "count": int(np.asarray(values, dtype=np.float64).size),
                "mean_grad_norm": float(np.asarray(values, dtype=np.float64).mean()) if values else None,
                "max_grad_norm": float(np.asarray(values, dtype=np.float64).max()) if values else None,
            }
            for key, values in by_group.items()
        },
        "by_param": by_param,
    }


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def holdout_filename(seed: int) -> str:
    return f"holdout_seed{seed}_t{cfg.TRAIN_RATIO:.2f}_v{cfg.VAL_RATIO:.2f}_te{cfg.TEST_RATIO:.2f}.json"


def cv_filename(k: int, seed: int) -> str:
    return f"cv_k{k}_seed{seed}.csv"


def _make_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    uniq = np.unique(y[~np.isnan(y)])
    b = int(min(n_bins, max(2, len(uniq))))
    s = pd.Series(y)
    try:
        bins = pd.qcut(s, q=b, labels=False, duplicates="drop")
    except Exception:
        ranks = s.rank(method="average", na_option="keep")
        bins = pd.qcut(ranks, q=b, labels=False, duplicates="drop")
    arr = bins.to_numpy()
    return np.where(pd.isna(arr), -1, arr).astype(int)


def stratified_train_val(ids: List[str], y: np.ndarray, val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    ids_arr = np.asarray(ids, dtype=object)
    bins = _make_bins(y, cfg.STRATIFY_BINS)
    rng = np.random.default_rng(seed)
    tr: List[str] = []
    va: List[str] = []
    for bucket in [x for x in np.unique(bins) if x >= 0]:
        idx = np.where(bins == bucket)[0]
        block = ids_arr[idx].copy()
        rng.shuffle(block)
        n = len(block)
        n_va = int(math.floor(n * val_ratio))
        tr.extend(block[: n - n_va].tolist())
        va.extend(block[n - n_va :].tolist())
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


def _normalize_raw_sequence(series: pd.Series, map_t_to_u: bool) -> pd.Series:
    out = series.fillna("").astype(str).str.upper()
    if map_t_to_u:
        out = out.str.replace("T", "U", regex=False)
    return out


def _sample_id_for_species(species: str, ensid: str, mixed_mode: bool) -> str:
    value = str(ensid)
    return f"{species}:{value}" if mixed_mode else value


def _read_single_species_data_and_split(
    species: str,
    mixed_mode: bool,
    run_cfg: Config,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    df_path = Path(run_cfg.PROCESSED_ROOT) / species / "normalized.csv.gz"
    if not df_path.exists():
        raise FileNotFoundError(f"missing normalized: {df_path}")
    df = pd.read_csv(df_path, compression="gzip")
    need = ["ENSID", "HALFLIFE", "5UTR", "ORF", "3UTR"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df["ENSID"] = df["ENSID"].astype(str)
    df["ORIG_ENSID"] = df["ENSID"]
    df["SOURCE_SPECIES"] = str(species)
    df["ENSID"] = df["ENSID"].map(lambda x: _sample_id_for_species(species, x, mixed_mode))
    df["HALFLIFE"] = pd.to_numeric(df["HALFLIFE"], errors="coerce")
    df = df.dropna(subset=["HALFLIFE"]).copy()
    df["5UTR"] = _normalize_raw_sequence(df["5UTR"], map_t_to_u=bool(run_cfg.MAP_T_TO_U_FOR_MODEL))
    df["ORF"] = _normalize_raw_sequence(df["ORF"], map_t_to_u=bool(run_cfg.MAP_T_TO_U_FOR_MODEL))
    df["3UTR"] = _normalize_raw_sequence(df["3UTR"], map_t_to_u=bool(run_cfg.MAP_T_TO_U_FOR_MODEL))

    sp_dir = Path(run_cfg.PROCESSED_ROOT) / species
    if run_cfg.SPLIT_PROTOCOL == "holdout":
        p = sp_dir / "splits" / holdout_filename(run_cfg.SEED)
        if not p.exists():
            raise FileNotFoundError(f"missing holdout split: {p}")
        with open(p, "r", encoding="utf-8") as f:
            split = json.load(f)
        return df, {
            k: [_sample_id_for_species(species, str(x), mixed_mode) for x in split[k]]
            for k in ("train", "val", "test")
        }

    p = sp_dir / "splits" / cv_filename(run_cfg.CV_K, run_cfg.SEED)
    if not p.exists():
        raise FileNotFoundError(f"missing cv split: {p}")
    cv = pd.read_csv(p)
    cv["ENSID"] = cv["ENSID"].astype(str)
    cv["ENSID"] = cv["ENSID"].map(lambda x: _sample_id_for_species(species, x, mixed_mode))
    cv["fold"] = pd.to_numeric(cv["fold"], errors="coerce").astype("Int64")
    cv = cv.dropna(subset=["fold"]).astype({"fold": int})
    test_ids = cv.loc[cv["fold"] == int(run_cfg.CV_FOLD), "ENSID"].astype(str).tolist()
    pool = cv.loc[cv["fold"] != int(run_cfg.CV_FOLD), "ENSID"].astype(str).tolist()
    dfi = df.set_index("ENSID", drop=False)
    y_pool = dfi.loc[pool, "HALFLIFE"].to_numpy(np.float64)
    train_ids, val_ids = stratified_train_val(
        pool,
        y_pool,
        run_cfg.INNER_VAL_RATIO,
        seed=run_cfg.SEED * 100 + int(run_cfg.CV_FOLD),
    )
    return df, {"train": train_ids, "val": val_ids, "test": test_ids}


def read_data_and_split(run_cfg: Config) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    species_list = normalize_species_config(run_cfg.SPECIES)
    mixed_mode = len(species_list) > 1

    if not mixed_mode:
        return _read_single_species_data_and_split(species_list[0], mixed_mode=False, run_cfg=run_cfg)

    frames: List[pd.DataFrame] = []
    merged_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for species in species_list:
        species_df, species_split = _read_single_species_data_and_split(species, mixed_mode=True, run_cfg=run_cfg)
        frames.append(species_df)
        for split_name in merged_split:
            merged_split[split_name].extend(species_split[split_name])

    df = pd.concat(frames, axis=0, ignore_index=True)
    if bool(df["ENSID"].duplicated().any()):
        dup_ids = df.loc[df["ENSID"].duplicated(), "ENSID"].astype(str).head(10).tolist()
        raise ValueError(f"duplicated ENSID values detected after species merge: {dup_ids}")

    rng = random.Random(int(run_cfg.SEED))
    for split_name in merged_split:
        rng.shuffle(merged_split[split_name])
    return df, merged_split


def batch_input_tensor_key(batch: Dict[str, Any]) -> str:
    if "utr5_input_ids" in batch:
        return "utr5_input_ids"
    if "input_ids" in batch:
        return "input_ids"
    raise KeyError(f"unknown batch input keys: {sorted(batch.keys())}")


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    for key, value in list(batch.items()):
        if torch.is_tensor(value) and key != "y_raw":
            batch[key] = value.to(device, non_blocking=True)
    return batch


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_fn,
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(**loader_kwargs)


def ensure_sqlite_storage_parent(storage_url: str | None) -> None:
    if storage_url is None:
        return
    storage = str(storage_url).strip()
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return
    path_str = storage[len(prefix):]
    if os.name == "nt" and path_str.startswith("/") and len(path_str) >= 3 and path_str[2] == ":":
        path_str = path_str[1:]
    if not path_str:
        return
    ensure_dir(Path(path_str).parent)


def normalize_sequence_for_model(seq: str, map_t_to_u: bool = True) -> str:
    out = str(seq or "").upper()
    if map_t_to_u:
        out = out.replace("T", "U")
    return out


def utr_tokens_for_mrnalm(seq: str, map_t_to_u: bool = True) -> List[str]:
    normalized = normalize_sequence_for_model(seq, map_t_to_u=map_t_to_u)
    return list(normalized)


def cds_tokens_for_mrnalm(
    seq: str,
    trim_to_multiple_of_3: bool = True,
    map_t_to_u: bool = True,
) -> List[str]:
    normalized = normalize_sequence_for_model(seq, map_t_to_u=map_t_to_u)
    if trim_to_multiple_of_3:
        normalized = normalized[: (len(normalized) // 3) * 3]
    return [normalized[i : i + 3] for i in range(0, len(normalized), 3) if len(normalized[i : i + 3]) == 3]


def tokenize_utr_for_mrnalm(seq: str, map_t_to_u: bool = True) -> str:
    return " ".join(utr_tokens_for_mrnalm(seq, map_t_to_u=map_t_to_u))


def tokenize_cds_for_mrnalm(seq: str, trim_to_multiple_of_3: bool = True, map_t_to_u: bool = True) -> str:
    return " ".join(
        cds_tokens_for_mrnalm(
            seq,
            trim_to_multiple_of_3=trim_to_multiple_of_3,
            map_t_to_u=map_t_to_u,
        )
    )


def make_token_chunks(tokens: List[str], chunk_len_no_special: int, stride_no_special: int) -> List[List[str]]:
    if int(chunk_len_no_special) <= 0:
        raise ValueError(f"chunk_len_no_special must be > 0, got {chunk_len_no_special!r}")
    if int(stride_no_special) <= 0:
        raise ValueError(f"stride_no_special must be > 0, got {stride_no_special!r}")
    if not tokens:
        return [[]]
    if len(tokens) <= int(chunk_len_no_special):
        return [list(tokens)]

    chunks: List[List[str]] = []
    start = 0
    while start < len(tokens):
        end = min(start + int(chunk_len_no_special), len(tokens))
        chunk = list(tokens[start:end])
        if not chunk:
            break
        chunks.append(chunk)
        if end >= len(tokens):
            break
        start += int(stride_no_special)
    return chunks


def token_chunks_to_strings(chunks: List[List[str]]) -> List[str]:
    return [" ".join(chunk) if chunk else "" for chunk in chunks]


def _build_wordlevel_tokenizer(vocab_tokens: List[str], model_max_length: int) -> PreTrainedTokenizerFast:
    vocab = dict(zip(vocab_tokens, range(len(vocab_tokens))))
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["[PAD]", "[CLS]", "[UNK]", "[SEP]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = BertProcessing(("[SEP]", vocab["[SEP]"]), ("[CLS]", vocab["[CLS]"]))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )
    fast.model_max_length = int(model_max_length)
    return fast


def build_mrnalm_tokenizers(run_cfg: Config) -> Tuple[PreTrainedTokenizerFast, PreTrainedTokenizerFast, PreTrainedTokenizerFast]:
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    utr_alphabet = list("AUGCN")
    cds_alphabet = list("AUGC")
    cds_tokens = list(special_tokens)
    for a1 in cds_alphabet:
        for a2 in cds_alphabet:
            for a3 in cds_alphabet:
                cds_tokens.append(f"{a1}{a2}{a3}")
    utr_tokens = list(special_tokens) + utr_alphabet
    tokenizer_utr5 = _build_wordlevel_tokenizer(utr_tokens, model_max_length=int(run_cfg.CHUNK_LEN_UTR5))
    tokenizer_cds = _build_wordlevel_tokenizer(cds_tokens, model_max_length=int(run_cfg.CHUNK_LEN_CDS))
    tokenizer_utr3 = _build_wordlevel_tokenizer(utr_tokens, model_max_length=int(run_cfg.CHUNK_LEN_UTR3))
    return tokenizer_utr5, tokenizer_cds, tokenizer_utr3


def get_model_vocab_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    config_vocab_size = getattr(config, "vocab_size", None) if config is not None else None
    if isinstance(config_vocab_size, int) and config_vocab_size > 0:
        return int(config_vocab_size)
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        num_embeddings = getattr(embeddings, "num_embeddings", None)
        if isinstance(num_embeddings, int) and num_embeddings > 0:
            return int(num_embeddings)
    raise ValueError(f"Unable to resolve vocab_size for model type={type(model).__name__}")


def get_model_max_position_embeddings(model: nn.Module) -> int | None:
    config = getattr(model, "config", None)
    max_pos = getattr(config, "max_position_embeddings", None) if config is not None else None
    if isinstance(max_pos, int) and max_pos > 0:
        return int(max_pos)
    return None


def validate_tokenizer_model_compatibility(
    tokenizer: PreTrainedTokenizerFast,
    model: nn.Module,
    branch_name: str,
) -> None:
    tok_vocab = int(tokenizer.vocab_size)
    model_vocab = int(get_model_vocab_size(model))
    max_pos = get_model_max_position_embeddings(model)
    print(
        f"[Compat][{branch_name}] tokenizer_vocab={tok_vocab} "
        f"model_vocab={model_vocab} max_position_embeddings={max_pos}",
        flush=True,
    )
    if tok_vocab > model_vocab:
        raise ValueError(
            f"{branch_name}: tokenizer vocab_size ({tok_vocab}) exceeds model vocab_size ({model_vocab})"
        )


def assert_input_ids_in_range(input_ids: torch.Tensor, model: nn.Module, branch_name: str) -> None:
    if int(input_ids.numel()) == 0:
        return
    min_id = int(input_ids.min().item())
    max_id = int(input_ids.max().item())
    vocab_size = int(get_model_vocab_size(model))
    if min_id < 0:
        raise ValueError(f"{branch_name}: negative token id {min_id}")
    if max_id >= vocab_size:
        raise ValueError(
            f"{branch_name}: token id out of range, max_id={max_id}, vocab_size={vocab_size}"
        )


def print_debug_batch_once(batch: Dict[str, torch.Tensor], model: nn.Module, trial_tag: str) -> None:
    cds_vocab_size = int(get_model_vocab_size(model.cds))
    cds_max_pos = get_model_max_position_embeddings(model.cds)
    print(
        f"[DebugBatch][{trial_tag}] utr5_input_ids.shape={tuple(batch['utr5_input_ids'].shape)} "
        f"cds_input_ids.shape={tuple(batch['cds_input_ids'].shape)} "
        f"utr3_input_ids.shape={tuple(batch['utr3_input_ids'].shape)}",
        flush=True,
    )
    print(
        f"[DebugBatch][{trial_tag}] cds_input_ids.min={int(batch['cds_input_ids'].min().item())} "
        f"cds_input_ids.max={int(batch['cds_input_ids'].max().item())} "
        f"cds_model_vocab_size={cds_vocab_size} "
        f"cds_max_position_embeddings={cds_max_pos}",
        flush=True,
    )


def _region_tokens_for_mrnalm(region_name: str, seq: str, run_cfg: Config) -> List[str]:
    if region_name == "cds":
        return cds_tokens_for_mrnalm(
            seq,
            trim_to_multiple_of_3=bool(run_cfg.TRIM_CDS_TO_MULTIPLE_OF_3),
            map_t_to_u=bool(run_cfg.MAP_T_TO_U_FOR_MODEL),
        )
    return utr_tokens_for_mrnalm(seq, map_t_to_u=bool(run_cfg.MAP_T_TO_U_FOR_MODEL))


def _requested_chunk_stride(region_name: str, run_cfg: Config) -> int:
    if region_name == "utr5":
        return int(run_cfg.CHUNK_STRIDE_UTR5)
    if region_name == "cds":
        return int(run_cfg.CHUNK_STRIDE_CDS)
    if region_name == "utr3":
        return int(run_cfg.CHUNK_STRIDE_UTR3)
    raise KeyError(f"unknown region_name={region_name!r}")


def _chunk_len_without_special(chunk_length_with_special: int) -> int:
    return max(1, int(chunk_length_with_special) - 2)


def _chunk_stride_without_special(region_name: str, run_cfg: Config, chunk_length_with_special: int) -> int:
    requested = max(1, _requested_chunk_stride(region_name, run_cfg) - 2)
    return min(requested, _chunk_len_without_special(chunk_length_with_special))


def collect_sequence_stats_mrnalm(
    df: pd.DataFrame,
    split_ids: Dict[str, List[str]],
    run_cfg: Config,
    chunk_lengths: Dict[str, int],
) -> Dict[str, Any]:
    dfi = df.set_index("ENSID", drop=False)
    payload: Dict[str, Any] = {
        "backbone": "mrna_lm",
        "mode": "chunked",
        "model_dirs": {"utr5": str(run_cfg.UTR5_MODEL_DIR), "cds": str(run_cfg.CDS_MODEL_DIR), "utr3": str(run_cfg.UTR3_MODEL_DIR)},
        "chunk_lengths_used": {k: int(v) for k, v in chunk_lengths.items()},
        "chunk_len_without_special": {k: _chunk_len_without_special(v) for k, v in chunk_lengths.items()},
        "chunk_stride_without_special": {
            region_name: _chunk_stride_without_special(region_name, run_cfg, chunk_lengths[region_name])
            for region_name in ("utr5", "cds", "utr3")
        },
        "trim_cds_to_multiple_of_3": bool(run_cfg.TRIM_CDS_TO_MULTIPLE_OF_3),
        "seq_head_global_pos_max": int(run_cfg.SEQ_HEAD_GLOBAL_POS_MAX),
        "splits": {},
    }
    field_map = {"utr5": "5UTR", "cds": "ORF", "utr3": "3UTR"}
    for split_name, ids in split_ids.items():
        split_payload: Dict[str, Any] = {"n_samples": int(len(ids))}
        region_chunk_count_map: Dict[str, List[int]] = {}
        for region_name, column_name in field_map.items():
            raw_nt_lengths: List[int] = []
            token_lengths: List[int] = []
            trimmed_nt_lengths: List[int] = []
            n_chunks_list: List[int] = []
            chunk_len_list: List[int] = []
            overflow_chunk_count = 0
            limit = int(chunk_lengths[region_name])
            chunk_len_no_special = _chunk_len_without_special(limit)
            stride_no_special = _chunk_stride_without_special(region_name, run_cfg, limit)
            for ensid in ids:
                row = dfi.loc[str(ensid)]
                seq = str(row[column_name])
                raw_nt_lengths.append(len(seq))
                tokens = _region_tokens_for_mrnalm(region_name, seq, run_cfg)
                token_len = int(len(tokens))
                chunks = make_token_chunks(tokens, chunk_len_no_special=chunk_len_no_special, stride_no_special=stride_no_special)
                token_lengths.append(token_len)
                trimmed_nt_lengths.append(token_len if region_name != "cds" else token_len * 3)
                n_chunks_list.append(len(chunks))
                for chunk in chunks:
                    chunk_len = int(len(chunk))
                    chunk_len_list.append(chunk_len)
                    if chunk_len + 2 > limit:
                        overflow_chunk_count += 1
            if overflow_chunk_count > 0:
                raise ValueError(
                    f"Detected {overflow_chunk_count} overflow chunks for split={split_name} region={region_name}; "
                    f"chunk_length_with_special={limit}"
                )
            region_chunk_count_map[region_name] = n_chunks_list
            split_payload[region_name] = {
                "raw_nt_len": summarize_numeric_series(raw_nt_lengths),
                "trimmed_nt_len_for_model": summarize_numeric_series(trimmed_nt_lengths),
                "token_len_without_special": summarize_numeric_series(token_lengths),
                "n_chunks": summarize_numeric_series(n_chunks_list),
                "chunk_len_without_special": summarize_numeric_series(chunk_len_list),
                "max_chunk_len_observed_with_special": int((max(chunk_len_list) + 2) if chunk_len_list else 2),
                "overflow_chunk_count": int(overflow_chunk_count),
                "chunk_length_with_special": int(limit),
                "chunk_stride_without_special": int(stride_no_special),
            }
        total_chunks_per_sample = [
            int(
                region_chunk_count_map["utr5"][i]
                + region_chunk_count_map["cds"][i]
                + region_chunk_count_map["utr3"][i]
            )
            for i in range(len(ids))
        ]
        total_summary = summarize_numeric_series(total_chunks_per_sample)
        global_pos_limit = int(run_cfg.SEQ_HEAD_GLOBAL_POS_MAX)
        n_samples_exceeding_global_pos_max = int(sum(x > global_pos_limit for x in total_chunks_per_sample))
        max_chunks_over_global_pos_max = int(max([0, *[x - global_pos_limit for x in total_chunks_per_sample]]))
        split_payload["total_chunks_per_sample"] = total_summary
        split_payload["global_pos_limit"] = global_pos_limit
        split_payload["n_samples_exceeding_global_pos_max"] = n_samples_exceeding_global_pos_max
        split_payload["max_chunks_over_global_pos_max"] = max_chunks_over_global_pos_max
        payload["splits"][split_name] = split_payload
    return payload


def build_global_pos_limit_warnings(sequence_stats: Dict[str, Any], run_cfg: Config) -> List[str]:
    warnings: List[str] = []
    global_pos_limit = int(run_cfg.SEQ_HEAD_GLOBAL_POS_MAX)
    splits = sequence_stats.get("splits", {})
    if not isinstance(splits, dict):
        return warnings
    for split_name, split_payload in splits.items():
        if not isinstance(split_payload, dict):
            continue
        n_exceed = int(split_payload.get("n_samples_exceeding_global_pos_max", 0))
        if n_exceed <= 0:
            continue
        total_stats = split_payload.get("total_chunks_per_sample", {})
        p95 = total_stats.get("p95") if isinstance(total_stats, dict) else None
        max_total = total_stats.get("max") if isinstance(total_stats, dict) else None
        max_over = int(split_payload.get("max_chunks_over_global_pos_max", 0))
        warnings.append(
            f"[SeqHeadWarn] split={split_name} global_pos_limit={global_pos_limit} "
            f"n_samples_exceeding={n_exceed} total_chunks_p95={p95} total_chunks_max={max_total} max_over={max_over}"
        )
    return warnings


def build_collate_mrnalm(
    tok_utr5: PreTrainedTokenizerFast,
    tok_cds: PreTrainedTokenizerFast,
    tok_utr3: PreTrainedTokenizerFast,
    run_cfg: Config,
    chunk_lengths: Dict[str, int],
):
    tokenizers = {"utr5": tok_utr5, "cds": tok_cds, "utr3": tok_utr3}
    region_specs = {
        region_name: {
            "chunk_len_no_special": _chunk_len_without_special(chunk_lengths[region_name]),
            "stride_no_special": _chunk_stride_without_special(region_name, run_cfg, chunk_lengths[region_name]),
        }
        for region_name in ("utr5", "cds", "utr3")
    }

    def encode_region_chunks(region_name: str, seqs: List[str], tokenizer: PreTrainedTokenizerFast) -> Dict[str, torch.Tensor]:
        chunk_strings: List[str] = []
        sample_indices: List[int] = []
        chunk_indices: List[int] = []
        n_chunks_per_sample: List[int] = []
        for sample_index, seq in enumerate(seqs):
            tokens = _region_tokens_for_mrnalm(region_name, seq, run_cfg)
            chunks = make_token_chunks(
                tokens,
                chunk_len_no_special=int(region_specs[region_name]["chunk_len_no_special"]),
                stride_no_special=int(region_specs[region_name]["stride_no_special"]),
            )
            chunk_texts = token_chunks_to_strings(chunks)
            chunk_strings.extend(chunk_texts)
            sample_indices.extend([sample_index] * len(chunks))
            chunk_indices.extend(list(range(len(chunks))))
            n_chunks_per_sample.append(len(chunks))

        encoded = tokenizer(chunk_strings, truncation=False, padding=True, return_tensors="pt")
        observed_seq_len = int(encoded["input_ids"].shape[1]) if int(encoded["input_ids"].shape[0]) > 0 else 0
        if observed_seq_len > int(chunk_lengths[region_name]):
            raise ValueError(
                f"{region_name} chunk encoding exceeded allowed length: observed={observed_seq_len} "
                f"allowed={int(chunk_lengths[region_name])}"
            )
        return {
            f"{region_name}_input_ids": encoded["input_ids"],
            f"{region_name}_attention_mask": encoded["attention_mask"],
            f"{region_name}_sample_index": torch.tensor(sample_indices, dtype=torch.long),
            f"{region_name}_chunk_index": torch.tensor(chunk_indices, dtype=torch.long),
            f"n_chunks_{region_name}_per_sample": torch.tensor(n_chunks_per_sample, dtype=torch.long),
        }

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload = {
            "ensid": [x["ensid"] for x in batch],
            "y_raw": torch.tensor([float(x["y_raw"]) for x in batch], dtype=torch.float32),
            "y": torch.tensor([float(x["y"]) for x in batch], dtype=torch.float32),
            "batch_size": int(len(batch)),
        }
        seq_fields = {"utr5": "utr5", "cds": "orf", "utr3": "utr3"}
        for region_name, field_name in seq_fields.items():
            payload.update(
                encode_region_chunks(
                    region_name,
                    [str(x[field_name]) for x in batch],
                    tokenizers[region_name],
                )
            )
        return payload

    return collate


class MRNALMChunkedSequenceRegressor(nn.Module):
    supports_aux_outputs = False

    def __init__(self, run_cfg: Config):
        super().__init__()
        self.run_cfg = run_cfg
        self.utr5 = BertForMaskedLM.from_pretrained(str(run_cfg.UTR5_MODEL_DIR), local_files_only=True)
        self.cds = BertForMaskedLM.from_pretrained(str(run_cfg.CDS_MODEL_DIR), local_files_only=True)
        self.utr3 = BertForMaskedLM.from_pretrained(str(run_cfg.UTR3_MODEL_DIR), local_files_only=True)
        for branch in self.backbone_modules():
            if hasattr(branch, "config"):
                try:
                    branch.config.use_cache = False
                except Exception:
                    pass

        hidden_sizes = [int(getattr(branch.config, "hidden_size", 768)) for branch in self.backbone_modules()]
        if len(set(hidden_sizes)) != 1:
            raise ValueError(f"Expected equal hidden sizes across mRNA-LM branches, got {hidden_sizes}")
        self.hidden_size = int(hidden_sizes[0])
        self.max_chunks_per_forward = int(run_cfg.MAX_CHUNKS_PER_FORWARD)

        self.region_embed = nn.Embedding(3, self.hidden_size)
        self.local_pos_embed = nn.Embedding(int(run_cfg.SEQ_HEAD_LOCAL_POS_MAX), self.hidden_size)
        self.global_pos_embed = nn.Embedding(int(run_cfg.SEQ_HEAD_GLOBAL_POS_MAX), self.hidden_size)

        self.scalar_meta_proj = nn.Sequential(
            nn.Linear(5, self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(run_cfg.SEQ_HEAD_META_DROPOUT)),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.seq_input_norm = nn.LayerNorm(self.hidden_size)
        self.seq_input_dropout = nn.Dropout(float(run_cfg.SEQ_HEAD_INPUT_DROPOUT))

        if bool(run_cfg.SEQ_HEAD_USE_CONV):
            self.pre_seq_conv: nn.Sequential | None = nn.Sequential(
                nn.Conv1d(
                    self.hidden_size,
                    self.hidden_size,
                    kernel_size=int(run_cfg.SEQ_HEAD_CONV_KERNEL_SIZE_1),
                    padding=int(run_cfg.SEQ_HEAD_CONV_KERNEL_SIZE_1) // 2,
                ),
                nn.GELU(),
                nn.Dropout(float(run_cfg.SEQ_HEAD_CONV_DROPOUT)),
                nn.Conv1d(
                    self.hidden_size,
                    self.hidden_size,
                    kernel_size=int(run_cfg.SEQ_HEAD_CONV_KERNEL_SIZE_2),
                    padding=int(run_cfg.SEQ_HEAD_CONV_KERNEL_SIZE_2) // 2,
                ),
                nn.GELU(),
            )
        else:
            self.pre_seq_conv = None

        self.chunk_seq_encoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=int(run_cfg.SEQ_HEAD_GRU_LAYERS),
            batch_first=True,
            bidirectional=True,
            dropout=float(run_cfg.SEQ_HEAD_GRU_DROPOUT) if int(run_cfg.SEQ_HEAD_GRU_LAYERS) > 1 else 0.0,
        )

        self.chunk_attn = nn.Sequential(
            nn.Linear(self.hidden_size, int(run_cfg.SEQ_HEAD_ATTN_HIDDEN)),
            nn.Tanh(),
            nn.Dropout(float(run_cfg.HEAD_DROPOUT)),
            nn.Linear(int(run_cfg.SEQ_HEAD_ATTN_HIDDEN), 1, bias=False),
        )

        self.final_dense = nn.Linear(self.hidden_size, int(run_cfg.HEAD_DIM))
        self.layer_norm = nn.LayerNorm(int(run_cfg.HEAD_DIM), eps=1e-12)
        self.dropout = nn.Dropout(float(run_cfg.HEAD_DROPOUT))
        self.decoder = nn.Linear(int(run_cfg.HEAD_DIM), 1)

    def backbone_modules(self) -> List[nn.Module]:
        return [self.utr5, self.cds, self.utr3]

    def head_modules(self) -> Dict[str, nn.Module]:
        modules = {
            "region_embed": self.region_embed,
            "local_pos_embed": self.local_pos_embed,
            "global_pos_embed": self.global_pos_embed,
            "scalar_meta_proj": self.scalar_meta_proj,
            "seq_input_norm": self.seq_input_norm,
            "chunk_seq_encoder": self.chunk_seq_encoder,
            "chunk_attn": self.chunk_attn,
            "final_dense": self.final_dense,
            "layer_norm": self.layer_norm,
            "decoder": self.decoder,
        }
        if self.pre_seq_conv is not None:
            modules["pre_seq_conv"] = self.pre_seq_conv
        return modules

    def mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = hidden[:, 1:-1, :]
        mask = attention_mask[:, 1:-1].unsqueeze(-1).to(dtype=hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def _forward_backbone(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)["hidden_states"][-1]

    def _encode_region_chunks(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        branch_name: str,
    ) -> torch.Tensor:
        if int(input_ids.shape[0]) == 0:
            return input_ids.new_zeros((0, self.hidden_size), dtype=torch.float32)
        assert_input_ids_in_range(input_ids, model, branch_name)
        chunk_embeds: List[torch.Tensor] = []
        for start in range(0, int(input_ids.shape[0]), self.max_chunks_per_forward):
            end = min(int(input_ids.shape[0]), start + self.max_chunks_per_forward)
            hidden = self._forward_backbone(model, input_ids[start:end], attention_mask[start:end])
            chunk_embeds.append(self.mean_pool(hidden, attention_mask[start:end]))
        return torch.cat(chunk_embeds, dim=0)

    def _build_region_meta(
        self,
        n_chunks: int,
        region_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        if n_chunks <= 0:
            return torch.zeros((0, self.hidden_size), device=device)

        idx = torch.arange(n_chunks, device=device, dtype=torch.long)
        denom = max(1, n_chunks - 1)

        rel_from_start = idx.float() / float(denom)
        rel_from_end = (denom - idx).float() / float(denom)
        is_first = (idx == 0).float()
        is_last = (idx == n_chunks - 1).float()
        log_len = torch.full_like(rel_from_start, float(math.log1p(n_chunks)))

        scalar = torch.stack(
            [rel_from_start, rel_from_end, is_first, is_last, log_len],
            dim=-1,
        )

        region_ids = torch.full((n_chunks,), int(region_id), device=device, dtype=torch.long)
        local_pos = idx.clamp_max(int(self.run_cfg.SEQ_HEAD_LOCAL_POS_MAX) - 1)

        meta = (
            self.region_embed(region_ids)
            + self.local_pos_embed(local_pos)
            + self.scalar_meta_proj(scalar)
        )
        return meta

    def _gather_region_chunks_for_sample(
        self,
        chunk_embeddings: torch.Tensor,
        sample_index: torch.Tensor,
        chunk_index: torch.Tensor,
        sample_id: int,
    ) -> torch.Tensor:
        mask = sample_index == int(sample_id)
        if not torch.any(mask):
            return chunk_embeddings.new_zeros((0, self.hidden_size))
        region_chunks = chunk_embeddings[mask]
        order = torch.argsort(chunk_index[mask])
        return region_chunks[order]

    def _build_sample_transcript_sequence(
        self,
        sample_id: int,
        utr5_chunk_embeddings: torch.Tensor,
        cds_chunk_embeddings: torch.Tensor,
        utr3_chunk_embeddings: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        device = utr5_chunk_embeddings.device

        utr5_seq = self._gather_region_chunks_for_sample(
            utr5_chunk_embeddings,
            batch["utr5_sample_index"],
            batch["utr5_chunk_index"],
            sample_id,
        )
        cds_seq = self._gather_region_chunks_for_sample(
            cds_chunk_embeddings,
            batch["cds_sample_index"],
            batch["cds_chunk_index"],
            sample_id,
        )
        utr3_seq = self._gather_region_chunks_for_sample(
            utr3_chunk_embeddings,
            batch["utr3_sample_index"],
            batch["utr3_chunk_index"],
            sample_id,
        )

        parts: List[torch.Tensor] = []

        if utr5_seq.shape[0] > 0:
            parts.append(
                utr5_seq
                + self._build_region_meta(utr5_seq.shape[0], region_id=0, device=device).to(dtype=utr5_seq.dtype)
            )
        if cds_seq.shape[0] > 0:
            parts.append(
                cds_seq
                + self._build_region_meta(cds_seq.shape[0], region_id=1, device=device).to(dtype=cds_seq.dtype)
            )
        if utr3_seq.shape[0] > 0:
            parts.append(
                utr3_seq
                + self._build_region_meta(utr3_seq.shape[0], region_id=2, device=device).to(dtype=utr3_seq.dtype)
            )

        if not parts:
            return utr5_chunk_embeddings.new_zeros((1, self.hidden_size))

        seq = torch.cat(parts, dim=0)

        global_pos = torch.arange(seq.shape[0], device=device, dtype=torch.long)
        global_pos = global_pos.clamp_max(int(self.run_cfg.SEQ_HEAD_GLOBAL_POS_MAX) - 1)
        seq = seq + self.global_pos_embed(global_pos).to(dtype=seq.dtype)

        return seq

    def _pad_chunk_sequences(
        self,
        sequences: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([seq.shape[0] for seq in sequences], device=padded.device, dtype=torch.long)
        max_len = int(padded.shape[1])
        mask = torch.arange(max_len, device=padded.device).unsqueeze(0) < lengths.unsqueeze(1)
        return padded, mask, lengths

    def _masked_attn_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.chunk_attn(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)

    def _run_sequence_head(
        self,
        padded: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = self.seq_input_norm(padded)
        x = self.seq_input_dropout(x)
        x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        if self.pre_seq_conv is not None:
            x_conv = x.transpose(1, 2)
            x_conv = self.pre_seq_conv(x_conv)
            x = x_conv.transpose(1, 2)
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.chunk_seq_encoder(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=int(padded.shape[1]),
        )

        z = self._masked_attn_pool(x, mask)
        return z

    def _decode_joint(self, joint_embed: torch.Tensor) -> torch.Tensor:
        x = self.final_dense(joint_embed)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.decoder(x).squeeze(-1)

    def forward(self, batch: Dict[str, torch.Tensor], return_aux: bool = False) -> Any:
        if return_aux:
            raise ValueError("Route-B sequence head does not use auxiliary outputs.")

        batch_size = int(batch["y"].shape[0])
        utr5_chunk_embeddings = self._encode_region_chunks(
            self.utr5,
            batch["utr5_input_ids"],
            batch["utr5_attention_mask"],
            branch_name="utr5",
        )
        cds_chunk_embeddings = self._encode_region_chunks(
            self.cds,
            batch["cds_input_ids"],
            batch["cds_attention_mask"],
            branch_name="cds",
        )
        utr3_chunk_embeddings = self._encode_region_chunks(
            self.utr3,
            batch["utr3_input_ids"],
            batch["utr3_attention_mask"],
            branch_name="utr3",
        )

        seq_list = [
            self._build_sample_transcript_sequence(
                sample_id=i,
                utr5_chunk_embeddings=utr5_chunk_embeddings,
                cds_chunk_embeddings=cds_chunk_embeddings,
                utr3_chunk_embeddings=utr3_chunk_embeddings,
                batch=batch,
            )
            for i in range(batch_size)
        ]

        padded, mask, lengths = self._pad_chunk_sequences(seq_list)
        z = self._run_sequence_head(padded, mask, lengths)
        pred = self._decode_joint(z)
        return pred


MRNALMTripletRegressor = MRNALMChunkedSequenceRegressor


def resolve_safe_max_length(tokenizer: PreTrainedTokenizerFast, model: nn.Module, requested_max_length: int) -> int:
    limits: List[int] = [int(requested_max_length)]
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
        limits.append(int(tok_max))
    config = getattr(model, "config", None)
    for key in ("max_position_embeddings", "n_positions", "input_size"):
        value = getattr(config, key, None) if config is not None else None
        if isinstance(value, int) and value > 0:
            limits.append(int(value))
    return max(8, int(min(limits)))


def extract_head_state_dict(model: MRNALMTripletRegressor) -> Dict[str, Any]:
    return {
        "head_arch": str(model.run_cfg.HEAD_ARCH),
        "modules": {name: module.state_dict() for name, module in model.head_modules().items()},
    }


def load_head_state_dict(model: MRNALMTripletRegressor, state: Dict[str, Any]) -> None:
    if "modules" not in state:
        raise KeyError("head_state missing 'modules'")
    if state.get("head_arch") != str(model.run_cfg.HEAD_ARCH):
        raise RuntimeError(f"head_arch mismatch: ckpt={state.get('head_arch')} current={model.run_cfg.HEAD_ARCH}")
    modules_state = state["modules"]
    for name, module in model.head_modules().items():
        if name not in modules_state:
            raise KeyError(f"head_state missing module {name!r}")
        module.load_state_dict(modules_state[name], strict=True)


def _linear_module_names(module: nn.Module) -> List[str]:
    return [name for name, child in module.named_modules() if isinstance(child, nn.Linear)]


def _count_target_hits(linear_names: List[str], target_modules: List[str]) -> Dict[str, int]:
    return {target: int(sum(name.endswith(target) for name in linear_names)) for target in target_modules}


def resolve_mrnalm_lora_targets(
    model: MRNALMTripletRegressor,
    manual_targets: List[str],
) -> Tuple[List[str], str, Dict[str, Any]]:
    branch_linear_names = {
        "utr5": _linear_module_names(model.utr5),
        "cds": _linear_module_names(model.cds),
        "utr3": _linear_module_names(model.utr3),
    }
    diagnostics: Dict[str, Any] = {
        "branch_linear_name_counts": {name: len(names) for name, names in branch_linear_names.items()},
        "branch_linear_name_samples": {name: names[:80] for name, names in branch_linear_names.items()},
    }
    if manual_targets:
        diagnostics["branch_hit_counts"] = {
            name: _count_target_hits(names, manual_targets) for name, names in branch_linear_names.items()
        }
        return list(manual_targets), "manual_override", diagnostics

    fallback_presets = [
        ("bert_query_value", ["query", "value"]),
        ("bert_qkv", ["query", "key", "value"]),
    ]
    diagnostics["preset_hit_counts"] = {
        preset_name: {
            branch_name: _count_target_hits(linear_names, targets)
            for branch_name, linear_names in branch_linear_names.items()
        }
        for preset_name, targets in fallback_presets
    }
    for preset_name, targets in fallback_presets:
        if all(all(count > 0 for count in diagnostics["preset_hit_counts"][preset_name][branch_name].values()) for branch_name in branch_linear_names):
            diagnostics["branch_hit_counts"] = {
                branch_name: diagnostics["preset_hit_counts"][preset_name][branch_name]
                for branch_name in branch_linear_names
            }
            return list(targets), preset_name, diagnostics

    diagnostics["branch_hit_counts"] = {}
    return [], "peft_auto", diagnostics


def maybe_apply_lora_to_mrnalm(model: MRNALMTripletRegressor, run_cfg: Config):
    if not run_cfg.USE_LORA:
        return model, None, None, "disabled", [], {}
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
    except Exception as e:
        raise RuntimeError("需要安装 peft 才能进行 LoRA 微调。") from e

    manual_targets = [str(x) for x in run_cfg.LORA_TARGET_MODULES if str(x).strip()]
    target_modules, resolution_mode, diagnostics = resolve_mrnalm_lora_targets(model, manual_targets)
    peft_kwargs: Dict[str, Any] = {
        "task_type": TaskType.TOKEN_CLS,
        "r": int(run_cfg.LORA_R),
        "lora_alpha": int(run_cfg.LORA_ALPHA),
        "lora_dropout": float(run_cfg.LORA_DROPOUT),
        "use_rslora": True,
        "bias": "none",
    }
    if target_modules:
        peft_kwargs["target_modules"] = list(target_modules)
    peft_cfg = LoraConfig(**peft_kwargs)
    model.utr5 = get_peft_model(model.utr5, peft_cfg)
    model.cds = get_peft_model(model.cds, peft_cfg)
    model.utr3 = get_peft_model(model.utr3, peft_cfg)
    return model, get_peft_model_state_dict, set_peft_model_state_dict, resolution_mode, target_modules, diagnostics


def freeze_backbone_modules(model: MRNALMTripletRegressor) -> None:
    for branch in model.backbone_modules():
        for param in branch.parameters():
            param.requires_grad_(False)


def unfreeze_backbone_modules(model: MRNALMTripletRegressor) -> None:
    for branch in model.backbone_modules():
        for param in branch.parameters():
            param.requires_grad_(True)


def maybe_enable_input_require_grads(model: MRNALMTripletRegressor) -> List[str]:
    enabled: List[str] = []
    for name, branch in (("utr5", model.utr5), ("cds", model.cds), ("utr3", model.utr3)):
        if hasattr(branch, "enable_input_require_grads"):
            try:
                branch.enable_input_require_grads()
                enabled.append(name)
            except Exception:
                pass
    return enabled


def maybe_enable_gradient_checkpointing(model: MRNALMTripletRegressor, run_cfg: Config) -> List[str]:
    enabled: List[str] = []
    if not bool(run_cfg.USE_GRAD_CHECKPOINTING):
        return enabled
    for name, branch in (("utr5", model.utr5), ("cds", model.cds), ("utr3", model.utr3)):
        if not hasattr(branch, "gradient_checkpointing_enable"):
            continue
        try:
            try:
                branch.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": bool(run_cfg.GRAD_CHECKPOINT_USE_REENTRANT)}
                )
            except TypeError:
                branch.gradient_checkpointing_enable()
            if hasattr(branch, "config"):
                try:
                    branch.config.use_cache = False
                except Exception:
                    pass
            enabled.append(name)
        except Exception:
            continue
    return enabled


def _unwrap_branch_flags(module: nn.Module) -> Dict[str, Any]:
    current = module
    for attr in ("module", "base_model", "model"):
        if hasattr(current, attr):
            current = getattr(current, attr)
    config = getattr(current, "config", None)
    return {
        "type": type(current).__name__,
        "is_gradient_checkpointing": getattr(current, "is_gradient_checkpointing", None),
        "use_cache": getattr(config, "use_cache", None) if config is not None else None,
    }


def print_model_flags(model: MRNALMTripletRegressor, tag: str = "") -> None:
    flags = {"utr5": _unwrap_branch_flags(model.utr5), "cds": _unwrap_branch_flags(model.cds), "utr3": _unwrap_branch_flags(model.utr3)}
    print(f"{tag} branch_flags={flags}", flush=True)


def unpack_model_output(output: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if isinstance(output, dict):
        if "pred" not in output:
            raise KeyError(f"model output dict missing 'pred': keys={sorted(output.keys())}")
        return output["pred"], {k: v for k, v in output.items() if k != "pred"}
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    return output, {}


def model_forward(model: nn.Module, batch: Dict[str, torch.Tensor], return_aux: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if return_aux and bool(getattr(model, "supports_aux_outputs", False)):
        output = model(batch, return_aux=True)
    else:
        output = model(batch)
    return unpack_model_output(output)


def pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    ar = pd.Series(a).rank(method="average").to_numpy(np.float64)
    br = pd.Series(b).rank(method="average").to_numpy(np.float64)
    return pearson_np(ar, br)


def r2_np(y: np.ndarray, yhat: np.ndarray) -> float:
    y = y.astype(np.float64)
    yhat = yhat.astype(np.float64)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    ss_res = float(np.sum((y - yhat) ** 2))
    return float(1.0 - ss_res / ss_tot)


def metrics_np(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    y = y.astype(np.float64)
    yhat = yhat.astype(np.float64)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {
        "r2": r2_np(y, yhat),
        "pearson": pearson_np(y, yhat),
        "spearman": spearman_np(y, yhat),
        "rmse": rmse,
        "mae": float(np.mean(np.abs(y - yhat))),
    }


def corr_loss_torch(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = yhat - yhat.mean()
    z = y - y.mean()
    den = torch.sqrt(torch.clamp(torch.sum(x * x) * torch.sum(z * z), min=1e-12))
    return 1.0 - torch.sum(x * z) / den


def compute_loss(yhat: torch.Tensor, y: torch.Tensor, run_cfg: Config) -> torch.Tensor:
    if run_cfg.LOSS_KIND == "mse":
        return F.mse_loss(yhat, y)
    if run_cfg.LOSS_KIND == "huber":
        return F.smooth_l1_loss(yhat, y, beta=float(run_cfg.HUBER_BETA))
    if run_cfg.LOSS_KIND == "mse_corr":
        return F.mse_loss(yhat, y) + float(run_cfg.CORR_LOSS_WEIGHT) * corr_loss_torch(yhat, y)
    raise ValueError(f"unknown LOSS_KIND={run_cfg.LOSS_KIND}")


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    y_scaler: Dict[str, float] | None,
    device: str,
    collect_region_weights: bool = False,
    split_name: str | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame | None]:
    del collect_region_weights, split_name
    model.eval()
    ys: List[np.ndarray] = []
    yhs: List[np.ndarray] = []
    ids: List[str] = []
    for batch in loader:
        move_batch_to_device(batch, device)
        yhat_tensor, _ = model_forward(model, batch, return_aux=False)
        yhat = yhat_tensor.detach().cpu().numpy()
        if y_scaler is not None:
            yhat = yhat * float(y_scaler["std"]) + float(y_scaler["mean"])
        ys.append(batch["y_raw"].numpy())
        yhs.append(yhat)
        ids.extend(batch["ensid"])
    y = np.concatenate(ys).reshape(-1)
    yhat = np.concatenate(yhs).reshape(-1)
    pred_df = pd.DataFrame({"ENSID": ids, "y_true": y, "y_pred": yhat})
    return metrics_np(y, yhat), pred_df, None


def _move_state_to_cpu(state: Any) -> Any:
    if torch.is_tensor(state):
        return state.detach().cpu()
    if isinstance(state, dict):
        return {k: _move_state_to_cpu(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_move_state_to_cpu(v) for v in state]
    if isinstance(state, tuple):
        return tuple(_move_state_to_cpu(v) for v in state)
    return state


def _move_state_to_device(state: Any, device: str) -> Any:
    if torch.is_tensor(state):
        return state.to(device)
    if isinstance(state, dict):
        return {k: _move_state_to_device(v, device) for k, v in state.items()}
    if isinstance(state, list):
        return [_move_state_to_device(v, device) for v in state]
    if isinstance(state, tuple):
        return tuple(_move_state_to_device(v, device) for v in state)
    return state


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = _move_state_to_device(value, device)


def save_mrnalm_checkpoint(
    model: MRNALMTripletRegressor,
    path: Path,
    epoch: int,
    score_cmp: float,
    score_raw: float,
    y_scaler: Dict[str, float] | None,
    trial_cfg: Config,
    use_lora: bool,
    peft_state_dict_fn=None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    training_state: Dict[str, Any] | None = None,
    checkpoint_role: str = "best",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "best_score_cmp": float(score_cmp),
        "best_score_raw": float(score_raw),
        "y_scaler": y_scaler,
        "cfg": asdict(trial_cfg),
        "checkpoint_role": str(checkpoint_role),
        "head_arch": str(trial_cfg.HEAD_ARCH),
    }
    if use_lora:
        if peft_state_dict_fn is None:
            raise RuntimeError("LoRA checkpoint requested but PEFT state-dict helper is unavailable.")
        payload["checkpoint_format"] = "mrnalm_seqhead_lora_head"
        payload["utr5_adapter_state"] = _move_state_to_cpu(peft_state_dict_fn(model.utr5))
        payload["cds_adapter_state"] = _move_state_to_cpu(peft_state_dict_fn(model.cds))
        payload["utr3_adapter_state"] = _move_state_to_cpu(peft_state_dict_fn(model.utr3))
        payload["head_state"] = _move_state_to_cpu(extract_head_state_dict(model))
    else:
        payload["checkpoint_format"] = "full_model_state"
        payload["model_state"] = _move_state_to_cpu(model.state_dict())
    if optimizer is not None or scheduler is not None or scaler is not None or training_state:
        resume_state: Dict[str, Any] = dict(training_state or {})
        if optimizer is not None:
            resume_state["optimizer_state"] = _move_state_to_cpu(optimizer.state_dict())
        if scheduler is not None:
            resume_state["scheduler_state"] = _move_state_to_cpu(scheduler.state_dict())
        if scaler is not None:
            resume_state["scaler_state"] = _move_state_to_cpu(scaler.state_dict())
        payload["resume_state"] = resume_state
    torch.save(payload, path)
    return payload


def _adapter_key_mismatch(current_keys: List[str], provided_keys: List[str], branch_name: str, trial_tag: str) -> None:
    missing = sorted(set(current_keys) - set(provided_keys))
    unexpected = sorted(set(provided_keys) - set(current_keys))
    if missing or unexpected:
        raise RuntimeError(
            f"adapter_state mismatch for {trial_tag}/{branch_name}: missing={missing[:20]} unexpected={unexpected[:20]}"
        )


def load_mrnalm_checkpoint(
    model: MRNALMTripletRegressor,
    ckpt_path: Path,
    use_lora: bool,
    peft_state_dict_fn=None,
    peft_load_state_dict_fn=None,
    trial_tag: str = "",
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    expected_head_arch = str(model.run_cfg.HEAD_ARCH)
    ckpt_head_arch = ckpt.get("head_arch")
    if ckpt_head_arch != expected_head_arch:
        raise RuntimeError(
            f"checkpoint head_arch mismatch for {ckpt_path}: ckpt={ckpt_head_arch!r} current={expected_head_arch!r}. "
            "Do not restore an old triplet-head checkpoint into the seqhead script."
        )
    checkpoint_format = ckpt.get("checkpoint_format", "full_model_state" if "model_state" in ckpt else None)
    if "model_state" in ckpt:
        if checkpoint_format != "full_model_state":
            raise RuntimeError(
                f"Unexpected checkpoint_format={checkpoint_format!r} for full-model checkpoint at {ckpt_path}. "
                "Do not restore an old triplet-head checkpoint into the seqhead script."
            )
        model.load_state_dict(ckpt["model_state"], strict=True)
        return ckpt
    if checkpoint_format != "mrnalm_seqhead_lora_head":
        raise RuntimeError(
            f"Unsupported checkpoint_format={checkpoint_format!r} for {ckpt_path}. "
            "Do not restore an old triplet-head checkpoint into the seqhead script."
        )
    if not use_lora:
        raise RuntimeError("LoRA-format checkpoint cannot be loaded when USE_LORA=False.")
    if peft_state_dict_fn is None or peft_load_state_dict_fn is None:
        raise RuntimeError("PEFT state-dict helpers are required to restore LoRA checkpoint.")

    for branch_name, branch in (("utr5", model.utr5), ("cds", model.cds), ("utr3", model.utr3)):
        state_key = f"{branch_name}_adapter_state"
        if state_key not in ckpt:
            raise KeyError(f"checkpoint missing key {state_key!r}")
        current_keys = list(peft_state_dict_fn(branch).keys())
        provided_keys = list(ckpt[state_key].keys())
        _adapter_key_mismatch(current_keys, provided_keys, branch_name=branch_name, trial_tag=trial_tag)
        peft_load_state_dict_fn(branch, ckpt[state_key])

    if "head_state" not in ckpt:
        raise KeyError("checkpoint missing 'head_state'")
    load_head_state_dict(model, ckpt["head_state"])
    return ckpt


def restore_training_state_from_checkpoint(
    ckpt: Dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    device: str,
) -> Dict[str, Any]:
    resume_state = ckpt.get("resume_state")
    if not isinstance(resume_state, dict):
        return {}
    if optimizer is not None and "optimizer_state" in resume_state:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        move_optimizer_state_to_device(optimizer, device)
    if scheduler is not None and "scheduler_state" in resume_state:
        scheduler.load_state_dict(resume_state["scheduler_state"])
    if scaler is not None and "scaler_state" in resume_state:
        scaler.load_state_dict(resume_state["scaler_state"])
    return resume_state


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    v = float(value)
    if np.isfinite(v):
        return v
    return None


def _metric_from_final(final: Dict[str, Any] | None, split_name: str, metric_name: str) -> float | None:
    if not isinstance(final, dict):
        return None
    split_metrics = final.get(split_name)
    if not isinstance(split_metrics, dict):
        return None
    return _float_or_none(split_metrics.get(metric_name))


def _trial_value_or_none(trial: "optuna.trial.FrozenTrial") -> float | None:
    if trial.value is not None:
        return _float_or_none(trial.value)
    if trial.intermediate_values:
        last_step = max(trial.intermediate_values)
        return _float_or_none(trial.intermediate_values[last_step])
    return None


def _trial_message(trial: "optuna.trial.FrozenTrial") -> str | None:
    message = trial.user_attrs.get("message")
    return str(message) if message is not None else None


def summarize_frozen_trial(trial: "optuna.trial.FrozenTrial") -> Dict[str, Any]:
    final = trial.user_attrs.get("final")
    return {
        "number": int(trial.number),
        "state": str(trial.state.name),
        "value": _trial_value_or_none(trial),
        "params": dict(trial.params),
        "trial_tag": trial.user_attrs.get("trial_tag"),
        "out_dir": trial.user_attrs.get("out_dir"),
        "adapter_dir": trial.user_attrs.get("adapter_dir"),
        "message": _trial_message(trial),
        "objective_metric": trial.user_attrs.get("objective_metric"),
        "tuning_mode": trial.user_attrs.get("tuning_mode"),
        "head_arch": trial.user_attrs.get("head_arch"),
        "val_r2": _metric_from_final(final, "val", "r2"),
        "val_pearson": _metric_from_final(final, "val", "pearson"),
        "val_spearman": _metric_from_final(final, "val", "spearman"),
        "val_rmse": _metric_from_final(final, "val", "rmse"),
        "val_mae": _metric_from_final(final, "val", "mae"),
        "test_r2": _metric_from_final(final, "test", "r2"),
        "test_pearson": _metric_from_final(final, "test", "pearson"),
        "test_spearman": _metric_from_final(final, "test", "spearman"),
        "test_rmse": _metric_from_final(final, "test", "rmse"),
        "test_mae": _metric_from_final(final, "test", "mae"),
    }


def build_optuna_trial_rows(study: "optuna.study.Study") -> List[Dict[str, Any]]:
    return [summarize_frozen_trial(trial) for trial in study.trials]


def format_trial_tag(trial_number: int, params: Dict[str, Any]) -> str:
    dropout_tag = f"{float(params['lora_dropout']):.2f}".replace(".", "p")
    loss_kind = str(params.get("loss_kind", "na")).lower()
    return f"trial_{int(trial_number):04d}_r{int(params['lora_r'])}_a{int(params['lora_alpha'])}_d{dropout_tag}_{loss_kind}"


def metric_for_compare(value: float, maximize: bool) -> float:
    if np.isfinite(value):
        return float(value)
    return -float("inf") if maximize else float("inf")


def metric_for_pruner(value: float, maximize: bool) -> float:
    if np.isfinite(value):
        return float(value)
    return -1.0e18 if maximize else 1.0e18


def suggest_optuna_params(trial: "optuna.trial.Trial") -> Dict[str, Any]:
    use_lora = bool(cfg.USE_LORA)
    if use_lora:
        lora_r = int(trial.suggest_categorical("lora_r", [16, 32]))
        lora_alpha_mult = float(trial.suggest_categorical("lora_alpha_mult", [4.0, 6.0, 8.0]))
        lora_alpha = max(1, int(round(float(lora_r) * float(lora_alpha_mult))))
        lora_dropout = float(trial.suggest_categorical("lora_dropout", [0.00, 0.02, 0.05]))
    else:
        lora_r = int(cfg.LORA_R)
        lora_alpha = int(cfg.LORA_ALPHA)
        lora_alpha_mult = float(lora_alpha / max(1, lora_r))
        lora_dropout = float(cfg.LORA_DROPOUT)

    return {
        "lora_r": lora_r,
        "lora_alpha_mult": lora_alpha_mult,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lr": float(trial.suggest_float("lr", 3e-5, 2e-4, log=True)),
        "weight_decay": float(trial.suggest_float("weight_decay", 1e-7, 3e-5, log=True)),
        "head_dim": int(trial.suggest_categorical("head_dim", [512, 768])),
        "head_dropout": float(trial.suggest_float("head_dropout", 0.25, 0.60)),
        "seq_head_attn_hidden": int(trial.suggest_categorical("seq_head_attn_hidden", [128, 256, 512])),
        "seq_head_gru_dropout": float(trial.suggest_float("seq_head_gru_dropout", 0.05, 0.20)),
        "seq_head_input_dropout": float(trial.suggest_float("seq_head_input_dropout", 0.05, 0.20)),
        "seq_head_conv_dropout": float(trial.suggest_float("seq_head_conv_dropout", 0.05, 0.20)),
        "loss_kind": "huber",
        "huber_beta": float(trial.suggest_float("huber_beta", 1.0, 1.5)),
    }


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ids: List[str], y_scaler: Dict[str, float] | None):
        self.df = df.set_index("ENSID", drop=False)
        self.ids = [str(x) for x in ids]
        self.y_scaler = y_scaler

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ensid = self.ids[idx]
        row = self.df.loc[ensid]
        y_raw = float(row["HALFLIFE"])
        y = y_raw
        if self.y_scaler is not None:
            y = (y_raw - float(self.y_scaler["mean"])) / float(self.y_scaler["std"])
        return {"ensid": ensid, "utr5": str(row["5UTR"]), "orf": str(row["ORF"]), "utr3": str(row["3UTR"]), "y_raw": y_raw, "y": y}


def _extract_effective_target_modules(branch: nn.Module) -> List[str]:
    peft_config = getattr(branch, "peft_config", None)
    if not isinstance(peft_config, dict) or not peft_config:
        return []
    adapter_cfg = next(iter(peft_config.values()))
    target_modules = getattr(adapter_cfg, "target_modules", None)
    if target_modules is None:
        return []
    return sorted(str(x) for x in target_modules)


def collect_branch_trainable_parameter_counts(model: MRNALMTripletRegressor) -> Dict[str, Dict[str, float | int]]:
    payload: Dict[str, Dict[str, float | int]] = {}
    for branch_name in ("utr5", "cds", "utr3"):
        branch = getattr(model, branch_name)
        total = int(sum(param.numel() for param in branch.parameters()))
        trainable = int(sum(param.numel() for param in branch.parameters() if param.requires_grad))
        payload[branch_name] = {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_ratio": float(trainable / total) if total > 0 else 0.0,
        }
    return payload


def run_one_trial(
    trial: "optuna.trial.Trial | None",
    trial_cfg: Config,
    trial_params: Dict[str, Any],
    session_root: Path,
    df: pd.DataFrame,
    tr_ids: List[str],
    va_ids: List[str],
    te_ids: List[str],
    y_scaler: Dict[str, float] | None,
) -> Dict[str, Any]:
    trial_cfg = validate_run_cfg(trial_cfg)
    set_seed(trial_cfg.SEED)

    requested_device = str(trial_cfg.DEVICE).strip()
    active_device = resolve_runtime_device(requested_device)
    if active_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {requested_device!r}, but CUDA is not available.")
        torch.cuda.set_device(0)

    script_metadata = collect_script_runtime_metadata()
    objective_metric = str(trial_cfg.OPTUNA_OBJECTIVE_METRIC).strip().lower()
    maximize_metric = metric_should_maximize(objective_metric)
    tuning_mode = resolve_tuning_mode(trial_cfg)
    use_lora = bool(trial_cfg.USE_LORA)

    trial_number = int(trial.number) if trial is not None else 0
    trial_tag = format_trial_tag(trial_number, trial_params) if trial is not None else resolve_experiment_slug(trial_cfg)
    out_dir = session_root / trial_tag
    ensure_dir(out_dir)

    with open(out_dir / "resolved_trial_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "trial_number": int(trial.number) if trial is not None else None,
                "trial_tag": trial_tag,
                "head_arch": str(trial_cfg.HEAD_ARCH),
                "objective_metric": objective_metric,
                "tuning_mode": tuning_mode,
                "params": trial_params,
                "script_metadata": script_metadata,
            },
            f,
            indent=2,
        )

    if trial is not None:
        trial.set_user_attr("trial_tag", trial_tag)
        trial.set_user_attr("out_dir", str(out_dir))
        trial.set_user_attr("objective_metric", objective_metric)
        trial.set_user_attr("tuning_mode", tuning_mode)
        trial.set_user_attr("head_arch", str(trial_cfg.HEAD_ARCH))

    config_path = out_dir / "config.json"

    def write_trial_config(best_value: float | None, state: str, message: str | None) -> None:
        config_dump = asdict(trial_cfg)
        config_dump.pop("HEAD_ARCH", None)
        config_dump.update(
            {
                "head_arch": str(trial_cfg.HEAD_ARCH),
                "backbone": "mrna_lm",
                "trial_tag": trial_tag,
                "objective_metric": objective_metric,
                "tuning_mode": tuning_mode,
                "requested_device": requested_device,
                "runtime_device": active_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "best_value": _float_or_none(best_value),
                "state": state,
                "message": message,
                "requested_input_require_grads_hook": bool(trial_cfg.USE_INPUT_REQUIRE_GRADS_HOOK),
                "effective_input_require_grads_branches": list(enabled_input_require_grads),
                "gradient_checkpointing_branches": list(grad_checkpoint_branches),
                "lora_target_resolution_mode": lora_target_resolution_mode,
                "resume_training_if_available": bool(trial_cfg.RESUME_TRAINING_IF_AVAILABLE),
                "resumed_from_last_checkpoint": bool(resumed_from_last_checkpoint),
                "last_checkpoint_path": str(last_path) if last_path is not None else "",
                "script_metadata": script_metadata,
            }
        )
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dump, f, indent=2)

    model: MRNALMChunkedSequenceRegressor | None = None
    optimizer = None
    scheduler = None
    writer = None
    scaler = None
    peft_state_dict_fn = None
    peft_load_state_dict_fn = None
    tok_utr5 = None
    tok_cds = None
    tok_utr3 = None
    dl_va = None
    dl_tr = None
    ds_tr = None
    ds_va = None
    ds_te = None
    enabled_input_require_grads: List[str] = []
    grad_checkpoint_branches: List[str] = []
    lora_diagnostics: Dict[str, Any] = {}
    lora_target_resolution_mode = "disabled"
    configured_target_modules = [str(x) for x in trial_cfg.LORA_TARGET_MODULES if str(x).strip()]
    selected_target_modules: List[str] = []
    resumed_from_last_checkpoint = False
    last_path: Path | None = None

    try:
        write_trial_config(best_value=None, state="RUNNING", message=None)
        with open(out_dir / "split_sizes.json", "w", encoding="utf-8") as f:
            json.dump({"n_train": len(tr_ids), "n_val": len(va_ids), "n_test": len(te_ids)}, f, indent=2)
        if y_scaler is not None:
            with open(out_dir / "target_scaler.json", "w", encoding="utf-8") as f:
                json.dump(y_scaler, f, indent=2)

        tok_utr5, tok_cds, tok_utr3 = build_mrnalm_tokenizers(trial_cfg)
        model = MRNALMChunkedSequenceRegressor(trial_cfg)
        validate_tokenizer_model_compatibility(tok_utr5, model.utr5, "utr5")
        validate_tokenizer_model_compatibility(tok_cds, model.cds, "cds")
        validate_tokenizer_model_compatibility(tok_utr3, model.utr3, "utr3")
        (
            model,
            peft_state_dict_fn,
            peft_load_state_dict_fn,
            lora_target_resolution_mode,
            selected_target_modules,
            lora_diagnostics,
        ) = maybe_apply_lora_to_mrnalm(model, trial_cfg)

        if use_lora and bool(trial_cfg.USE_INPUT_REQUIRE_GRADS_HOOK):
            enabled_input_require_grads = maybe_enable_input_require_grads(model)
        else:
            enabled_input_require_grads = []
        if use_lora:
            print(
                f"[LoRA][{trial_tag}] enabled_input_require_grads={enabled_input_require_grads} "
                f"requested_by_config={trial_cfg.USE_INPUT_REQUIRE_GRADS_HOOK}",
                flush=True,
            )
        elif trial_cfg.UNFREEZE_BACKBONE:
            unfreeze_backbone_modules(model)
        else:
            freeze_backbone_modules(model)

        if bool(trial_cfg.USE_GRAD_CHECKPOINTING) and (use_lora or bool(trial_cfg.UNFREEZE_BACKBONE)):
            grad_checkpoint_branches = maybe_enable_gradient_checkpointing(model, trial_cfg)
        print(
            f"[Model][{trial_tag}] backbone=mrna_lm tuning_mode={tuning_mode} "
            f"head_arch={trial_cfg.HEAD_ARCH} grad_checkpoint_branches={grad_checkpoint_branches}",
            flush=True,
        )
        print_model_flags(model, tag=f"[Flags][{trial_tag}]")

        chunk_lengths = {
            "utr5": resolve_safe_max_length(tok_utr5, model.utr5, trial_cfg.CHUNK_LEN_UTR5),
            "cds": resolve_safe_max_length(tok_cds, model.cds, trial_cfg.CHUNK_LEN_CDS),
            "utr3": resolve_safe_max_length(tok_utr3, model.utr3, trial_cfg.CHUNK_LEN_UTR3),
        }
        with open(out_dir / "tokenizer_model_limits.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "backbone": "mrna_lm",
                    "model_dirs": {"utr5": str(trial_cfg.UTR5_MODEL_DIR), "cds": str(trial_cfg.CDS_MODEL_DIR), "utr3": str(trial_cfg.UTR3_MODEL_DIR)},
                    "requested_chunk_lengths": {
                        "utr5": int(trial_cfg.CHUNK_LEN_UTR5),
                        "cds": int(trial_cfg.CHUNK_LEN_CDS),
                        "utr3": int(trial_cfg.CHUNK_LEN_UTR3),
                    },
                    "requested_chunk_strides": {
                        "utr5": int(trial_cfg.CHUNK_STRIDE_UTR5),
                        "cds": int(trial_cfg.CHUNK_STRIDE_CDS),
                        "utr3": int(trial_cfg.CHUNK_STRIDE_UTR3),
                    },
                    "safe_chunk_lengths": {k: int(v) for k, v in chunk_lengths.items()},
                    "tokenizer_vocab_sizes": {
                        "utr5": int(tok_utr5.vocab_size),
                        "cds": int(tok_cds.vocab_size),
                        "utr3": int(tok_utr3.vocab_size),
                    },
                    "model_vocab_sizes": {
                        "utr5": int(get_model_vocab_size(model.utr5)),
                        "cds": int(get_model_vocab_size(model.cds)),
                        "utr3": int(get_model_vocab_size(model.utr3)),
                    },
                    "model_hidden_sizes": {
                        "utr5": int(getattr(model.utr5.config, "hidden_size", -1)),
                        "cds": int(getattr(model.cds.config, "hidden_size", -1)),
                        "utr3": int(getattr(model.utr3.config, "hidden_size", -1)),
                    },
                    "model_max_position_embeddings": {
                        "utr5": int(getattr(model.utr5.config, "max_position_embeddings", -1)),
                        "cds": int(getattr(model.cds.config, "max_position_embeddings", -1)),
                        "utr3": int(getattr(model.utr3.config, "max_position_embeddings", -1)),
                    },
                },
                f,
                indent=2,
            )

        sequence_stats = collect_sequence_stats_mrnalm(
            df=df,
            split_ids={"train": tr_ids, "val": va_ids, "test": te_ids},
            run_cfg=trial_cfg,
            chunk_lengths=chunk_lengths,
        )
        with open(out_dir / "sequence_stats_mrnalm_chunked.json", "w", encoding="utf-8") as f:
            json.dump(sequence_stats, f, indent=2)
        for warning in build_global_pos_limit_warnings(sequence_stats, trial_cfg):
            print(warning, flush=True)

        collate = build_collate_mrnalm(tok_utr5, tok_cds, tok_utr3, trial_cfg, chunk_lengths=chunk_lengths)
        ds_tr = SeqDataset(df, tr_ids, y_scaler)
        ds_va = SeqDataset(df, va_ids, y_scaler)
        ds_te = SeqDataset(df, te_ids, y_scaler)

        with open(out_dir / "lora_target_modules.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "backbone": "mrna_lm",
                    "use_lora": use_lora,
                    "tuning_mode": tuning_mode,
                    "lora_target_resolution_mode": lora_target_resolution_mode,
                    "configured_lora_target_modules": list(configured_target_modules),
                    "selected_lora_target_modules": list(selected_target_modules),
                    "precheck": lora_diagnostics,
                    "resolved_target_modules": {
                        "utr5": _extract_effective_target_modules(model.utr5),
                        "cds": _extract_effective_target_modules(model.cds),
                        "utr3": _extract_effective_target_modules(model.utr3),
                    },
                },
                f,
                indent=2,
            )

        model = model.to(active_device)
        branch_trainable_counts = collect_branch_trainable_parameter_counts(model)

        trainable_param_rows, trainable_param_summary = collect_trainable_parameter_details(model)
        with open(out_dir / "trainable_param_summary.json", "w", encoding="utf-8") as f:
            json.dump(trainable_param_summary, f, indent=2)
        with open(out_dir / "trainable_param_names.txt", "w", encoding="utf-8") as f:
            for row in trainable_param_rows:
                f.write(f"{row['name']}\tshape={row['shape']}\tnumel={row['numel']}\tgroup={row['group']}\n")
        with open(out_dir / "lora_scope_stats.json", "w", encoding="utf-8") as f:
            summary_payload = dict(trainable_param_summary)
            summary_payload.update(count_trainable_parameters(model))
            summary_payload.update(
                {
                    "head_arch": str(trial_cfg.HEAD_ARCH),
                    "backbone": "mrna_lm",
                    "use_lora": use_lora,
                    "tuning_mode": tuning_mode,
                    "lora_target_resolution_mode": lora_target_resolution_mode,
                    "configured_lora_target_modules": list(configured_target_modules),
                    "selected_lora_target_modules": list(selected_target_modules),
                    "lora_precheck": lora_diagnostics,
                    "branch_trainable_parameter_counts": branch_trainable_counts,
                }
            )
            json.dump(summary_payload, f, indent=2)

        trainable = [param for param in model.parameters() if param.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters found.")

        if bool(trial_cfg.UNFREEZE_BACKBONE) and not use_lora:
            backbone_params = [
                p
                for name, p in model.named_parameters()
                if p.requires_grad and classify_trainable_param_group(name) == "backbone"
            ]
            head_params = [
                p
                for name, p in model.named_parameters()
                if p.requires_grad and classify_trainable_param_group(name) != "backbone"
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": float(trial_cfg.LR) * 0.1},
                    {"params": head_params, "lr": float(trial_cfg.LR)},
                ],
                weight_decay=float(trial_cfg.WEIGHT_DECAY),
            )
            print(
                f"[Optim][{trial_tag}] full fine-tuning: backbone_lr={trial_cfg.LR * 0.1:.2e} head_lr={trial_cfg.LR:.2e}",
                flush=True,
            )
        else:
            head_params = []
            lora_params = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                group = classify_trainable_param_group(name)
                if group == "head":
                    head_params.append(param)
                elif group == "lora_ab":
                    lora_params.append(param)
                else:
                    head_params.append(param)

            param_groups = []
            if head_params:
                param_groups.append(
                    {
                        "params": head_params,
                        "lr": float(trial_cfg.LR) * float(trial_cfg.LORA_HEAD_LR_RATIO),
                        "weight_decay": float(trial_cfg.WEIGHT_DECAY),
                    }
                )
            if lora_params:
                param_groups.append(
                    {
                        "params": lora_params,
                        "lr": float(trial_cfg.LR) * float(trial_cfg.LORA_BODY_LR_RATIO),
                        "weight_decay": float(trial_cfg.WEIGHT_DECAY),
                    }
                )

            optimizer = torch.optim.AdamW(param_groups)

        amp_device = "cuda" if str(active_device).startswith("cuda") else "cpu"
        amp_enabled = bool(trial_cfg.USE_AMP and amp_device == "cuda")
        use_bf16 = bool(amp_device == "cuda" and torch.cuda.is_bf16_supported())
        amp_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if amp_device == "cuda" else torch.bfloat16)
        amp_dtype_name = "bfloat16" if amp_dtype == torch.bfloat16 else "float16"
        grad_scaler_enabled = bool(amp_enabled and not use_bf16)
        scaler = torch.amp.GradScaler(amp_device, enabled=grad_scaler_enabled)
        print(
            f"[AMP][{trial_tag}] enabled={amp_enabled} amp_dtype={amp_dtype_name} grad_scaler_enabled={grad_scaler_enabled}",
            flush=True,
        )

        dl_tr = make_loader(
            ds_tr,
            batch_size=trial_cfg.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate,
            num_workers=trial_cfg.NUM_WORKERS_TRAIN,
            pin_memory=trial_cfg.PIN_MEMORY_TRAIN,
            persistent_workers=trial_cfg.PERSISTENT_WORKERS_TRAIN,
        )
        dl_va = make_loader(
            ds_va,
            batch_size=trial_cfg.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate,
            num_workers=trial_cfg.NUM_WORKERS_EVAL,
            pin_memory=trial_cfg.PIN_MEMORY_EVAL,
            persistent_workers=trial_cfg.PERSISTENT_WORKERS_EVAL,
        )

        num_update_steps_per_epoch = max(1, math.ceil(len(dl_tr) / max(1, int(trial_cfg.GRAD_ACCUM_STEPS))))
        total_steps = max(1, int(trial_cfg.EPOCHS) * num_update_steps_per_epoch)
        warmup_steps = max(1, int(float(trial_cfg.WARMUP_RATIO) * total_steps))

        def lr_lambda(step: int) -> float:
            step_index = int(step) + 1
            if step_index <= warmup_steps:
                return float(step_index) / float(max(1, warmup_steps))
            progress = float(step_index - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        initial_lr_scale = float(lr_lambda(0))
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * initial_lr_scale

        writer = SummaryWriter(log_dir=str(out_dir))
        best_score = -float("inf") if maximize_metric else float("inf")
        bad = 0
        best_path = out_dir / "best.pt"
        last_path = out_dir / "last.pt"
        did_grad_sanity_check = False
        printed_debug_batch = False
        start_epoch = 1

        if bool(trial_cfg.RESUME_TRAINING_IF_AVAILABLE) and last_path.exists():
            last_ckpt = load_mrnalm_checkpoint(
                model=model,
                ckpt_path=last_path,
                use_lora=use_lora,
                peft_state_dict_fn=peft_state_dict_fn,
                peft_load_state_dict_fn=peft_load_state_dict_fn,
                trial_tag=trial_tag,
            )
            model = model.to(active_device)
            resume_state = restore_training_state_from_checkpoint(
                ckpt=last_ckpt,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=active_device,
            )
            resumed_from_last_checkpoint = True
            start_epoch = int(last_ckpt.get("epoch", 0)) + 1
            loaded_best_score = float(last_ckpt.get("best_score_cmp", best_score))
            if np.isfinite(loaded_best_score):
                best_score = loaded_best_score
            bad = int(resume_state.get("bad_epochs", 0))
            did_grad_sanity_check = bool(resume_state.get("did_grad_sanity_check", False))
            print(
                f"[Resume][{trial_tag}] last_checkpoint={last_path} start_epoch={start_epoch} "
                f"best_score_cmp={best_score:.6f} bad={bad}",
                flush=True,
            )

        for epoch in range(start_epoch, int(trial_cfg.EPOCHS) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses: List[float] = []
            main_losses: List[float] = []
            last_step = 0

            for step, batch in enumerate(dl_tr, start=1):
                last_step = step
                move_batch_to_device(batch, active_device)
                if not printed_debug_batch:
                    print_debug_batch_once(batch, model, trial_tag)
                    printed_debug_batch = True
                with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enabled):
                    yhat, _ = model_forward(model, batch, return_aux=False)
                    base_loss = compute_loss(yhat, batch["y"], trial_cfg)
                    total_loss = base_loss
                    loss = total_loss / float(trial_cfg.GRAD_ACCUM_STEPS)

                scaler.scale(loss).backward()
                losses.append(float(total_loss.detach().cpu().item()))
                main_losses.append(float(base_loss.detach().cpu().item()))

                if not did_grad_sanity_check:
                    n_grad = sum(int(param.grad is not None) for param in trainable)
                    if n_grad == 0:
                        raise RuntimeError(
                            "No gradients found after backward on first step. "
                            "Try setting USE_GRAD_CHECKPOINTING=False or disable LoRA."
                        )
                    grad_scale = float(scaler.get_scale()) if grad_scaler_enabled else 1.0
                    first_step_grad_payload = collect_first_step_grad_norms(model, scale_factor=grad_scale)
                    with open(out_dir / "first_step_grad_norms.json", "w", encoding="utf-8") as f:
                        json.dump(first_step_grad_payload, f, indent=2)
                    did_grad_sanity_check = True

                if step % int(trial_cfg.GRAD_ACCUM_STEPS) == 0:
                    scaler.unscale_(optimizer)
                    if float(trial_cfg.GRAD_CLIP_NORM) > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, float(trial_cfg.GRAD_CLIP_NORM))
                    prev_scale = float(scaler.get_scale()) if grad_scaler_enabled else None
                    scaler.step(optimizer)
                    scaler.update()
                    did_optimizer_step = True
                    if grad_scaler_enabled:
                        did_optimizer_step = float(scaler.get_scale()) >= float(prev_scale)
                    if scheduler is not None and did_optimizer_step:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if last_step > 0 and (last_step % int(trial_cfg.GRAD_ACCUM_STEPS) != 0):
                scaler.unscale_(optimizer)
                if float(trial_cfg.GRAD_CLIP_NORM) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, float(trial_cfg.GRAD_CLIP_NORM))
                prev_scale = float(scaler.get_scale()) if grad_scaler_enabled else None
                scaler.step(optimizer)
                scaler.update()
                did_optimizer_step = True
                if grad_scaler_enabled:
                    did_optimizer_step = float(scaler.get_scale()) >= float(prev_scale)
                if scheduler is not None and did_optimizer_step:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            val_met, _, _ = run_eval(model, dl_va, y_scaler, device=active_device)
            train_loss = float(np.mean(losses)) if losses else float("nan")
            train_main_loss = float(np.mean(main_losses)) if main_losses else float("nan")
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/main_loss", train_main_loss, epoch)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), epoch)
            for metric_name, metric_value in val_met.items():
                writer.add_scalar(f"val/{metric_name}", metric_value, epoch)

            score_raw = float(val_met.get(objective_metric, float("nan")))
            score_cmp = metric_for_compare(score_raw, maximize_metric)
            if trial is not None:
                trial.report(metric_for_pruner(score_raw, maximize_metric), epoch)
                if epoch >= int(trial_cfg.OPTUNA_PRUNER_WARMUP_EPOCHS) and trial.should_prune():
                    message = f"trial pruned at epoch={epoch} with val_{objective_metric}={score_raw}"
                    trial.set_user_attr("message", message)
                    raise optuna.TrialPruned(message)

            improved = (score_cmp > best_score + 1e-8) if maximize_metric else (score_cmp < best_score - 1e-8)
            if improved or (not best_path.exists()):
                best_score = score_cmp
                bad = 0
                save_mrnalm_checkpoint(
                    model=model,
                    path=best_path,
                    epoch=epoch,
                    score_cmp=best_score,
                    score_raw=score_raw,
                    y_scaler=y_scaler,
                    trial_cfg=trial_cfg,
                    use_lora=use_lora,
                    peft_state_dict_fn=peft_state_dict_fn,
                    checkpoint_role="best",
                )
            else:
                bad += 1

            save_mrnalm_checkpoint(
                model=model,
                path=last_path,
                epoch=epoch,
                score_cmp=best_score,
                score_raw=score_raw,
                y_scaler=y_scaler,
                trial_cfg=trial_cfg,
                use_lora=use_lora,
                peft_state_dict_fn=peft_state_dict_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                training_state={
                    "bad_epochs": int(bad),
                    "did_grad_sanity_check": bool(did_grad_sanity_check),
                },
                checkpoint_role="last",
            )

            print(
                f"[mRNA-LM][{trial_tag}] e={epoch:03d}/{trial_cfg.EPOCHS} "
                f"loss={train_loss:.4f} main={train_main_loss:.4f} "
                f"val_r2={val_met['r2']:.4f} val_p={val_met['pearson']:.4f} "
                f"val_rmse={val_met['rmse']:.4f} best_{objective_metric}={best_score:.4f} "
                f"bad={bad}/{trial_cfg.PATIENCE}",
                flush=True,
            )
            if bad >= int(trial_cfg.PATIENCE):
                break

        if not best_path.exists():
            raise RuntimeError("best checkpoint was not created; training did not produce a valid snapshot.")

        ckpt = load_mrnalm_checkpoint(
            model=model,
            ckpt_path=best_path,
            use_lora=use_lora,
            peft_state_dict_fn=peft_state_dict_fn,
            peft_load_state_dict_fn=peft_load_state_dict_fn,
            trial_tag=trial_tag,
        )
        model = model.to(active_device)

        dl_tr_eval = make_loader(
            ds_tr,
            batch_size=trial_cfg.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate,
            num_workers=trial_cfg.NUM_WORKERS_EVAL,
            pin_memory=trial_cfg.PIN_MEMORY_EVAL,
            persistent_workers=trial_cfg.PERSISTENT_WORKERS_EVAL,
        )
        dl_te = make_loader(
            ds_te,
            batch_size=trial_cfg.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate,
            num_workers=trial_cfg.NUM_WORKERS_EVAL,
            pin_memory=trial_cfg.PIN_MEMORY_EVAL,
            persistent_workers=trial_cfg.PERSISTENT_WORKERS_EVAL,
        )

        tr_met, pred_tr, _ = run_eval(model, dl_tr_eval, y_scaler, device=active_device, split_name="train")
        va_met, pred_va, _ = run_eval(model, dl_va, y_scaler, device=active_device, split_name="val")
        te_met, pred_te, _ = run_eval(model, dl_te, y_scaler, device=active_device, split_name="test")

        objective_value = float(va_met.get(objective_metric, float("nan")))
        best_metric_raw = float(ckpt.get("best_score_raw", float("nan")))
        if not np.isfinite(objective_value):
            raise RuntimeError(f"final val_{objective_metric} is non-finite; refusing to return invalid objective")
        if not np.isfinite(best_metric_raw):
            raise RuntimeError(f"best checkpoint val_{objective_metric} is non-finite; refusing to export invalid metrics")

        final = {
            "head_arch": str(trial_cfg.HEAD_ARCH),
            "train": tr_met,
            "val": va_met,
            "test": te_met,
            "best_epoch": int(ckpt["epoch"]),
            "best_val_metric_name": objective_metric,
            "best_val_metric_cmp": float(ckpt.get("best_score_cmp", float("nan"))),
            "best_val_metric_raw": best_metric_raw,
            "objective_metric": objective_metric,
            "tuning_mode": tuning_mode,
            "lora_target_resolution_mode": lora_target_resolution_mode,
            "configured_lora_target_modules": list(configured_target_modules),
            "selected_lora_target_modules": list(selected_target_modules),
            "resumed_from_last_checkpoint": bool(resumed_from_last_checkpoint),
            "last_checkpoint_path": str(last_path) if last_path is not None else "",
            "best_checkpoint_format": str(ckpt.get("checkpoint_format", "full_model_state")),
        }
        with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
        pred_tr.to_csv(out_dir / "pred_train.csv", index=False)
        pred_va.to_csv(out_dir / "pred_val.csv", index=False)
        pred_te.to_csv(out_dir / "pred_test.csv", index=False)

        adapter_dir = out_dir / "adapter"
        ensure_dir(adapter_dir)
        if use_lora:
            model.utr5.save_pretrained(adapter_dir / "utr5")
            model.cds.save_pretrained(adapter_dir / "cds")
            model.utr3.save_pretrained(adapter_dir / "utr3")
        else:
            with open(adapter_dir / f"{tuning_mode}.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "backbone": "mrna_lm",
                        "tuning_mode": tuning_mode,
                        "head_arch": str(trial_cfg.HEAD_ARCH),
                        "model_dirs": {"utr5": str(trial_cfg.UTR5_MODEL_DIR), "cds": str(trial_cfg.CDS_MODEL_DIR), "utr3": str(trial_cfg.UTR3_MODEL_DIR)},
                    },
                    f,
                    indent=2,
                )

        tokenizer_root = out_dir / "tokenizer"
        tok_utr5.save_pretrained(tokenizer_root / "utr5")
        tok_cds.save_pretrained(tokenizer_root / "cds")
        tok_utr3.save_pretrained(tokenizer_root / "utr3")
        torch.save(extract_head_state_dict(model), out_dir / "reg_head.pt")

        result = {
            "trial_tag": trial_tag,
            "out_dir": str(out_dir),
            "adapter_dir": str(adapter_dir),
            "objective": float(objective_value),
            "best_epoch": int(final["best_epoch"]),
            "tuning_mode": tuning_mode,
            "objective_metric": objective_metric,
            "head_arch": str(trial_cfg.HEAD_ARCH),
            "lora_target_resolution_mode": lora_target_resolution_mode,
            "resumed_from_last_checkpoint": bool(resumed_from_last_checkpoint),
            "final": final,
        }
        if trial is not None:
            trial.set_user_attr("adapter_dir", str(adapter_dir))
            trial.set_user_attr("best_epoch", int(final["best_epoch"]))
            trial.set_user_attr("final", final)
            trial.set_user_attr("objective", float(objective_value))
        write_trial_config(best_value=objective_value, state="COMPLETE", message=None)
        return result

    except Exception as exc:
        message = str(exc)
        write_trial_config(best_value=None, state="FAILED", message=message)
        if trial is not None:
            trial.set_user_attr("message", message)
        raise

    finally:
        if writer is not None:
            writer.close()
        del model, optimizer, scheduler, writer, scaler, tok_utr5, tok_cds, tok_utr3
        del dl_tr, dl_va, ds_tr, ds_va, ds_te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_single_session(run_cfg: Config, session_root: Path) -> Dict[str, Any]:
    set_seed(run_cfg.SEED)
    experiment_slug = resolve_experiment_slug(run_cfg)
    ensure_dir(session_root)
    script_metadata = collect_script_runtime_metadata()
    print(
        f"[Script] path={script_metadata['absolute_script_path']} "
        f"sha256={script_metadata['script_sha256']} git_commit={script_metadata.get('git_commit')}",
        flush=True,
    )

    df, split = read_data_and_split(run_cfg)
    dfi = df.set_index("ENSID", drop=False)
    tr_ids = [x for x in split["train"] if x in dfi.index]
    va_ids = [x for x in split["val"] if x in dfi.index]
    te_ids = [x for x in split["test"] if x in dfi.index]
    if len(tr_ids) == 0 or len(va_ids) == 0 or len(te_ids) == 0:
        raise RuntimeError(f"empty split after ENSID filter: train={len(tr_ids)}, val={len(va_ids)}, test={len(te_ids)}")

    y_scaler = None
    if bool(run_cfg.STANDARDIZE_TARGET):
        arr = dfi.loc[tr_ids, "HALFLIFE"].to_numpy(np.float64)
        mu = float(np.mean(arr))
        sd = float(np.std(arr))
        if (not np.isfinite(sd)) or sd < float(run_cfg.TARGET_STD_EPS):
            sd = 1.0
        y_scaler = {"mean": mu, "std": sd}

    with open(session_root / "resolved_run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "head_arch": str(run_cfg.HEAD_ARCH),
                "experiment_slug": experiment_slug,
                "session_root": str(session_root),
                "config": asdict(run_cfg),
                "split_sizes": {"n_train": len(tr_ids), "n_val": len(va_ids), "n_test": len(te_ids)},
                "target_scaler": y_scaler,
                "script_metadata": script_metadata,
            },
            f,
            indent=2,
        )

    fixed_params = {
        "lora_r": int(run_cfg.LORA_R),
        "lora_alpha_mult": float(run_cfg.LORA_ALPHA / max(1, run_cfg.LORA_R)),
        "lora_alpha": int(run_cfg.LORA_ALPHA),
        "lora_dropout": float(run_cfg.LORA_DROPOUT),
        "lr": float(run_cfg.LR),
        "weight_decay": float(run_cfg.WEIGHT_DECAY),
        "head_dim": int(run_cfg.HEAD_DIM),
        "head_dropout": float(run_cfg.HEAD_DROPOUT),
        "seq_head_attn_hidden": int(run_cfg.SEQ_HEAD_ATTN_HIDDEN),
        "seq_head_gru_dropout": float(run_cfg.SEQ_HEAD_GRU_DROPOUT),
        "seq_head_input_dropout": float(run_cfg.SEQ_HEAD_INPUT_DROPOUT),
        "seq_head_conv_dropout": float(run_cfg.SEQ_HEAD_CONV_DROPOUT),
        "huber_beta": float(run_cfg.HUBER_BETA),
        "loss_kind": str(run_cfg.LOSS_KIND),
    }

    if not bool(run_cfg.USE_OPTUNA):
        with open(session_root / "resolved_search_space.json", "w", encoding="utf-8") as f:
            json.dump({"mode": "fixed_params", "fixed_params": fixed_params, "script_metadata": script_metadata}, f, indent=2)
        print(f"[Run] tuning_mode={resolve_tuning_mode(run_cfg)} session={session_root}", flush=True)
        result = run_one_trial(
            trial=None,
            trial_cfg=run_cfg,
            trial_params=fixed_params,
            session_root=session_root,
            df=df,
            tr_ids=tr_ids,
            va_ids=va_ids,
            te_ids=te_ids,
            y_scaler=y_scaler,
        )
        with open(session_root / "baseline_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_root": str(session_root),
                    "experiment_name": experiment_slug,
                    "tuning_mode": resolve_tuning_mode(run_cfg),
                    "script_metadata": script_metadata,
                    "objective_metric": str(run_cfg.OPTUNA_OBJECTIVE_METRIC).strip().lower(),
                    "result": result,
                },
                f,
                indent=2,
            )
        with open(session_root / "best_run.json", "w", encoding="utf-8") as f:
            json.dump({"script_metadata": script_metadata, "best_run": result}, f, indent=2)
        print(f"[DONE] session={session_root} run={result['trial_tag']}", flush=True)
        return {
            "mode": "fixed_params",
            "session_root": str(session_root),
            "experiment_slug": experiment_slug,
            "cv_fold": int(run_cfg.CV_FOLD) if run_cfg.SPLIT_PROTOCOL == "cv" else None,
            "trial_tag": result["trial_tag"],
            "objective": float(result["objective"]),
            "objective_metric": str(run_cfg.OPTUNA_OBJECTIVE_METRIC).strip().lower(),
            "best_epoch": int(result["best_epoch"]),
            "head_arch": str(run_cfg.HEAD_ARCH),
            "val_r2": result["final"]["val"].get("r2"),
            "test_r2": result["final"]["test"].get("r2"),
            "result": result,
        }

    if optuna is None:
        raise RuntimeError("需要安装 optuna 才能进行超参数搜索。") from OPTUNA_IMPORT_ERROR

    if str(run_cfg.OPTUNA_STORAGE).strip():
        print(
            "[Optuna] OPTUNA_STORAGE is ignored; storage now follows session_root/optuna automatically.",
            flush=True,
        )

    optuna_paths = resolve_optuna_runtime_paths(session_root, run_cfg)
    optuna_dir = Path(optuna_paths["optuna_dir"])
    optuna_trials_dir = Path(optuna_paths["trials_dir"])
    storage_path = Path(optuna_paths["db_path"])
    storage = str(optuna_paths["storage_url"])
    study_name = str(optuna_paths["study_name"])
    ensure_dir(optuna_dir)
    ensure_dir(optuna_trials_dir)

    search_space = {
        "lr": {"low": 3e-5, "high": 2e-4, "log": True},
        "weight_decay": {"low": 1e-7, "high": 3e-5, "log": True},
        "head_dim": [512, 768],
        "head_dropout": {"low": 0.25, "high": 0.60},
        "seq_head_attn_hidden": [128, 256, 512],
        "seq_head_gru_dropout": {"low": 0.05, "high": 0.20},
        "seq_head_input_dropout": {"low": 0.05, "high": 0.20},
        "seq_head_conv_dropout": {"low": 0.05, "high": 0.20},
        "loss_kind": ["huber"],
        "huber_beta": {"low": 1.0, "high": 1.5, "active_if": {"loss_kind": "huber"}},
    }
    if bool(run_cfg.USE_LORA):
        search_space.update(
            {
                "lora_r": [16, 32],
                "lora_alpha_mult": [4.0, 6.0, 8.0],
                "lora_dropout": [0.00, 0.02, 0.05],
            }
        )
    else:
        search_space.update(
            {
                "lora_r": {"fixed": int(run_cfg.LORA_R)},
                "lora_alpha_mult": {"fixed": float(run_cfg.LORA_ALPHA / max(1, run_cfg.LORA_R))},
                "lora_dropout": {"fixed": float(run_cfg.LORA_DROPOUT)},
            }
        )
    with open(optuna_dir / "resolved_search_space.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "optuna",
                "optuna_dir": str(optuna_dir),
                "storage_path": str(storage_path),
                "search_space": search_space,
                "script_metadata": script_metadata,
            },
            f,
            indent=2,
        )

    ensure_sqlite_storage_parent(storage)
    direction = "maximize" if metric_should_maximize(run_cfg.OPTUNA_OBJECTIVE_METRIC) else "minimize"
    print(
        f"[Optuna] n_trials={run_cfg.OPTUNA_N_TRIALS} session={session_root} "
        f"optuna_dir={optuna_dir} study_name={study_name} storage_path={storage_path} direction={direction}",
        flush=True,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(run_cfg.OPTUNA_RESUME_IF_EXISTS),
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=run_cfg.SEED, n_startup_trials=run_cfg.OPTUNA_STARTUP_TRIALS),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=run_cfg.OPTUNA_STARTUP_TRIALS,
            n_warmup_steps=run_cfg.OPTUNA_PRUNER_WARMUP_EPOCHS,
        ),
    )
    resumed = bool(storage is not None and run_cfg.OPTUNA_RESUME_IF_EXISTS and len(study.trials) > 0)

    def objective(trial: "optuna.trial.Trial") -> float:
        trial_params = suggest_optuna_params(trial)
        trial_tag = format_trial_tag(int(trial.number), trial_params)
        with open(optuna_trials_dir / f"{trial_tag}_resolved_trial_params.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trial_number": int(trial.number),
                    "trial_tag": trial_tag,
                    "head_arch": str(run_cfg.HEAD_ARCH),
                    "params": trial_params,
                    "script_metadata": script_metadata,
                },
                f,
                indent=2,
            )
        trial_cfg = validate_run_cfg(
            replace(
                run_cfg,
                LORA_R=int(trial_params["lora_r"]),
                LORA_ALPHA=int(trial_params["lora_alpha"]),
                LORA_DROPOUT=float(trial_params["lora_dropout"]),
                LR=float(trial_params["lr"]),
                WEIGHT_DECAY=float(trial_params["weight_decay"]),
                HEAD_DIM=int(trial_params["head_dim"]),
                HEAD_DROPOUT=float(trial_params["head_dropout"]),
                SEQ_HEAD_ATTN_HIDDEN=int(trial_params["seq_head_attn_hidden"]),
                SEQ_HEAD_GRU_DROPOUT=float(trial_params["seq_head_gru_dropout"]),
                SEQ_HEAD_INPUT_DROPOUT=float(trial_params["seq_head_input_dropout"]),
                SEQ_HEAD_CONV_DROPOUT=float(trial_params["seq_head_conv_dropout"]),
                LOSS_KIND=str(trial_params["loss_kind"]),
                HUBER_BETA=float(trial_params["huber_beta"]),
            )
        )
        result = run_one_trial(
            trial=trial,
            trial_cfg=trial_cfg,
            trial_params=trial_params,
            session_root=session_root,
            df=df,
            tr_ids=tr_ids,
            va_ids=va_ids,
            te_ids=te_ids,
            y_scaler=y_scaler,
        )
        return float(result["objective"])

    timeout_sec = None if int(run_cfg.OPTUNA_TIMEOUT_SEC) <= 0 else int(run_cfg.OPTUNA_TIMEOUT_SEC)
    study.optimize(objective, n_trials=int(run_cfg.OPTUNA_N_TRIALS), timeout=timeout_sec, gc_after_trial=False)

    trial_rows = build_optuna_trial_rows(study)
    pd.DataFrame(trial_rows).to_csv(optuna_dir / "optuna_trials.csv", index=False)

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(
        completed,
        key=lambda trial: metric_for_compare(float(trial.value), maximize=metric_should_maximize(run_cfg.OPTUNA_OBJECTIVE_METRIC)),
        reverse=metric_should_maximize(run_cfg.OPTUNA_OBJECTIVE_METRIC),
    )
    top_k = [summarize_frozen_trial(trial) for trial in ranked[: max(1, int(run_cfg.OPTUNA_TOP_K))]]
    best_trial = summarize_frozen_trial(ranked[0]) if ranked else None

    with open(optuna_dir / "optuna_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "head_arch": str(run_cfg.HEAD_ARCH),
                "session_root": str(session_root),
                "optuna_dir": str(optuna_dir),
                "study_name": study.study_name,
                "storage_backend": "sqlite",
                "storage_path": str(storage_path),
                "resumed": resumed,
                "script_metadata": script_metadata,
                "direction": direction,
                "objective_metric": str(run_cfg.OPTUNA_OBJECTIVE_METRIC).strip().lower(),
                "n_trials_total": len(study.trials),
                "n_trials_completed": sum(int(trial.state == optuna.trial.TrialState.COMPLETE) for trial in study.trials),
                "n_trials_pruned": sum(int(trial.state == optuna.trial.TrialState.PRUNED) for trial in study.trials),
                "n_trials_failed": sum(int(trial.state == optuna.trial.TrialState.FAIL) for trial in study.trials),
                "search_space": search_space,
                "best_trial": best_trial,
                "top_k_trials": top_k,
            },
            f,
            indent=2,
        )
    with open(optuna_dir / "best_trial.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "head_arch": str(run_cfg.HEAD_ARCH),
                "study_name": study.study_name,
                "optuna_dir": str(optuna_dir),
                "storage_path": str(storage_path),
                "script_metadata": script_metadata,
                "best_trial": best_trial,
            },
            f,
            indent=2,
        )

    print(f"[DONE] session={session_root} best={best_trial['trial_tag'] if best_trial else 'NA'}", flush=True)
    return {
        "mode": "optuna",
        "session_root": str(session_root),
        "experiment_slug": experiment_slug,
        "cv_fold": int(run_cfg.CV_FOLD) if run_cfg.SPLIT_PROTOCOL == "cv" else None,
        "study_name": study.study_name,
        "objective_metric": str(run_cfg.OPTUNA_OBJECTIVE_METRIC).strip().lower(),
        "head_arch": str(run_cfg.HEAD_ARCH),
        "best_trial": best_trial,
        "top_k_trials": top_k,
    }


def main() -> None:
    run_cfg = validate_run_cfg(cfg)
    experiment_slug = resolve_experiment_slug(run_cfg)
    batch_root = resolve_session_root(run_cfg)
    ensure_dir(batch_root)

    if run_cfg.SPLIT_PROTOCOL != "cv":
        run_single_session(run_cfg, batch_root)
        return

    cv_folds = resolve_cv_fold_values(run_cfg)
    if len(cv_folds) <= 1:
        run_single_session(validate_run_cfg(replace(run_cfg, CV_FOLD=int(cv_folds[0]))), batch_root)
        return

    script_metadata = collect_script_runtime_metadata()
    with open(batch_root / "resolved_run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "head_arch": str(run_cfg.HEAD_ARCH),
                "batch_mode": "cv_multi_fold",
                "experiment_slug": experiment_slug,
                "batch_root": str(batch_root),
                "requested_cv_folds": list(cv_folds),
                "config": asdict(run_cfg),
                "script_metadata": script_metadata,
            },
            f,
            indent=2,
        )

    batch_rows: List[Dict[str, Any]] = []
    for fold in cv_folds:
        fold_cfg = validate_run_cfg(replace(run_cfg, CV_FOLD=int(fold)))
        fold_root = batch_root / f"cv_fold_{int(fold):02d}"
        print(f"[CV-Batch] start fold={fold} session={fold_root}", flush=True)
        fold_summary = run_single_session(fold_cfg, fold_root)
        row = {
            "cv_fold": int(fold),
            "mode": str(fold_summary.get("mode")),
            "session_root": str(fold_summary.get("session_root")),
            "objective_metric": str(fold_summary.get("objective_metric")),
            "head_arch": str(fold_summary.get("head_arch", run_cfg.HEAD_ARCH)),
        }
        if fold_summary.get("mode") == "fixed_params":
            row.update(
                {
                    "trial_tag": fold_summary.get("trial_tag"),
                    "objective": fold_summary.get("objective"),
                    "best_epoch": fold_summary.get("best_epoch"),
                    "val_r2": fold_summary.get("val_r2"),
                    "test_r2": fold_summary.get("test_r2"),
                }
            )
        else:
            best_trial = fold_summary.get("best_trial") or {}
            row.update(
                {
                    "study_name": fold_summary.get("study_name"),
                    "trial_tag": best_trial.get("trial_tag"),
                    "objective": best_trial.get("value"),
                    "best_epoch": best_trial.get("best_epoch"),
                    "val_r2": best_trial.get("val_r2"),
                    "test_r2": best_trial.get("test_r2"),
                }
            )
        batch_rows.append(row)

    with open(batch_root / "cv_batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "head_arch": str(run_cfg.HEAD_ARCH),
                "batch_mode": "cv_multi_fold",
                "experiment_slug": experiment_slug,
                "batch_root": str(batch_root),
                "requested_cv_folds": list(cv_folds),
                "objective_metric": str(run_cfg.OPTUNA_OBJECTIVE_METRIC).strip().lower(),
                "use_optuna": bool(run_cfg.USE_OPTUNA),
                "rows": batch_rows,
            },
            f,
            indent=2,
        )
    pd.DataFrame(batch_rows).to_csv(batch_root / "cv_batch_summary.csv", index=False)
    print(f"[DONE] batch_root={batch_root}", flush=True)


if __name__ == "__main__":
    main()
