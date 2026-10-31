"""Optimizer and validation primitives also used by formal PD4."""
import hashlib
import math
import numpy as np
import pandas as pd
import torch
from training.support import require
from training.distillation.student import to_device

def order_digest(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()

def lr_multiplier(step, total_steps, warmup_ratio):
    warmup = max(1, int(warmup_ratio * total_steps))
    index = int(step) + 1
    if index <= warmup:
        return float(index) / warmup
    progress = min(max(float(index - warmup) / max(1, total_steps - warmup), 0.), 1.)
    return .5 * (1. + math.cos(math.pi * progress))

def make_optimizer(parameters, cfg, batches):
    total = cfg.EPOCHS * math.ceil(batches / cfg.GRAD_ACCUM_STEPS)
    optimizer = torch.optim.AdamW(parameters, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    fn = lambda step: lr_multiplier(step, total, cfg.WARMUP_RATIO)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fn)
    # Preserve the frozen legacy implementation, including its extra first-step
    # multiplication after LambdaLR initialization. Do not silently "optimize" it.
    for group in optimizer.param_groups:
        group["lr"] *= fn(0)
    return optimizer, scheduler, total

def parameter_digest(student, trainable=None):
    digest = hashlib.sha256()
    for name, p in student.named_parameters():
        if trainable is not None and p.requires_grad != trainable:
            continue
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def metrics(y, pred):
    y, pred = np.asarray(y, np.float64), np.asarray(pred, np.float64)
    require(y.ndim == pred.ndim == 1 and y.shape == pred.shape and len(y) > 1
            and np.isfinite(y).all() and np.isfinite(pred).all(), "Invalid validation arrays")
    residual = y - pred
    result = {"r2": float(1 - residual.dot(residual) / np.square(y - y.mean()).sum()),
              "rmse": float(np.sqrt(np.square(residual).mean())), "mae": float(np.abs(residual).mean()),
              "pearson": float(np.corrcoef(y, pred)[0, 1]),
              "spearman": float(np.corrcoef(pd.Series(y).rank(), pd.Series(pred).rank())[0, 1])}
    require(np.isfinite(list(result.values())).all(), "Nonfinite validation metrics")
    return result

@torch.inference_mode()
def evaluate_validation(student, loader, dataset, device):
    require(dataset.teacher is None, "Validation teacher attached")
    student.eval()
    ids, values = [], []
    for batch in loader:
        require("teacher_y" not in batch, "Validation teacher leakage")
        # Match the frozen legacy's FP32 evaluation path (no training autocast).
        pred = student(to_device(batch["inputs"], device)).float()
        raw = pred * dataset.scaler["std"] + dataset.scaler["mean"]
        ids.extend(batch["ensid"])
        values.extend(raw.cpu().tolist())
    require(ids == dataset.ids, "Validation ID/order drift")
    # Use the frozen float64 labels rather than rounding them via batch y_raw.
    frame = pd.DataFrame({"ENSID": ids, "y_true": dataset.labels, "y_pred": values})
    return metrics(frame.y_true, frame.y_pred), frame
