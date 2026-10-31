"""Formal scientific epoch loop, separated from cluster authorization/history.
Inputs must be prepared under the frozen human-only protocol. This is not a
standalone trainer and is not run by scripts/reproduce.py.
"""
import math
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from training.support import require,csv,js
from training.distillation.student import ARMS,distillation_loss,to_device
from training.distillation.optimization import make_optimizer,evaluate_validation,parameter_digest,order_digest
from training.distillation.checkpoint import epoch_orders,roundtrip

def train_student(student,cfg,legacy,collator,train_set,val_set,arm,seed,root,provenance,device,paired_state):
    from training.distillation.checkpoint import restore_trainables
    require(cfg.EPOCHS==16 and cfg.PATIENCE==5 and cfg.BATCH_SIZE==8 and cfg.GRAD_ACCUM_STEPS==2,"Frozen training budget changed")
    require(seed in (42,2026,3407) and arm in ARMS,"Unknown arm/seed")
    require(val_set.teacher is None and not set(train_set.ids)&set(val_set.ids),"Validation leakage")
    restore_trainables(student,paired_state)
    frozen_before=parameter_digest(student,False)
    probe=collator.inference([{k:train_set[i][k] for k in ("utr5","orf","utr3")} for i in range(2)])
    roundtrip(root,"checkpoints/initial.pt",student,cfg,legacy,provenance,0,None,probe,device)
    trainable=[p for p in student.parameters() if p.requires_grad]
    batches=math.ceil(len(train_set)/cfg.BATCH_SIZE)
    optimizer,scheduler,total_planned=make_optimizer(trainable,cfg,batches)
    val_loader=DataLoader(val_set,batch_size=cfg.BATCH_SIZE,shuffle=False,num_workers=0,collate_fn=collator,pin_memory=cfg.PIN_MEMORY_EVAL,generator=torch.Generator().manual_seed(seed))
    orders=epoch_orders(len(train_set),seed)
    best_epoch,best_score,bad,successful,attempted,history=None,-float("inf"),0,0,0,[]
    legacy.set_seed(seed)
    last_log=time.monotonic()
    job_name=f"fold{cfg.CV_FOLD}_seed{seed}_{arm}"
    for epoch, indices in enumerate(orders, 1):
        epoch_start = time.monotonic()
        loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, sampler=indices, num_workers=0, collate_fn=collator,
            pin_memory=cfg.PIN_MEMORY_TRAIN, generator=torch.Generator().manual_seed(seed))
        student.train()
        optimizer.zero_grad(set_to_none=True)
        seen, batch_rows = [], []
        for step, batch in enumerate(loader, 1):
            seen.extend(batch["ensid"])
            y = batch["y"].to(device)
            teacher_tensor = batch.get("teacher_y")
            teacher_tensor = teacher_tensor.to(device) if teacher_tensor is not None else None
            require(teacher_tensor is None or not teacher_tensor.requires_grad, "Teacher must remain detached")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.USE_AMP):
                pred = student(to_device(batch["inputs"], device))
                loss = distillation_loss(pred, y, arm, teacher_tensor)
                supervised = F.smooth_l1_loss(pred, y.detach(), beta=cfg.HUBER_BETA)
                teacher_loss = F.smooth_l1_loss(pred, teacher_tensor.detach(), beta=cfg.HUBER_BETA) if teacher_tensor is not None else None
                scaled_loss = loss / cfg.GRAD_ACCUM_STEPS
            scaled_loss.backward()
            if epoch == step == 1:
                grads = {name: float(p.grad.detach().float().norm()) for name, p in student.named_parameters() if p.requires_grad and p.grad is not None}
                require(len(grads) == 228 and np.isfinite(list(grads.values())).all(), "Gradient sanity failed")
                require(all(p.grad is None for p in student.parameters() if not p.requires_grad), "Backbone gradient leak")
                js(root, "audit/first_step_gradients.json", {"trainable_gradient_tensors": len(grads), "norms": grads, "teacher_gradient": False})
            update = step % cfg.GRAD_ACCUM_STEPS == 0 or step == batches
            lr_used = optimizer.param_groups[0]["lr"]
            if update:
                attempted += 1
                torch.nn.utils.clip_grad_norm_(trainable, cfg.GRAD_CLIP_NORM, error_if_nonfinite=True)
                optimizer.step()
                successful += 1
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            batch_rows.append({"epoch": epoch, "batch": step, "n": len(batch["ensid"]), "supervised_loss": float(supervised.detach()),
                "teacher_loss": float(teacher_loss.detach()) if teacher_loss is not None else None,
                "weighted_total_loss": float(loss.detach()), "lambda": ARMS[arm][0], "optimizer_attempted": update,
                "optimizer_succeeded": update, "skipped_updates": 0, "lr_before_update": lr_used})
            if time.monotonic() - last_log >= 120:
                print(f"{job_name} epoch={epoch}/16 batch={step}/{batches} updates={successful}", flush=True)
                last_log = time.monotonic()
        require(seen == [train_set.ids[i] for i in indices], "Consumed data order differs from paired plan")
        csv(root, f"results/train_batches_epoch_{epoch}.csv", pd.DataFrame(batch_rows))
        val_metrics, val_predictions = evaluate_validation(student, val_loader, val_set, device)
        csv(root, f"results/validation_epoch_{epoch}.csv", val_predictions)
        roundtrip(root, f"checkpoints/epoch_{epoch}.pt", student, cfg, legacy, provenance, epoch, val_metrics["r2"], probe, device)
        require(parameter_digest(student, False) == frozen_before, "Frozen backbone changed")
        if val_metrics["r2"] > best_score + 1e-8:
            best_score, best_epoch, bad = val_metrics["r2"], epoch, 0
        else:
            bad += 1
        entry = {"epoch": epoch, "validation": val_metrics, "train_loss_batch_mean": float(np.mean([r["weighted_total_loss"] for r in batch_rows])),
            "samples": len(seen), "train_order_sha256": order_digest(seen), "optimizer_steps": successful,
            "skipped_updates": attempted - successful, "best_epoch_so_far": best_epoch, "bad_epochs": bad, "elapsed_seconds": time.monotonic() - epoch_start}
        history.append(entry)
        js(root, f"results/epoch_{epoch}.json", entry)
        print(f"{job_name} epoch={epoch} V_R2={val_metrics['r2']:.6f} best_epoch={best_epoch} bad_epochs={bad}; roundtrip PASS", flush=True)
        if bad >= cfg.PATIENCE:
            break

    return {"best_epoch":best_epoch,"best_validation_r2":best_score,"history":history}
