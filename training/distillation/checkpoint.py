"""Paired initial-state and checkpoint serialization primitives."""
from dataclasses import asdict
from training.support import require,safe,sha,js
from training.distillation.student import REGIONS
from training.distillation.optimization import parameter_digest

def epoch_orders(n, seed):
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randperm(n, generator=gen).tolist() for _ in range(16)]

def trainable_state(student):
    return {name: p.detach().cpu().clone() for name, p in student.named_parameters() if p.requires_grad}

def restore_trainables(student, state):
    import torch
    current = {name: p for name, p in student.named_parameters() if p.requires_grad}
    require(set(current) == set(state), "Trainable state keys differ")
    with torch.no_grad():
        for name, p in current.items():
            require(p.shape == state[name].shape and p.dtype == state[name].dtype, "Tensor shape/dtype differs")
            p.copy_(state[name].to(p.device))
    require(all(torch.equal(p.detach().cpu(), state[name]) for name, p in current.items()), "Trainable restore failed")

def save_tensor_payload(root, relative, payload):
    import torch
    path = safe(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        torch.save(payload, f)
    return path

def export_checkpoint(root, relative, student, cfg, legacy, provenance, epoch, score):
    from peft.utils.save_and_load import get_peft_model_state_dict
    payload = {"checkpoint_format": "mrnalm_triplet_lora_head", "pd4_provenance": provenance,
        "cfg": asdict(cfg), "epoch": epoch, "best_score_raw": score, "y_scaler": student.scaler}
    for region in REGIONS:
        state = get_peft_model_state_dict(getattr(student.model, region), save_embedding_layers=(region == "cds"))
        payload[region + "_adapter_state"] = {k: v.detach().cpu().clone() for k, v in state.items()}
    payload["head_state"] = legacy._move_state_to_cpu(legacy.extract_head_state_dict(student.model))
    return save_tensor_payload(root, relative, payload)

def restore_checkpoint(student, path, expected, legacy):
    import torch
    from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["pd4_provenance"] == expected and payload["y_scaler"] == student.scaler, "Checkpoint provenance/scaler mismatch")
    for region in REGIONS:
        branch = getattr(student.model, region)
        current = get_peft_model_state_dict(branch, save_embedding_layers=(region == "cds"))
        state = payload[region + "_adapter_state"]
        require(set(state) == set(current), "Adapter keys mismatch")
        # Non-LoRA tensors serialized by PEFT must remain byte-identical to base.
        require(all(torch.equal(current[k].cpu(), v) for k, v in state.items() if "lora_" not in k), "Frozen CDS tensor altered")
        set_peft_model_state_dict(branch, state)
        restored = get_peft_model_state_dict(branch, save_embedding_layers=(region == "cds"))
        require(all(torch.equal(restored[k].cpu(), v) for k, v in state.items()), "Adapter restore failed")
    legacy.load_head_state_dict(student.model, payload["head_state"])
    for name, state in payload["head_state"].items():
        require(all(torch.equal(getattr(student.model, name).state_dict()[k].cpu(), v) for k, v in state.items()), "Head restore failed")
    return payload

def roundtrip(root, relative, student, cfg, legacy, provenance, epoch, score, probe, device):
    import torch
    from training.distillation.student import to_device
    student.eval()
    with torch.inference_mode():
        before = student(to_device(probe, device)).float().cpu().clone()
    digest = parameter_digest(student)
    objects = [(n, id(p), p.requires_grad) for n, p in student.named_parameters()]
    path = export_checkpoint(root, relative, student, cfg, legacy, provenance, epoch, score)
    with torch.no_grad():
        for p in student.parameters():
            if p.requires_grad:
                p.zero_()
    require(parameter_digest(student) != digest, "Ineffective zeroing test")
    restore_checkpoint(student, path, provenance, legacy)
    require(parameter_digest(student) == digest and objects == [(n, id(p), p.requires_grad) for n, p in student.named_parameters()], "Checkpoint roundtrip mismatch")
    with torch.inference_mode():
        after = student(to_device(probe, device)).float().cpu()
    require(torch.equal(before, after), "Sequence prediction roundtrip mismatch")
    js(root, f"audit/checkpoint_roundtrip_epoch_{epoch}.json", {"status": "PASS", "checkpoint": str(path), "sha256": sha(path),
        "all_trainables_zeroed_before_restore": True, "parameter_object_ids_preserved": True, "ordered_sequence_predictions_bitwise_equal": True})
    return path
