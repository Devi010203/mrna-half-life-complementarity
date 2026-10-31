"""Teacher-separated, sequence-only PD4 student and frozen loss."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from training.support import require
ARMS={"A_LABEL_ONLY":(0.0,None),"B_BIOCHEM_KD":(0.25,"biochemical_559"),"C_COMPACT_KD":(0.25,"common_compact_1027")}
BETA=1.2783
REGIONS=("utr5","cds","utr3")
MODEL_KEYS={"batch_size"}|{f"{r}_{s}" for r in REGIONS for s in ("input_ids","attention_mask","sample_index","chunk_index")}|{f"n_chunks_{r}_per_sample" for r in REGIONS}

def aligned_teacher(table, ids, outer_fold, teacher, receivers):
    required = {"ENSID", "row_index", "teacher_pred", "outer_fold", "inner_fold",
                "teacher", "student_train_order"}
    require(set(table.columns) == required, "Unexpected teacher columns, including possible labels")
    require(table.ENSID.is_unique and len(ids) == len(set(ids)) and set(table.ENSID) == set(ids),
            "Teacher IDs must cover exactly the current student train IDs")
    require(table.outer_fold.eq(outer_fold).all() and table.teacher.eq(teacher).all(),
            "Wrong outer fold or teacher")
    aligned = table.set_index("ENSID").loc[ids]
    require(np.array_equal(aligned.student_train_order, np.arange(len(ids))), "Student order metadata mismatch")
    require(np.array_equal(aligned.inner_fold, receivers.set_index("ENSID").loc[ids, "teacher_receiver_fold"]),
            "Cross-fitting receiver fold mismatch")
    values = aligned.teacher_pred.to_numpy(np.float64)
    require(np.isfinite(values).all(), "Nonfinite teacher supervision")
    return values

def distillation_loss(pred, target, arm, teacher=None):
    require(arm in ARMS, "Arm is not preregistered")
    require(pred.ndim == target.ndim == 1 and pred.shape == target.shape and pred.numel() > 0,
            "Prediction/label shape mismatch (broadcasting prohibited)")
    require(torch.isfinite(pred).all().item() and torch.isfinite(target).all().item(), "Nonfinite loss inputs")
    weight, _ = ARMS[arm]
    supervised = F.smooth_l1_loss(pred, target.detach(), beta=BETA)
    if weight == 0:
        # Deliberately do not inspect/read the teacher argument in this path.
        return supervised
    require(teacher is not None and teacher.shape == pred.shape, "Missing or wrong-shaped teacher target")
    require(torch.isfinite(teacher).all().item(), "Nonfinite teacher target")
    return (1.0 - weight) * supervised + weight * F.smooth_l1_loss(pred, teacher.detach(), beta=BETA)

class StudentDataset(Dataset):
    def __init__(self, sequences, ids, labels, scaler, teacher=None, split="train"):
        require(split in {"train", "val", "test"} and (split == "train" or teacher is None), "Teacher allowed only in training")
        require(len(ids) == len(set(ids)) == len(labels) and sequences.index.tolist() == ids, "Dataset alignment error")
        require(np.isfinite(labels).all() and np.isfinite(list(scaler.values())).all() and scaler["std"] > 0, "Invalid target/scaler")
        require(teacher is None or (len(teacher) == len(ids) and np.isfinite(teacher).all()), "Teacher alignment error")
        self.ids, self.sequences, self.labels, self.scaler, self.teacher = ids, sequences, labels, scaler, teacher

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        row = self.sequences.iloc[i]
        z = lambda v: (float(v) - self.scaler["mean"]) / self.scaler["std"]
        item = {"ensid": self.ids[i], "utr5": str(row["5UTR"]), "orf": str(row["ORF"]),
                "utr3": str(row["3UTR"]), "y_raw": float(self.labels[i]), "y": z(self.labels[i])}
        if self.teacher is not None:
            item["teacher_y"] = z(self.teacher[i])
        return item

class SequenceCollator:
    def __init__(self, legacy, tokenizers, cfg, lengths):
        self.legacy_collate = legacy.build_collate_mrnalm(*tokenizers, cfg, chunk_lengths=lengths)

    def inference(self, sequences):
        require(len(sequences) > 0 and all(set(x) == {"utr5", "orf", "utr3"} for x in sequences),
                "Inference accepts only three sequence fields, no ID/label/teacher")
        payload = self.legacy_collate([dict(x, ensid="", y=0.0, y_raw=0.0) for x in sequences])
        return {k: payload[k] for k in MODEL_KEYS}

    def __call__(self, samples):
        present = ["teacher_y" in x for x in samples]
        require(all(present) or not any(present), "Mixed teacher availability in a batch")
        result = {"inputs": self.inference([{k: x[k] for k in ("utr5", "orf", "utr3")} for x in samples]),
                  "ensid": [x["ensid"] for x in samples],
                  "y": torch.tensor([x["y"] for x in samples], dtype=torch.float32),
                  "y_raw": torch.tensor([x["y_raw"] for x in samples], dtype=torch.float32)}
        if all(present):
            result["teacher_y"] = torch.tensor([x["teacher_y"] for x in samples], dtype=torch.float32)
        return result

class SequenceStudent(nn.Module):
    def __init__(self, model, scaler):
        super().__init__()
        self.model, self.scaler = model, dict(scaler)

    def forward(self, inputs):
        require(set(inputs) == MODEL_KEYS, "Unexpected model inputs: IDs/targets/teacher are prohibited")
        first = inputs["utr5_input_ids"]
        internal = dict(inputs, y=torch.zeros(int(inputs["batch_size"]), device=first.device))
        return self.model(internal)

    @torch.inference_mode()
    def predict_sequences(self, sequences, collator, device):
        self.eval()
        values = self(to_device(collator.inference(sequences), device)).float()
        return values * self.scaler["std"] + self.scaler["mean"]

def to_device(inputs, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
