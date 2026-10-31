"""Label-blind human-to-mouse sequence inference primitives."""
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from training.support import require
from dataclasses import replace

def infer_human_checkpoint(sequences, checkpoint, backbone_dirs, device='cuda:0', batch_size=8):
    """Sequence-only adapter of the original five-fold transfer inference.

    Requires externally supplied trained checkpoint and matching three pretrained
    backbones. Call once per frozen human fold, then use fusion.five_fold_mean.
    Never accepts mouse targets or uses them to select a checkpoint.
    """
    from training.human import train_mrnalm as formal
    require(set(sequences.columns)=={'ENSID','5UTR','ORF','3UTR'},'Only ID and three sequences accepted')
    require(sequences.ENSID.is_unique and not sequences.isna().any().any(),'Invalid sequence table')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(payload['checkpoint_format']=='mrnalm_triplet_lora_head','Unexpected checkpoint format')
    require(list(payload['cfg']['SPECIES'])==['human'],'Human-only checkpoint required')
    cfg=replace(formal.Config(**payload['cfg']),DEVICE=device,BATCH_SIZE=batch_size,
        UTR5_MODEL_DIR=str(backbone_dirs['utr5']),CDS_MODEL_DIR=str(backbone_dirs['cds']),UTR3_MODEL_DIR=str(backbone_dirs['utr3']),
        USE_GRAD_CHECKPOINTING=False,USE_INPUT_REQUIRE_GRADS_HOOK=False,RESUME_TRAINING_IF_AVAILABLE=False,
        NUM_WORKERS_EVAL=0,PIN_MEMORY_EVAL=device.startswith('cuda'),PERSISTENT_WORKERS_EVAL=False)
    formal.set_seed(cfg.SEED)
    tokenizers=formal.build_mrnalm_tokenizers(cfg)
    model=formal.MRNALMChunkedTripletRegressor(cfg)
    for tok,branch,name in zip(tokenizers,model.backbone_modules(),('utr5','cds','utr3')):
        formal.validate_tokenizer_model_compatibility(tok,branch,name)
    model,get_state,set_state,*_=formal.maybe_apply_lora_to_mrnalm(model,cfg)
    restored=formal.load_mrnalm_checkpoint(model,checkpoint,use_lora=True,peft_state_dict_fn=get_state,peft_load_state_dict_fn=set_state,trial_tag='human_to_mouse')
    scaler=restored['y_scaler'];require(scaler is not None and scaler['std']>0,'Invalid human target scaler')
    model=model.to(device).eval()
    lengths={r:formal.resolve_safe_max_length(t,getattr(model,r),getattr(cfg,'CHUNK_LEN_'+r.upper())) for t,r in zip(tokenizers,('utr5','cds','utr3'))}
    require(lengths=={'utr5':512,'cds':1024,'utr3':1024},'Chunk length changed')
    collate=formal.build_collate_mrnalm(*tokenizers,cfg,chunk_lengths=lengths)
    loader=DataLoader(MouseSequenceDataset(sequences),batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=device.startswith('cuda'),collate_fn=collate)
    ids,values=[],[]
    with torch.inference_mode():
        for batch in loader:
            formal.move_batch_to_device(batch,device)
            raw=formal.model_forward(model,batch,return_aux=False)[0]
            values.append(raw.detach().float().cpu().numpy().reshape(-1)*float(scaler['std'])+float(scaler['mean']))
            ids.extend(batch['ensid'])
    pred=np.concatenate(values).astype(np.float64,copy=False)
    require(ids==sequences.ENSID.tolist() and np.isfinite(pred).all(),'Inference ID/value mismatch')
    return pd.DataFrame({'ENSID':ids,'prediction':pred})

class MouseSequenceDataset(Dataset):
    """Sequence-only dataset; dummy zero tensors exist solely for model API shape."""

    def __init__(self, frame: pd.DataFrame):
        self.records = frame.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        return {
            "ensid": str(row["ENSID"]),
            "utr5": str(row["5UTR"]),
            "orf": str(row["ORF"]),
            "utr3": str(row["3UTR"]),
            "y_raw": 0.0,
            "y": 0.0,
        }
