"""Fresh PD4 student construction from explicit pretrained backbones."""
import json
from dataclasses import replace
from pathlib import Path
from training.support import require
from training.distillation.student import REGIONS,SequenceCollator,SequenceStudent

def fresh_student(backbones, scaler, fold, seed, device):
    from training.human import train_mrnalm as legacy
    raw=json.loads((Path(__file__).parents[1]/'config/lm.json').read_text())['config']
    require(fold in range(5) and seed in (42,2026,3407),'Invalid fold/seed')
    require(set(backbones)==set(REGIONS),'Explicit three-branch backbones required')
    cfg=replace(legacy.Config(**raw),CV_FOLD=fold,SEED=seed,DEVICE=device,
        UTR5_MODEL_DIR=str(backbones['utr5']),CDS_MODEL_DIR=str(backbones['cds']),UTR3_MODEL_DIR=str(backbones['utr3']),
        RESUME_TRAINING_IF_AVAILABLE=False,NUM_WORKERS_TRAIN=0,NUM_WORKERS_EVAL=0)
    legacy.set_seed(seed)
    tokenizers=legacy.build_mrnalm_tokenizers(cfg)
    model=legacy.MRNALMChunkedTripletRegressor(cfg)
    for tokenizer,branch,name in zip(tokenizers,model.backbone_modules(),REGIONS):
        legacy.validate_tokenizer_model_compatibility(tokenizer,branch,name)
    model,_,_,_,targets,_=legacy.maybe_apply_lora_to_mrnalm(model,cfg)
    require(targets==['query','key','value'],'LoRA target drift')
    require(all(getattr(model,r).peft_config['default'].use_rslora for r in REGIONS),'RSLoRA drift')
    require(legacy.maybe_enable_input_require_grads(model)==list(REGIONS),'Gradient hook drift')
    require(legacy.maybe_enable_gradient_checkpointing(model,cfg)==list(REGIONS),'Checkpointing drift')
    _,scope=legacy.collect_trainable_parameter_details(model)
    require(scope['total_trainable_params']==6786305 and scope['n_trainable_tensors']==228 and scope['backbone_params']==scope['other_requires_grad_params']==0,'Trainable scope drift')
    lengths={r:legacy.resolve_safe_max_length(t,getattr(model,r),getattr(cfg,'CHUNK_LEN_'+r.upper())) for t,r in zip(tokenizers,REGIONS)}
    require(lengths=={'utr5':512,'cds':1024,'utr3':1024},'Chunk length drift')
    return SequenceStudent(model,scaler).to(device),SequenceCollator(legacy,tokenizers,cfg,lengths),cfg,legacy
