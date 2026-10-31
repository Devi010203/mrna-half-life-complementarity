# Optional scientific implementations

These modules expose the study's learning and transfer logic. They are **not**
part of the default statistical reproduction and are not self-contained LEVEL3
training. No training or weight inference was run during release assembly.

| Module | Preserved implementation | Resources not bundled |
|---|---|---|
| `human/train_mrnalm.py` | Three-region tokenization/chunking, pretrained backbones, RSLoRA, aggregation/head, loss, optimization, checkpoint handling | Three exact pretrained backbones/tokenizers, normalized sequences, original fold artifacts |
| `human/train_xgboost.py` | Feature loading, Bio-XGB fitting/early stopping and controls; exact human split IDs supported | Full human feature matrices and normalized input table |
| `human/train_ridge.py`, `human/train_elasticnet.py` | Fold preprocessing, validation-based regularization and feature-group controls | Matching raw/processed feature and sequence tables |
| `common_xgb.py` | Frozen Compact booster parameters and best-iteration prediction | Human/mouse1027-feature matrices and trained boosters |
| `fusion.py` | Human inner-validation OLS with intercept; fixed application; five-fold averaging | Human validation predictions to refit coefficients (saved final coefficients are in results/) |
| `transfer.py` | Label-blind three-sequence human-checkpoint inference | Five human LM checkpoints and exact pretrained resources |
| `distillation/` | Cross-fitted teacher roles/fitting, detached mixture loss, sequence-only student, fresh construction, optimizer, paired epoch order, checkpoint save/reload, formal epoch loop | Frozen teacher role/receiver packets and soft targets, sequence/feature inputs, pretrained resources and paired initialization states |

`config/` contains source configuration snapshots, all15 human split ID lists and
all1,027 Compact feature definitions. Path placeholders identify unbundled inputs;
they are not download promises. The formal LM configuration is **human only**,
LoRA rank16/alpha64/dropout0, maximum16 epochs, patience5. The copied general
trainer defaults were made human-only with those LoRA settings; implicit resume
and import-time CUDA remasking were disabled. The historical snapshot still
records its original resume setting. The snapshot, not an arbitrary invocation
of a configurable trainer, defines the paper run.

The formal PD4 arms are A label-only, B biochemical teacher and C Compact teacher;
lambda0.25 applies to B/C. The original Huber mixture, extra first-step LR
multiplication, seed order, validation-only stopping and serialization are kept.
PD4 student inputs contain sequences only; teacher scores appear only in the
training loss. The epoch-loop adapter accepts explicit already-paired initial
trainable tensors. It is not a replacement for preparing and validating all
historical protocol inputs. No mouse labels are accepted by the transfer/fusion
fitting interfaces.

Local imports resolve inside this repository (`training.support`, `human`, and
`distillation`); the model implementation is not an unresolved import from the
research workspace. Syntax and local-import closure are tested. Optional
PyTorch/Transformers/PEFT/XGBoost imports and GPU execution were **not** tested in
the clean analysis environment, which deliberately does not install them.

`environment/pyproject.toml`, `uv.lock` and `.python-version` preserve the separate
Python3.11 training environment. If supplying all required resources, install it
separately with `uv sync --project training/environment --locked`, then import
these modules from the repository root using that environment's Python. The
analysis uv lock is not a substitute for the training lock. Hardware/runtime and
upstream resource availability remain additional reproducibility constraints.
