"""Common-Compact booster primitives; feature matrices are not bundled."""
import numpy as np
from typing import Any
import json
from pathlib import Path
from training.support import require

FORMAL_XGB_PARAMS=json.loads((Path(__file__).parent/'config/common_xgb.json').read_text())

def fit_human_compact(train_x, train_y, val_x, val_y, fold, n_jobs=8):
    """Fixed booster, early stopped only on the human inner-validation set.

    Inputs must follow config/common_compact_features.csv and the supplied human
    split ID lists. No mouse or human outer-test labels are accepted here.
    """
    from xgboost import XGBRegressor
    require(fold in range(5), 'Invalid human fold')
    train_x=np.asarray(train_x,dtype=np.float32);val_x=np.asarray(val_x,dtype=np.float32)
    require(train_x.ndim==val_x.ndim==2 and train_x.shape[1]==val_x.shape[1]==1027,'Expected1027 frozen features')
    require(np.isfinite(train_x).all() and np.isfinite(val_x).all(),'Features must be finite')
    params=dict(FORMAL_XGB_PARAMS,n_jobs=n_jobs,random_state=42+fold,verbosity=1)
    model=XGBRegressor(**params)
    model.fit(train_x,np.asarray(train_y,np.float64),eval_set=[(val_x,np.asarray(val_y,np.float64))],verbose=False)
    return model

def best_iteration(model: Any) -> int:
    value = getattr(model, "best_iteration", None)
    return int(value) if value is not None else int(FORMAL_XGB_PARAMS["n_estimators"] - 1)

def predict_best(model: Any, matrix: np.ndarray) -> np.ndarray:
    iteration = best_iteration(model)
    return np.asarray(model.predict(matrix, iteration_range=(0, iteration + 1)), dtype=np.float64)
