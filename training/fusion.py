"""Human inner-validation fitted OLS fusion; no target-domain fitting."""
import numpy as np
from training.support import require

def fit_human_validation(lm, xgb, target, validation_ids, excluded_test_ids):
    from sklearn.linear_model import LinearRegression
    require(len(validation_ids)==len(set(validation_ids)), 'Duplicate validation IDs')
    require(not set(validation_ids)&set(excluded_test_ids), 'Outer-test leakage')
    matrix=np.column_stack([xgb,lm]).astype(np.float64)
    target=np.asarray(target,np.float64)
    require(matrix.shape==(len(validation_ids),2) and target.shape==(len(validation_ids),),'Alignment error')
    require(np.isfinite(matrix).all() and np.isfinite(target).all(),'Nonfinite fusion inputs')
    model=LinearRegression(fit_intercept=True).fit(matrix,target)
    return {'intercept':float(model.intercept_),'xgb':float(model.coef_[0]),'lm':float(model.coef_[1])}

def apply_locked(lm, xgb, coefficients):
    lm,xgb=np.asarray(lm,np.float64),np.asarray(xgb,np.float64)
    require(lm.shape==xgb.shape and lm.ndim==1,'Prediction alignment error')
    return coefficients['intercept']+coefficients['xgb']*xgb+coefficients['lm']*lm

def five_fold_mean(predictions):
    predictions=np.asarray(predictions,np.float64)
    require(predictions.ndim==2 and predictions.shape[1]==5 and np.isfinite(predictions).all(),'Expected gene-by-five-fold predictions')
    return predictions.mean(axis=1)
