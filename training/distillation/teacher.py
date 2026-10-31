"""Inner-cross-fitted teacher core; prediction receivers never supply labels."""
import numpy as np
import pandas as pd
import xgboost as xgb
from training.support import require

def validate_packets(packets, roles, ids):
    require(roles.ENSID.is_unique, "Duplicate role IDs")
    sets = {}
    for key in ("fit", "early_stop", "predict"):
        table = packets[key]
        expected = ["ENSID", "row_index"] + ([] if key == "predict" else ["HALFLIFE"])
        require(table.columns.tolist() == expected, f"Unexpected columns / label exposure in {key}")
        require(table.ENSID.is_unique, f"Duplicate {key} ID")
        positions = table.row_index.to_numpy()
        require(np.issubdtype(positions.dtype, np.integer) and (positions >= 0).all() and (positions < len(ids)).all(), "Invalid feature row positions")
        require([ids[i] for i in positions] == table.ENSID.tolist(), f"Prediction/feature ID misalignment: {key}")
        require(table.ENSID.tolist() == roles.loc[roles.role.eq(key), "ENSID"].tolist(), f"Original role/order mismatch: {key}")
        if key != "predict":
            require(np.isfinite(table.HALFLIFE.to_numpy()).all(), "Nonfinite fit/early-stop label")
        sets[key] = set(table.ENSID)
    require(not (sets["fit"] & sets["early_stop"] or sets["fit"] & sets["predict"] or sets["early_stop"] & sets["predict"]), "Teacher role overlap")
    require(set.union(*sets.values()) == set(roles.ENSID), "Teacher role universe mismatch")

def train_arrays(matrix, packets, names, params, seed, n_jobs):
    """Only fit/es labels accepted by signature; prediction table has no labels."""
    data = {key: pd.DataFrame(matrix[table.row_index.to_numpy()], columns=names) for key, table in packets.items()}
    model = xgb.XGBRegressor(**params, n_jobs=n_jobs, random_state=seed, verbosity=0)
    model.fit(data["fit"], packets["fit"].HALFLIFE.to_numpy(np.float64),
              eval_set=[(data["early_stop"], packets["early_stop"].HALFLIFE.to_numpy(np.float64))], verbose=False)
    best = int(model.best_iteration)
    prediction = np.asarray(model.predict(data["predict"], iteration_range=(0, best + 1)), dtype=np.float64)
    require(np.isfinite(prediction).all(), "Nonfinite teacher predictions")
    return model, prediction
