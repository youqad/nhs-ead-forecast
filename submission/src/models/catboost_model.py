"""Per-horizon CatBoost regressors."""
from __future__ import annotations

import catboost as cb
import numpy as np
import pandas as pd

from submission.src.data.schema import SEED


FEATURE_DROP_COLS = {"origin", "horizon", "y"}


def _feature_matrix(ds: pd.DataFrame) -> pd.DataFrame:
    return ds[[c for c in ds.columns if c not in FEATURE_DROP_COLS]].astype(float)


def fit_catboost_per_horizon(
    ds: pd.DataFrame,
    params: dict,
    horizons: list[int],
    early_stopping_days: int = 60,
) -> dict[int, cb.CatBoostRegressor]:
    models: dict[int, cb.CatBoostRegressor] = {}
    for h in horizons:
        sub = ds[ds["horizon"] == h].sort_values("origin")
        if sub.empty:
            continue
        cutoff = sub["origin"].max() - pd.Timedelta(days=early_stopping_days)
        tr = sub[sub["origin"] <= cutoff]
        va = sub[sub["origin"] > cutoff]
        X_tr, y_tr = _feature_matrix(tr), tr["y"].values
        m = cb.CatBoostRegressor(**{**params, "random_seed": SEED, "verbose": 0})
        eval_set = (_feature_matrix(va), va["y"].values) if len(va) > 5 else None
        m.fit(X_tr, y_tr, eval_set=eval_set, early_stopping_rounds=50 if eval_set else None)
        models[h] = m
    return models


def predict_catboost_per_horizon(
    models: dict[int, cb.CatBoostRegressor],
    feats: dict,
    horizons: list[int],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for h in horizons:
        if h not in models:
            out[h] = float("nan")
            continue
        X = pd.DataFrame([feats]).astype(float).reindex(columns=models[h].feature_names_, fill_value=np.nan)
        out[h] = float(models[h].predict(X)[0])
    return out
