"""Per-horizon LightGBM regressors."""
from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from submission.src.data.schema import SEED


FEATURE_DROP_COLS = {"origin", "horizon", "y"}
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_]")


def _sanitize_name(s: str) -> str:
    out = _UNSAFE_CHARS.sub("_", s)
    if not out or not (out[0].isalpha() or out[0] == "_"):
        out = "f_" + out
    return out


def _feature_matrix(ds: pd.DataFrame) -> pd.DataFrame:
    X = ds[[c for c in ds.columns if c not in FEATURE_DROP_COLS]].astype(float)
    X.columns = [_sanitize_name(c) for c in X.columns]
    return X


def fit_lgbm_per_horizon(
    ds: pd.DataFrame,
    params: dict,
    horizons: list[int],
    early_stopping_days: int = 60,
) -> dict[int, lgb.Booster]:
    models: dict[int, lgb.Booster] = {}
    for h in horizons:
        sub = ds[ds["horizon"] == h].sort_values("origin")
        if sub.empty:
            continue
        cutoff = sub["origin"].max() - pd.Timedelta(days=early_stopping_days)
        train_part = sub[sub["origin"] <= cutoff]
        val_part = sub[sub["origin"] > cutoff]
        X_tr, y_tr = _feature_matrix(train_part), train_part["y"].values
        X_va, y_va = _feature_matrix(val_part), val_part["y"].values
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain) if len(val_part) > 5 else None
        cb = [lgb.early_stopping(50, verbose=False)] if dvalid else []
        m = lgb.train(
            params={**params, "seed": SEED, "verbose": -1},
            train_set=dtrain,
            valid_sets=[dvalid] if dvalid else None,
            num_boost_round=params.get("n_estimators", 3000),
            callbacks=cb,
        )
        models[h] = m
    return models


def predict_per_horizon(
    models: dict[int, lgb.Booster],
    feats_at_origin: dict,
    horizons: list[int],
) -> dict[int, float]:
    preds: dict[int, float] = {}
    for h in horizons:
        if h not in models:
            preds[h] = float("nan")
            continue
        X = pd.DataFrame([feats_at_origin]).astype(float)
        X.columns = [_sanitize_name(c) for c in X.columns]
        X = X.reindex(columns=models[h].feature_name(), fill_value=np.nan)
        preds[h] = float(models[h].predict(X)[0])
    return preds
