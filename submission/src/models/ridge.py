"""Per-horizon Ridge regressors with per-origin StandardScaler."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from submission.src.data.schema import SEED


FEATURE_DROP_COLS = {"origin", "horizon", "y"}


def _feature_matrix(ds: pd.DataFrame) -> pd.DataFrame:
    return ds[[c for c in ds.columns if c not in FEATURE_DROP_COLS]].astype(float)


def fit_ridge_per_horizon(
    ds: pd.DataFrame,
    params: dict,
    horizons: list[int],
) -> dict[int, Pipeline]:
    models: dict[int, Pipeline] = {}
    for h in horizons:
        sub = ds[ds["horizon"] == h]
        if sub.empty:
            continue
        X = _feature_matrix(sub).fillna(0.0).values
        y = sub["y"].values
        pipe = Pipeline([
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=params.get("alpha", 1.0), random_state=SEED)),
        ])
        pipe.fit(X, y)
        pipe._feature_names = _feature_matrix(sub).columns.tolist()  # type: ignore[attr-defined]
        models[h] = pipe
    return models


def predict_ridge_per_horizon(
    models: dict[int, Pipeline],
    feats: dict,
    horizons: list[int],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for h in horizons:
        if h not in models:
            out[h] = float("nan")
            continue
        cols = models[h]._feature_names  # type: ignore[attr-defined]
        X = pd.DataFrame([feats]).reindex(columns=cols).fillna(0.0).astype(float).values
        out[h] = float(models[h].predict(X)[0])
    return out
