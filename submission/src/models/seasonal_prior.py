"""Apply the shrunk seasonal Ridge residual prior.

The prior was fit on validation-block ensemble residuals using
Ridge(alpha=10) on structural calendar + regime features. At inference
time we reconstruct the same feature vector and apply
``lambda * ridge.predict(...)`` to the ensemble's per-horizon prediction.

The Ridge model is stored as plain coefficients + intercept (no sklearn
pickling), so this stays decoupled from the training script's Python
environment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from submission.src.data.schema import REPORTING_LAG_DAYS
from submission.src.features.regime import _infer_state_probs, fit_state_model


# Must match the FEATURE_COLS order in fit_seasonal_prior.py.
FEATURE_COLS = [
    "t_dow", "t_is_weekend", "t_month", "t_doy_sin", "t_doy_cos",
    "t_is_xmas_week", "t_is_xmas_to_ny", "t_is_post_ny_week", "t_is_jan_first_half",
    "t_days_to_dec25", "t_days_to_jan1",
    "cov_state_prob_0", "cov_state_prob_1", "cov_state_prob_2", "cov_state_prob_3",
    "cov_state_entropy",
    "t_is_long_horizon",
    "t_p3_x_post_ny", "t_p3_x_jan_first_half", "t_entropy_x_xmas",
]


def _signed_days(target: pd.Timestamp, mm: int, dd: int) -> int:
    y = target.year
    candidates = [pd.Timestamp(y - 1, mm, dd), pd.Timestamp(y, mm, dd), pd.Timestamp(y + 1, mm, dd)]
    return min((c - target).days for c in candidates if abs((c - target).days) <= 366)


def _feature_vector(
    origin: pd.Timestamp,
    horizon: int,
    state_probs: np.ndarray,
    state_entropy: float,
) -> np.ndarray:
    td = origin + pd.Timedelta(days=horizon)
    f = {
        "t_dow": float(td.dayofweek),
        "t_is_weekend": float(td.dayofweek >= 5),
        "t_month": float(td.month),
        "t_doy_sin": float(np.sin(2 * np.pi * td.dayofyear / 365.25)),
        "t_doy_cos": float(np.cos(2 * np.pi * td.dayofyear / 365.25)),
        "t_is_xmas_week": float(td.month == 12 and 22 <= td.day <= 28),
        "t_is_xmas_to_ny": float((td.month == 12 and td.day >= 25) or (td.month == 1 and td.day == 1)),
        "t_is_post_ny_week": float(td.month == 1 and td.day <= 7),
        "t_is_jan_first_half": float(td.month == 1 and td.day <= 15),
        "t_days_to_dec25": float(_signed_days(td, 12, 25)),
        "t_days_to_jan1": float(_signed_days(td, 1, 1)),
        "cov_state_prob_0": float(state_probs[0]),
        "cov_state_prob_1": float(state_probs[1]),
        "cov_state_prob_2": float(state_probs[2]),
        "cov_state_prob_3": float(state_probs[3]),
        "cov_state_entropy": state_entropy,
        "t_is_long_horizon": float(horizon >= 6),
    }
    f["t_p3_x_post_ny"] = f["cov_state_prob_3"] * f["t_is_post_ny_week"]
    f["t_p3_x_jan_first_half"] = f["cov_state_prob_3"] * f["t_is_jan_first_half"]
    f["t_entropy_x_xmas"] = f["cov_state_entropy"] * f["t_is_xmas_to_ny"]
    return np.array([f[c] for c in FEATURE_COLS], dtype=float)


def load_prior(path: str | Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    coefs = np.array([d["coefficients"][c] for c in FEATURE_COLS], dtype=float)
    return {"coefs": coefs, "intercept": float(d["intercept"]), "raw": d}


def apply_seasonal_prior(
    preds: dict[int, float],
    origin: pd.Timestamp,
    daily_df: pd.DataFrame,
    prior: dict,
    lambda_: float,
    n_states: int = 4,
) -> dict[int, float]:
    """Add ``lambda_ * ridge.predict(features(o, h))`` to each ``preds[h]``.

    Fits the covariate-pressure-state model at origin O on data <= O - 4
    (mirroring the rest of the pipeline's two-clock embargo) and infers
    state posteriors at O, which feed the structural feature vector.
    """
    train_end = origin - pd.Timedelta(days=REPORTING_LAG_DAYS + 1)
    state_model = fit_state_model(daily_df, n_states=n_states, train_end=train_end)
    cov_known = daily_df.loc[:origin, state_model["cols"]].ffill().fillna(0.0)
    if cov_known.empty:
        return preds  # bail safely
    p_now = _infer_state_probs(state_model, cov_known.iloc[-1].values)
    entropy = float(-(p_now * np.log(p_now + 1e-9)).sum())

    out: dict[int, float] = {}
    for h, base in preds.items():
        x = _feature_vector(origin, h, p_now, entropy)
        delta = float(x @ prior["coefs"] + prior["intercept"])
        out[h] = float(base + lambda_ * delta)
    return out
