"""P3a Markov pressure-regime features.

Hospital pressure is regime-driven (normal -> elevated -> crisis -> recovery).
We fit K latent states with KMeans on operational covariates, then surface:
  - current state posteriors at origin O
  - expected state distribution at O + h via P^h
  - state momentum (days since last state change)

Two-clock discipline (mirrors test_leakage_invariants.py invariant 9):
  - covariate-driven state uses inputs <= midday(o); h-step forecast = state @ P^h
  - target-driven state    uses inputs <= o - 3;    h-step forecast = state @ P^(h + 3)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from submission.src.data.schema import REPORTING_LAG_DAYS, TARGET_COL


PRESSURE_PROXIES = [
    "OPEL", "Patients in A&E", "Bed Occupancy", "Ambulance Handovers",
    "Decision to Admit", "4hr Breach", "Total Breaches", "Resus",
]


def select_pressure_columns(daily_df: pd.DataFrame) -> list[str]:
    """Pick covariate columns whose names contain pressure-proxy keywords."""
    cols = []
    for c in daily_df.columns:
        if c == TARGET_COL:
            continue
        if any(p.lower() in c.lower() for p in PRESSURE_PROXIES):
            cols.append(c)
    return cols


def _estimate_transition_matrix(states: np.ndarray, n_states: int, alpha: float) -> np.ndarray:
    """Laplace-smoothed transition matrix from a state sequence."""
    counts = np.full((n_states, n_states), alpha)
    for a, b in zip(states[:-1], states[1:]):
        counts[a, b] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def fit_state_model(
    daily_df: pd.DataFrame,
    n_states: int,
    train_end: pd.Timestamp,
    cols: list[str] | None = None,
) -> dict:
    """Fit KMeans pressure states on training-pool data only (data <= train_end)."""
    cols = cols or select_pressure_columns(daily_df)
    train = daily_df.loc[:train_end, cols].dropna(how="all").ffill().fillna(0.0)
    scaler = StandardScaler().fit(train.values)
    km = KMeans(n_clusters=n_states, n_init=10, random_state=2026)
    km.fit(scaler.transform(train.values))
    states = km.predict(scaler.transform(train.values))
    P = _estimate_transition_matrix(states, n_states=n_states, alpha=1.0)
    return {"cols": cols, "scaler": scaler, "kmeans": km, "P": P, "n_states": n_states}


def fit_target_state_model(
    daily_df: pd.DataFrame, n_states: int, train_end: pd.Timestamp
) -> dict:
    """Univariate Y-history state model (small K, e.g. K=3)."""
    cols = [TARGET_COL]
    train = daily_df.loc[:train_end, cols].dropna()
    scaler = StandardScaler().fit(train.values)
    km = KMeans(n_clusters=n_states, n_init=10, random_state=2026)
    km.fit(scaler.transform(train.values))
    states = km.predict(scaler.transform(train.values))
    P = _estimate_transition_matrix(states, n_states=n_states, alpha=1.0)
    return {"cols": cols, "scaler": scaler, "kmeans": km, "P": P, "n_states": n_states}


def _infer_state_probs(model: dict, x_row: np.ndarray) -> np.ndarray:
    distances = model["kmeans"].transform(model["scaler"].transform(x_row.reshape(1, -1)))[0]
    inv = 1.0 / (distances + 1e-6)
    return inv / inv.sum()


def covariate_state_features(
    daily_df: pd.DataFrame, origin: pd.Timestamp, model: dict, horizons: list[int]
) -> dict[str, float]:
    """Covariate-clock Markov features at origin O (data <= midday O)."""
    cols = model["cols"]
    cov_known = daily_df.loc[:origin, cols].ffill().fillna(0.0)
    if cov_known.empty:
        return {}
    p_now = _infer_state_probs(model, cov_known.iloc[-1].values)
    feats: dict[str, float] = {}
    for k in range(model["n_states"]):
        feats[f"cov_state_prob_{k}"] = float(p_now[k])
    feats["cov_most_likely_state"] = float(p_now.argmax())
    feats["cov_state_entropy"] = float(-(p_now * np.log(p_now + 1e-9)).sum())

    P = model["P"]
    for h in horizons:
        p_h = p_now @ np.linalg.matrix_power(P, h)
        for k in range(model["n_states"]):
            feats[f"cov_state_prob_{k}_h{h}"] = float(p_h[k])
        feats[f"cov_expected_state_h{h}"] = float(p_h.argmax())
        feats[f"cov_prob_high_pressure_h{h}"] = float(p_h[-2:].sum())

    # State momentum: how long has the system been in its current regime?
    last_30 = cov_known.iloc[-30:]
    if len(last_30) >= 2:
        recent_states = model["kmeans"].predict(model["scaler"].transform(last_30.values))
        cur = recent_states[-1]
        # walk backwards until state changes
        run = 0
        for s in recent_states[::-1]:
            if s == cur:
                run += 1
            else:
                break
        feats["cov_days_in_current_state"] = float(run)
    else:
        feats["cov_days_in_current_state"] = 0.0
    return feats


def target_state_features(
    daily_df: pd.DataFrame, origin: pd.Timestamp, target_state_model: dict, horizons: list[int]
) -> dict[str, float]:
    """Y-clock Markov features anchored at O - REPORTING_LAG_DAYS.

    h-step forecast uses P^(h + REPORTING_LAG_DAYS).
    """
    y_cutoff = origin - pd.Timedelta(days=REPORTING_LAG_DAYS)
    y_known = daily_df.loc[:y_cutoff, [TARGET_COL]].dropna()
    if y_known.empty:
        return {}
    p_now = _infer_state_probs(target_state_model, y_known.iloc[-1].values)
    feats: dict[str, float] = {}
    for k in range(target_state_model["n_states"]):
        feats[f"y_state_prob_{k}"] = float(p_now[k])
    P = target_state_model["P"]
    for h in horizons:
        steps = h + REPORTING_LAG_DAYS
        p_h = p_now @ np.linalg.matrix_power(P, steps)
        for k in range(target_state_model["n_states"]):
            feats[f"y_state_prob_{k}_h{h}"] = float(p_h[k])
    return feats
