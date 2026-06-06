"""Per-prize convex (NNLS) blending of per-horizon learner outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls


DEFAULT_LEARNER_COLS = [
    "lgbm",
    "catboost",
    "ridge",
    "chronos",
    "tuominen_lgbm",
    "pmdarima_auto",
]


def infer_learner_cols(
    df: pd.DataFrame,
    requested: list[str] | None = None,
) -> list[str]:
    """Return learner columns present in a prediction frame."""
    candidates = requested or DEFAULT_LEARNER_COLS
    return [c for c in candidates if c in df.columns and df[c].notna().any()]


def nnls_simplex(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve argmin ||Mw - y||^2 s.t. w>=0, sum(w)=1."""
    M_aug = np.vstack([M, np.ones((1, M.shape[1])) * 1e3])
    y_aug = np.concatenate([y, np.array([1e3])])
    w, _ = nnls(M_aug, y_aug)
    if w.sum() == 0:
        return np.full(M.shape[1], 1.0 / M.shape[1])
    return w / w.sum()


def fit_per_horizon_weights(
    stacked: pd.DataFrame, horizons: list[int], learner_cols: list[str]
) -> dict[int, np.ndarray]:
    """For each h, fit NNLS over per-learner predictions vs y."""
    weights: dict[int, np.ndarray] = {}
    for h in horizons:
        sub = stacked[stacked["horizon"] == h].dropna(subset=learner_cols + ["y"])
        if sub.empty:
            weights[h] = np.full(len(learner_cols), 1.0 / len(learner_cols))
            continue
        M = sub[learner_cols].values
        y = sub["y"].values
        weights[h] = nnls_simplex(M, y)
    return weights


def fit_prize_weights(
    stacked: pd.DataFrame,
    prize_horizons: list[int],
    learner_cols: list[str],
    strategy: str = "mean",
) -> np.ndarray:
    """Aggregate per-horizon weights for one prize.

    strategy: 'mean' = average per-h weights; 'joint' = NNLS on stacked rows.
    """
    if strategy == "mean":
        per_h = fit_per_horizon_weights(stacked, prize_horizons, learner_cols)
        return np.mean([per_h[h] for h in prize_horizons if h in per_h], axis=0)
    if strategy == "joint":
        sub = stacked[stacked["horizon"].isin(prize_horizons)].dropna(subset=learner_cols + ["y"])
        return nnls_simplex(sub[learner_cols].values, sub["y"].values)
    raise ValueError(f"unknown strategy: {strategy}")


def apply_weights(preds: dict[str, float], weights: np.ndarray, learner_cols: list[str]) -> float:
    return float(sum(weights[i] * preds[c] for i, c in enumerate(learner_cols)))
