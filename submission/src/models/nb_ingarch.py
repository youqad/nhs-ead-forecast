"""Negative Binomial INGARCH for NHS-EAD forecasting.

Based on the Xu Negative Binomial INGARCH variant from Reboredo et al. (2023),
"Forecasting emergency department arrivals using INGARCH models". The paper's
winning specification — NB beats Poisson on out-of-sample MSE and especially on
tail calibration (PIT bars stay within 99% bounds, vs Poisson's U-shape).

Model:
    λ_t = (ω + Σ α_i·z_{t-i} + Σ β_j·λ_{t-j}) · exp(γ · X_t)
    z_t | λ_t ~ NegBin(mean=λ_t, dispersion=φ)
    E[z_t]   = λ_t
    Var[z_t] = λ_t + λ_t² / φ                                # overdispersion when φ < ∞

Parameters: ω (baseline), α_i (shock), β_j (persistence), γ (5 calendar coeffs), φ (dispersion).
φ → ∞ recovers Poisson INGARCH.

Continuous target adaptation: same `z = round(y × count_scale)` trick as the
Poisson variant so the discrete-count likelihood applies.

LEAKAGE SAFETY: identical to `ed_baselines.fit_ingarch` — fits only on target
data up to `forecast_date − REPORTING_LAG_DAYS`; calendar features are
deterministic functions of date; predict step only uses fitted parameters and
future calendar.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from submission.src.data.schema import REPORTING_LAG_DAYS, TARGET_COL


def _calendar_design(index: pd.DatetimeIndex) -> np.ndarray:
    """Monday, weekend, winter (DJF), doy sin/cos. Identical to ed_baselines."""
    dow = index.dayofweek.to_numpy(dtype=float)
    month = index.month.to_numpy(dtype=float)
    doy = index.dayofyear.to_numpy(dtype=float)
    return np.column_stack(
        [
            (dow == 0).astype(float),
            (dow >= 5).astype(float),
            np.isin(month, [12.0, 1.0, 2.0]).astype(float),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ]
    )


def _ingarch_orders(params: dict[str, Any]) -> tuple[int, int]:
    p = int(params.get("p", params.get("ar_order", 1)))
    q = int(params.get("q", params.get("ma_order", 1)))
    if p < 0 or q < 0 or p + q == 0:
        raise ValueError(f"NB-INGARCH orders must be non-negative with p+q>0, got p={p}, q={q}")
    return p, q


def _lag_init(params: dict[str, Any], key: str, total_key: str, order: int) -> np.ndarray:
    if order == 0:
        return np.array([], dtype=float)
    explicit = params.get(key)
    if explicit is not None:
        values = np.asarray(explicit, dtype=float)
        if len(values) != order:
            raise ValueError(f"{key} must have length {order}, got {len(values)}")
        return values
    total = float(params.get(total_key, 0.0))
    return np.full(order, total / order, dtype=float)


def _split_theta(theta: np.ndarray, p: int, q: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    omega = float(theta[0])
    alphas = np.asarray(theta[1 : 1 + p], dtype=float)
    betas = np.asarray(theta[1 + p : 1 + p + q], dtype=float)
    gamma = np.asarray(theta[1 + p + q :], dtype=float)
    return omega, alphas, betas, gamma


def _ingarch_filter(z: np.ndarray, x: np.ndarray, theta: np.ndarray, p: int, q: int) -> np.ndarray:
    """Forward NB-INGARCH(p,q) filter."""
    omega, alphas, betas, gamma = _split_theta(theta, p, q)
    lambdas = np.empty_like(z, dtype=float)
    pre_sample_mean = max(float(np.nanmean(z)), 1e-3)
    lambdas[0] = pre_sample_mean
    for t in range(1, len(z)):
        observed_part = sum(
            float(alphas[i]) * (float(z[t - i - 1]) if t - i - 1 >= 0 else pre_sample_mean)
            for i in range(p)
        )
        intensity_part = sum(
            float(betas[j]) * (float(lambdas[t - j - 1]) if t - j - 1 >= 0 else pre_sample_mean)
            for j in range(q)
        )
        base = omega + observed_part + intensity_part
        multiplier = float(np.exp(np.clip(x[t] @ gamma, -3.0, 3.0)))
        lambdas[t] = max(base * multiplier, 1e-6)
    return lambdas


def _nb_loss(theta_phi: np.ndarray, z: np.ndarray, x: np.ndarray, p: int, q: int) -> float:
    """Negative-binomial INGARCH NLL.

    theta_phi[:-1] are the INGARCH parameters (omega, alphas, betas, *gamma).
    theta_phi[-1] is log(φ); we optimise in log-space to keep φ > 0 and
    cover several orders of magnitude smoothly.

    Likelihood (dropping the lgamma(z+1) term that is constant w.r.t. params):
        ll_t = lgamma(z_t + φ) − lgamma(φ)
             + φ · log(φ / (λ_t + φ))
             + z_t · (log λ_t − log(λ_t + φ))
    """
    theta = theta_phi[:-1]
    log_phi = theta_phi[-1]
    omega, alphas, betas, _ = _split_theta(theta, p, q)
    persistence = float(alphas.sum() + betas.sum())
    if omega <= 0 or (alphas < 0).any() or (betas < 0).any() or persistence >= 0.995:
        return 1e12 + 1e10 * max(persistence - 0.995, 0.0)

    phi = float(np.exp(np.clip(log_phi, -8.0, 16.0)))
    if not np.isfinite(phi) or phi <= 0:
        return 1e12

    lambdas = _ingarch_filter(z, x, theta, p, q)
    lam_plus_phi = lambdas + phi
    log_lik = (
        gammaln(z + phi)
        - gammaln(phi)
        + phi * np.log(phi / lam_plus_phi)
        + z * (np.log(lambdas) - np.log(lam_plus_phi))
    )
    nll = -float(np.sum(log_lik))
    if not np.isfinite(nll):
        return 1e12
    return nll


def fit_nb_ingarch(
    daily_df: pd.DataFrame,
    params: dict[str, Any],
    forecast_date: pd.Timestamp,
    horizons: list[int],
) -> dict[str, Any]:
    """Fit NB INGARCH on target history available by D − REPORTING_LAG_DAYS."""
    del horizons
    known_end = forecast_date - pd.Timedelta(days=REPORTING_LAG_DAYS)
    y = daily_df.loc[:known_end, TARGET_COL].dropna().astype(float).clip(lower=0.0)
    min_train_days = int(params.get("min_train_days", 180))
    if len(y) < min_train_days:
        return {"kind": "constant", "value": float(y.mean()) if len(y) else 0.0}

    count_scale = float(params.get("count_scale", 100.0))
    p, q = _ingarch_orders(params)
    z = np.rint(y.to_numpy() * count_scale).clip(min=0.0)
    x = _calendar_design(pd.DatetimeIndex(y.index))
    mean_z = max(float(z.mean()), 1e-3)
    var_z = max(float(z.var()), mean_z + 1e-3)

    alpha0 = _lag_init(params, "alphas_init", "alpha_init", p)
    beta0 = _lag_init(params, "betas_init", "beta_init", q)
    initial_persistence = float(alpha0.sum() + beta0.sum())
    if initial_persistence >= 0.995:
        shrink = 0.9 / initial_persistence
        alpha0 = alpha0 * shrink
        beta0 = beta0 * shrink
        initial_persistence = float(alpha0.sum() + beta0.sum())
    omega0 = max((1.0 - initial_persistence) * mean_z, 1e-3)
    # Method-of-moments init for φ: Var = mean + mean^2 / φ  →  φ = mean^2 / (Var − mean)
    phi0 = float(np.clip(mean_z * mean_z / max(var_z - mean_z, 1e-3), 1.0, 1e5))
    theta0 = np.r_[omega0, alpha0, beta0, np.zeros(x.shape[1]), np.log(phi0)]

    bounds = [(1e-6, max(mean_z * 5.0, 10.0))]
    bounds += [(0.0, 0.98)] * (p + q)
    bounds += [(-1.5, 1.5)] * x.shape[1]
    bounds += [(-2.0, 12.0)]  # log φ → φ ∈ [~0.14, ~163000]

    result = minimize(
        _nb_loss,
        theta0,
        args=(z, x, p, q),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(params.get("max_iter", 300))},
    )
    theta_phi = result.x if result.success else theta0
    theta = theta_phi[:-1]
    phi = float(np.exp(np.clip(theta_phi[-1], -8.0, 16.0)))
    lambdas = _ingarch_filter(z, x, theta, p, q)
    return {
        "kind": "nb_ingarch",
        "theta": theta,
        "phi": phi,
        "p": p,
        "q": q,
        "last_lambda": float(lambdas[-1]),
        "last_z": float(z[-1]),
        "z_history": z[-max(p, 1):].astype(float).tolist(),
        "lambda_history": lambdas[-max(q, 1):].astype(float).tolist(),
        "last_date": pd.Timestamp(y.index[-1]),
        "count_scale": count_scale,
        "success": bool(result.success),
        "loss": float(result.fun) if np.isfinite(result.fun) else None,
    }


def predict_nb_ingarch(
    model: dict[str, Any],
    forecast_date: pd.Timestamp,
    horizons: list[int],
) -> dict[int, float]:
    """Multi-horizon point forecast (E[z_t] = λ_t).

    Uses the same forward-cascade simplification as the Poisson INGARCH:
    propagate λ_t forward by feeding the predicted intensity in as the next
    "observation". A closed-form multi-step is left for a follow-up.
    """
    if model.get("kind") == "constant":
        return {h: float(model["value"]) for h in horizons}

    theta = np.asarray(model["theta"], dtype=float)
    p = int(model.get("p", 1))
    q = int(model.get("q", 1))
    omega, alphas, betas, gamma = _split_theta(theta, p, q)
    last_date = pd.Timestamp(model["last_date"])
    end_date = forecast_date + pd.Timedelta(days=max(horizons))
    future_index = pd.date_range(last_date + pd.Timedelta(days=1), end_date, freq="D")
    x_future = _calendar_design(future_index)

    z_history = [float(v) for v in model.get("z_history", [model["last_z"]])]
    lambda_history = [float(v) for v in model.get("lambda_history", [model["last_lambda"]])]
    forecast_counts: dict[pd.Timestamp, float] = {}
    for date, x_row in zip(future_index, x_future, strict=True):
        observed_part = sum(float(alphas[i]) * z_history[-i - 1] for i in range(p))
        intensity_part = sum(float(betas[j]) * lambda_history[-j - 1] for j in range(q))
        base = omega + observed_part + intensity_part
        multiplier = float(np.exp(np.clip(x_row @ gamma, -3.0, 3.0)))
        lambda_t = max(base * multiplier, 1e-6)
        forecast_counts[pd.Timestamp(date)] = lambda_t
        z_history.append(lambda_t)
        lambda_history.append(lambda_t)

    scale = float(model["count_scale"])
    return {
        h: float(forecast_counts[forecast_date + pd.Timedelta(days=h)] / scale)
        for h in horizons
    }
