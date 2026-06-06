"""Leakage-safe INGARCH count baselines.

The target is continuous, so both models fit a count-time-series likelihood to
``round(target * count_scale)`` and scale forecasts back down.  The only
exogenous inputs are deterministic calendar features, which are known for all
future horizons.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from submission.src.data.schema import DUMMY_VALUE, REPORTING_LAG_DAYS, TARGET_COL


def _calendar_design(index: pd.DatetimeIndex) -> np.ndarray:
    """Small INGARCH-X design matrix: Monday, weekend, winter, yearly phase."""
    dow = index.dayofweek.to_numpy(dtype=float)
    month = index.month.to_numpy(dtype=float)
    doy = index.dayofyear.to_numpy(dtype=float)
    return np.column_stack(
        [
            (dow == 0).astype(float),
            (dow >= 5).astype(float),
            np.isin(month, [12.0, 1.0, 2.0]).astype(float),
            np.sin(2.0 * np.pi * doy / 365.25),
            np.cos(2.0 * np.pi * doy / 365.25),
        ]
    )


def _orders(params: dict[str, Any]) -> tuple[int, int]:
    p = int(params.get("p", params.get("ar_order", 1)))
    q = int(params.get("q", params.get("ma_order", 1)))
    if p < 0 or q < 0 or p + q == 0:
        raise ValueError(f"INGARCH orders must satisfy p>=0, q>=0, p+q>0; got p={p}, q={q}")
    return p, q


def _lag_init(params: dict[str, Any], explicit_key: str, total_key: str, order: int) -> np.ndarray:
    if order == 0:
        return np.array([], dtype=float)
    explicit = params.get(explicit_key)
    if explicit is not None:
        values = np.asarray(explicit, dtype=float)
        if len(values) != order:
            raise ValueError(f"{explicit_key} must have length {order}, got {len(values)}")
        return values
    total = float(params.get(total_key, 0.0))
    return np.full(order, total / order, dtype=float)


def _split_theta(theta: np.ndarray, p: int, q: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    omega = float(theta[0])
    alphas = np.asarray(theta[1 : 1 + p], dtype=float)
    betas = np.asarray(theta[1 + p : 1 + p + q], dtype=float)
    gamma = np.asarray(theta[1 + p + q :], dtype=float)
    return omega, alphas, betas, gamma


def _filter_counts(
    z: np.ndarray,
    x: np.ndarray,
    theta: np.ndarray,
    p: int,
    q: int,
    *,
    seed_first_with_mean: bool,
) -> np.ndarray:
    """Forward INGARCH filter for conditional means."""
    omega, alphas, betas, gamma = _split_theta(theta, p, q)
    lambdas = np.empty_like(z, dtype=float)
    pre_sample = max(float(np.nanmean(z)), 1e-3)
    start = 0
    if seed_first_with_mean:
        lambdas[0] = pre_sample
        start = 1
    for t in range(start, len(z)):
        observed = sum(
            float(alphas[i]) * (float(z[t - i - 1]) if t - i - 1 >= 0 else pre_sample)
            for i in range(p)
        )
        intensity = sum(
            float(betas[j]) * (float(lambdas[t - j - 1]) if t - j - 1 >= 0 else pre_sample)
            for j in range(q)
        )
        base = omega + observed + intensity
        calendar = float(np.exp(np.clip(x[t] @ gamma, -3.0, 3.0)))
        lambdas[t] = max(base * calendar, 1e-6)
    return lambdas


def _initial_theta(z: np.ndarray, x: np.ndarray, params: dict[str, Any], p: int, q: int) -> np.ndarray:
    mean_z = max(float(np.nanmean(z)), 1e-3)
    alpha0 = _lag_init(params, "alphas_init", "alpha_init", p)
    beta0 = _lag_init(params, "betas_init", "beta_init", q)
    persistence = float(alpha0.sum() + beta0.sum())
    if persistence >= 0.995:
        shrink = 0.9 / persistence
        alpha0 = alpha0 * shrink
        beta0 = beta0 * shrink
        persistence = float(alpha0.sum() + beta0.sum())
    omega0 = max((1.0 - persistence) * mean_z, 1e-3)
    return np.r_[omega0, alpha0, beta0, np.zeros(x.shape[1])]


def _bounds(z: np.ndarray, p: int, q: int) -> list[tuple[float, float]]:
    mean_z = max(float(np.nanmean(z)), 1e-3)
    out = [(1e-6, max(mean_z * 5.0, 10.0))]
    out += [(0.0, 0.98)] * (p + q)
    out += [(-1.5, 1.5)] * _calendar_design(pd.date_range("2024-01-01", periods=1)).shape[1]
    return out


def _target_counts(
    daily_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    params: dict[str, Any],
) -> tuple[pd.Series, np.ndarray]:
    deadline = forecast_date - pd.Timedelta(days=REPORTING_LAG_DAYS)
    y = daily_df[TARGET_COL].mask(daily_df[TARGET_COL] == DUMMY_VALUE).loc[:deadline]
    y = y.dropna().astype(float).clip(lower=0.0)
    min_train_days = int(params.get("min_train_days", 180))
    if len(y) < min_train_days:
        return y, np.array([], dtype=float)
    count_scale = float(params.get("count_scale", 100.0))
    z = np.rint(y.to_numpy(dtype=float) * count_scale).clip(min=0.0)
    return y, z


def _poisson_loss(theta: np.ndarray, z: np.ndarray, x: np.ndarray, p: int, q: int) -> float:
    omega, alphas, betas, _ = _split_theta(theta, p, q)
    persistence = float(alphas.sum() + betas.sum())
    if omega <= 0.0 or (alphas < 0.0).any() or (betas < 0.0).any() or persistence >= 0.995:
        return 1e12 + 1e10 * max(persistence - 0.995, 0.0)
    lambdas = _filter_counts(z, x, theta, p, q, seed_first_with_mean=False)
    return float(np.sum(lambdas - z * np.log(lambdas + 1e-12)))


def _nb_loss(theta_phi: np.ndarray, z: np.ndarray, x: np.ndarray, p: int, q: int) -> float:
    theta = theta_phi[:-1]
    log_phi = float(theta_phi[-1])
    omega, alphas, betas, _ = _split_theta(theta, p, q)
    persistence = float(alphas.sum() + betas.sum())
    if omega <= 0.0 or (alphas < 0.0).any() or (betas < 0.0).any() or persistence >= 0.995:
        return 1e12 + 1e10 * max(persistence - 0.995, 0.0)
    phi = float(np.exp(np.clip(log_phi, -8.0, 16.0)))
    lambdas = _filter_counts(z, x, theta, p, q, seed_first_with_mean=True)
    lam_plus_phi = lambdas + phi
    log_lik = (
        gammaln(z + phi)
        - gammaln(phi)
        + phi * np.log(phi / lam_plus_phi)
        + z * (np.log(lambdas) - np.log(lam_plus_phi))
    )
    nll = -float(np.sum(log_lik))
    return nll if np.isfinite(nll) else 1e12


def fit_ingarch(
    daily_df: pd.DataFrame,
    params: dict[str, Any],
    forecast_date: pd.Timestamp,
    horizons: list[int],
) -> dict[str, Any]:
    """Fit Poisson INGARCH-X on target history available by ``forecast_date - 3``."""
    del horizons
    y, z = _target_counts(daily_df, forecast_date, params)
    if len(z) == 0:
        return {"kind": "constant", "value": float(y.mean()) if len(y) else 0.0}
    p, q = _orders(params)
    x = _calendar_design(pd.DatetimeIndex(y.index))
    theta0 = _initial_theta(z, x, params, p, q)
    result = minimize(
        _poisson_loss,
        theta0,
        args=(z, x, p, q),
        method="L-BFGS-B",
        bounds=_bounds(z, p, q),
        options={"maxiter": int(params.get("max_iter", 250))},
    )
    theta = result.x if result.success else theta0
    lambdas = _filter_counts(z, x, theta, p, q, seed_first_with_mean=False)
    return _pack_model("ingarch", y, z, lambdas, theta, p, q, params, result)


def fit_nb_ingarch(
    daily_df: pd.DataFrame,
    params: dict[str, Any],
    forecast_date: pd.Timestamp,
    horizons: list[int],
) -> dict[str, Any]:
    """Fit negative-binomial INGARCH-X on target history available by ``forecast_date - 3``."""
    del horizons
    y, z = _target_counts(daily_df, forecast_date, params)
    if len(z) == 0:
        return {"kind": "constant", "value": float(y.mean()) if len(y) else 0.0}
    p, q = _orders(params)
    x = _calendar_design(pd.DatetimeIndex(y.index))
    theta0 = _initial_theta(z, x, params, p, q)
    mean_z = max(float(np.nanmean(z)), 1e-3)
    var_z = max(float(np.nanvar(z)), mean_z + 1e-3)
    phi0 = float(np.clip(mean_z * mean_z / max(var_z - mean_z, 1e-3), 1.0, 1e5))
    start = np.r_[theta0, np.log(phi0)]
    bounds = _bounds(z, p, q) + [(-2.0, 12.0)]
    result = minimize(
        _nb_loss,
        start,
        args=(z, x, p, q),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(params.get("max_iter", 300))},
    )
    theta_phi = result.x if result.success else start
    theta = theta_phi[:-1]
    phi = float(np.exp(np.clip(theta_phi[-1], -8.0, 16.0)))
    lambdas = _filter_counts(z, x, theta, p, q, seed_first_with_mean=True)
    model = _pack_model("nb_ingarch", y, z, lambdas, theta, p, q, params, result)
    model["phi"] = phi
    return model


def _pack_model(
    kind: str,
    y: pd.Series,
    z: np.ndarray,
    lambdas: np.ndarray,
    theta: np.ndarray,
    p: int,
    q: int,
    params: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "theta": np.asarray(theta, dtype=float),
        "p": int(p),
        "q": int(q),
        "z_history": z[-max(p, 1) :].astype(float).tolist(),
        "lambda_history": lambdas[-max(q, 1) :].astype(float).tolist(),
        "last_z": float(z[-1]),
        "last_lambda": float(lambdas[-1]),
        "last_date": pd.Timestamp(y.index[-1]),
        "count_scale": float(params.get("count_scale", 100.0)),
        "success": bool(result.success),
        "loss": float(result.fun) if np.isfinite(result.fun) else None,
    }


def _predict(model: dict[str, Any], forecast_date: pd.Timestamp, horizons: list[int]) -> dict[int, float]:
    if model.get("kind") == "constant":
        return {h: float(model["value"]) for h in horizons}
    theta = np.asarray(model["theta"], dtype=float)
    p = int(model["p"])
    q = int(model["q"])
    omega, alphas, betas, gamma = _split_theta(theta, p, q)
    last_date = pd.Timestamp(model["last_date"])
    end_date = forecast_date + pd.Timedelta(days=max(horizons))
    future_index = pd.date_range(last_date + pd.Timedelta(days=1), end_date, freq="D")
    x_future = _calendar_design(future_index)

    z_history = [float(v) for v in model.get("z_history", [model["last_z"]])]
    lambda_history = [float(v) for v in model.get("lambda_history", [model["last_lambda"]])]
    forecast_counts: dict[pd.Timestamp, float] = {}
    for date, x_row in zip(future_index, x_future, strict=True):
        observed = sum(float(alphas[i]) * z_history[-i - 1] for i in range(p))
        intensity = sum(float(betas[j]) * lambda_history[-j - 1] for j in range(q))
        base = omega + observed + intensity
        calendar = float(np.exp(np.clip(x_row @ gamma, -3.0, 3.0)))
        lambda_t = max(base * calendar, 1e-6)
        forecast_counts[pd.Timestamp(date)] = lambda_t
        z_history.append(lambda_t)
        lambda_history.append(lambda_t)

    scale = float(model["count_scale"])
    return {
        h: float(forecast_counts[forecast_date + pd.Timedelta(days=h)] / scale)
        for h in horizons
    }


def predict_ingarch(model: dict[str, Any], forecast_date: pd.Timestamp, horizons: list[int]) -> dict[int, float]:
    return _predict(model, forecast_date, horizons)


def predict_nb_ingarch(model: dict[str, Any], forecast_date: pd.Timestamp, horizons: list[int]) -> dict[int, float]:
    return _predict(model, forecast_date, horizons)
