"""Negative Binomial INGARCH(p,q) with adaptive-quantile point estimation.

Clean reading version of the model in `nb_ingarch.py`, reduced to the parts
that empirically improve forecast performance.

What's here:
    - NB INGARCH(p, q) on continuous mortality target via integer pseudo-counts
    - Calendar covariates in the mean intensity (Monday / weekend / winter / doy sin-cos)
    - Adaptive quantile prediction: instead of E[y] = λ_t, return a quantile q ∈
      [q_calm, q_volatile] of NB(λ_t, φ) chosen by a spike-risk score at the forecast origin
    - Spike-risk score combines:
          y-volatility ratio (recent σ_y vs training-window median σ_y)
          z-scores of EDA-validated leading covariates (NHS 111 calls etc.)
      Combined as max(·) so any single signal can elevate the quantile.

What's NOT here (vs `nb_ingarch.py`):
    - Operational covariates in γ·X (the mean-shift experiments that regressed)
    - Heteroscedastic dispersion (φ_t depending on covariates — modest, not the win)
    - Lagged-covariate rolling means (only relevant to γ·X)

Thresholds (`q_calm = 0.5`, `q_volatile = 0.7`, `spike_score_threshold = 1.0`)
are mechanistic, not val-tuned.

LEAKAGE SAFETY: identical to the original NB INGARCH. Fit on target data up to
`forecast_date − REPORTING_LAG_DAYS`. Calendar features are deterministic. The
spike-risk covariate values at the forecast origin use a rolling-window mean
ending at midday(O) — i.e. exactly the most recent leakage-safe observation —
and the standardisation stats are computed on the training window only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from submission.src.data.schema import REPORTING_LAG_DAYS, TARGET_COL


# ──────────────────────────────────────────────────────────────────────────────
# Design matrix (calendar only)
# ──────────────────────────────────────────────────────────────────────────────

def _calendar_design(index: pd.DatetimeIndex) -> np.ndarray:
    """Monday, weekend, winter (DJF), doy sin/cos. Same as the Reboredo paper
    spec with COVID removed and a smooth annual harmonic added."""
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


# ──────────────────────────────────────────────────────────────────────────────
# INGARCH(p, q) recursion and NB likelihood
# ──────────────────────────────────────────────────────────────────────────────

def _ingarch_orders(params: dict[str, Any]) -> tuple[int, int]:
    p = int(params.get("p", 1))
    q = int(params.get("q", 1))
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
    """Forward NB-INGARCH(p,q) filter:  λ_t = (ω + Σ α_i z_{t-i} + Σ β_j λ_{t-j}) · exp(γ·x_t)."""
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
    """Negative-binomial INGARCH NLL (constant φ).

    Layout of theta_phi: [omega, alphas[p], betas[q], gamma[n_x], log_phi].

    Drops the lgamma(z+1) term that is constant w.r.t. parameters:
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
    return nll if np.isfinite(nll) else 1e12


# ──────────────────────────────────────────────────────────────────────────────
# Spike-risk gate (covariate-driven; used only for quantile selection)
# ──────────────────────────────────────────────────────────────────────────────

def _spike_risk_covariate_means(
    daily_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    cov_names: list[str],
    window: int,
) -> np.ndarray:
    """Rolling-`window`-day mean of each cov ending at each date in `dates`.

    Each row uses only data ≤ its own date — leakage-safe at midday(date)."""
    out = np.zeros((len(dates), len(cov_names)), dtype=float)
    for j, name in enumerate(cov_names):
        if name not in daily_df.columns:
            continue
        s = daily_df[name].astype(float).ffill()
        rolled = s.rolling(window=window, min_periods=1).mean()
        vals = rolled.reindex(pd.DatetimeIndex(dates)).ffill().bfill()
        out[:, j] = vals.to_numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Fit and predict
# ──────────────────────────────────────────────────────────────────────────────

def fit_nb_ingarch_aq(
    daily_df: pd.DataFrame,
    params: dict[str, Any],
    forecast_date: pd.Timestamp,
    horizons: list[int],
) -> dict[str, Any]:
    """Fit NB INGARCH(p,q) on target history available by `forecast_date − 3`.

    Also captures the diagnostics needed by the adaptive-quantile gate at
    predict time: y-volatility at origin, training-window typical y-volatility,
    and z-scores of the EDA-selected spike-risk covariates at origin.
    """
    del horizons  # NB INGARCH is fit per origin, not per horizon
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
        alpha0 *= shrink
        beta0 *= shrink
        initial_persistence = float(alpha0.sum() + beta0.sum())
    omega0 = max((1.0 - initial_persistence) * mean_z, 1e-3)
    phi0 = float(np.clip(mean_z * mean_z / max(var_z - mean_z, 1e-3), 1.0, 1e5))

    theta0 = np.r_[omega0, alpha0, beta0, np.zeros(x.shape[1]), np.log(phi0)]
    bounds = [(1e-6, max(mean_z * 5.0, 10.0))]
    bounds += [(0.0, 0.98)] * (p + q)
    bounds += [(-1.5, 1.5)] * x.shape[1]
    bounds += [(-2.0, 12.0)]  # log φ

    result = minimize(
        _nb_loss, theta0, args=(z, x, p, q),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": int(params.get("max_iter", 300))},
    )
    theta_phi = result.x if result.success else theta0
    theta = theta_phi[:-1]
    phi = float(np.exp(np.clip(theta_phi[-1], -8.0, 16.0)))
    lambdas = _ingarch_filter(z, x, theta, p, q)

    # ── Adaptive-quantile diagnostics ────────────────────────────────────────
    vol_window = int(params.get("volatility_window", 14))
    y_vals = y.to_numpy()
    if len(y_vals) >= vol_window:
        context_std = float(np.std(y_vals[-vol_window:]))
        rolling_std = pd.Series(y_vals).rolling(vol_window, min_periods=vol_window).std()
        training_std_median = float(rolling_std.median())
    else:
        context_std = float(np.std(y_vals)) if len(y_vals) > 1 else 0.0
        training_std_median = max(context_std, 1e-6)

    # Covariate z-scores: standardise origin rolling mean against training rolling-mean distribution
    spike_risk_covariates: list[str] = list(params.get("spike_risk_covariates", []) or [])
    spike_risk_window = int(params.get("spike_risk_window", 7))
    spike_cov_z: list[float] = []
    if spike_risk_covariates:
        train_op = _spike_risk_covariate_means(
            daily_df, pd.DatetimeIndex(y.index), spike_risk_covariates, spike_risk_window
        )
        train_mu = train_op.mean(axis=0)
        train_sd = np.where(train_op.std(axis=0) > 1e-8, train_op.std(axis=0), 1.0)
        origin_op = _spike_risk_covariate_means(
            daily_df, pd.DatetimeIndex([forecast_date]), spike_risk_covariates, spike_risk_window
        )[0]
        spike_cov_z = ((origin_op - train_mu) / train_sd).tolist()

    return {
        "kind": "nb_ingarch_aq",
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
        # Adaptive-quantile gate state — defaults preserve mean-of-NB behaviour
        "predict_quantile": params.get("predict_quantile", None),
        "adaptive_quantile": bool(params.get("adaptive_quantile", False)),
        "quantile_calm": float(params.get("quantile_calm", 0.5)),
        "quantile_volatile": float(params.get("quantile_volatile", 0.7)),
        "spike_score_threshold": float(params.get("spike_score_threshold", 1.0)),
        "context_std": context_std,
        "training_std_median": training_std_median,
        "spike_cov_z": spike_cov_z,
        "success": bool(result.success),
        "loss": float(result.fun) if np.isfinite(result.fun) else None,
    }


def predict_nb_ingarch_aq(
    model: dict[str, Any],
    forecast_date: pd.Timestamp,
    horizons: list[int],
) -> dict[int, float]:
    """Multi-horizon point forecast.

    Default: returns E[z_t] = λ_t / count_scale (same as standard NB INGARCH mean prediction).

    When `adaptive_quantile = True` (or `predict_quantile = q`) was passed in fit params,
    instead returns a quantile of NB(mean=λ_t, dispersion=φ) at each horizon. The quantile
    is chosen by the spike-risk gate.
    """
    if model.get("kind") == "constant":
        return {h: float(model["value"]) for h in horizons}

    theta = np.asarray(model["theta"], dtype=float)
    p = int(model.get("p", 1))
    q_ord = int(model.get("q", 1))
    omega, alphas, betas, gamma = _split_theta(theta, p, q_ord)
    last_date = pd.Timestamp(model["last_date"])
    end_date = forecast_date + pd.Timedelta(days=max(horizons))
    future_index = pd.date_range(last_date + pd.Timedelta(days=1), end_date, freq="D")
    x_future = _calendar_design(future_index)

    z_history = [float(v) for v in model.get("z_history", [model["last_z"]])]
    lambda_history = [float(v) for v in model.get("lambda_history", [model["last_lambda"]])]
    forecast_counts: dict[pd.Timestamp, float] = {}
    for date, x_row in zip(future_index, x_future, strict=True):
        observed_part = sum(float(alphas[i]) * z_history[-i - 1] for i in range(p))
        intensity_part = sum(float(betas[j]) * lambda_history[-j - 1] for j in range(q_ord))
        base = omega + observed_part + intensity_part
        multiplier = float(np.exp(np.clip(x_row @ gamma, -3.0, 3.0)))
        lambda_t = max(base * multiplier, 1e-6)
        forecast_counts[pd.Timestamp(date)] = lambda_t
        z_history.append(lambda_t)
        lambda_history.append(lambda_t)

    scale = float(model["count_scale"])
    phi_val = float(model.get("phi", 1.0))

    # ── Quantile selection ──────────────────────────────────────────────────
    adaptive_q = bool(model.get("adaptive_quantile", False))
    predict_q = model.get("predict_quantile")
    if adaptive_q:
        ctx_std = float(model.get("context_std", 0.0))
        ref_std = max(float(model.get("training_std_median", 1.0)), 1e-6)
        y_signal = max(0.0, ctx_std / ref_std - 1.0)
        cov_zs = model.get("spike_cov_z", []) or []
        cov_signal = max([0.0] + [float(z) for z in cov_zs])
        spike_score = max(y_signal, cov_signal)
        threshold = float(model.get("spike_score_threshold", 1.0))
        q_calm = float(model.get("quantile_calm", 0.5))
        q_vol = float(model.get("quantile_volatile", 0.7))
        q_eff: float | None = q_calm + (q_vol - q_calm) * min(spike_score / max(threshold, 1e-6), 1.0)
    elif predict_q is not None:
        q_eff = float(predict_q)
    else:
        q_eff = None

    if q_eff is None:
        # Default: mean prediction
        return {
            h: float(forecast_counts[forecast_date + pd.Timedelta(days=h)] / scale)
            for h in horizons
        }

    # Quantile prediction via NB(mean = λ, dispersion = φ): scipy nbinom with n=φ, p=φ/(λ+φ).
    from scipy.stats import nbinom
    out: dict[int, float] = {}
    for h in horizons:
        lam = float(forecast_counts[forecast_date + pd.Timedelta(days=h)])
        p_param = phi_val / (lam + phi_val)
        pred_scaled = float(nbinom.ppf(q_eff, phi_val, p_param))
        out[h] = max(0.0, pred_scaled / scale)
    return out
