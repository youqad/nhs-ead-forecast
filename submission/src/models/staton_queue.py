"""Staton-style PyMC queueing baseline for the contest API.

The model is self-contained: it builds its own operational covariate frame from
``daily_df``, fits a posterior per forecast origin, forecasts future covariates
with Chronos or last-value carry-forward, and returns horizon point forecasts.

The learner owns its two-clock filtering: target values after
``origin - REPORTING_LAG_DAYS`` are hidden, and covariates after ``origin`` are
absent before any model frame is built.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from submission.src.data.schema import DUMMY_VALUE, REPORTING_LAG_DAYS, SEED, TARGET_COL


OCCUPANCY_NAME = "G&A Bed occupancy"
CAPACITY_NAME = "G&A beds, core stock open"
SIRONA_NAME = "Sirona BNSSG % Staffing Reduction"
ADJUST_NAMES = [
    "Patients in A&E",
    "New Arrivals in Last Hour",
    "Ambulance Queue",
    SIRONA_NAME,
    "Number of empty beds on assessment units 8am (today)",
]
MODEL_COLS = [OCCUPANCY_NAME, CAPACITY_NAME, *ADJUST_NAMES]
AR_LAG = REPORTING_LAG_DAYS + 1
WARMUP_STEPS = AR_LAG - 1
HARM_EPS = 0.05

_CHRONOS_FAILED = False


def _staton_origin_frame(daily_df: pd.DataFrame, forecast_date: pd.Timestamp) -> pd.DataFrame:
    """Learner-local as-of view.

    The contest permits covariates through the forecast origin but target only
    through ``origin - REPORTING_LAG_DAYS``.
    """
    origin = pd.Timestamp(forecast_date)
    target_cutoff = origin - pd.Timedelta(days=REPORTING_LAG_DAYS)
    observed = daily_df.loc[:origin].copy()
    observed = observed.replace(DUMMY_VALUE, np.nan)
    observed.loc[observed.index > target_cutoff, TARGET_COL] = np.nan
    return observed


def _metric_columns(daily_df: pd.DataFrame, metric_name: str) -> list[str]:
    prefix = metric_name.lower()
    return sorted(
        c
        for c in daily_df.columns
        if c.lower() == prefix or c.lower().startswith(prefix + " -")
    )


def _aggregate_metric(daily_df: pd.DataFrame, metric_name: str) -> pd.Series:
    cols = _metric_columns(daily_df, metric_name)
    if not cols:
        raise ValueError(f"No columns found for Staton metric {metric_name!r}")
    values = daily_df[cols].replace([np.inf, -np.inf], np.nan).astype(float)
    out = values.mean(axis=1, skipna=True)
    out.name = metric_name
    return out


def build_staton_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Return target plus aggregate operational covariates used by the queue model."""
    frame = pd.DataFrame(index=daily_df.index)
    frame[TARGET_COL] = daily_df[TARGET_COL].astype(float)
    for name in MODEL_COLS:
        frame[name] = _aggregate_metric(daily_df, name)
    feature_cols = [c for c in frame.columns if c != TARGET_COL]
    frame[feature_cols] = frame[feature_cols].ffill().bfill()
    return frame


def calendar_features(dates: pd.DatetimeIndex) -> np.ndarray:
    doy = np.asarray(dates.dayofyear, dtype=np.float32)
    dow = np.asarray(dates.dayofweek, dtype=np.float32)
    return np.column_stack(
        [
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
        ]
    ).astype(np.float32)


def _fallback(frame: pd.DataFrame, train_end: pd.Timestamp, horizons: list[int]) -> dict[int, float]:
    y = frame.loc[:train_end, TARGET_COL].dropna()
    value = float(y.tail(min(7, len(y))).mean()) if len(y) else 0.0
    return {h: value for h in horizons}


def _prepare_training(
    frame: pd.DataFrame,
    train_end: pd.Timestamp,
    train_days: int,
    min_train_days: int,
) -> pd.DataFrame:
    train = frame.loc[:train_end, [TARGET_COL, *MODEL_COLS]].copy()
    train = train.replace([np.inf, -np.inf], np.nan)
    train[MODEL_COLS] = train[MODEL_COLS].ffill().bfill()
    train = train.dropna(subset=[TARGET_COL, *MODEL_COLS])
    if train_days > 0:
        train = train.tail(train_days)
    if len(train) < min_train_days:
        return train.iloc[0:0]
    return train


def fit_staton_queue(
    daily_df: pd.DataFrame,
    params: dict[str, Any],
    forecast_date: pd.Timestamp,
    horizons: list[int],
) -> dict[str, Any]:
    """Fit one PyMC posterior at ``forecast_date`` using only admissible data."""
    del horizons
    observed = _staton_origin_frame(daily_df, forecast_date)
    frame = build_staton_frame(observed)
    train_end = pd.Timestamp(forecast_date) - pd.Timedelta(days=REPORTING_LAG_DAYS)
    train = _prepare_training(
        frame,
        train_end,
        train_days=int(params.get("train_days", 365)),
        min_train_days=int(params.get("min_train_days", 180)),
    )
    if train.empty:
        return {
            "kind": "constant",
            "frame": frame,
            "train_end": train_end,
            "params": params,
            "value": _fallback(frame, train_end, [1])[1],
        }

    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    occupancy = train[OCCUPANCY_NAME].to_numpy(dtype=np.float32)
    capacity = train[CAPACITY_NAME].to_numpy(dtype=np.float32)
    base_util_np = (occupancy / (capacity + 1e-8)).astype(np.float32)
    adjust = train[ADJUST_NAMES].to_numpy(dtype=np.float32)
    adjust_mean = adjust.mean(axis=0)
    adjust_std = adjust.std(axis=0) + 1e-8
    adjust_np = ((adjust - adjust_mean) / adjust_std).astype(np.float32)
    cal_np = calendar_features(pd.DatetimeIndex(train.index))
    log_y_np = np.log(train[TARGET_COL].to_numpy(dtype=np.float32).clip(min=0.0) + HARM_EPS)
    log_y_lag_np = np.concatenate([np.repeat(log_y_np[0], AR_LAG), log_y_np[:-AR_LAG]])

    coords = {
        "adjust": ADJUST_NAMES,
        "cal": ["sin_year", "cos_year", "sin_week", "cos_week"],
    }
    with pm.Model(coords=coords):
        w_adjust = pm.Normal("w_adjust", mu=0.0, sigma=1.0, dims="adjust")
        w_cal = pm.Normal("w_cal", mu=0.0, sigma=0.3, dims="cal")
        beta = pm.HalfNormal("beta", sigma=float(params.get("beta_prior_scale", 3.0)))
        rho = pm.Uniform("rho", lower=0.0, upper=0.99)
        alpha = pm.HalfNormal("alpha", sigma=0.3)
        baseline = pm.Normal("baseline", mu=0.0, sigma=1.0)
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        base_util = pt.clip(base_util_np, 0.01, 0.99)
        adj = pt.dot(adjust_np, w_adjust)
        util = pm.math.sigmoid(pt.log(base_util / (1.0 - base_util)) + adj)
        util = pt.clip(util, 0.01, 0.99)
        pressure = pt.power(1.0 / (1.0 - util), beta)
        mu = baseline + pt.dot(cal_np, w_cal) + rho * log_y_lag_np + alpha * pressure
        pm.Normal("log_harm", mu=mu, sigma=sigma, observed=log_y_np)

        trace = pm.sample(
            draws=int(params.get("draws", 1000)),
            tune=int(params.get("tune", 500)),
            chains=int(params.get("chains", 2)),
            cores=int(params.get("cores", min(2, int(params.get("chains", 2))))),
            target_accept=float(params.get("target_accept", 0.9)),
            random_seed=int(params.get("seed", SEED)) + pd.Timestamp(forecast_date).dayofyear,
            progressbar=bool(params.get("progressbar", False)),
            compute_convergence_checks=bool(params.get("compute_convergence_checks", False)),
        )

    posterior: dict[str, np.ndarray] = {}
    for name in ["w_adjust", "w_cal", "beta", "rho", "alpha", "baseline", "sigma"]:
        da = az.extract(trace, var_names=[name])
        values = np.asarray(da.values)
        posterior[name] = values.T if values.ndim > 1 else values

    return {
        "kind": "staton_queue",
        "frame": frame,
        "train_end": train_end,
        "forecast_date": pd.Timestamp(forecast_date),
        "params": params,
        "posterior": posterior,
        "adjust_mean": adjust_mean.astype(float),
        "adjust_std": adjust_std.astype(float),
    }


def _last_value_paths(history: np.ndarray, horizon: int, n_samples: int) -> np.ndarray:
    clean = pd.Series(history).replace([np.inf, -np.inf], np.nan).ffill().bfill().to_numpy(dtype=float)
    value = float(clean[-1]) if len(clean) else 0.0
    return np.full((n_samples, horizon), value, dtype=float)


def _chronos_paths(
    history: np.ndarray,
    params: dict[str, Any],
    horizon: int,
    n_samples: int,
) -> np.ndarray:
    global _CHRONOS_FAILED
    if params.get("covariate_method", "chronos") != "chronos" or _CHRONOS_FAILED:
        return _last_value_paths(history, horizon, n_samples)

    try:
        import torch

        from submission.src.models.chronos import load_chronos2_pipeline

        chronos = load_chronos2_pipeline(
            {
                "model_name": params.get("chronos_model_name", "amazon/chronos-2"),
                "device": params.get("chronos_device", "cpu"),
            }
        )
        clean = pd.Series(history).replace([np.inf, -np.inf], np.nan).ffill().bfill().to_numpy(dtype=float)
        if len(clean) == 0:
            return np.zeros((n_samples, horizon), dtype=float)
        context_days = int(params.get("chronos_context_days", 250))
        clean = clean[-context_days:]
        tensor = torch.tensor(clean, dtype=torch.float32).reshape(1, 1, -1)
        q_levels = np.linspace(0.02, 0.98, n_samples).tolist()
        quantiles, _ = chronos.predict_quantiles(
            tensor,
            prediction_length=horizon,
            quantile_levels=q_levels,
        )
        arr = quantiles[0][0].detach().cpu().numpy().T
        return np.asarray(arr, dtype=float)
    except Exception:
        _CHRONOS_FAILED = True
        return _last_value_paths(history, horizon, n_samples)


def _covariate_forecasts(
    frame: pd.DataFrame,
    forecast_date: pd.Timestamp,
    params: dict[str, Any],
    horizon: int,
    n_samples: int,
) -> dict[str, np.ndarray]:
    hist = frame.loc[:forecast_date, MODEL_COLS].copy()
    hist[MODEL_COLS] = hist[MODEL_COLS].ffill().bfill()
    return {
        col: _chronos_paths(hist[col].to_numpy(dtype=float), params, horizon, n_samples)
        for col in MODEL_COLS
    }


def _extend_paths(warmup: np.ndarray, forecast_paths: np.ndarray, cov_ix: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.tile(np.asarray(warmup, dtype=float)[None, :], (len(cov_ix), 1)),
            forecast_paths[cov_ix],
        ],
        axis=1,
    )


def _rollout(
    model: dict[str, Any],
    cov_paths: dict[str, np.ndarray],
    horizons: list[int],
) -> np.ndarray:
    posterior = model["posterior"]
    params = model["params"]
    frame = model["frame"]
    forecast_date = pd.Timestamp(model["forecast_date"])
    train_end = pd.Timestamp(model["train_end"])
    first_target_date = forecast_date + pd.Timedelta(days=1)
    max_h = max(horizons)
    n_rollout = int(params.get("n_rollout", 500))
    rng = np.random.default_rng(int(params.get("rollout_seed", SEED)) + forecast_date.dayofyear)
    n_post = len(posterior["baseline"])
    post_ix = rng.choice(n_post, size=n_rollout, replace=True)
    cov_ix = rng.integers(0, cov_paths[OCCUPANCY_NAME].shape[0], size=n_rollout)

    w_adjust = posterior["w_adjust"][post_ix]
    w_cal = posterior["w_cal"][post_ix]
    beta = posterior["beta"][post_ix]
    rho = posterior["rho"][post_ix]
    alpha = posterior["alpha"][post_ix]
    baseline = posterior["baseline"][post_ix]
    sigma = posterior["sigma"][post_ix]

    warmup_dates = pd.date_range(
        first_target_date - pd.Timedelta(days=WARMUP_STEPS),
        periods=WARMUP_STEPS,
        freq="D",
    )
    target_dates = pd.date_range(first_target_date, periods=max_h, freq="D")
    warmup = frame.loc[warmup_dates, MODEL_COLS].ffill().bfill()
    ext = {
        col: _extend_paths(warmup[col].to_numpy(dtype=float), cov_paths[col], cov_ix)
        for col in MODEL_COLS
    }
    cal_ext = np.concatenate([calendar_features(warmup_dates), calendar_features(target_dates)], axis=0)

    seed_dates = pd.date_range(first_target_date - pd.Timedelta(days=(AR_LAG + WARMUP_STEPS)), periods=AR_LAG)
    y_seed = frame.loc[seed_dates, TARGET_COL].ffill().bfill().to_numpy(dtype=float)
    log_y_seed = np.log(np.clip(y_seed, 0.0, None) + HARM_EPS)
    clean_mu = np.zeros((n_rollout, WARMUP_STEPS + max_h), dtype=float)

    for i in range(WARMUP_STEPS + max_h):
        if i < AR_LAG:
            log_y_prev = np.full(n_rollout, float(log_y_seed[i]))
        else:
            log_y_prev = clean_mu[:, i - AR_LAG]

        base_util = np.clip(ext[OCCUPANCY_NAME][:, i] / (ext[CAPACITY_NAME][:, i] + 1e-8), 0.01, 0.99)
        adjust = np.stack([ext[col][:, i] for col in ADJUST_NAMES], axis=1)
        adjust = (adjust - model["adjust_mean"]) / model["adjust_std"]
        adj = (adjust * w_adjust).sum(axis=1)
        util = 1.0 / (1.0 + np.exp(-(np.log(base_util / (1.0 - base_util)) + adj)))
        util = np.clip(util, 0.01, 0.99)
        pressure = np.power(1.0 / (1.0 - util), beta)
        clean_mu[:, i] = baseline + (cal_ext[i] * w_cal).sum(axis=1) + rho * log_y_prev + alpha * pressure

    pred_log = clean_mu + rng.normal(size=clean_mu.shape) * sigma[:, None]
    hist_y = frame.loc[:train_end, TARGET_COL].dropna()
    hist_max = float(hist_y.max()) if len(hist_y) else 1.0
    cap = max(
        float(params.get("prediction_cap_min", 3.0)),
        hist_max * float(params.get("prediction_cap_multiplier", 2.0)),
    )
    log_floor = np.log(HARM_EPS)
    log_cap = np.log(cap + HARM_EPS)
    pred_log = np.nan_to_num(pred_log, nan=log_cap, posinf=log_cap, neginf=log_floor)
    pred_log = np.clip(pred_log, log_floor, log_cap)
    return np.clip(np.exp(pred_log[:, WARMUP_STEPS:]) - HARM_EPS, 0.0, cap)


def predict_staton_queue(
    model: dict[str, Any],
    forecast_date: pd.Timestamp,
    horizons: list[int],
) -> dict[int, float]:
    if model.get("kind") == "constant":
        return _fallback(model["frame"], pd.Timestamp(model["train_end"]), horizons)

    params = model["params"]
    max_h = max(horizons)
    n_cov_samples = int(params.get("n_cov_samples", 50))
    cov_paths = _covariate_forecasts(
        model["frame"],
        pd.Timestamp(forecast_date),
        params,
        horizon=max_h,
        n_samples=n_cov_samples,
    )
    samples = _rollout(model, cov_paths, horizons)
    means = np.nanmean(samples, axis=0)
    return {h: float(max(means[h - 1], 0.0)) for h in horizons}


__all__ = [
    "build_staton_frame",
    "fit_staton_queue",
    "predict_staton_queue",
]
