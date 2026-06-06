"""Date + DTA-sum time-series baselines.

These learners intentionally use only:
  - the target's own history,
  - calendar/date structure supplied by the model,
  - one exogenous covariate: total "No. of DTAs" across locations.

Zeros in the location-level DTA columns are treated as missing before summing;
training rows with missing total DTA are dropped.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

from submission.src.data.schema import DUMMY_VALUE, REPORTING_LAG_DAYS, TARGET_COL


DTA_PREFIX = "No. of DTAs - "
DEFAULT_PMDARIMA_VARIANTS = [
    {"name": "pmdarima_auto", "mode": "auto", "seasonal": False},
]


def select_dta_columns(daily_df: pd.DataFrame) -> list[str]:
    """Return exact total-DTA columns, excluding derived '> 8hrs' variants."""
    return sorted(c for c in daily_df.columns if c.startswith(DTA_PREFIX))


def dta_sum_series(daily_df: pd.DataFrame, zero_is_missing: bool = True) -> pd.Series:
    """Sum location-level DTA counts after masking dummy and zero-missing values."""
    cols = select_dta_columns(daily_df)
    if not cols:
        raise ValueError(f"No DTA columns found with prefix {DTA_PREFIX!r}")
    dta = daily_df[cols].mask(daily_df[cols] == DUMMY_VALUE).astype(float)
    if zero_is_missing:
        dta = dta.mask(dta == 0.0)
    out = dta.sum(axis=1, min_count=1)
    out.name = "dta_sum"
    return out


def build_ts_training_frame(
    daily_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    *,
    zero_is_missing: bool = True,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Series]:
    """Return train frame with columns ds, y, dta_sum.

    Target history ends at ``forecast_date - REPORTING_LAG_DAYS``. DTA values
    are aligned to their own dates for training; rows with missing/zero DTA are
    dropped as requested.
    """
    deadline = forecast_date - pd.Timedelta(days=REPORTING_LAG_DAYS)
    dta = dta_sum_series(daily_df, zero_is_missing=zero_is_missing)
    y = daily_df[TARGET_COL].mask(daily_df[TARGET_COL] == DUMMY_VALUE).astype(float)
    train = pd.DataFrame({
        "ds": daily_df.index,
        "y": y.values,
        "dta_sum": dta.values,
    }, index=daily_df.index)
    train = train.loc[:deadline].replace([np.inf, -np.inf], np.nan)
    train = train.dropna(subset=["y", "dta_sum"])
    return train.reset_index(drop=True), deadline, dta


def future_dta_frame(
    dta: pd.Series,
    *,
    deadline: pd.Timestamp,
    forecast_date: pd.Timestamp,
    n_steps: int,
) -> pd.DataFrame:
    """Build leakage-safe future exogenous values for the reporting-lag bridge.

    For dates after the target deadline but up to the forecast date, real DTA
    values are available and may be used. For dates after the forecast date,
    the latest valid DTA value known at the forecast date is carried forward.
    """
    future_dates = pd.date_range(deadline + pd.Timedelta(days=1), periods=n_steps, freq="D")
    known = dta.loc[:forecast_date].dropna()
    if known.empty:
        fill = float(dta.dropna().iloc[-1]) if dta.notna().any() else 0.0
    else:
        fill = float(known.iloc[-1])

    vals: list[float] = []
    for dt in future_dates:
        if dt <= forecast_date and dt in dta.index and pd.notna(dta.loc[dt]):
            vals.append(float(dta.loc[dt]))
        else:
            vals.append(fill)
    return pd.DataFrame({"ds": future_dates, "dta_sum": vals})


def _fallback_forecast(train: pd.DataFrame, horizons: Iterable[int]) -> dict[int, float]:
    y = train["y"].dropna()
    if y.empty:
        value = 0.0
    else:
        value = float(y.tail(min(7, len(y))).mean())
    return {int(h): value for h in horizons}


def _horizon_values(pred_values: np.ndarray, horizons: list[int]) -> dict[int, float]:
    return {
        h: float(pred_values[REPORTING_LAG_DAYS + h - 1])
        for h in horizons
    }


def _auto_arima_kwargs(variant: dict) -> dict:
    keys = {
        "d", "D", "seasonal", "m", "start_p", "start_q", "max_p", "max_q",
        "start_P", "start_Q", "max_P", "max_Q", "information_criterion",
        "stepwise", "maxiter",
    }
    kwargs = {k: variant[k] for k in keys if k in variant and variant[k] is not None}
    kwargs.setdefault("seasonal", False)
    kwargs.setdefault("m", 7 if kwargs["seasonal"] else 1)
    kwargs.setdefault("start_p", 0)
    kwargs.setdefault("start_q", 0)
    kwargs.setdefault("max_p", 3)
    kwargs.setdefault("max_q", 3)
    kwargs.setdefault("stepwise", True)
    kwargs.setdefault("maxiter", 50)
    kwargs["suppress_warnings"] = True
    kwargs["error_action"] = "ignore"
    kwargs["trace"] = False
    return kwargs


def predict_pmdarima_variant(
    train: pd.DataFrame,
    future: pd.DataFrame,
    variant: dict,
    horizons: list[int],
) -> dict[int, float]:
    if len(train) < int(variant.get("min_train_rows", 30)):
        return _fallback_forecast(train, horizons)

    import pmdarima as pm

    y = train["y"].to_numpy(dtype=float)
    x = train[["dta_sum"]].to_numpy(dtype=float)
    future_x = future[["dta_sum"]].to_numpy(dtype=float)

    mode = variant.get("mode", "auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode == "auto":
            model = pm.auto_arima(y, X=x, **_auto_arima_kwargs(variant))
        elif mode == "fixed":
            order = tuple(int(v) for v in variant.get("order", [1, 1, 1]))
            seasonal_order = tuple(int(v) for v in variant.get("seasonal_order", [0, 0, 0, 0]))
            model = pm.ARIMA(
                order=order,
                seasonal_order=seasonal_order,
                suppress_warnings=True,
            ).fit(y, X=x)
        else:
            raise ValueError(f"Unknown pmdarima variant mode: {mode}")
        pred = model.predict(n_periods=len(future), X=future_x)
    return _horizon_values(np.asarray(pred, dtype=float), horizons)


def predict_ts_baseline_learners(
    daily_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    config: dict,
    horizons: list[int],
) -> dict[str, dict[int, float]]:
    """Return learner-name -> per-horizon predictions."""
    zero_is_missing = bool(config.get("dta_zero_is_missing", True))
    train, deadline, dta = build_ts_training_frame(
        daily_df,
        forecast_date,
        zero_is_missing=zero_is_missing,
    )
    n_steps = REPORTING_LAG_DAYS + max(horizons)
    future = future_dta_frame(
        dta,
        deadline=deadline,
        forecast_date=forecast_date,
        n_steps=n_steps,
    )

    out: dict[str, dict[int, float]] = {}
    pmdarima_cfg = config.get("pmdarima", {})
    if pmdarima_cfg.get("enabled", False):
        variants = pmdarima_cfg.get("variants", DEFAULT_PMDARIMA_VARIANTS)
        for variant in variants:
            name = variant.get("name")
            if not name:
                raise ValueError("Every pmdarima variant must define a name")
            try:
                out[name] = predict_pmdarima_variant(train, future, variant, horizons)
            except Exception:
                out[name] = _fallback_forecast(train, horizons)

    return out
