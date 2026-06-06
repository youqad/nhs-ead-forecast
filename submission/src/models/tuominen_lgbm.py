"""Tuominen et al.-style LightGBM adaptation.

The reference ED occupancy model uses Darts' LightGBMModel with a 7-day
autoregressive window, multivariable past covariates, known future covariates,
and a 24-step output chunk. This module keeps the same modeling idea but adapts
it to the contest's daily, 10-horizon, reporting-lag setting without adding
Darts as a runtime dependency.
"""
from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from submission.src.data.features import append_horizon_calendar
from submission.src.data.schema import DUMMY_VALUE, REPORTING_LAG_DAYS, SEED, TARGET_COL


TUOMINEN_HISTORY_DAYS = 7
CONTROL_PARAMS = {
    "history_days",
    "max_covariates",
    "covariate_keywords",
    "early_stopping_days",
}
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_]")


# Name-only mapping from the paper's multivariable ED occupancy inputs to the
# contest data. This avoids fitting a selector on future covariate values.
TUOMINEN_COVARIATE_KEYWORDS = [
    "patients in a&e",
    "occupancy",
    "bed",
    "opel",
    "dta",
    "decision to admit",
    "breach",
    "ambulance",
    "handover",
    "queue",
    "arrival",
    "discharge",
    "triage",
    "assessment",
    "resuscitation",
    "resus",
    "majors",
    "minors",
    "capacity",
    "call stack",
    "waiting calls",
    "response",
]


def _sanitize_name(s: str) -> str:
    out = _UNSAFE_CHARS.sub("_", s)
    if not out or not (out[0].isalpha() or out[0] == "_"):
        out = "f_" + out
    return out


def _lightgbm_params(params: dict) -> tuple[dict, int]:
    model_params = {k: v for k, v in params.items() if k not in CONTROL_PARAMS}
    n_estimators = int(model_params.pop("n_estimators", model_params.pop("num_boost_round", 100)))
    model_params = {
        "objective": "quantile",
        "alpha": 0.5,
        "metric": "quantile",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample_for_bin": 200_000,
        "subsample": 1.0,
        **model_params,
        "seed": SEED,
        "verbose": -1,
    }
    return model_params, n_estimators


def select_tuominen_covariates(
    daily_df: pd.DataFrame,
    *,
    max_covariates: int | None = None,
    keywords: list[str] | None = None,
) -> list[str]:
    """Select ED pressure/capacity covariates using Tuominen-style groups.

    The source paper groups multivariable inputs into historical operational
    covariates such as beds, traffic, occupancy, and demand proxies. The NHS
    contest has different column names, so the adaptation uses conservative
    name matching and a deterministic keyword-priority order.
    """
    kws = [k.lower() for k in (keywords or TUOMINEN_COVARIATE_KEYWORDS)]
    cols = [
        c
        for c in daily_df.columns
        if c != TARGET_COL and any(k in c.lower() for k in kws)
    ]
    cols = sorted(
        cols,
        key=lambda c: (
            min(i for i, k in enumerate(kws) if k in c.lower()),
            c.lower(),
        ),
    )
    if not cols:
        cols = [c for c in daily_df.columns if c != TARGET_COL]
    if max_covariates is not None and max_covariates > 0:
        cols = cols[: int(max_covariates)]
    return cols


def precompute_tuominen_origin_panel(
    daily_df: pd.DataFrame,
    covariate_cols: list[str],
    history_days: int = TUOMINEN_HISTORY_DAYS,
) -> pd.DataFrame:
    """Return origin-level autoregressive features.

    Row ``O`` contains:
      - target lags ending at ``O - REPORTING_LAG_DAYS``;
      - covariate lags ending at ``O``;
      - origin calendar features.

    Horizon calendar features are appended separately because they depend on
    ``O + h``.
    """
    daily = daily_df.mask(daily_df == DUMMY_VALUE).replace([np.inf, -np.inf], np.nan)
    cols: dict[str, pd.Series] = {}

    y = daily[TARGET_COL].ffill()
    for lag in range(history_days):
        cols[f"tu_y_lag_{lag + 1}"] = y.shift(REPORTING_LAG_DAYS + lag)

    for col in covariate_cols:
        if col not in daily.columns:
            continue
        s = daily[col]
        s_ffilled = s.ffill()
        for lag in range(history_days):
            cols[f"{col}__tu_cov_lag_{lag}"] = s_ffilled.shift(lag)
        cols[f"{col}__tu_missing_now"] = s.isna().astype(float)

    panel = pd.concat(cols, axis=1) if cols else pd.DataFrame(index=daily.index)
    panel["tu_origin_dow"] = panel.index.dayofweek.astype(float)
    panel["tu_origin_month"] = panel.index.month.astype(float)
    panel["tu_origin_doy_sin"] = np.sin(2 * np.pi * panel.index.dayofyear / 365.25)
    panel["tu_origin_doy_cos"] = np.cos(2 * np.pi * panel.index.dayofyear / 365.25)
    return panel


def _records_for_rows(rows: pd.DataFrame, panel: pd.DataFrame) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for row in rows.itertuples(index=False):
        origin = pd.Timestamp(row.origin)
        base = panel.loc[origin].to_dict()
        feats = append_horizon_calendar(base, origin=origin, horizon=int(row.horizon))
        feats["tu_horizon"] = float(row.horizon)
        records.append(feats)
    return records


def _feature_matrix(rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(_records_for_rows(rows, panel)).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0).astype(float)
    X.columns = [_sanitize_name(c) for c in X.columns]
    return X


def fit_tuominen_lgbm_per_horizon(
    ds: pd.DataFrame,
    daily_df: pd.DataFrame,
    params: dict,
    horizons: list[int],
) -> dict:
    """Fit one Tuominen-style LightGBM booster per contest horizon."""
    history_days = int(params.get("history_days", TUOMINEN_HISTORY_DAYS))
    covariate_cols = select_tuominen_covariates(
        daily_df,
        max_covariates=params.get("max_covariates"),
        keywords=params.get("covariate_keywords"),
    )
    panel = precompute_tuominen_origin_panel(daily_df, covariate_cols, history_days)
    lgb_params, n_estimators = _lightgbm_params(params)
    early_stopping_days = int(params.get("early_stopping_days", 60))

    boosters: dict[int, lgb.Booster] = {}
    for h in horizons:
        sub = ds[ds["horizon"] == h].sort_values("origin")
        if sub.empty:
            continue

        use_validation = len(sub) > early_stopping_days + 5
        if use_validation:
            cutoff = sub["origin"].max() - pd.Timedelta(days=early_stopping_days)
            train_part = sub[sub["origin"] <= cutoff]
            val_part = sub[sub["origin"] > cutoff]
        else:
            train_part = sub
            val_part = sub.iloc[0:0]

        X_tr = _feature_matrix(train_part, panel)
        y_tr = train_part["y"].values
        dtrain = lgb.Dataset(X_tr, label=y_tr)

        valid_sets = None
        callbacks = None
        if len(val_part) > 5:
            X_va = _feature_matrix(val_part, panel).reindex(columns=X_tr.columns, fill_value=0.0)
            dvalid = lgb.Dataset(X_va, label=val_part["y"].values, reference=dtrain)
            valid_sets = [dvalid]
            callbacks = [lgb.early_stopping(20, verbose=False)]

        boosters[h] = lgb.train(
            params=lgb_params,
            train_set=dtrain,
            valid_sets=valid_sets,
            num_boost_round=n_estimators,
            callbacks=callbacks,
        )

    return {
        "boosters": boosters,
        "covariate_cols": covariate_cols,
        "history_days": history_days,
        "panel": panel,
    }


def predict_tuominen_lgbm_per_horizon(
    model: dict,
    daily_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    horizons: list[int],
) -> dict[int, float]:
    """Predict all horizons for one forecast origin."""
    panel = model.get("panel")
    if panel is None:
        panel = precompute_tuominen_origin_panel(
            daily_df,
            model["covariate_cols"],
            history_days=model.get("history_days", TUOMINEN_HISTORY_DAYS),
        )

    out: dict[int, float] = {}
    boosters: dict[int, lgb.Booster] = model["boosters"]
    for h in horizons:
        booster = boosters.get(h)
        if booster is None:
            out[h] = float("nan")
            continue
        row = pd.DataFrame(
            _records_for_rows(
                pd.DataFrame([{"origin": forecast_date, "horizon": h}]),
                panel,
            )
        ).replace([np.inf, -np.inf], np.nan)
        row = row.fillna(0.0).astype(float)
        row.columns = [_sanitize_name(c) for c in row.columns]
        row = row.reindex(columns=booster.feature_name(), fill_value=0.0)
        out[h] = float(booster.predict(row)[0])
    return out
