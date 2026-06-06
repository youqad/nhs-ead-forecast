"""Two-clock feature engineering at a single forecast origin."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from submission.src.data.schema import DUMMY_VALUE, REPORTING_LAG_DAYS, TARGET_COL
from submission.src.data.imputation import online_ffill

Y_LAGS = [3, 4, 5, 7, 14, 28]
Y_ROLL = [7, 14, 28, 56]
Y_TREND = [7, 14, 28]

COV_LAGS = [1, 2, 3, 7, 14]
COV_ROLL = [3, 7, 14, 28]


def build_features_at_origin(daily_df: pd.DataFrame, origin: pd.Timestamp) -> dict:
    """Return a flat dict of feature_name -> value for this origin."""
    feats: dict[str, float] = {}
    y_cutoff = origin - pd.Timedelta(days=REPORTING_LAG_DAYS)
    cov_cutoff = origin

    # Defensive masking: the loader already converts DUMMY_VALUE to NaN, but
    # mask again here so downstream stats never inherit the sentinel even if
    # callers pass a frame straight from another source.
    daily_df = daily_df.mask(daily_df == DUMMY_VALUE)

    y = daily_df[TARGET_COL]
    y_known = online_ffill(y, cutoff=y_cutoff)
    for lag in Y_LAGS:
        date = origin - pd.Timedelta(days=lag)
        feats[f"y_lag_{lag}"] = float(y_known.loc[date]) if date in y_known.index else np.nan
    for w in Y_ROLL:
        window = y_known.loc[:y_cutoff].iloc[-w:]
        feats[f"y_roll_mean_{w}"] = float(window.mean()) if len(window) else np.nan
        feats[f"y_roll_std_{w}"] = float(window.std()) if len(window) > 1 else np.nan
        feats[f"y_roll_min_{w}"] = float(window.min()) if len(window) else np.nan
        feats[f"y_roll_max_{w}"] = float(window.max()) if len(window) else np.nan
    for w in Y_TREND:
        window = y_known.loc[:y_cutoff].iloc[-w:].dropna()
        if len(window) >= 2:
            x = np.arange(len(window), dtype=float)
            feats[f"y_slope_{w}"] = float(np.polyfit(x, window.values, 1)[0])
        else:
            feats[f"y_slope_{w}"] = np.nan

    cov_cols = [c for c in daily_df.columns if c != TARGET_COL]
    for col in cov_cols:
        s = daily_df[col]
        s_known = online_ffill(s, cutoff=cov_cutoff)
        latest = s_known.loc[:cov_cutoff].dropna()
        feats[f"{col}__latest"] = float(latest.iloc[-1]) if len(latest) else np.nan
        feats[f"{col}__is_missing"] = float(pd.isna(s.loc[cov_cutoff])) if cov_cutoff in s.index else 1.0
        for lag in COV_LAGS:
            date = cov_cutoff - pd.Timedelta(days=lag)
            feats[f"{col}__lag_{lag}"] = float(s_known.loc[date]) if date in s_known.index else np.nan
        for w in COV_ROLL:
            window = s_known.loc[:cov_cutoff].iloc[-w:]
            feats[f"{col}__roll_mean_{w}"] = float(window.mean()) if len(window) else np.nan
            feats[f"{col}__roll_std_{w}"] = float(window.std()) if len(window) > 1 else np.nan
        if len(s_known.loc[:cov_cutoff].dropna()) >= 8:
            recent7 = s_known.loc[:cov_cutoff].iloc[-7:].mean()
            prev7 = s_known.loc[:cov_cutoff].iloc[-14:-7].mean()
            feats[f"{col}__delta_7d"] = float(recent7 - prev7) if pd.notna(recent7) and pd.notna(prev7) else np.nan
        else:
            feats[f"{col}__delta_7d"] = np.nan

    feats["origin_dow"] = float(origin.dayofweek)
    feats["origin_month"] = float(origin.month)
    feats["origin_doy_sin"] = float(np.sin(2 * np.pi * origin.dayofyear / 365.25))
    feats["origin_doy_cos"] = float(np.cos(2 * np.pi * origin.dayofyear / 365.25))
    return feats


def append_horizon_calendar(feats: dict, origin: pd.Timestamp, horizon: int) -> dict:
    """Add per-horizon calendar features anchored to target_date = origin + h.

    Includes target-date winter / post-holiday recovery flags and distances.
    These are pure calendar constants, leakage-free; they encode prior
    structural knowledge that the NHS-EAD target has predictable seasonality
    around Christmas and the New Year that smooth Fourier features can only
    approximate. Interactions with `cov_state_prob_*` (already in `feats`)
    activate the prior only when the system is in a high-pressure regime.
    """
    import holidays
    target_date = origin + pd.Timedelta(days=horizon)
    uk = holidays.UnitedKingdom()
    out = dict(feats)
    out["h_dow"] = float(target_date.dayofweek)
    out["h_is_weekend"] = float(target_date.dayofweek >= 5)
    out["h_month"] = float(target_date.month)
    out["h_doy_sin"] = float(np.sin(2 * np.pi * target_date.dayofyear / 365.25))
    out["h_doy_cos"] = float(np.cos(2 * np.pi * target_date.dayofyear / 365.25))
    out["h_is_bank_holiday"] = float(target_date.date() in uk)

    # Winter / holiday recovery binary flags (target-date).
    m, d = target_date.month, target_date.day
    is_christmas_week = (m == 12 and 22 <= d <= 28)
    # "Christmas to new year" - includes Dec 25 through Jan 1 inclusive.
    is_xmas_to_ny = (m == 12 and d >= 25) or (m == 1 and d == 1)
    is_post_ny_week = (m == 1 and d <= 7)
    is_jan_first_half = (m == 1 and d <= 15)
    out["h_is_christmas_week"] = float(is_christmas_week)
    out["h_is_xmas_to_new_year"] = float(is_xmas_to_ny)
    out["h_is_post_new_year_week"] = float(is_post_ny_week)
    out["h_is_jan_first_half"] = float(is_jan_first_half)

    # Signed distances (in days) to the nearest Dec 25 / Jan 1.
    # Positive = future, negative = past. Use target_date's year as anchor.
    y_ = target_date.year
    candidates_dec25 = [pd.Timestamp(y_ - 1, 12, 25), pd.Timestamp(y_, 12, 25)]
    candidates_jan1 = [pd.Timestamp(y_, 1, 1), pd.Timestamp(y_ + 1, 1, 1)]
    def _signed_min(anchor: pd.Timestamp, opts: list[pd.Timestamp]) -> float:
        return float(min((d_ - anchor).days for d_ in opts if abs((d_ - anchor).days) <= 366))
    nearest_dec25 = min(candidates_dec25, key=lambda c: abs((target_date - c).days))
    nearest_jan1 = min(candidates_jan1, key=lambda c: abs((target_date - c).days))
    out["h_days_to_dec25"] = float((nearest_dec25 - target_date).days)
    out["h_days_to_jan1"] = float((nearest_jan1 - target_date).days)

    # Pressure-state × holiday interactions (only fire when both signals present).
    # Uses cov_state_prob_3 if 4-state model is enabled (highest-index state is
    # typically the highest-pressure cluster after KMeans seeded by SEED=2026).
    high_p = float(feats.get("cov_state_prob_3", 0.0))
    out["h_state_high_x_post_ny_week"] = high_p * out["h_is_post_new_year_week"]
    out["h_state_high_x_jan_first_half"] = high_p * out["h_is_jan_first_half"]

    return out


def _slope_apply(y_arr: np.ndarray) -> float:
    """OLS slope on the available (non-NaN) values, evenly-spaced x."""
    y_arr = y_arr[~np.isnan(y_arr)]
    n = len(y_arr)
    if n < 2:
        return float("nan")
    x = np.arange(n, dtype=float)
    x_centered = x - x.mean()
    return float(np.dot(x_centered, y_arr) / np.dot(x_centered, x_centered))


def precompute_feature_panel(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized counterpart to `build_features_at_origin`.

    Returns a panel indexed by date whose row at index O equals (modulo
    floating-point error) the dict produced by build_features_at_origin(daily_df, O).
    Leakage safe: every transformation depends only on data at indices <= O
    (ffill is monotonic; shift(k) reads only past rows; rolling looks backward).
    """
    daily = daily_df.mask(daily_df == DUMMY_VALUE)
    cols: dict[str, pd.Series] = {}

    # Y features anchored to (origin - REPORTING_LAG_DAYS).
    y_ffilled = daily[TARGET_COL].ffill()
    y_at_cutoff = y_ffilled.shift(REPORTING_LAG_DAYS)  # row O carries y_ffilled[O - 3]
    for lag in Y_LAGS:
        cols[f"y_lag_{lag}"] = y_ffilled.shift(lag)
    for w in Y_ROLL:
        win = y_at_cutoff.rolling(w, min_periods=1)
        cols[f"y_roll_mean_{w}"] = win.mean()
        cols[f"y_roll_std_{w}"] = y_at_cutoff.rolling(w, min_periods=2).std()
        cols[f"y_roll_min_{w}"] = win.min()
        cols[f"y_roll_max_{w}"] = win.max()
    for w in Y_TREND:
        cols[f"y_slope_{w}"] = y_at_cutoff.rolling(w, min_periods=2).apply(_slope_apply, raw=True)

    # Covariate features anchored to origin itself.
    cov_cols = [c for c in daily.columns if c != TARGET_COL]
    for col in cov_cols:
        s = daily[col]
        s_ffilled = s.ffill()
        cols[f"{col}__latest"] = s_ffilled
        cols[f"{col}__is_missing"] = s.isna().astype(float)
        for lag in COV_LAGS:
            cols[f"{col}__lag_{lag}"] = s_ffilled.shift(lag)
        for w in COV_ROLL:
            cols[f"{col}__roll_mean_{w}"] = s_ffilled.rolling(w, min_periods=1).mean()
            cols[f"{col}__roll_std_{w}"] = s_ffilled.rolling(w, min_periods=2).std()
        recent7 = s_ffilled.rolling(7, min_periods=1).mean()
        prev7 = recent7.shift(7)
        cols[f"{col}__delta_7d"] = recent7 - prev7

    panel = pd.concat(cols, axis=1)

    # Origin calendar (no lookahead - derived from the index itself).
    panel["origin_dow"] = panel.index.dayofweek.astype(float)
    panel["origin_month"] = panel.index.month.astype(float)
    panel["origin_doy_sin"] = np.sin(2 * np.pi * panel.index.dayofyear / 365.25)
    panel["origin_doy_cos"] = np.cos(2 * np.pi * panel.index.dayofyear / 365.25)

    return panel


def features_from_panel(panel: pd.DataFrame, origin: pd.Timestamp) -> dict:
    """Row lookup matching build_features_at_origin's dict output."""
    return panel.loc[origin].to_dict()


def load_or_build_panel(daily_df: pd.DataFrame, cache_dir: str | Path) -> pd.DataFrame:
    """Parquet-cached feature panel keyed by daily_df shape.

    Invalidation: cache is reused only when daily_df shape (n_rows × n_cols)
    matches the stored sidecar. Delete `feature_panel.parquet` to force rebuild.
    """
    cache_dir = Path(cache_dir)
    parquet_path = cache_dir / "feature_panel.parquet"
    meta_path = cache_dir / "feature_panel.meta.json"

    if parquet_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("daily_rows") == int(daily_df.shape[0])
            and meta.get("daily_cols") == int(daily_df.shape[1])
        ):
            return pd.read_parquet(parquet_path)

    panel = precompute_feature_panel(daily_df)
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(parquet_path)
    meta_path.write_text(json.dumps({
        "daily_rows": int(daily_df.shape[0]),
        "daily_cols": int(daily_df.shape[1]),
        "panel_rows": int(panel.shape[0]),
        "panel_cols": int(panel.shape[1]),
    }))
    return panel
