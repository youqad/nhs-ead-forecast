"""Online imputation primitives - never use data past the per-origin cutoff."""
from __future__ import annotations

import pandas as pd


def online_ffill(series: pd.Series, cutoff: pd.Timestamp) -> pd.Series:
    truncated = series.loc[:cutoff]
    filled = truncated.ffill()
    full = pd.Series(index=series.index, dtype=series.dtype)
    full.loc[:cutoff] = filled
    return full


def online_rolling_median(series: pd.Series, window: int, cutoff: pd.Timestamp) -> pd.Series:
    truncated = series.loc[:cutoff]
    rolled = truncated.rolling(window=window, min_periods=1).median()
    full = pd.Series(index=series.index, dtype=series.dtype)
    full.loc[:cutoff] = rolled
    return full
