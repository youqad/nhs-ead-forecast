"""as_of_day dataset constructor - enforces the contest's two-clock leakage rules.

For each candidate (origin o, horizon h) admissible at forecast date D:
    - Covariate features use timestamps <= midday(o)
    - Y-derived features use dates <= o - 3
    - Label y = Y[o + h], admitted only when o + h <= D - 3
"""
from __future__ import annotations

import pandas as pd

from submission.src.data.schema import (
    DUMMY_VALUE,
    REPORTING_LAG_DAYS,
    TARGET_COL,
)
from submission.src.data.features import (
    append_horizon_calendar,
    build_features_at_origin,
    features_from_panel,
    precompute_feature_panel,
)
from submission.src.features.regime import (
    covariate_state_features,
    target_state_features,
)


def build_as_of_dataset(
    daily_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    candidate_origins: pd.DatetimeIndex,
    horizons: list[int],
    panel: pd.DataFrame | None = None,
    cov_state_model: dict | None = None,
    y_state_model: dict | None = None,
) -> pd.DataFrame:
    """Construct (o, h) rows admissible at forecast_date D.

    `panel`: optional precomputed feature panel from
    `precompute_feature_panel(daily_df)`. If None, computed locally.

    `cov_state_model`, `y_state_model`: optional fitted Markov state models
    (see submission.src.features.regime). If provided, their per-origin /
    per-(origin,horizon) features are injected into every row.

    Returns columns: origin, horizon, y, <features...>
    """
    if panel is None:
        panel = precompute_feature_panel(daily_df)
    deadline = forecast_date - pd.Timedelta(days=REPORTING_LAG_DAYS)
    rows = []
    for o in candidate_origins:
        if o not in panel.index:
            continue
        feats = features_from_panel(panel, origin=o)
        if cov_state_model is not None:
            feats.update(covariate_state_features(daily_df, o, cov_state_model, horizons))
        for h in horizons:
            label_date = o + pd.Timedelta(days=h)
            if label_date > deadline:
                continue
            if label_date not in daily_df.index:
                continue
            y = daily_df.loc[label_date, TARGET_COL]
            if pd.isna(y) or y == DUMMY_VALUE:
                continue
            row_feats = append_horizon_calendar(feats, origin=o, horizon=h)
            if y_state_model is not None:
                row_feats.update(target_state_features(daily_df, o, y_state_model, [h]))
            row = {"origin": o, "horizon": h, "y": float(y), **row_feats}
            rows.append(row)

    ds = pd.DataFrame(rows)
    _assert_invariants(ds, forecast_date, daily_df)
    return ds


def _assert_invariants(ds: pd.DataFrame, D: pd.Timestamp, daily_df: pd.DataFrame) -> None:
    if ds.empty:
        return
    deadline = D - pd.Timedelta(days=REPORTING_LAG_DAYS)
    label_dates = ds["origin"] + pd.to_timedelta(ds["horizon"], unit="D")
    assert (label_dates <= deadline).all(), "per-horizon embargo violated"
    feat_cols = [c for c in ds.columns if c not in {"origin", "horizon", "y"}]
    assert not (ds[feat_cols] == DUMMY_VALUE).any().any(), "DUMMY_VALUE leaked into features"
    assert ds["y"].notna().all(), "label NaNs admitted"
