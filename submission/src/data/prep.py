"""Pre-built training matrix for fast hyperparameter tuning.

Long-format DataFrame with one row per admissible (origin, horizon) pair,
labelled with a `block` column ('train' or 'val'). All features pre-joined.
Tuning loops just load this parquet and fit - no feature engineering required.
"""
from __future__ import annotations

import pandas as pd

from submission.src.data.as_of import build_as_of_dataset
from submission.src.data.schema import HORIZONS, REPORTING_LAG_DAYS, TRAIN_START


def build_tuning_matrix(
    daily_df: pd.DataFrame,
    panel: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build (origin, horizon, y, block, ...features) DataFrame for tuning.

    Train block: origins in [TRAIN_START, train_end - 4] with admissible labels
    Val block:   origins in (train_end, val_end - 4] with admissible labels

    The -4 offset enforces the per-horizon embargo o + h <= block_end - 3 at
    the maximum horizon h=1 (so any earlier horizon is automatically admissible).
    `build_as_of_dataset` rechecks per-row, so the offset only sets the
    candidate-origins ceiling.
    """
    train_candidates = pd.date_range(
        TRAIN_START, train_end - pd.Timedelta(days=REPORTING_LAG_DAYS + 1), freq="D"
    )
    val_candidates = pd.date_range(
        train_end + pd.Timedelta(days=1),
        val_end - pd.Timedelta(days=REPORTING_LAG_DAYS + 1),
        freq="D",
    )

    train_ds = build_as_of_dataset(
        daily_df=daily_df,
        forecast_date=train_end,
        candidate_origins=train_candidates,
        horizons=HORIZONS,
        panel=panel,
    )
    train_ds["block"] = "train"

    val_ds = build_as_of_dataset(
        daily_df=daily_df,
        forecast_date=val_end,
        candidate_origins=val_candidates,
        horizons=HORIZONS,
        panel=panel,
    )
    val_ds["block"] = "val"

    return pd.concat([train_ds, val_ds], ignore_index=True)
