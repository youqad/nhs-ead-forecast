"""Chronos zero-shot univariate forecaster (Amazon)."""
from __future__ import annotations

import pandas as pd

from chronos import Chronos2Pipeline

from submission.src.data.schema import REPORTING_LAG_DAYS, TARGET_COL


_pipeline_cache: dict[tuple[str, str], Chronos2Pipeline] = {}


def load_chronos2_pipeline(params: dict) -> Chronos2Pipeline:
    """Load the Chronos pipeline from the model name and device."""
    model_name = params.get("model_name", "amazon/chronos-t5-small")
    device = params.get("device", "cpu")
    key = (model_name, device)
    if key not in _pipeline_cache:
        _pipeline_cache[key] = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device,
        )
    return _pipeline_cache[key]


def predict_chronos(
    daily_df: pd.DataFrame,
    params: dict,
    forecast_date: pd.Timestamp,
    pipeline: Chronos2Pipeline,
    horizons: list[int],
) -> dict[int, float]:
    """Zero-shot Chronos prediction for all horizons.

    Context ends at deadline = forecast_date - REPORTING_LAG_DAYS (last date
    with an observed target). Chronos is asked for REPORTING_LAG_DAYS +
    max(horizons) steps; the first REPORTING_LAG_DAYS steps bridge deadline →
    forecast_date and are discarded.

    Step mapping (0-indexed into the prediction array):
        step REPORTING_LAG_DAYS + h - 1  →  forecast_date + h  (horizon h)
    """

    context_length = params.get("context_length", None)
    deadline = forecast_date - pd.Timedelta(days=REPORTING_LAG_DAYS)

    index_target = daily_df.loc[:deadline, [TARGET_COL]].ffill().dropna()
    if context_length is not None:
        index_target = index_target.iloc[-int(context_length):]

    index_target = index_target.reset_index(names="date")
    index_target["item_id"] = "ead"

    n_steps = REPORTING_LAG_DAYS + max(horizons)

    forecast = pipeline.predict_df(index_target, prediction_length=n_steps, \
                                   timestamp_column="date", id_column="item_id", \
                                   target=TARGET_COL, quantile_levels=[0.5])


    predictions = forecast["predictions"]
    return {h: float(predictions[REPORTING_LAG_DAYS + h - 1]) for h in horizons}
