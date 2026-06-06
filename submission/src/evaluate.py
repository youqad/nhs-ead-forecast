"""Rolling-origin scoring that mirrors the contest's MSE aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY_COLS = [f"day_{i}" for i in range(1, 11)]
PRIZE_HORIZONS = {"1to5": [1, 2, 3, 4, 5], "6to10": [6, 7, 8, 9, 10]}


def score_predictions(pred_matrix: pd.DataFrame, actuals: pd.Series) -> pd.DataFrame:
    """Score a (forecast_id, origin, day_1..day_10) matrix.

    Returns one row per origin with mse_1_5, mse_6_10.
    """
    rows = []
    for _, p in pred_matrix.iterrows():
        o = p["origin"]
        sq = []
        for h in range(1, 11):
            target_date = o + pd.Timedelta(days=h)
            if target_date in actuals.index and pd.notna(actuals.loc[target_date]):
                err = float(p[f"day_{h}"]) - float(actuals.loc[target_date])
                sq.append((h, err * err))
        sq_arr = dict(sq)
        mse_1_5 = float(np.mean([sq_arr[h] for h in [1, 2, 3, 4, 5] if h in sq_arr]))
        mse_6_10 = float(np.mean([sq_arr[h] for h in [6, 7, 8, 9, 10] if h in sq_arr]))
        rows.append({"origin": o, "mse_1_5": mse_1_5, "mse_6_10": mse_6_10})
    return pd.DataFrame(rows)


def score_overlapping_windows(pred_matrix: pd.DataFrame, actuals: pd.Series, prize: str) -> dict:
    """Score each (period, horizon) cell independently - no de-dup by date.

    Returns dict with mse and n_cells.
    """
    horizons = PRIZE_HORIZONS[prize]
    sq = []
    for _, p in pred_matrix.iterrows():
        for h in horizons:
            if f"day_{h}" not in p:
                continue
            target_date = p["origin"] + pd.Timedelta(days=h)
            if target_date in actuals.index and pd.notna(actuals.loc[target_date]):
                err = float(p[f"day_{h}"]) - float(actuals.loc[target_date])
                sq.append(err * err)
    return {"mse": float(np.mean(sq)) if sq else float("nan"), "n_cells": len(sq)}


def aggregate_prize_mse(per_origin_mse: pd.DataFrame) -> dict:
    return {
        "mse_1to5_mean": float(per_origin_mse["mse_1_5"].mean()),
        "mse_6to10_mean": float(per_origin_mse["mse_6_10"].mean()),
        "n_origins": int(len(per_origin_mse)),
    }
