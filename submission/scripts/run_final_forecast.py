"""Run the full 173-origin assessment matrix.

Reuses run_forecast.py logic per origin and writes the contest deliverable:
    submission/pred_matrix.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from submission.src.data.loader import load_or_build_daily
from submission.src.data.schema import ASSESSMENT_ORIGINS, HORIZONS, TARGET_COL
from submission.src.evaluate import score_predictions
from submission.src.utils import get_logger, load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--weights", default=None, help="Optional override for short-horizon blend weights.")
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing per-origin forecast CSVs instead of recomputing them.",
    )
    p.add_argument(
        "--write-mse-summary",
        action="store_true",
        help="Write submission/mse_summary.csv only for local post-assessment diagnostics.",
    )
    args = p.parse_args()

    log = get_logger()
    cfg = load_config(args.config)

    forecast_dir = Path("submission/artifacts/forecasts")
    forecast_dir.mkdir(parents=True, exist_ok=True)

    for origin in ASSESSMENT_ORIGINS:
        out = forecast_dir / f"{origin.date()}.csv"
        if out.exists() and args.reuse_existing:
            continue
        cmd = [
            sys.executable, "-m", "submission.run_forecast",
            "--config", args.config,
            "--origin", str(origin.date()),
        ]
        if args.weights:
            cmd.extend(["--weights", args.weights])
        log.info(f"running {origin.date()}")
        subprocess.run(cmd, check=True)

    rows = []
    for origin in ASSESSMENT_ORIGINS:
        row = pd.read_csv(forecast_dir / f"{origin.date()}.csv", parse_dates=["origin"])
        rows.append(row)
    pred_matrix = pd.concat(rows, ignore_index=True)
    pred_matrix.insert(0, "forecast_id", range(1, len(pred_matrix) + 1))
    deliverable_cols = ["forecast_id"] + [f"day_{h}" for h in HORIZONS]
    pred_matrix[deliverable_cols].to_csv("submission/pred_matrix.csv", index=False)
    log.info(
        "wrote submission/pred_matrix.csv with "
        f"{len(pred_matrix)} rows in official 11-column format"
    )

    daily = load_or_build_daily(cfg["data"]["raw_zip"], cfg["data"]["cache_dir"])
    targets = daily[TARGET_COL]
    have_targets = any(
        (origin + pd.Timedelta(days=h)) in targets.index
        and pd.notna(targets.loc[origin + pd.Timedelta(days=h)])
        for origin in ASSESSMENT_ORIGINS for h in HORIZONS
    )
    if have_targets and args.write_mse_summary:
        mse = score_predictions(pred_matrix, targets)
        mse.insert(0, "forecast_id", range(1, len(mse) + 1))
        mse[["forecast_id", "mse_1_5", "mse_6_10"]].to_csv(
            "submission/mse_summary.csv", index=False
        )
        log.info("wrote submission/mse_summary.csv (targets available)")
    elif have_targets:
        log.info("targets are available; not writing mse_summary.csv without --write-mse-summary")
    else:
        log.info("targets not yet available; mse_summary.csv is a post-assessment diagnostic")


if __name__ == "__main__":
    main()
