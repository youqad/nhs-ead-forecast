"""Pre-push deliverable checks. Exits non-zero on any failure."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from submission.src.data.schema import ASSESSMENT_ORIGINS, HORIZONS
from submission.src.utils import get_logger, load_config

EXPECTED_PRED_MATRIX_SHA256 = (
    "513dcaa45cb761c25b43c24c3835e9cb5212d21e68b57f11512367d06be8de3b"
)


def fail(msg: str) -> None:
    log = get_logger()
    log.error(f"VALIDATION FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="submission/config/default.yaml")
    args = p.parse_args()
    log = get_logger()
    cfg = load_config(args.config)

    sub = Path("submission")
    for required in [sub / "pred_matrix.csv", sub / "report.pdf", sub / "forecast.csv"]:
        if not required.exists():
            fail(f"missing required file: {required}")
    if (sub / "mse_summary.csv").exists():
        fail("submission/mse_summary.csv is a local diagnostic, not a contest deliverable")

    blend = cfg.get("blend", {})
    if blend.get("mode") != "chronos_nb_ingarch_aq_seasonal_split":
        fail(f"default blend mode is {blend.get('mode')!r}; expected chronos_nb_ingarch_aq_seasonal_split")
    models = cfg.get("models", {})
    for learner in ["chronos", "nb_ingarch_aq"]:
        if not models.get(learner, {}).get("enabled", False):
            fail(f"default config must enable {learner}")
    seasonal_cfg = models.get("seasonal_prior", {})
    if not seasonal_cfg.get("enabled", False):
        fail("default config must enable the seasonal residual prior")

    pred = pd.read_csv(sub / "pred_matrix.csv")
    expected_cols = ["forecast_id"] + [f"day_{h}" for h in HORIZONS]
    if list(pred.columns) != expected_cols:
        fail(f"pred_matrix columns are {list(pred.columns)}; expected {expected_cols}")
    if len(pred) != len(ASSESSMENT_ORIGINS):
        fail(f"pred_matrix has {len(pred)} rows; expected {len(ASSESSMENT_ORIGINS)}")
    expected_ids = list(range(1, len(ASSESSMENT_ORIGINS) + 1))
    if pred["forecast_id"].tolist() != expected_ids:
        fail("pred_matrix forecast_id must be sequential from 1 to 173")
    for h in HORIZONS:
        col = f"day_{h}"
        if pred[col].isna().any():
            fail(f"pred_matrix has NaNs in {col}")
        if not np.isfinite(pred[col]).all():
            fail(f"pred_matrix has non-finite values in {col}")
        if (pred[col] < 0).any():
            fail(f"pred_matrix has negative values in {col}")

    actual_sha = hashlib.sha256((sub / "pred_matrix.csv").read_bytes()).hexdigest()
    if actual_sha != EXPECTED_PRED_MATRIX_SHA256:
        fail(
            "pred_matrix.csv SHA256 mismatch: "
            f"{actual_sha}; expected {EXPECTED_PRED_MATRIX_SHA256}"
        )

    # forecast.csv is a byte-identical copy of pred_matrix.csv, shipped so the
    # aggregator collates this entry whether its target filename is pred_matrix.csv
    # (the NHS-EAD template name) or forecast.csv (the aggregator default). It must
    # never drift from pred_matrix.csv.
    if (sub / "forecast.csv").read_bytes() != (sub / "pred_matrix.csv").read_bytes():
        fail("forecast.csv must be byte-identical to pred_matrix.csv")

    rpath = sub / "report.pdf"
    if rpath.stat().st_size == 0:
        fail("report.pdf is empty")

    log.info("submission validation PASSED")


if __name__ == "__main__":
    main()
