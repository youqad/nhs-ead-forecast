"""Official single-origin forecast entry point.

Usage:
    python -m submission.run_forecast --config submission/config/default.yaml --origin 2025-10-01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from submission.src.data.loader import load_or_build_daily
from submission.src.data.schema import HORIZONS
from submission.src.models.seasonal_prior import apply_seasonal_prior, load_prior
from submission.src.pipeline import forecast_learners
from submission.src.utils import config_hash, get_logger, load_config, set_seeds, timer


def _load_blend_weights(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _required_prediction(per_h: dict[int, dict[str, float]], horizon: int, learner: str) -> float:
    if horizon not in per_h or learner not in per_h[horizon]:
        available = sorted(per_h.get(horizon, {}).keys())
        raise RuntimeError(
            f"missing required learner {learner!r} for horizon {horizon}; "
            f"available learners: {available}"
        )
    value = float(per_h[horizon][learner])
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite prediction from {learner!r} at horizon {horizon}: {value}")
    return value


def _blend_block(
    per_h: dict[int, dict[str, float]],
    weights: dict,
    horizons: list[int],
    block: str,
) -> dict[int, float]:
    learners = weights[block]["learners"]
    w = np.array(weights[block]["weights"], dtype=float)
    if len(learners) != len(w):
        raise RuntimeError(f"blend block {block!r} has {len(learners)} learners but {len(w)} weights")

    out: dict[int, float] = {}
    for h in horizons:
        vec = np.array([_required_prediction(per_h, h, learner) for learner in learners], dtype=float)
        out[h] = float((w * vec).sum())
    return out


def _blend(per_h: dict[int, dict[str, float]], weights: dict) -> dict[int, float]:
    out: dict[int, float] = {}
    for h in HORIZONS:
        prize = "1to5" if h <= 5 else "6to10"
        w = np.array(weights[prize]["weights"])
        learners = weights[prize]["learners"]
        vec = np.array([per_h[h].get(learner, np.nan) for learner in learners])
        if np.isnan(vec).any():
            out[h] = float(np.nanmean(vec))
        else:
            out[h] = float((w * vec).sum())
    return out


def _fixed_chronos_nb_ingarch_aq_seasonal_split(
    *,
    per_h: dict[int, dict[str, float]],
    cfg: dict,
    weights_override: str | None,
    origin: pd.Timestamp,
    daily: pd.DataFrame,
    log,
) -> tuple[dict[int, float], dict]:
    """Final selected system: short Chronos/NB-AQ seasonal blend, long NB-AQ alone."""
    blend_cfg = cfg.get("blend", {})
    short_horizons = [int(h) for h in blend_cfg.get("short_horizons", [1, 2, 3, 4, 5])]
    long_horizons = [int(h) for h in blend_cfg.get("long_horizons", [6, 7, 8, 9, 10])]
    weights_path = (
        weights_override
        or blend_cfg.get("short_weights")
        or "submission/artifacts/cv_predictions/blend_chronos_plus_nb_ingarch_aq.json"
    )
    weights = _load_blend_weights(weights_path)

    preds = _blend_block(
        per_h=per_h,
        weights=weights,
        horizons=short_horizons,
        block=blend_cfg.get("short_weight_block", "1to5"),
    )

    long_learner = blend_cfg.get("long_learner", "nb_ingarch_aq")
    for h in long_horizons:
        preds[h] = _required_prediction(per_h, h, long_learner)

    seasonal_cfg = cfg.get("models", {}).get("seasonal_prior", {})
    prior_path = seasonal_cfg.get(
        "prior_path",
        "submission/artifacts/cv_predictions/seasonal_residual_chronos_plus_nb_ingarch_aq_TESTAB.json",
    )
    lam = float(seasonal_cfg.get("lambda", 1.0))
    if seasonal_cfg.get("enabled", False):
        prior = load_prior(prior_path)
        n_states = int(seasonal_cfg.get("n_states", prior["raw"].get("n_states", 4)))
        short_preds = {h: preds[h] for h in short_horizons}
        preds.update(apply_seasonal_prior(short_preds, origin, daily, prior, lambda_=lam, n_states=n_states))
        log.info(
            "applied seasonal residual prior to short horizons only "
            f"with lambda={lam} and prior_path={prior_path}"
        )

    missing = [h for h in HORIZONS if h not in preds]
    if missing:
        raise RuntimeError(f"fixed horizon split did not produce horizons: {missing}")

    return preds, {
        "mode": "chronos_nb_ingarch_aq_seasonal_split",
        "short_horizons": short_horizons,
        "short_weights_path": str(weights_path),
        "short_weights": weights.get(blend_cfg.get("short_weight_block", "1to5")),
        "long_horizons": long_horizons,
        "long_learner": long_learner,
        "seasonal_prior": {
            "enabled": bool(seasonal_cfg.get("enabled", False)),
            "lambda": lam,
            "prior_path": str(prior_path),
            "applied_horizons": short_horizons if seasonal_cfg.get("enabled", False) else [],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--origin", required=True, help="forecast origin date YYYY-MM-DD")
    p.add_argument(
        "--weights",
        default=None,
        help="Optional override for the short-horizon blend weights JSON.",
    )
    p.add_argument(
        "--out-dir",
        default="submission/artifacts/forecasts",
    )
    args = p.parse_args()

    log = get_logger()
    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    origin = pd.Timestamp(args.origin)

    daily = load_or_build_daily(cfg["data"]["raw_zip"], cfg["data"]["cache_dir"])
    timings: list[dict] = []
    with timer("total", sink=timings):
        per_h = forecast_learners(daily, origin, cfg)

    if cfg.get("blend", {}).get("mode") == "chronos_nb_ingarch_aq_seasonal_split":
        preds, blend_record = _fixed_chronos_nb_ingarch_aq_seasonal_split(
            per_h=per_h,
            cfg=cfg,
            weights_override=args.weights,
            origin=origin,
            daily=daily,
            log=log,
        )
    else:
        weights_path = args.weights or cfg.get("blend", {}).get(
            "weights", "submission/artifacts/cv_predictions/blend_weights.json"
        )
        weights = _load_blend_weights(weights_path)
        preds = _blend(per_h, weights)

        # Legacy seasonal-prior path: apply the configured prior to all horizons.
        seasonal_cfg = cfg.get("models", {}).get("seasonal_prior", {})
        if seasonal_cfg.get("enabled", False):
            prior_path = seasonal_cfg.get(
                "prior_path", "submission/artifacts/cv_predictions/seasonal_prior.json"
            )
            lam = float(seasonal_cfg.get("lambda", 0.7))
            prior = load_prior(prior_path)
            preds = apply_seasonal_prior(preds, origin, daily, prior, lambda_=lam)
            log.info(f"applied seasonal prior with lambda={lam}")
        blend_record = {"mode": "generic_blend", "weights_path": str(weights_path), "weights": weights}

    # Target is non-negative; predictions should be too.
    # In val + test our learners never went negative (min 0.264), so this is
    # defensive insurance against an unforeseen assessment-block input.
    preds = {h: max(v, 0.0) for h, v in preds.items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"origin": origin, **{f"day_{h}": preds[h] for h in HORIZONS}}])
    fpath = out_dir / f"{origin.date()}.csv"
    row.to_csv(fpath, index=False)

    log_path = Path("submission/artifacts/runtime_logs") / f"{origin.date()}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "origin": str(origin.date()),
            "config_hash": config_hash(cfg),
            "seed": cfg["seed"],
            "blend": blend_record,
            "learner_outputs": {h: sorted(per_h[h].keys()) for h in HORIZONS},
            "timings": timings,
        }, f, indent=2, default=str)
    log.info(f"wrote {fpath} and {log_path}")


if __name__ == "__main__":
    main()
