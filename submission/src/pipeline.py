"""Single-origin forecast pipeline producing per-learner predictions."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from submission.src.data.as_of import build_as_of_dataset
from submission.src.data.features import (
    append_horizon_calendar,
    features_from_panel,
    load_or_build_panel,
)
from submission.src.data.schema import HORIZONS, REPORTING_LAG_DAYS, TRAIN_START
from submission.src.features.regime import (
    covariate_state_features,
    fit_state_model,
    fit_target_state_model,
    target_state_features,
)
from submission.src.models.catboost_model import fit_catboost_per_horizon, predict_catboost_per_horizon
from submission.src.models.lgbm import fit_lgbm_per_horizon, predict_per_horizon
from submission.src.models.ridge import fit_ridge_per_horizon, predict_ridge_per_horizon
from submission.src.models.chronos import (
    load_chronos2_pipeline,
    predict_chronos,
)
from submission.src.models.ingarch import fit_ingarch, predict_ingarch
from submission.src.models.nb_ingarch import fit_nb_ingarch, predict_nb_ingarch
from submission.src.models.nb_ingarch_aq import fit_nb_ingarch_aq, predict_nb_ingarch_aq

def _maybe_filter_panel(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """If `data.feature_subset` points to a selected_features.json, keep only those panel cols."""
    subset_path = config.get("data", {}).get("feature_subset")
    if not subset_path:
        return panel
    selected = json.loads(Path(subset_path).read_text())["selected"]
    keep = [c for c in selected if c in panel.columns]
    return panel[keep]


def forecast_learners(
    daily_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    config: dict,
) -> dict[int, dict[str, float]]:
    """Return {h: {learner_name: pred}} at forecast_date."""
    deadline = forecast_date - pd.Timedelta(days=REPORTING_LAG_DAYS)
    candidate_origins = pd.date_range(TRAIN_START, deadline - pd.Timedelta(days=1), freq="D")
    panel = load_or_build_panel(daily_df, config["data"]["cache_dir"])
    panel = _maybe_filter_panel(panel, config)

    # P3a Markov regime models: fit on data <= deadline - 1 so the state
    # model itself never sees data beyond what the per-horizon embargo allows.
    cov_state_model = None
    y_state_model = None
    regime_cfg = config.get("models", {}).get("regime", {})
    if regime_cfg.get("enabled", False):
        state_train_end = deadline - pd.Timedelta(days=1)
        cov_state_model = fit_state_model(
            daily_df,
            n_states=regime_cfg.get("cov_n_states", 4),
            train_end=state_train_end,
        )
        y_state_model = fit_target_state_model(
            daily_df,
            n_states=regime_cfg.get("y_n_states", 3),
            train_end=state_train_end,
        )

    ds = build_as_of_dataset(
        daily_df=daily_df,
        forecast_date=forecast_date,
        candidate_origins=candidate_origins,
        horizons=HORIZONS,
        panel=panel,
        cov_state_model=cov_state_model,
        y_state_model=y_state_model,
    )

    enabled = {k for k, v in config["models"].items() if v.get("enabled", False)}
    models: dict[str, dict] = {}
    if "lgbm" in enabled:
        models["lgbm"] = fit_lgbm_per_horizon(ds, config["models"]["lgbm"]["params"], HORIZONS)
    if "catboost" in enabled:
        models["catboost"] = fit_catboost_per_horizon(ds, config["models"]["catboost"]["params"], HORIZONS)
    if "ridge" in enabled:
        models["ridge"] = fit_ridge_per_horizon(ds, config["models"]["ridge"]["params"], HORIZONS)
    if "ingarch" in enabled:
        models["ingarch"] = fit_ingarch(daily_df, config["models"]["ingarch"]["params"], forecast_date, HORIZONS)
    if "nb_ingarch" in enabled:
        models["nb_ingarch"] = fit_nb_ingarch(
            daily_df,
            config["models"]["nb_ingarch"]["params"],
            forecast_date,
            HORIZONS,
        )
    if "staton_queue" in enabled:
        from submission.src.models.staton_queue import fit_staton_queue

        models["staton_queue"] = fit_staton_queue(
            daily_df,
            config["models"]["staton_queue"]["params"],
            forecast_date,
            HORIZONS,
        )
    if "chronos" in enabled:
        models["chronos"] = load_chronos2_pipeline(config["models"]["chronos"]["params"])
    if "nb_ingarch_aq" in enabled:
        models["nb_ingarch_aq"] = fit_nb_ingarch_aq(
            daily_df, config["models"]["nb_ingarch_aq"]["params"], forecast_date, HORIZONS,
        )



    feats_o = features_from_panel(panel, origin=forecast_date)
    if cov_state_model is not None:
        feats_o.update(covariate_state_features(daily_df, forecast_date, cov_state_model, HORIZONS))
    out: dict[int, dict[str, float]] = {h: {} for h in HORIZONS}
    for h in HORIZONS:
        feats_oh = append_horizon_calendar(feats_o, origin=forecast_date, horizon=h)
        if y_state_model is not None:
            feats_oh.update(target_state_features(daily_df, forecast_date, y_state_model, [h]))
        if "lgbm" in models:
            out[h]["lgbm"] = predict_per_horizon(models["lgbm"], feats_oh, [h])[h]
        if "catboost" in models:
            out[h]["catboost"] = predict_catboost_per_horizon(models["catboost"], feats_oh, [h])[h]
        if "ridge" in models:
            out[h]["ridge"] = predict_ridge_per_horizon(models["ridge"], feats_oh, [h])[h]

    if "tuominen_lgbm" in models:
        from submission.src.models.tuominen_lgbm import predict_tuominen_lgbm_per_horizon

        tuominen_preds = predict_tuominen_lgbm_per_horizon(
            models["tuominen_lgbm"],
            daily_df,
            forecast_date,
            HORIZONS,
        )
        for h in HORIZONS:
            out[h]["tuominen_lgbm"] = tuominen_preds[h]

    if "ts_baselines" in enabled:
        from submission.src.models.ts_baselines import predict_ts_baseline_learners

        ts_preds = predict_ts_baseline_learners(
            daily_df,
            forecast_date,
            config["models"]["ts_baselines"],
            HORIZONS,
        )
        for learner_name, per_h in ts_preds.items():
            for h in HORIZONS:
                out[h][learner_name] = per_h[h]

    if "ingarch" in models:
        ingarch_preds = predict_ingarch(models["ingarch"], forecast_date, HORIZONS)
        for h in HORIZONS:
            out[h]["ingarch"] = ingarch_preds[h]

    if "nb_ingarch" in models:
        nb_preds = predict_nb_ingarch(models["nb_ingarch"], forecast_date, HORIZONS)
        for h in HORIZONS:
            out[h]["nb_ingarch"] = nb_preds[h]

    if "staton_queue" in models:
        from submission.src.models.staton_queue import predict_staton_queue

        staton_preds = predict_staton_queue(models["staton_queue"], forecast_date, HORIZONS)
        for h in HORIZONS:
            out[h]["staton_queue"] = staton_preds[h]

    if "chronos" in models:
        chronos_preds = predict_chronos(
            daily_df,
            config["models"]["chronos"]["params"],
            forecast_date,
            models["chronos"],
            HORIZONS,
        )
        for h in HORIZONS:
            out[h]["chronos"] = chronos_preds[h]

    if "nb_ingarch_aq" in models:
        aq_preds = predict_nb_ingarch_aq(models["nb_ingarch_aq"], forecast_date, HORIZONS)
        for h in HORIZONS:
            out[h]["nb_ingarch_aq"] = aq_preds[h]

    return out
