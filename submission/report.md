# NHS-EAD Forecast Submission Report

Team: Daniella Ye, Munib Mesinovic, Jacek Karwowski, Younesse Kaddar, Esmeralda S. Whitammer, Sam Staton

## Task

We forecast estimated avoidable emergency-department deaths for the 173 rolling
10-day windows of the assessment period. The deliverable is
`submission/pred_matrix.csv`: one row per forecast window, columns
`forecast_id, day_1, ..., day_10`.

Each origin uses only information available at that origin, and the target series
is treated with the contest's three-day reporting lag, so at origin D the most
recent observed target value is from D-3.

## Data and leakage discipline

The pipeline uses only the provided BNSSG operational series; no external data is
introduced. Same-day operational covariates respect the midday cut-off (values
timestamped after midday on the origin day are assigned to the next day).
Imputation and rolling summaries are computed per origin from past values only,
so no future information leaks into a forecast. The raw challenge dataset is not
bundled in this repository.

## Model

The final forecast uses a fixed split by horizon:

- `day_1` to `day_5`: Chronos + NB-INGARCH-AQ + seasonal residual correction.
- `day_6` to `day_10`: NB-INGARCH-AQ.

NB-INGARCH is a negative-binomial integer-valued GARCH model: a count time-series
process that is self-exciting across days and absorbs the overdispersion of the
boarding series. The AQ ("adaptive quantile") variant shifts toward a higher
predictive quantile when a small set of operational covariates, known at the
origin, indicate elevated pressure; this responds to the asymmetric cost of
under-predicting winter spikes under squared error. Chronos is a pretrained,
open-source time-series foundation model applied zero-shot to the target history,
used only in the shorter-horizon block. The seasonal residual correction is
calendar-driven and applied to the `day_1` to `day_5` block.

The horizon split is fixed before the assessment run and does not change from
one assessment origin to another except through the horizon number. The committed
JSON artifacts record the validation and evaluation quantities used to freeze the
candidate.

## Validation

We tuned the small number of free parameters (the blend weights and the
seasonal-prior scale) on a held-out validation block carved from the
pre-assessment period, then evaluated on the combined winter block (Test-A plus
Test-B), the closest available mirror of the October-March assessment season.
Significance was assessed with a moving-block bootstrap over forecast origins.
The selected per-prize configuration:

| Horizon block | Model | A+B MSE |
|---|---|---:|
| days 1-5 | Chronos + NB-INGARCH-AQ + seasonal residual correction | 0.1157 |
| days 6-10 | NB-INGARCH-AQ | 0.1293 |
| combined | fixed horizon split | 0.2450 |

The comparison set also included plain Chronos, NB-INGARCH without AQ,
NB-INGARCH-AQ alone, blended Chronos/NB variants, and gradient-boosted and other
tree learners. The tree learners generalised poorly across seasons and were not
selected. Short and long horizons are dominated by different signals, which is
why the per-prize split outperforms any single model across both prizes.

## Runtime

The pipeline runs on CPU. One 10-day forecast completes well within the one-hour
limit on a standard desktop; the foundation-model component is small and does not
require a GPU.

## Reproducibility

The repository contains the model-development code, configuration, frozen
evaluation artifacts, this report, the forecast matrix, and the validation
script. The submitted matrix has 173 rows, the template columns, sequential
`forecast_id` values from 1 to 173, and finite numeric forecasts.

`submission/pred_matrix.csv` SHA256:

`513dcaa45cb761c25b43c24c3835e9cb5212d21e68b57f11512367d06be8de3b`

## Limitations

The assessment winter included supply-side shocks (a seasonal influenza surge and
periods of industrial action) that the provided demand-side covariates capture
only partially. The adaptive-quantile gate is the main mechanism that lifts
forecasts when system pressure is elevated. The seasonal-residual component is
calendar-driven and is applied only to the short-horizon block.
