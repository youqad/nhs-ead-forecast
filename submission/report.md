# NHS-EAD Forecast Submission Report

Team: Daniella Ye, Munib Mesinovic, Jacek Karwowski, Younesse Kaddar, Esmeralda S. Whitammer, Sam Staton

## Task

We forecast estimated avoidable deaths from emergency-department admission
delays over the 131 rolling 10-day windows of the assessment period. The deliverable,
`submission/pred_matrix.csv`, has one row per forecast window and columns
`forecast_id, day_1, ..., day_10`.

## Model

The final forecast splits by horizon:

- `day_1` to `day_5`: Chronos + NB-INGARCH-AQ + seasonal residual correction.
- `day_6` to `day_10`: NB-INGARCH-AQ.

NB-INGARCH is a negative-binomial integer-valued GARCH model (cf.
[@ferland2006integer; @zhu2011negative]). The target is rescaled to a
pseudo-count $z_t = \mathrm{round}(50 y_t)$ (so that a count likelihood
applies to a continuous series) and filtered by

$$\lambda_t = \bigl(\omega + \alpha z_{t-1} + \beta_1 \lambda_{t-1} + \beta_2
\lambda_{t-2}\bigr) e^{\gamma^{\top} x_t}, \qquad z_t \mid \lambda_t \sim
\mathrm{NB}(\lambda_t, \phi),$$

where $x_t$ collects five calendar indicators and the dispersion $\phi$ is
estimated jointly by maximum likelihood. Since the
intensity feeds back on the previous count, the process is self-exciting (one
day's pressure carries into the next), while the negative-binomial observation
absorbs the overdispersion of the series; multi-step forecasts come
from iterating the same recursion, with $\lambda$ plugged in for the counts
that have not yet been observed. The AQ
("adaptive quantile") variant leaves this mean model untouched and only moves
the point forecast. Its spike score $s$ is the larger of two signals, both
computed at the origin: how volatile the target has recently been, and the
maximum z-score of three pressure covariates (SevernSide calls received,
SWASFT NHS-111 incidents, and BRI paediatric A&E attendances). Whenever $s$
is positive, the point forecast is taken at a higher quantile of the fitted
distribution,

$$\hat{y} = \tfrac{1}{50} F^{-1}_{\mathrm{NB}}\bigl(q;\ \lambda, \phi\bigr),
\qquad q = 0.5 + 0.2 \min(s, 1).$$

A self-exciting model's conditional mean lags the onset of a surge, so on
days where the spike score is positive, taking an upper quantile corrects for
a mean that would otherwise sit too low. (The
rounding inside $z_t$ is also why the `day_6` to `day_10` forecasts land on a
grid with steps of 0.02.) Chronos [@ansari2025chronos2], a pretrained
open-weights time-series foundation model (\hf{amazon/chronos-2}; inference
code \gh{amazon-science/chronos-forecasting}, Apache-2.0), is applied
zero-shot to the target history (at inference it is given only the
series it forecasts) and is used only for the `day_1`
to `day_5` block. The seasonal residual correction, finally, is a ridge
regression of the ensemble's residual on calendar features and on a coarse
pressure state derived from the covariates; its prediction, scaled by a factor
lambda chosen on the validation block, is added to those same horizons.

## Validation

We tuned the few free parameters (the blend weights and the seasonal-prior
scale lambda) on a held-out validation block carved out of the pre-assessment
period, and evaluated the candidates on the combined winter block (Test-A plus
Test-B), the nearest available winter to the assessment period. Since model
development predates the June 2026 amendment of the assessment data, the
figures below are computed over the pre-amendment Test-A+B block; the first
row is recorded in `seasonal_residual_chronos_plus_nb_ingarch_aq_TESTAB.json`
(at the deployed lambda = 1.0), the second in `standalone_eval_TESTAB.json`.

| Horizon block | Model | Test-A+B MSE |
|---|---|---:|
| days 1-5 | Chronos + NB-INGARCH-AQ + seasonal residual correction | 0.1156 |
| days 6-10 | NB-INGARCH-AQ | 0.1293 |

Scoring the submitted matrix against the released assessment targets
(`submission/mse_summary.csv`) gives realized MSEs of 0.0957 for days 1-5 and
0.1354 for days 6-10: better than those Test-A+B figures anticipated on the
short block, worse on the long one.

We compared plain Chronos, NB-INGARCH without the adaptive quantile, and
NB-INGARCH-AQ over all ten horizons; further blends and tree learners were
tried during development, though their evaluations are not part of this
repository. The blend with the seasonal correction came out best over days
1-5 and plain NB-INGARCH-AQ over days 6-10, so the split deploys each winner:
on the short block it beats every single-model candidate in the committed
comparison, and on the long block it coincides with the best one.

## Runtime

The pipeline runs on CPU. The per-origin logs committed under
`submission/artifacts/runtime_logs/` (131 in all) record a median of 14
seconds per forecast, far inside the one-hour limit.

## Data and leakage discipline

Every forecast uses only information that was available at its origin. The
target series carries the contest's three-day reporting lag, so at origin D
the latest observed value dates from D-3. Covariates recorded after midday on
the origin day only become visible the next day, and imputation and rolling
summaries are computed per origin from past values alone. We used no data
beyond what the contest provides.[^holidays]

[^holidays]: The code base also computes a bank-holiday feature from a public
calendar, but only for tree-based models we did not deploy; nothing derived
from it is in the submitted forecasts.

## Reproducibility

The submitted matrix has 131 rows, sequential `forecast_id` values, the
template columns, and finite forecasts throughout, all of which the validation
script checks.

`submission/pred_matrix.csv` SHA256:

`c57862850bd758b8af399d522f694e4c50b83b8b7df7a30cfcc47dcfc73e6699`

## Limitations

The assessment winter included supply-side shocks (a seasonal influenza surge,
periods of industrial action) that the provided demand-side covariates capture
only partially. On top of that, the adaptive-quantile gate reads covariates
observed at the origin, so pressure that first emerges inside the 10-day
forecast window cannot lift the forecast, however severe it becomes. And the
seasonal correction is estimated from past winters: a winter whose calendar
profile departs from theirs (as one with widespread industrial action
plausibly does) receives a correction fitted to the wrong shape, and since
that correction covers `day_1` to `day_5` only, the longest horizons meet
Christmas and New Year without it.

## References

::: {#refs}
:::
