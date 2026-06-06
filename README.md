<div align="center">

# NHS Estimated Avoidable Deaths: 10-Day Forecast

### An entry to the [SPHERE-PPL NHS Acute Patient Harm Forecasting Contest](https://github.com/SPHERE-PPL/NHS-EAD-forecast)

[![Challenge](https://img.shields.io/badge/SPHERE--PPL-NHS--EAD_forecast-0b5394?style=for-the-badge)](https://sphere-ppl.org)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Chronos-2](https://img.shields.io/badge/Chronos--2-foundation_model-FF9900?style=for-the-badge)](https://github.com/amazon-science/chronos-forecasting)
[![License](https://img.shields.io/badge/license-CC--BY--NC--SA_4.0-A6CE39?style=for-the-badge)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

Forecasting daily **estimated avoidable deaths** from emergency-department admission delays across the Bristol NHS, ten days ahead, so managers get advance warning of system pressure.

</div>

---

## Team

Daniella Ye &middot; Munib Mesinovic &middot; Jacek Karwowski &middot; Younesse Kaddar &middot; Esmeralda S. Whitammer &middot; Sam Staton

## The challenge

Every four hours of delay admitting a patient from the emergency department raises 30-day mortality odds by about 8% (Howlett et al., 2026): an estimated 25 avoidable deaths a month in the Bristol NHS. The contest asks for a daily forecast of that count over the next ten days, across 173 sliding windows of the Oct 2025 to Mar 2026 winter, scored by mean squared error separately for **days 1-5** and **days 6-10** (two prizes).

## Our approach

The two horizons are scored independently, so each prize gets the strongest validated system rather than one compromise:

| Horizon | System | Winter MSE |
| :-- | :-- | --: |
| **Days 1-5** | Chronos-2 + NB-INGARCH-AQ + seasonal residual prior | **0.116** |
| **Days 6-10** | NB-INGARCH-AQ | **0.129** |

- **Chronos-2**: a pretrained time-series foundation model, run zero-shot; carries the near horizon across the three-day target reporting lag.
- **NB-INGARCH-AQ**: a Negative-Binomial INGARCH count model with an adaptive-quantile surge response; steadiest at the far horizon.
- **Seasonal residual prior**: a ridge correction for calendar and holiday structure.

The split is chosen on a held-out winter (the one-year-back mirror of the assessment period). Every feature respects the three-day target lag, and only the contest's provided data is used.

## Reproduce

```bash
# one official 10-day forecast (under the contest's one-hour rule)
uv run python -m submission.run_forecast --config submission/config/default.yaml --origin 2025-10-01

# the full 173-origin assessment matrix
uv run python -m submission.scripts.run_final_forecast --config submission/config/default.yaml

# validate the deliverables
uv run python -m submission.scripts.validate_submission --config submission/config/default.yaml
```

Full method and per-prize rationale live in [`submission/README.md`](submission/README.md) and the report at [`submission/report.pdf`](submission/report.pdf).

## Submission

| File | Contents |
| :-- | :-- |
| [`submission/pred_matrix.csv`](submission/pred_matrix.csv) | 173 rows of `forecast_id, day_1, ..., day_10` |
| `submission/forecast.csv` | byte-identical copy of `pred_matrix.csv` (collated under either filename) |
| [`submission/report.pdf`](submission/report.pdf) | report, max 1000 words |

Entries are collated by the [Forecast AggregatoR](https://github.com/SPHERE-PPL/Forecast-AggregatoR) the day after the contest closes.

## License

Content is licensed [CC-BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/), following the upstream contest. The Bristol NHS data under `data/` is the contest's own (Git LFS); see [`data/README.md`](data/README.md) for its terms.
