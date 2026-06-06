"""Domain constants. Single source of truth for column names, dates, dummies."""
from __future__ import annotations

import pandas as pd

TARGET_COL = "estimated_avoidable_deaths"
DATE_COL = "date"
DT_COL = "dt"
DUMMY_VALUE = -9999

# Assessment data period (the contest's "182-day window")
ASSESSMENT_START = pd.Timestamp("2025-10-01")
ASSESSMENT_END = pd.Timestamp("2026-03-31")

# Forecast origins D where D+1..D+10 fits inside the assessment window.
# Period 1 covers Oct 1-10  → D = 2025-09-30
# Period 173 covers Mar 22-31 → D = 2026-03-21
ASSESSMENT_FIRST_ORIGIN = pd.Timestamp("2025-09-30")
ASSESSMENT_LAST_ORIGIN = pd.Timestamp("2026-03-21")
ASSESSMENT_ORIGINS = pd.date_range(
    ASSESSMENT_FIRST_ORIGIN, ASSESSMENT_LAST_ORIGIN, freq="D"
)
assert len(ASSESSMENT_ORIGINS) == 173, (
    f"ASSESSMENT_ORIGINS must have exactly 173 entries, got {len(ASSESSMENT_ORIGINS)}"
)

HORIZONS = list(range(1, 11))
REPORTING_LAG_DAYS = 3
MIDDAY_HOUR = 12

TRAIN_START = pd.Timestamp("2023-03-16")

# Validation = Dec 2023 – Sep 2024 (fits blend, seasonal prior, feature selection, tuning).
# Window-end convention: scripts build origins as date_range(start, END - 10d),
# so the last val origin is 2024-09-20 and its forecast window ends <= 2024-09-30.
VAL_START = pd.Timestamp("2023-12-01")
VAL_END = pd.Timestamp("2024-09-30")

# TEST = exact one-year-back mirror of the assessment window (held-out).
# Origins are used DIRECTLY (no -10 cap) so the count matches the assessment.
TEST_FIRST_ORIGIN = ASSESSMENT_FIRST_ORIGIN - pd.DateOffset(years=1)   # 2024-09-30
TEST_LAST_ORIGIN = ASSESSMENT_LAST_ORIGIN - pd.DateOffset(years=1)     # 2025-03-21
TEST_ORIGINS = pd.date_range(TEST_FIRST_ORIGIN, TEST_LAST_ORIGIN, freq="D")
assert len(TEST_ORIGINS) == 173, (
    f"TEST_ORIGINS must mirror the 173 assessment origins, got {len(TEST_ORIGINS)}"
)

# Test-A (results table) vs Test-B (submission-day lockbox), temporal split at
# end-of-January 2025. A is the Christmas-peak block; B is post-peak.
TEST_A_FIRST_ORIGIN = TEST_FIRST_ORIGIN                  # 2024-09-30
TEST_A_LAST_ORIGIN = pd.Timestamp("2025-01-31")
TEST_B_FIRST_ORIGIN = pd.Timestamp("2025-02-01")
TEST_B_LAST_ORIGIN = TEST_LAST_ORIGIN                    # 2025-03-21

SEED = 2026
