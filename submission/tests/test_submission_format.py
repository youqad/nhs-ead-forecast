import numpy as np
import pandas as pd

from submission.src.data.schema import ASSESSMENT_ORIGINS, HORIZONS


def test_assessment_origin_count():
    assert len(ASSESSMENT_ORIGINS) == 173


def test_horizons_one_to_ten():
    assert HORIZONS == list(range(1, 11))


def test_pred_matrix_shape_contract():
    expected_cols = ["forecast_id"] + [f"day_{h}" for h in HORIZONS]
    sample = pd.DataFrame(
        {"forecast_id": [1], **{f"day_{h}": [0.5] for h in HORIZONS}}
    )
    for c in expected_cols:
        assert c in sample.columns
