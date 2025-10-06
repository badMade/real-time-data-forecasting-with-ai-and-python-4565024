from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = PROJECT_ROOT / "real-time-data-forecasting-with-ai-and-python-4565024-03_10" / "real-time"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

import inference  # noqa: E402


def test_merge_feature_records_combines_partial_rows():
    feature_records = [
        {
            "id": "row-1",
            "lag_1": 1.0,
            "lag_2": 2.0,
            "timestamp": "2024-03-01T00:00:00Z",
        },
        {
            "id": "row-1",
            "temperature_forecast": 11.5,
            "timestamp": "2024-03-01T00:05:00Z",
        },
    ]

    merged = inference.merge_feature_records(feature_records)

    expected = pd.DataFrame(
        {
            "lag_1": [1.0],
            "lag_2": [2.0],
            "timestamp": pd.to_datetime(["2024-03-01T00:05:00Z"]),
            "temperature_forecast": [11.5],
        },
        index=pd.Index(["row-1"], name="id"),
    )

    assert_frame_equal(merged.loc[:, expected.columns], expected)


def test_predict_next_24_hours_merges_features_and_calls_model_once():
    feature_records = [
        {
            "id": "row-1",
            "timestamp": "2024-03-01T00:00:00Z",
            "lag_1": 1.0,
            "lag_2": 2.0,
            "lag_6": 6.0,
            "lag_12": 12.0,
            "lag_24": 24.0,
            "rolling_mean_7": 7.0,
            "rolling_std_7": 0.7,
            "hour": 0,
            "day_of_week": 5,
            "month": 3,
        },
        {
            "id": "row-1",
            "timestamp": "2024-03-01T00:05:00Z",
            "temperature_forecast": 11.5,
        },
    ]

    dummy_model = MagicMock()
    dummy_model.predict.return_value = np.array([42.0])

    with (
        patch.object(inference, "fetch_all_feature_records", return_value=feature_records),
        patch.object(inference.joblib, "load", return_value=dummy_model),
    ):
        prediction, latest_feature_record = inference.predict_next_24_hours()

    assert dummy_model.predict.call_count == 1
    assert prediction.tolist() == [42.0]
    assert latest_feature_record.index.tolist() == ["row-1"]
    assert set(latest_feature_record.columns) == set(inference.FEATURE_NAMES)
    assert latest_feature_record.loc["row-1", "temperature_forecast"] == 11.5
