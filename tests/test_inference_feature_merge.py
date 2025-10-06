from __future__ import annotations

import sys
from pathlib import Path

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
