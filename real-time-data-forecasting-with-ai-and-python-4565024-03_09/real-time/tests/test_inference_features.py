from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import inference


def test_build_feature_frame_merges_partial_records():
    required_columns = [
        "lag_1",
        "lag_2",
        "lag_6",
        "lag_12",
        "lag_24",
        "rolling_mean_7",
        "rolling_std_7",
        "hour",
        "day_of_week",
        "month",
        "temperature_forecast",
    ]

    records = [
        {
            "id": "2024-05-12T01:00:00",
            "lag_1": 100,
            "lag_2": 110,
            "lag_6": 150,
            "lag_12": 170,
            "lag_24": 200,
            "rolling_mean_7": 145.0,
            "rolling_std_7": 5.5,
            "hour": 1,
            "day_of_week": 6,
            "month": 5,
        },
        {
            "id": "2024-05-12T01:00:00",
            "temperature_forecast": 22.5,
        },
    ]

    feature_frame = inference.build_feature_frame(records, required_columns)

    assert feature_frame.shape[0] == 1
    assert list(feature_frame.index) == ["2024-05-12T01:00:00"]
    assert list(feature_frame.columns) == required_columns
    assert feature_frame.notna().all().all()
