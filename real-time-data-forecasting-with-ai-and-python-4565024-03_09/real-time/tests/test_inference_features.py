from importlib import util
from pathlib import Path

import pandas as pd
import pytest


def load_inference_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "inference.py"
    )
    spec = util.spec_from_file_location("inference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load inference module for testing.")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_feature_frame_merges_partial_records():
    inference = load_inference_module()
    lag_record = {
        "id": "2024-03-01T00:00:00Z",
        "lag_1": 100.0,
        "lag_2": 90.0,
        "lag_6": 80.0,
        "lag_12": 70.0,
        "lag_24": 60.0,
        "rolling_mean_7": 85.0,
        "rolling_std_7": 5.0,
        "hour": 0,
        "day_of_week": 4,
        "month": 3,
    }
    temperature_record = {
        "id": "2024-03-01T00:00:00Z",
        "temperature_forecast": -2.0,
    }

    frame = inference.build_feature_frame(
        records=[lag_record, temperature_record],
        required_columns=inference.FEATURE_NAMES,
    )

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == inference.FEATURE_NAMES
    assert len(frame) == 1
    assert frame.iloc[0].to_dict() == pytest.approx({
        "lag_1": 100.0,
        "lag_2": 90.0,
        "lag_6": 80.0,
        "lag_12": 70.0,
        "lag_24": 60.0,
        "rolling_mean_7": 85.0,
        "rolling_std_7": 5.0,
        "hour": 0,
        "day_of_week": 4,
        "month": 3,
        "temperature_forecast": -2.0,
    })
