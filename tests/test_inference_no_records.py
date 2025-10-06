from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = PROJECT_ROOT / "real-time-data-forecasting-with-ai-and-python-4565024-03_10" / "real-time"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

import inference  # noqa: E402


def test_predict_next_24_hours_with_no_feature_records(monkeypatch, caplog):
    monkeypatch.setattr(inference, "fetch_all_feature_records", lambda: [])

    with caplog.at_level("WARNING"):
        with pytest.raises(ValueError) as excinfo:
            inference.predict_next_24_hours()

    assert "No feature records retrieved for inference" in str(excinfo.value)
    assert any(
        "No feature records retrieved for inference" in message
        for message in caplog.messages
    )
