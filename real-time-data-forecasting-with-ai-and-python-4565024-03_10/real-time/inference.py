from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from confluent_kafka import Consumer

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = [
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
MODEL_PATH = Path("models/energy_demand_model_v4.pkl")


def load_model(model_path: Path):
    return joblib.load(model_path)


def fetch_all_feature_records() -> List[dict]:
    """Read feature records from Kafka until no new messages arrive."""
    conf = {
        'bootstrap.servers': "localhost:9092",
        'group.id': "feature_store_reader",
        'auto.offset.reset': 'latest',
    }

    consumer = Consumer(conf)
    consumer.subscribe(['feature_store'])

    feature_records: List[dict] = []

    try:
        while True:
            msg = consumer.poll(1)
            if msg is None:
                break
            if msg.error():
                break
            feature_records.append(json.loads(msg.value().decode('utf-8')))
    finally:
        consumer.close()

    return feature_records


def _last_valid_value(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[-1]


def _normalize_timestamp_series(series: pd.Series) -> Optional[pd.Series]:
    """Return a timezone-naive datetime series when possible.

    Values are parsed individually to gracefully handle collections that mix
    tz-aware and tz-naive timestamps. When parsing fails, ``None`` is returned
    so callers can fall back to a more permissive ordering strategy.
    """

    def _convert(value: object) -> pd.Timestamp:
        if pd.isna(value):
            return pd.NaT
        try:
            timestamp = pd.to_datetime(value, utc=True)
        except (TypeError, ValueError, OverflowError):
            return pd.NaT
        if isinstance(timestamp, pd.Timestamp):
            return timestamp.tz_localize(None)
        return pd.NaT

    try:
        normalized_objects = series.map(_convert)
    except Exception:
        LOGGER.debug("Failed to map timestamp values; falling back to arrival order.", exc_info=True)
        return None

    try:
        return pd.to_datetime(normalized_objects, errors="coerce")
    except (TypeError, ValueError, OverflowError):
        LOGGER.debug("Failed to coerce normalized timestamps; falling back to arrival order.", exc_info=True)
        return None


def merge_feature_records(feature_records: Iterable[dict]) -> pd.DataFrame:
    """Merge partial feature rows so each id has a single row with the latest values."""
    records = list(feature_records)
    if not records:
        empty_frame = pd.DataFrame(columns=FEATURE_NAMES)
        empty_frame.index.name = "id"
        return empty_frame

    frame = pd.DataFrame(records)
    if "id" not in frame.columns:
        raise KeyError("Feature records must include an 'id' column.")

    frame = frame.copy()
    frame["_arrival_index"] = range(len(frame))

    sort_columns: List[str] = ["id"]
    if "timestamp" in frame.columns:
        normalized_timestamps = _normalize_timestamp_series(frame["timestamp"])
        if normalized_timestamps is not None:
            frame["timestamp"] = normalized_timestamps
            if normalized_timestamps.notna().any():
                sort_columns.append("timestamp")
        else:
            LOGGER.debug("Timestamp normalization failed; using arrival order only for sorting.")
    sort_columns.append("_arrival_index")

    sorted_frame = frame.sort_values(sort_columns)
    sorted_frame = sorted_frame.drop(columns=["_arrival_index"])

    merged = (
        sorted_frame.groupby("id", sort=False).agg(_last_valid_value)
    )
    merged.index.name = "id"
    return merged


def prepare_latest_feature_record(feature_records: Iterable[dict]) -> pd.DataFrame:
    merged = merge_feature_records(feature_records)
    if merged.empty:
        return merged

    missing_columns = [name for name in FEATURE_NAMES if name not in merged.columns]
    if missing_columns:
        raise ValueError(f"Missing expected feature columns: {missing_columns}")

    latest_feature_record = (
        merged[FEATURE_NAMES]
        .dropna()
        .sort_index(ascending=False)
        .head(1)
    )
    return latest_feature_record


def predict_next_24_hours() -> tuple[np.ndarray, pd.DataFrame]:
    feature_records = fetch_all_feature_records()
    if not feature_records:
        message = "No feature records retrieved for inference; skipping prediction."
        LOGGER.warning(message)
        raise ValueError(message)

    latest_feature_record = prepare_latest_feature_record(feature_records)
    if latest_feature_record.empty:
        message = "No aggregated feature rows available for inference; skipping prediction."
        LOGGER.warning(message)
        raise ValueError(message)

    model = load_model(MODEL_PATH)
    prediction = model.predict(latest_feature_record)
    return prediction, latest_feature_record


def main() -> None:
    prediction, latest_feature_record = predict_next_24_hours()
    print(
        "Next 24 hours energy prediction = "
        f"{prediction.tolist()} "
        f"Last updated:{latest_feature_record.index.to_list()}"
    )


if __name__ == "__main__":
    main()
