from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Iterable, List, Mapping

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
        "bootstrap.servers": "localhost:9092",
        "group.id": "feature_store_reader",
        "auto.offset.reset": "latest",
    }

    consumer = Consumer(conf)
    consumer.subscribe(["feature_store"])

    feature_records: List[dict] = []

    try:
        while True:
            msg = consumer.poll(1)
            if msg is None:
                break
            if msg.error():
                break
            feature_records.append(json.loads(msg.value().decode("utf-8")))
    finally:
        consumer.close()

    return feature_records


def _last_valid_value(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[-1]


def _normalize_timestamp_series(series: pd.Series) -> pd.Series | None:
    """Return timezone-naive timestamps or ``None`` when normalization fails."""

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


def merge_feature_records(feature_records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
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
    frame["id"] = frame["id"].astype(str)
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

    merged = sorted_frame.groupby("id", sort=False).agg(_last_valid_value)
    merged.index.name = "id"
    return merged


def build_feature_frame(
    records: Iterable[Mapping[str, object]],
    required_columns: Iterable[str],
) -> pd.DataFrame:
    merged = merge_feature_records(records)
    if merged.empty:
        empty_frame = pd.DataFrame(columns=list(required_columns))
        empty_frame.index.name = "id"
        return empty_frame

    frame = merged.copy()
    column_order = list(required_columns)
    for column in column_order:
        if column not in frame.columns:
            frame[column] = pd.NA

    return frame[column_order]


def prepare_latest_feature_record(feature_records: Iterable[dict]) -> pd.DataFrame:
    feature_frame = build_feature_frame(feature_records, FEATURE_NAMES)
    if feature_frame.empty:
        return feature_frame

    complete_rows = feature_frame.dropna(subset=FEATURE_NAMES)
    if complete_rows.empty:
        return complete_rows

    return complete_rows.sort_index(ascending=False).head(1)


def predict_next_24_hours() -> tuple[np.ndarray, pd.DataFrame]:
    latest_feature_record = prepare_latest_feature_record(fetch_all_feature_records())
    if latest_feature_record.empty:
        raise ValueError("No complete feature rows available for inference.")

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
