from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd
from confluent_kafka import Consumer

warnings.filterwarnings("ignore")

FEATURE_NAMES: List[str] = [
    'lag_1',
    'lag_2',
    'lag_6',
    'lag_12',
    'lag_24',
    'rolling_mean_7',
    'rolling_std_7',
    'hour',
    'day_of_week',
    'month',
    'temperature_forecast',
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


def merge_feature_records(feature_records: Iterable[dict]) -> pd.DataFrame:
    """Merge partial feature rows so each id has a single row with non-null values."""
    records = list(feature_records)
    if not records:
        empty_frame = pd.DataFrame(columns=FEATURE_NAMES)
        empty_frame.index.name = 'id'
        return empty_frame

    frame = pd.DataFrame(records)
    if 'id' not in frame.columns:
        raise KeyError("Feature records must include an 'id' column.")

    sort_columns = ['id']
    if 'timestamp' in frame.columns:
        frame = frame.copy()
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], errors='coerce')
        sort_columns.append('timestamp')

    sorted_frame = frame.sort_values(sort_columns)
    merged = (
        sorted_frame
        .set_index('id')
        .groupby(level=0)
        .ffill()
        .bfill()
        .groupby(level=0)
        .last()
    )
    merged.index.name = 'id'
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
