from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping

import joblib
import numpy as np
import pandas as pd
from confluent_kafka import Consumer

warnings.filterwarnings("ignore")

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


def build_feature_frame(
    records: Iterable[Mapping[str, object]],
    required_columns: Iterable[str],
) -> pd.DataFrame:
    """Merge partial records into complete feature rows indexed by identifier."""

    merged: MutableMapping[str, dict] = defaultdict(dict)
    for record in records:
        if "id" not in record:
            raise KeyError("Feature records must include an 'id' column.")
        identifier = record["id"]
        if not isinstance(identifier, str):
            identifier = str(identifier)

        current = merged[identifier]
        for key, value in record.items():
            if key == "id":
                continue
            current[key] = value

    if not merged:
        empty_frame = pd.DataFrame(columns=list(required_columns))
        empty_frame.index.name = "id"
        return empty_frame

    frame = pd.DataFrame.from_dict(merged, orient="index")
    frame.index.name = "id"

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
