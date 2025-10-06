from collections import defaultdict
import json
from typing import Iterable, Mapping, Sequence, Any, Dict

from confluent_kafka import Consumer
import joblib
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


REQUIRED_FEATURE_COLUMNS = [
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


def load_model(model_path: str):
    return joblib.load(model_path)


def fetch_all_feature_records():
    conf = {
        "bootstrap.servers": "localhost:9092",
        "group.id": "feature_store_reader",
        "auto.offset.reset": "latest",
    }

    consumer = Consumer(conf)
    consumer.subscribe(["feature_store"])

    feature_records = []

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
    records: Iterable[Mapping[str, Any]],
    required_columns: Sequence[str],
) -> pd.DataFrame:
    if required_columns is None:
        raise ValueError("required_columns must be provided")

    merged_records: Dict[Any, Dict[str, Any]] = defaultdict(dict)

    for payload in records or []:
        if not isinstance(payload, Mapping):
            raise TypeError("Feature records must be mappings")
        if "id" not in payload:
            raise ValueError("Feature record is missing required 'id' field")

        record_id = payload["id"]
        if record_id is None:
            raise ValueError("Feature record has null 'id'")

        for key, value in payload.items():
            if key == "id":
                continue
            merged_records[record_id][key] = value

    frame = pd.DataFrame.from_dict(merged_records, orient="index")
    if frame.empty:
        return frame.reindex(columns=list(required_columns))

    frame.index.name = "id"
    frame = frame.reindex(columns=list(required_columns))
    frame = frame.dropna(subset=required_columns)
    return frame


def prepare_latest_feature_record() -> pd.DataFrame:
    feature_records = fetch_all_feature_records()
    feature_frame = build_feature_frame(feature_records, REQUIRED_FEATURE_COLUMNS)
    if feature_frame.empty:
        raise ValueError("No feature records available with all required columns")

    return feature_frame.sort_index(ascending=False).head(1)


def run_inference(model_path: str = "models/energy_demand_model_v4.pkl") -> pd.DataFrame:
    latest_feature_record = prepare_latest_feature_record()
    model = load_model(model_path)
    prediction = model.predict(latest_feature_record)
    print(
        "Next 24 hours energy prediction = {prediction} Last updated:{timestamps}".format(
            prediction=str(prediction),
            timestamps=str(latest_feature_record.index.to_list()),
        )
    )
    return latest_feature_record


if __name__ == "__main__":
    run_inference()
