from datetime import datetime
from typing import Any

from app.domains.mlops.data_processing import build_training_dataset
from app.domains.mlops.schemas import TrainingContext, TrainingDataset


class DummyProcessor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def load_training_data(self, context: TrainingContext) -> list[int]:
        self.calls.append("load")
        return [1, 2, 3, 4]

    def preprocess(self, raw_data: list[int], context: TrainingContext) -> list[int]:
        self.calls.append("preprocess")
        return [value * 2 for value in raw_data]

    def build_features(self, processed_data: list[int], context: TrainingContext) -> dict[str, Any]:
        self.calls.append("features")
        return {"values": processed_data}

    def split_validation(self, features: dict[str, Any], context: TrainingContext) -> TrainingDataset:
        self.calls.append("split")
        return TrainingDataset(
            train_features=features["values"][:2],
            validation_features=features["values"][2:],
            metadata={"source": "dummy"},
        )


def test_build_training_dataset_runs_processor_steps_in_order() -> None:
    processor = DummyProcessor()
    context = TrainingContext(
        model_type="xgboost",
        train_start_at=datetime(2026, 1, 1),
        train_end_at=datetime(2026, 1, 2),
        validation_start_at=datetime(2026, 1, 3),
        validation_end_at=datetime(2026, 1, 4),
    )

    dataset = build_training_dataset(context, processor)

    assert processor.calls == ["load", "preprocess", "features", "split"]
    assert dataset.train_features == [2, 4]
    assert dataset.validation_features == [6, 8]
    assert dataset.metadata == {"source": "dummy"}
