from collections.abc import Callable
from typing import Generic, TypeVar

from app.domains.mlops.schemas import ModelTrainer, TrainingDataProcessor
from app.domains.prediction.validation import PredictionInputValidator

T = TypeVar("T")


class ModelTypeRegistry(Generic[T]):
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], T]] = {}

    def register(self, model_type: str, factory: Callable[[], T]) -> None:
        normalized_model_type = _normalize_model_type(model_type)
        self._factories[normalized_model_type] = factory

    def get(self, model_type: str) -> T:
        normalized_model_type = _normalize_model_type(model_type)
        factory = self._factories.get(normalized_model_type)
        if factory is None:
            raise KeyError(f"No factory registered for model_type: {model_type}")
        return factory()

    def registered_model_types(self) -> list[str]:
        return sorted(self._factories)


def _normalize_model_type(model_type: str) -> str:
    return model_type.strip().lower()


trainer_registry: ModelTypeRegistry[ModelTrainer] = ModelTypeRegistry()
data_processor_registry: ModelTypeRegistry[TrainingDataProcessor] = ModelTypeRegistry()
prediction_input_validator_registry: ModelTypeRegistry[PredictionInputValidator] = ModelTypeRegistry()
