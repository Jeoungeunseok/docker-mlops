from collections.abc import Callable
from dataclasses import dataclass

from app.domains.mlops.data_processing import PassthroughTrainingDataProcessor
from app.domains.mlops.registry import (
    data_processor_registry,
    prediction_input_validator_registry,
    trainer_registry,
)
from app.domains.mlops.schemas import ModelTrainer, TrainingDataProcessor
from app.domains.prediction.services.gru_service import GruTrainer
from app.domains.prediction.services.xgboost_service import XGBoostTrainer
from app.domains.prediction.validation import PredictionInputValidator


@dataclass(frozen=True)
class ModelComponentRegistration:
    model_type: str
    trainer_factory: Callable[[], ModelTrainer] | None = None
    data_processor_factory: Callable[[], TrainingDataProcessor] | None = None
    prediction_input_validator_factory: Callable[[], PredictionInputValidator] | None = None


DEFAULT_MODEL_COMPONENTS: tuple[ModelComponentRegistration, ...] = (
    ModelComponentRegistration(
        model_type="xgboost",
        trainer_factory=XGBoostTrainer,
        data_processor_factory=PassthroughTrainingDataProcessor,
    ),
    ModelComponentRegistration(
        model_type="gru",
        trainer_factory=GruTrainer,
        data_processor_factory=PassthroughTrainingDataProcessor,
    ),
)


def bootstrap_mlops_components(
    registrations: tuple[ModelComponentRegistration, ...] = DEFAULT_MODEL_COMPONENTS,
) -> None:
    for registration in registrations:
        register_model_components(registration)


def register_model_components(registration: ModelComponentRegistration) -> None:
    if registration.trainer_factory is not None:
        trainer_registry.register(registration.model_type, registration.trainer_factory)
    if registration.data_processor_factory is not None:
        data_processor_registry.register(registration.model_type, registration.data_processor_factory)
    if registration.prediction_input_validator_factory is not None:
        prediction_input_validator_registry.register(
            registration.model_type,
            registration.prediction_input_validator_factory,
        )
