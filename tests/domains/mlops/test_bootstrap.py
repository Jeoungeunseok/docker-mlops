from app.domains.mlops import bootstrap
from app.domains.mlops.bootstrap import ModelComponentRegistration, register_model_components
from app.domains.mlops.data_processing import PassthroughTrainingDataProcessor
from app.domains.mlops.registry import ModelTypeRegistry
from app.domains.mlops.schemas import TrainingContext
from app.domains.prediction.validation import RequiredFieldsValidator


class DummyTrainer:
    def train_model(self, context: TrainingContext) -> object:
        return object()

    def evaluate_model(self, model: object, context: TrainingContext) -> object:
        return object()

    def log_model(self, model: object, artifact_path: str) -> str:
        return f"runs:/run-1/{artifact_path}"


def test_register_model_components_registers_all_component_factories(monkeypatch) -> None:
    trainer_registry = ModelTypeRegistry()
    data_processor_registry = ModelTypeRegistry()
    validator_registry = ModelTypeRegistry()
    monkeypatch.setattr(bootstrap, "trainer_registry", trainer_registry)
    monkeypatch.setattr(bootstrap, "data_processor_registry", data_processor_registry)
    monkeypatch.setattr(bootstrap, "prediction_input_validator_registry", validator_registry)

    register_model_components(
        ModelComponentRegistration(
            model_type="forecast",
            trainer_factory=DummyTrainer,
            data_processor_factory=PassthroughTrainingDataProcessor,
            prediction_input_validator_factory=lambda: RequiredFieldsValidator({"x"}),
        )
    )

    assert isinstance(trainer_registry.get("forecast"), DummyTrainer)
    assert isinstance(data_processor_registry.get("forecast"), PassthroughTrainingDataProcessor)
    assert isinstance(validator_registry.get("forecast"), RequiredFieldsValidator)


def test_bootstrap_mlops_components_registers_default_model_types(monkeypatch) -> None:
    trainer_registry = ModelTypeRegistry()
    data_processor_registry = ModelTypeRegistry()
    validator_registry = ModelTypeRegistry()
    monkeypatch.setattr(bootstrap, "trainer_registry", trainer_registry)
    monkeypatch.setattr(bootstrap, "data_processor_registry", data_processor_registry)
    monkeypatch.setattr(bootstrap, "prediction_input_validator_registry", validator_registry)

    bootstrap.bootstrap_mlops_components()

    assert trainer_registry.registered_model_types() == ["gru", "xgboost"]
    assert data_processor_registry.registered_model_types() == ["gru", "xgboost"]
    assert validator_registry.registered_model_types() == []
