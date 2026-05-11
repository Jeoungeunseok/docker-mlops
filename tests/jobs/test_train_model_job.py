from datetime import datetime
from typing import Any

from app.domains.mlops.schemas import EvaluationMetrics, TrainingContext
from app.jobs import train_model_job


class DummyTrainer:
    def train_model(self, context: TrainingContext) -> dict[str, str]:
        return {"model": context.model_type}

    def evaluate_model(self, model: Any, context: TrainingContext) -> EvaluationMetrics:
        return EvaluationMetrics(mae=1.0, rmse=1.0, mape=1.0, validation_samples=100)

    def log_model(self, model: Any, artifact_path: str) -> str:
        return f"runs:/run-1/{artifact_path}"


def test_train_model_job_uses_registry_when_trainer_is_not_provided(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class DummyRegistry:
        def get(self, model_type: str) -> DummyTrainer:
            captured["model_type"] = model_type
            return DummyTrainer()

    def fake_run_training_pipeline(context: TrainingContext, trainer: DummyTrainer) -> str:
        captured["trainer"] = trainer
        return "ok"

    monkeypatch.setattr(train_model_job, "trainer_registry", DummyRegistry())
    monkeypatch.setattr(train_model_job, "run_training_pipeline", fake_run_training_pipeline)

    result = train_model_job.train_model_job(_context())

    assert result == "ok"
    assert captured["model_type"] == "xgboost"
    assert isinstance(captured["trainer"], DummyTrainer)


def test_train_model_job_prefers_explicit_trainer(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    explicit_trainer = DummyTrainer()

    def fake_run_training_pipeline(context: TrainingContext, trainer: DummyTrainer) -> str:
        captured["trainer"] = trainer
        return "ok"

    monkeypatch.setattr(train_model_job, "run_training_pipeline", fake_run_training_pipeline)

    result = train_model_job.train_model_job(_context(), trainer=explicit_trainer)

    assert result == "ok"
    assert captured["trainer"] is explicit_trainer


def _context() -> TrainingContext:
    return TrainingContext(
        model_type="xgboost",
        train_start_at=datetime(2026, 1, 1),
        train_end_at=datetime(2026, 1, 2),
        validation_start_at=datetime(2026, 1, 3),
        validation_end_at=datetime(2026, 1, 4),
    )
