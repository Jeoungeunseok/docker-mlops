from datetime import datetime
from typing import Any

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints import mlops
from app.domains.mlops.schemas import (
    EvaluationMetrics,
    ModelRollbackResult,
    PredictionLogPayload,
    TrainingContext,
    TrainingJobRecord,
    TrainingResult,
)


@pytest.mark.asyncio
async def test_create_training_job_returns_training_result(monkeypatch: Any) -> None:
    context = _context()
    expected = TrainingResult(
        model_name="xgboost_global",
        run_id="run-1",
        model_uri="runs:/run-1/model",
        metrics=EvaluationMetrics(mae=1.0, rmse=1.0, mape=1.0, validation_samples=100),
        promoted=True,
    )

    def fake_train_model_job(received_context: TrainingContext) -> TrainingResult:
        assert received_context == context
        return expected

    monkeypatch.setattr(mlops, "train_model_job", fake_train_model_job)

    result = await mlops.create_training_job(context)

    assert result == expected


@pytest.mark.asyncio
async def test_create_training_job_returns_400_when_model_type_is_not_registered(monkeypatch: Any) -> None:
    def fake_train_model_job(context: TrainingContext) -> TrainingResult:
        raise KeyError("No factory registered for model_type: unknown")

    monkeypatch.setattr(mlops, "train_model_job", fake_train_model_job)

    with pytest.raises(HTTPException) as exc:
        await mlops.create_training_job(_context(model_type="unknown"))

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_submit_async_training_job_returns_job_record(monkeypatch: Any) -> None:
    expected = TrainingJobRecord(
        job_id="job-1",
        status="pending",
        context=_context(),
        created_at=datetime(2026, 1, 1),
    )

    class DummyTrainingJobRunner:
        def submit(self, context: TrainingContext, max_attempts: int = 1) -> TrainingJobRecord:
            assert max_attempts == 2
            return expected

    monkeypatch.setattr(mlops, "training_job_runner", DummyTrainingJobRunner())

    result = await mlops.submit_async_training_job(_context(), max_attempts=2)

    assert result == expected


@pytest.mark.asyncio
async def test_get_training_job_returns_404_when_job_is_missing(monkeypatch: Any) -> None:
    class DummyTrainingJobRunner:
        def get(self, job_id: str) -> None:
            return None

    monkeypatch.setattr(mlops, "training_job_runner", DummyTrainingJobRunner())

    with pytest.raises(HTTPException) as exc:
        await mlops.get_training_job("missing")

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_retry_training_job_returns_409_when_job_cannot_be_retried(monkeypatch: Any) -> None:
    class DummyTrainingJobRunner:
        def retry(self, job_id: str) -> TrainingJobRecord:
            raise ValueError("Only failed training jobs can be retried")

    monkeypatch.setattr(mlops, "training_job_runner", DummyTrainingJobRunner())

    with pytest.raises(HTTPException) as exc:
        await mlops.retry_training_job("job-1")

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_rollback_model_returns_result_and_reloads_champion(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_rollback_champion(model_name: str, version: str) -> ModelRollbackResult:
        captured["rollback"] = (model_name, version)
        return ModelRollbackResult(model_name=model_name, champion_version=version)

    class DummyModelLoader:
        def load(self, model_name: str, force_reload: bool = False) -> mlops.ModelLoadResult:
            captured["reload"] = (model_name, force_reload)
            return mlops.ModelLoadResult(
                model_name=model_name,
                model_uri=f"models:/{model_name}@champion",
                version="2",
                run_id="run-2",
                loaded_at=datetime(2026, 1, 1, 12, 0, 0),
            )

    monkeypatch.setattr(mlops, "rollback_champion", fake_rollback_champion)
    monkeypatch.setattr(mlops, "model_loader", DummyModelLoader())
    monkeypatch.setattr(mlops.notification_dispatcher, "notify", lambda event: captured.setdefault("event", event))

    result = await mlops.rollback_model("xgboost_global", mlops.ModelRollbackRequest(version="2"))

    assert result == ModelRollbackResult(model_name="xgboost_global", champion_version="2")
    assert captured["rollback"] == ("xgboost_global", "2")
    assert captured["reload"] == ("xgboost_global", True)
    assert captured["event"].event_type == "rollback_completed"


@pytest.mark.asyncio
async def test_get_model_drift_returns_drift_result(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class DummyPredictionLogStore:
        def list_by_model(self, model_name: str) -> list[PredictionLogPayload]:
            assert model_name == "xgboost_global"
            return [
                PredictionLogPayload(
                    model_name=model_name,
                    predicted_at=datetime(2026, 1, 1, 12, 0, 0),
                    target_timestamp=datetime(2026, 1, 1, 13, 0, 0),
                    predicted_value=12.0,
                    actual_value=10.0,
                    error_value=2.0,
                    error_metrics={"mape": 20.0},
                )
            ]

    monkeypatch.setattr(mlops, "prediction_log_store", DummyPredictionLogStore())
    monkeypatch.setattr(mlops.notification_dispatcher, "notify", lambda event: captured.setdefault("event", event))

    result = await mlops.get_model_drift(
        "xgboost_global",
        min_samples=1,
        max_mean_metric_value=15.0,
    )

    assert result.drift_detected
    assert result.evaluated_samples == 1
    assert result.mean_metric_value == 20.0
    assert captured["event"].event_type == "drift_detected"


@pytest.mark.asyncio
async def test_reload_model_notifies_when_reload_fails(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class DummyModelLoader:
        def load(self, model_name: str, force_reload: bool = False) -> mlops.ModelLoadResult:
            raise RuntimeError("reload failed")

    monkeypatch.setattr(mlops, "model_loader", DummyModelLoader())
    monkeypatch.setattr(mlops.notification_dispatcher, "notify", lambda event: captured.setdefault("event", event))

    with pytest.raises(HTTPException) as exc:
        await mlops.reload_model("xgboost_global")

    assert exc.value.status_code == 502
    assert captured["event"].event_type == "model_reload_failed"
    assert captured["event"].payload["error_message"] == "reload failed"


def _context(model_type: str = "xgboost") -> TrainingContext:
    return TrainingContext(
        model_type=model_type,
        train_start_at=datetime(2026, 1, 1),
        train_end_at=datetime(2026, 1, 2),
        validation_start_at=datetime(2026, 1, 3),
        validation_end_at=datetime(2026, 1, 4),
    )
