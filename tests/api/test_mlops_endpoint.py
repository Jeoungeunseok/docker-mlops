from datetime import datetime
from typing import Any

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints import mlops
from app.domains.mlops.schemas import EvaluationMetrics, ModelRollbackResult, TrainingContext, TrainingJobRecord, TrainingResult


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
async def test_rollback_model_returns_result(monkeypatch: Any) -> None:
    def fake_rollback_champion(model_name: str, version: str) -> ModelRollbackResult:
        return ModelRollbackResult(model_name=model_name, champion_version=version)

    monkeypatch.setattr(mlops, "rollback_champion", fake_rollback_champion)

    result = await mlops.rollback_model("xgboost_global", mlops.ModelRollbackRequest(version="2"))

    assert result == ModelRollbackResult(model_name="xgboost_global", champion_version="2")


def _context(model_type: str = "xgboost") -> TrainingContext:
    return TrainingContext(
        model_type=model_type,
        train_start_at=datetime(2026, 1, 1),
        train_end_at=datetime(2026, 1, 2),
        validation_start_at=datetime(2026, 1, 3),
        validation_end_at=datetime(2026, 1, 4),
    )
