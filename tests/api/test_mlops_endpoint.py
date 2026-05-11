from datetime import datetime
from typing import Any

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints import mlops
from app.domains.mlops.schemas import EvaluationMetrics, TrainingContext, TrainingResult


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


def _context(model_type: str = "xgboost") -> TrainingContext:
    return TrainingContext(
        model_type=model_type,
        train_start_at=datetime(2026, 1, 1),
        train_end_at=datetime(2026, 1, 2),
        validation_start_at=datetime(2026, 1, 3),
        validation_end_at=datetime(2026, 1, 4),
    )
