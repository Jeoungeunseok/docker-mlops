from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints import mlops
from app.domains.mlops.schemas import (
    EvaluationMetrics,
    MlopsEventRecord,
    MlopsNotificationTestRequest,
    MlopsSchedulerJobStatus,
    MlopsSchedulerTickRequest,
    ModelRollbackResult,
    PredictionLogPayload,
    TrainingContext,
    TrainingJobRecord,
    TrainingResult,
)
from app.domains.mlops.registry import ModelTypeRegistry


@pytest.mark.asyncio
async def test_get_mlops_status_returns_operational_summary(monkeypatch: Any) -> None:
    trainer_registry = ModelTypeRegistry()
    data_processor_registry = ModelTypeRegistry()
    validator_registry = ModelTypeRegistry()
    trainer_registry.register("xgboost", lambda: object())
    data_processor_registry.register("xgboost", lambda: object())
    validator_registry.register("forecast", lambda: object())

    class DummyScheduler:
        def states(self) -> list[MlopsSchedulerJobStatus]:
            return [
                MlopsSchedulerJobStatus(
                    model_type="xgboost",
                    target_type="global",
                    next_run_at=datetime(2026, 1, 1, 12, 0, 0),
                    last_submitted_job_id="job-1",
                )
            ]

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(training_scheduler=DummyScheduler())))
    monkeypatch.setattr(mlops, "trainer_registry", trainer_registry)
    monkeypatch.setattr(mlops, "data_processor_registry", data_processor_registry)
    monkeypatch.setattr(mlops, "prediction_input_validator_registry", validator_registry)
    monkeypatch.setattr(
        mlops,
        "settings",
        SimpleNamespace(training_job_store="postgres", prediction_log_store="postgres", mlops_event_store="postgres"),
    )
    monkeypatch.setattr(
        mlops,
        "mlops_settings",
        SimpleNamespace(
            enable_scheduled_retraining=True,
            scheduled_retraining_jobs=(
                '[{"model_type":"xgboost","interval_seconds":3600,'
                '"train_window_hours":24,"validation_window_hours":6}]'
            ),
            notification_sink="webhook",
            notification_webhook_url="https://example.com/hook",
            drift_min_samples=30,
            drift_metric_name="mape",
            drift_max_mean_error_value=10.0,
            drift_max_mean_metric_value=15.0,
        ),
    )

    result = await mlops.get_mlops_status(request)

    assert result.registries.trainers == ["xgboost"]
    assert result.registries.data_processors == ["xgboost"]
    assert result.registries.prediction_input_validators == ["forecast"]
    assert result.stores.training_job_store == "postgres"
    assert result.stores.mlops_event_store == "postgres"
    assert result.scheduler.enabled
    assert result.scheduler.active
    assert result.scheduler.configured_jobs == 1
    assert result.scheduler.jobs[0].last_submitted_job_id == "job-1"
    assert result.notifications.sink == "webhook"
    assert result.notifications.webhook_configured
    assert result.drift.max_mean_metric_value == 15.0


@pytest.mark.asyncio
async def test_get_mlops_events_returns_recent_events(monkeypatch: Any) -> None:
    expected = [
        MlopsEventRecord(
            event_id="event-1",
            event_type="drift_detected",
            severity="warning",
            message="Model drift detected.",
            occurred_at=datetime(2026, 1, 1, 12, 0, 0),
            payload={"model_name": "xgboost_global"},
        )
    ]

    class DummyEventStore:
        def list_recent(self, limit: int = 100, event_type: str | None = None) -> list[MlopsEventRecord]:
            assert limit == 10
            assert event_type == "drift_detected"
            return expected

    monkeypatch.setattr(mlops, "mlops_event_store", DummyEventStore())

    result = await mlops.get_mlops_events(limit=10, event_type="drift_detected")

    assert result == expected


@pytest.mark.asyncio
async def test_tick_scheduler_returns_dry_run_contexts() -> None:
    class DummyScheduler:
        def dry_run(self, now: datetime | None = None) -> list[TrainingContext]:
            assert now == datetime(2026, 1, 2, 12, 0, 0)
            return [_context()]

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(training_scheduler=DummyScheduler())))

    result = await mlops.tick_scheduler(
        request,
        MlopsSchedulerTickRequest(dry_run=True, now=datetime(2026, 1, 2, 12, 0, 0)),
    )

    assert result.dry_run
    assert result.due_jobs == 1
    assert result.submitted_jobs == 0
    assert result.preview_contexts[0].model_type == "xgboost"


@pytest.mark.asyncio
async def test_tick_scheduler_runs_active_scheduler() -> None:
    class DummyScheduler:
        def tick(self, now: datetime | None = None) -> list[MlopsSchedulerJobStatus]:
            assert now == datetime(2026, 1, 2, 12, 0, 0)
            return [
                MlopsSchedulerJobStatus(
                    model_type="xgboost",
                    target_type="global",
                    next_run_at=datetime(2026, 1, 2, 13, 0, 0),
                    last_submitted_job_id="job-1",
                    last_submitted_at=now,
                )
            ]

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(training_scheduler=DummyScheduler())))

    result = await mlops.tick_scheduler(
        request,
        MlopsSchedulerTickRequest(now=datetime(2026, 1, 2, 12, 0, 0)),
    )

    assert not result.dry_run
    assert result.due_jobs == 1
    assert result.submitted_jobs == 1
    assert result.jobs[0].last_submitted_job_id == "job-1"


@pytest.mark.asyncio
async def test_tick_scheduler_returns_409_when_scheduler_is_not_active() -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    with pytest.raises(HTTPException) as exc:
        await mlops.tick_scheduler(request, MlopsSchedulerTickRequest())

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_send_test_notification_dispatches_event(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(mlops.notification_dispatcher, "notify", lambda event: captured.setdefault("event", event))

    result = await mlops.send_test_notification(
        MlopsNotificationTestRequest(
            event_type="notification_test",
            severity="warning",
            message="test",
            payload={"source": "api"},
        )
    )

    assert result.dispatched
    assert result.event_type == "notification_test"
    assert captured["event"].severity == "warning"
    assert captured["event"].payload == {"source": "api"}


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
