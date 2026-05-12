from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.logging import app_logger
from app.core.exceptions import ModelNotLoadedError
from app.core.timezone import now_in_app_timezone
from app.domains.mlops.config import mlops_settings
from app.domains.mlops.event_store import mlops_event_store
from app.domains.mlops.model_loader import model_loader
from app.domains.mlops.model_registry import get_model_by_alias, rollback_champion
from app.domains.mlops.drift import check_model_drift, default_drift_check_request
from app.domains.mlops.notifications import NotificationEvent, notification_dispatcher
from app.domains.mlops.registry import (
    data_processor_registry,
    prediction_input_validator_registry,
    trainer_registry,
)
from app.domains.mlops.scheduler import build_scheduled_training_jobs
from app.domains.mlops.schemas import (
    DriftCheckResult,
    MlopsDriftStatus,
    MlopsEventRecord,
    MlopsNotificationStatus,
    MlopsNotificationTestRequest,
    MlopsNotificationTestResult,
    MlopsRegistryStatus,
    MlopsSchedulerStatus,
    MlopsSchedulerTickRequest,
    MlopsSchedulerTickResult,
    MlopsStatus,
    MlopsStoreStatus,
    ModelLoadResult,
    ModelRegistryInfo,
    ModelRollbackRequest,
    ModelRollbackResult,
    TrainingContext,
    TrainingJobRecord,
    TrainingResult,
)
from app.domains.prediction.log_store import prediction_log_store
from app.domains.mlops.training_jobs import training_job_runner
from app.jobs.train_model_job import train_model_job

router = APIRouter()


@router.get("/status", response_model=MlopsStatus)
async def get_mlops_status(request: Request) -> MlopsStatus:
    scheduler = getattr(request.app.state, "training_scheduler", None)
    scheduler_status = _build_scheduler_status(scheduler)
    return MlopsStatus(
        registries=MlopsRegistryStatus(
            trainers=trainer_registry.registered_model_types(),
            data_processors=data_processor_registry.registered_model_types(),
            prediction_input_validators=prediction_input_validator_registry.registered_model_types(),
        ),
        stores=MlopsStoreStatus(
            training_job_store=settings.training_job_store,
            prediction_log_store=settings.prediction_log_store,
            mlops_event_store=settings.mlops_event_store,
        ),
        scheduler=scheduler_status,
        notifications=MlopsNotificationStatus(
            sink=mlops_settings.notification_sink,
            webhook_configured=bool(mlops_settings.notification_webhook_url),
        ),
        drift=MlopsDriftStatus(
            min_samples=mlops_settings.drift_min_samples,
            metric_name=mlops_settings.drift_metric_name,
            max_mean_error_value=mlops_settings.drift_max_mean_error_value,
            max_mean_metric_value=mlops_settings.drift_max_mean_metric_value,
        ),
    )


@router.get("/events", response_model=list[MlopsEventRecord])
async def get_mlops_events(limit: int = 100, event_type: str | None = None) -> list[MlopsEventRecord]:
    return await run_in_threadpool(mlops_event_store.list_recent, limit, event_type)


@router.post("/scheduler/tick", response_model=MlopsSchedulerTickResult)
async def tick_scheduler(request: Request, payload: MlopsSchedulerTickRequest) -> MlopsSchedulerTickResult:
    scheduler = getattr(request.app.state, "training_scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=409, detail="Scheduled retraining scheduler is not active.")
    checked_at = payload.now or now_in_app_timezone()
    if payload.dry_run:
        preview_contexts = await run_in_threadpool(scheduler.dry_run, checked_at)
        return MlopsSchedulerTickResult(
            dry_run=True,
            checked_at=checked_at,
            due_jobs=len(preview_contexts),
            preview_contexts=preview_contexts,
        )

    submitted_states = await run_in_threadpool(scheduler.tick, checked_at)
    failed_jobs = sum(1 for state in submitted_states if state.last_error_message is not None)
    return MlopsSchedulerTickResult(
        dry_run=False,
        checked_at=checked_at,
        due_jobs=len(submitted_states),
        submitted_jobs=len(submitted_states) - failed_jobs,
        failed_jobs=failed_jobs,
        jobs=submitted_states,
    )


@router.post("/notifications/test", response_model=MlopsNotificationTestResult)
async def send_test_notification(payload: MlopsNotificationTestRequest) -> MlopsNotificationTestResult:
    notification_dispatcher.notify(
        NotificationEvent(
            event_type=payload.event_type,
            severity=payload.severity,
            message=payload.message,
            payload=payload.payload,
        )
    )
    return MlopsNotificationTestResult(
        event_type=payload.event_type,
        severity=payload.severity,
        dispatched=True,
    )


@router.post("/training-jobs", response_model=TrainingResult)
async def create_training_job(context: TrainingContext) -> TrainingResult:
    app_logger.info(
        "Starting training job from API",
        extra={"model_type": context.model_type, "target_type": context.target_type, "target_id": context.target_id},
    )
    try:
        return await run_in_threadpool(train_model_job, context)
    except KeyError as exc:
        app_logger.warning(
            "Training job failed because model_type is not registered",
            extra={"model_type": context.model_type},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        app_logger.exception("Training job failed", extra={"model_type": context.model_type})
        raise HTTPException(status_code=502, detail=f"Failed to run training job: {exc}") from exc


@router.post("/training-jobs/async", response_model=TrainingJobRecord, status_code=202)
async def submit_async_training_job(context: TrainingContext, max_attempts: int = 1) -> TrainingJobRecord:
    app_logger.info(
        "Submitting async training job from API",
        extra={"model_type": context.model_type, "target_type": context.target_type, "target_id": context.target_id},
    )
    if max_attempts < 1:
        raise HTTPException(status_code=400, detail="max_attempts must be greater than or equal to 1.")
    return await run_in_threadpool(training_job_runner.submit, context, max_attempts)


@router.get("/training-jobs/{job_id}", response_model=TrainingJobRecord)
async def get_training_job(job_id: str) -> TrainingJobRecord:
    job = await run_in_threadpool(training_job_runner.get, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job was not found.")
    return job


@router.post("/training-jobs/{job_id}/retry", response_model=TrainingJobRecord, status_code=202)
async def retry_training_job(job_id: str) -> TrainingJobRecord:
    try:
        return await run_in_threadpool(training_job_runner.retry, job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/models/{model_name}", response_model=ModelRegistryInfo)
async def get_model_status(model_name: str) -> ModelRegistryInfo:
    app_logger.info("Fetching champion model status", extra={"model_name": model_name})
    model = await run_in_threadpool(get_model_by_alias, model_name)
    if model is None:
        app_logger.warning("Champion model was not found", extra={"model_name": model_name})
        raise HTTPException(status_code=404, detail="Champion model was not found.")
    return model


@router.get("/models/{model_name}/loaded", response_model=ModelLoadResult)
async def get_loaded_model(model_name: str) -> ModelLoadResult:
    app_logger.info("Fetching loaded model status", extra={"model_name": model_name})
    loaded = await run_in_threadpool(model_loader.status, model_name)
    if loaded is None:
        app_logger.warning("Model is not loaded in API process", extra={"model_name": model_name})
        raise HTTPException(status_code=404, detail="Model is not loaded in this API process.")
    return loaded


@router.post("/models/{model_name}/reload", response_model=ModelLoadResult)
async def reload_model(model_name: str) -> ModelLoadResult:
    app_logger.info("Reloading champion model", extra={"model_name": model_name})
    try:
        loaded = await run_in_threadpool(model_loader.load, model_name, True)
        app_logger.info(
            "Champion model reloaded",
            extra={"model_name": model_name, "model_version": loaded.version, "run_id": loaded.run_id},
        )
        return loaded
    except ModelNotLoadedError as exc:
        app_logger.warning("Model reload failed because model is not loaded", extra={"model_name": model_name})
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        app_logger.exception("Model reload failed", extra={"model_name": model_name})
        notification_dispatcher.notify(
            NotificationEvent(
                event_type="model_reload_failed",
                severity="error",
                message="Model reload failed.",
                payload={"model_name": model_name, "error_message": str(exc)},
            )
        )
        raise HTTPException(status_code=502, detail=f"Failed to reload model: {exc}") from exc


@router.post("/models/{model_name}/rollback", response_model=ModelRollbackResult)
async def rollback_model(model_name: str, request: ModelRollbackRequest) -> ModelRollbackResult:
    app_logger.info("Rolling back champion model", extra={"model_name": model_name, "version": request.version})
    try:
        result = await run_in_threadpool(rollback_champion, model_name, request.version)
        loaded = await run_in_threadpool(model_loader.load, model_name, True)
        app_logger.info(
            "Champion model reloaded after rollback",
            extra={"model_name": model_name, "model_version": loaded.version, "run_id": loaded.run_id},
        )
        notification_dispatcher.notify(
            NotificationEvent(
                event_type="rollback_completed",
                severity="warning",
                message="Champion model rollback completed.",
                payload={
                    "model_name": model_name,
                    "champion_version": result.champion_version,
                    "loaded_version": loaded.version,
                    "run_id": loaded.run_id,
                },
            )
        )
        return result
    except Exception as exc:
        app_logger.exception(
            "Model rollback failed",
            extra={"model_name": model_name, "version": request.version},
        )
        raise HTTPException(status_code=502, detail=f"Failed to rollback model: {exc}") from exc


@router.get("/models/{model_name}/drift", response_model=DriftCheckResult)
async def get_model_drift(
    model_name: str,
    limit: int = 100,
    min_samples: int | None = None,
    max_mean_error_value: float | None = None,
    metric_name: str | None = None,
    max_mean_metric_value: float | None = None,
) -> DriftCheckResult:
    app_logger.info("Checking model drift", extra={"model_name": model_name})
    try:
        request = default_drift_check_request(
            limit=limit,
            min_samples=min_samples,
            max_mean_error_value=max_mean_error_value,
            metric_name=metric_name,
            max_mean_metric_value=max_mean_metric_value,
        )
        logs = await run_in_threadpool(prediction_log_store.list_by_model, model_name)
        result = await run_in_threadpool(check_model_drift, model_name, logs, request)
        if result.drift_detected:
            notification_dispatcher.notify(
                NotificationEvent(
                    event_type="drift_detected",
                    severity="warning",
                    message="Model drift detected.",
                    payload=result.model_dump(mode="json"),
                )
            )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        app_logger.exception("Model drift check failed", extra={"model_name": model_name})
        raise HTTPException(status_code=502, detail=f"Failed to check model drift: {exc}") from exc


def _build_scheduler_status(scheduler: object | None) -> MlopsSchedulerStatus:
    if not mlops_settings.enable_scheduled_retraining:
        return MlopsSchedulerStatus(enabled=False, active=False, config_valid=True)

    try:
        configured_jobs = len(build_scheduled_training_jobs(mlops_settings.scheduled_retraining_jobs))
    except Exception as exc:
        return MlopsSchedulerStatus(
            enabled=True,
            active=False,
            config_valid=False,
            config_error=str(exc),
        )

    if scheduler is None:
        return MlopsSchedulerStatus(
            enabled=True,
            active=False,
            config_valid=True,
            configured_jobs=configured_jobs,
        )

    return MlopsSchedulerStatus(
        enabled=True,
        active=True,
        config_valid=True,
        configured_jobs=configured_jobs,
        jobs=scheduler.states(),
    )
