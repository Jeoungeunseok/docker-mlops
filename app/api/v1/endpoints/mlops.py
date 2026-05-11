from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.core.logging import app_logger
from app.core.exceptions import ModelNotLoadedError
from app.domains.mlops.model_loader import model_loader
from app.domains.mlops.model_registry import get_model_by_alias, rollback_champion
from app.domains.mlops.schemas import (
    ModelLoadResult,
    ModelRegistryInfo,
    ModelRollbackRequest,
    ModelRollbackResult,
    TrainingContext,
    TrainingJobRecord,
    TrainingResult,
)
from app.domains.mlops.training_jobs import training_job_runner
from app.jobs.train_model_job import train_model_job

router = APIRouter()


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
        raise HTTPException(status_code=502, detail=f"Failed to reload model: {exc}") from exc


@router.post("/models/{model_name}/rollback", response_model=ModelRollbackResult)
async def rollback_model(model_name: str, request: ModelRollbackRequest) -> ModelRollbackResult:
    app_logger.info("Rolling back champion model", extra={"model_name": model_name, "version": request.version})
    try:
        return await run_in_threadpool(rollback_champion, model_name, request.version)
    except Exception as exc:
        app_logger.exception(
            "Model rollback failed",
            extra={"model_name": model_name, "version": request.version},
        )
        raise HTTPException(status_code=502, detail=f"Failed to rollback model: {exc}") from exc
