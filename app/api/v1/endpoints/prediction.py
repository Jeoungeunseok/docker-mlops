from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.domains.mlops.schemas import PredictionLogPayload
from app.domains.prediction.schemas import PredictionActualUpdateRequest, PredictionRequest, PredictionResponse
from app.domains.prediction.services.prediction_service import list_prediction_logs, predict, update_prediction_actual

router = APIRouter()


@router.post("", response_model=PredictionResponse)
async def create_prediction(request: PredictionRequest) -> PredictionResponse:
    return await run_in_threadpool(predict, request)


@router.patch("/{request_id}/actual", status_code=204)
async def update_actual_value(request_id: str, request: PredictionActualUpdateRequest) -> None:
    await run_in_threadpool(update_prediction_actual, request_id, request)


@router.get("/logs", response_model=list[PredictionLogPayload])
async def get_prediction_logs(model_name: str) -> list[PredictionLogPayload]:
    return await run_in_threadpool(list_prediction_logs, model_name)
