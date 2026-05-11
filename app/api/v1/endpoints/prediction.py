from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.domains.prediction.schemas import PredictionRequest, PredictionResponse
from app.domains.prediction.services.prediction_service import predict

router = APIRouter()


@router.post("", response_model=PredictionResponse)
async def create_prediction(request: PredictionRequest) -> PredictionResponse:
    return await run_in_threadpool(predict, request)
