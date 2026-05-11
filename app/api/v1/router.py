from fastapi import APIRouter

from app.api.v1.endpoints import health, mlops, prediction

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(prediction.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(mlops.router, prefix="/mlops", tags=["mlops"])
