from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    model_name: str = Field(examples=["classifier_customer_42_region_apac"])
    request_id: str | None = None
    target_type: str = "global"
    target_id: str | None = None
    qualifiers: dict[str, str | int | float | bool] = Field(default_factory=dict)
    target_timestamp: datetime | None = None
    inputs: list[dict[str, Any]]


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str | None = None
    run_id: str | None = None
    request_id: str | None = None
    predicted_at: datetime
    predictions: Any


class PredictionActualUpdateRequest(BaseModel):
    actual_value: Any
    error_value: float | None = None
    error_metrics: dict[str, float] = Field(default_factory=dict)
