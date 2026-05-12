from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class TrainingContext(BaseModel):
    model_type: str
    target_type: str = "global"
    target_id: str | None = None
    qualifiers: dict[str, str | int | float | bool] = Field(default_factory=dict)
    train_start_at: datetime
    train_end_at: datetime
    validation_start_at: datetime
    validation_end_at: datetime
    extra_params: dict[str, Any] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    mae: float
    rmse: float
    mape: float
    peak_error: float | None = None
    validation_samples: int

    def as_mlflow_metrics(self) -> dict[str, float]:
        metrics = {
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "validation_samples": float(self.validation_samples),
        }
        if self.peak_error is not None:
            metrics["peak_error"] = self.peak_error
        return metrics


class ModelRegistryInfo(BaseModel):
    name: str
    version: str
    run_id: str | None = None
    source: str | None = None
    aliases: list[str] = Field(default_factory=list)


class TrainingResult(BaseModel):
    model_name: str
    run_id: str
    model_uri: str
    metrics: EvaluationMetrics
    registered_model: ModelRegistryInfo | None = None
    promoted: bool = False


class TrainingJobRecord(BaseModel):
    job_id: str
    status: Literal["pending", "running", "succeeded", "failed"]
    context: TrainingContext
    attempts: int = 0
    max_attempts: int = 1
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: TrainingResult | None = None
    error_message: str | None = None


class TrainingDataset(BaseModel):
    train_features: Any
    train_labels: Any | None = None
    validation_features: Any
    validation_labels: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelLoadResult(BaseModel):
    model_name: str
    model_uri: str
    version: str | None = None
    run_id: str | None = None
    loaded_at: datetime


class ModelRollbackRequest(BaseModel):
    version: str


class ModelRollbackResult(BaseModel):
    model_name: str
    champion_version: str


class DriftCheckRequest(BaseModel):
    limit: int = Field(default=100, ge=1)
    min_samples: int = Field(default=30, ge=1)
    max_mean_error_value: float | None = None
    metric_name: str = "mape"
    max_mean_metric_value: float | None = None


class DriftCheckResult(BaseModel):
    model_name: str
    drift_detected: bool
    evaluated_samples: int
    min_samples: int
    mean_error_value: float | None = None
    max_mean_error_value: float | None = None
    metric_name: str
    mean_metric_value: float | None = None
    max_mean_metric_value: float | None = None
    reason: str


class MlopsStoreStatus(BaseModel):
    training_job_store: str
    prediction_log_store: str
    mlops_event_store: str


class MlopsRegistryStatus(BaseModel):
    trainers: list[str] = Field(default_factory=list)
    data_processors: list[str] = Field(default_factory=list)
    prediction_input_validators: list[str] = Field(default_factory=list)


class MlopsSchedulerJobStatus(BaseModel):
    model_type: str
    target_type: str
    target_id: str | None = None
    next_run_at: datetime
    last_submitted_job_id: str | None = None
    last_submitted_at: datetime | None = None
    last_error_message: str | None = None


class MlopsSchedulerStatus(BaseModel):
    enabled: bool
    active: bool
    config_valid: bool
    configured_jobs: int = 0
    config_error: str | None = None
    jobs: list[MlopsSchedulerJobStatus] = Field(default_factory=list)


class MlopsNotificationStatus(BaseModel):
    sink: str
    webhook_configured: bool = False


class MlopsDriftStatus(BaseModel):
    min_samples: int
    metric_name: str
    max_mean_error_value: float | None = None
    max_mean_metric_value: float | None = None


class MlopsStatus(BaseModel):
    registries: MlopsRegistryStatus
    stores: MlopsStoreStatus
    scheduler: MlopsSchedulerStatus
    notifications: MlopsNotificationStatus
    drift: MlopsDriftStatus


class MlopsEventRecord(BaseModel):
    event_id: str
    event_type: str
    severity: str
    message: str
    occurred_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class PredictionLogPayload(BaseModel):
    model_name: str
    model_version: str | None = None
    run_id: str | None = None
    request_id: str | None = None
    target_type: str = "global"
    target_id: str | None = None
    qualifiers: dict[str, str | int | float | bool] = Field(default_factory=dict)
    predicted_at: datetime
    target_timestamp: datetime
    predicted_value: Any
    actual_value: Any | None = None
    error_value: float | None = None
    error_metrics: dict[str, float] = Field(default_factory=dict)
    input_features: dict[str, Any] = Field(default_factory=dict)
    output_metadata: dict[str, Any] = Field(default_factory=dict)


class ModelTrainer(Protocol):
    def train_model(self, context: TrainingContext) -> Any:
        ...

    def evaluate_model(self, model: Any, context: TrainingContext) -> EvaluationMetrics:
        ...

    def log_model(self, model: Any, artifact_path: str) -> str:
        ...


class TrainingDataProcessor(Protocol):
    def load_training_data(self, context: TrainingContext) -> Any:
        ...

    def preprocess(self, raw_data: Any, context: TrainingContext) -> Any:
        ...

    def build_features(self, processed_data: Any, context: TrainingContext) -> Any:
        ...

    def split_validation(self, features: Any, context: TrainingContext) -> TrainingDataset:
        ...
