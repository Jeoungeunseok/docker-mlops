from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MlopsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_experiment: str = Field(default="default-model-serving", alias="MLFLOW_EXPERIMENT")
    model_champion_alias: str = Field(default="champion", alias="MODEL_CHAMPION_ALIAS")
    model_candidate_alias: str = Field(default="candidate", alias="MODEL_CANDIDATE_ALIAS")

    min_validation_samples: int = Field(default=100, alias="MLOPS_MIN_VALIDATION_SAMPLES")
    max_mape_for_promotion: float = Field(default=15.0, alias="MLOPS_MAX_MAPE_FOR_PROMOTION")
    metric_for_promotion: str = Field(default="rmse", alias="MLOPS_METRIC_FOR_PROMOTION")
    enable_scheduled_retraining: bool = Field(default=False, alias="MLOPS_ENABLE_SCHEDULED_RETRAINING")
    scheduled_retraining_jobs: str = Field(default="[]", alias="MLOPS_SCHEDULED_RETRAINING_JOBS")
    drift_min_samples: int = Field(default=30, alias="MLOPS_DRIFT_MIN_SAMPLES")
    drift_max_mean_error_value: float | None = Field(default=None, alias="MLOPS_DRIFT_MAX_MEAN_ERROR_VALUE")
    drift_metric_name: str = Field(default="mape", alias="MLOPS_DRIFT_METRIC_NAME")
    drift_max_mean_metric_value: float | None = Field(default=None, alias="MLOPS_DRIFT_MAX_MEAN_METRIC_VALUE")


@lru_cache
def get_mlops_settings() -> MlopsSettings:
    return MlopsSettings()


mlops_settings = get_mlops_settings()
