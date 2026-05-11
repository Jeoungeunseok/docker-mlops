from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="MLOps API")
    app_version: str = Field(default="0.1.0")
    app_env: str = Field(default="local")
    app_timezone: str = Field(default="Asia/Seoul")
    enable_docs: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    log_to_file: bool = Field(default=False)
    log_file_path: str = Field(default="logs/app.log")
    log_retention_days: int = Field(default=30)
    app_database_url: str | None = Field(default=None, alias="APP_DATABASE_URL")
    prediction_log_store: str = Field(default="in_memory", alias="PREDICTION_LOG_STORE")


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()


settings = get_settings()
