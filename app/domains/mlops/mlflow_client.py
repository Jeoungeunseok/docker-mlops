from contextlib import contextmanager
from typing import Any, Iterator

import mlflow
from mlflow.entities import Run

from app.domains.mlops.config import mlops_settings


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(mlops_settings.mlflow_tracking_uri)
    mlflow.set_experiment(mlops_settings.mlflow_experiment)


@contextmanager
def start_run(run_name: str | None = None, tags: dict[str, str] | None = None) -> Iterator[Run]:
    configure_mlflow()
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def log_params(params: dict[str, Any]) -> None:
    clean_params = {key: value for key, value in params.items() if value is not None}
    if clean_params:
        mlflow.log_params(clean_params)


def log_metrics(metrics: dict[str, float]) -> None:
    clean_metrics = {key: value for key, value in metrics.items() if value is not None}
    if clean_metrics:
        mlflow.log_metrics(clean_metrics)


def log_tags(tags: dict[str, str]) -> None:
    if tags:
        mlflow.set_tags(tags)


def log_artifact(path: str, artifact_path: str | None = None) -> None:
    mlflow.log_artifact(path, artifact_path=artifact_path)
