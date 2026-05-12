import mlflow
from mlflow import MlflowClient

from app.core.logging import app_logger
from app.domains.mlops.config import mlops_settings
from app.domains.mlops.notifications import NotificationEvent, notification_dispatcher
from app.domains.mlops.schemas import EvaluationMetrics, ModelRegistryInfo, ModelRollbackResult


def build_model_name(
    model_type: str,
    target_type: str = "global",
    target_id: str | None = None,
    qualifiers: dict[str, str | int | float | bool] | None = None,
) -> str:
    parts = [model_type]
    if target_id is not None:
        parts.extend([target_type, str(target_id)])
        for key, value in sorted((qualifiers or {}).items()):
            parts.extend([key, str(value)])
    else:
        parts.append("global")
    return "_".join(parts)


def get_client() -> MlflowClient:
    return MlflowClient(tracking_uri=mlops_settings.mlflow_tracking_uri)


def register_model(model_uri: str, model_name: str) -> ModelRegistryInfo:
    mlflow.set_tracking_uri(mlops_settings.mlflow_tracking_uri)
    app_logger.info("Registering model in MLflow", extra={"model_name": model_name, "model_uri": model_uri})
    try:
        version = mlflow.register_model(model_uri=model_uri, name=model_name)
    except Exception:
        app_logger.exception("Failed to register model in MLflow", extra={"model_name": model_name})
        raise
    app_logger.info(
        "Model registered in MLflow",
        extra={"model_name": version.name, "model_version": version.version, "run_id": version.run_id},
    )
    return ModelRegistryInfo(
        name=version.name,
        version=version.version,
        run_id=version.run_id,
        source=version.source,
    )


def set_alias(model_name: str, version: str, alias: str) -> None:
    try:
        get_client().set_registered_model_alias(model_name, alias, version)
    except Exception:
        app_logger.exception(
            "Failed to set MLflow model alias",
            extra={"model_name": model_name, "model_version": version, "alias": alias},
        )
        raise
    app_logger.info(
        "MLflow model alias updated",
        extra={"model_name": model_name, "model_version": version, "alias": alias},
    )


def set_candidate(model_name: str, version: str) -> None:
    set_alias(model_name, version, mlops_settings.model_candidate_alias)


def promote_to_champion(model_name: str, version: str) -> None:
    set_alias(model_name, version, mlops_settings.model_champion_alias)
    app_logger.info(
        "Model promoted to champion",
        extra={"model_name": model_name, "model_version": version},
    )
    notification_dispatcher.notify(
        NotificationEvent(
            event_type="champion_promoted",
            severity="info",
            message="Model promoted to champion.",
            payload={"model_name": model_name, "model_version": version},
        )
    )


def rollback_champion(model_name: str, version: str) -> ModelRollbackResult:
    set_alias(model_name, version, mlops_settings.model_champion_alias)
    app_logger.info(
        "Model champion rolled back",
        extra={"model_name": model_name, "model_version": version},
    )
    return ModelRollbackResult(model_name=model_name, champion_version=version)


def get_model_by_alias(model_name: str, alias: str | None = None) -> ModelRegistryInfo | None:
    selected_alias = alias or mlops_settings.model_champion_alias
    client = get_client()
    try:
        version = client.get_model_version_by_alias(model_name, selected_alias)
    except Exception:
        app_logger.exception(
            "Failed to fetch MLflow model by alias",
            extra={"model_name": model_name, "alias": selected_alias},
        )
        return None
    return ModelRegistryInfo(
        name=version.name,
        version=version.version,
        run_id=version.run_id,
        source=version.source,
        aliases=list(version.aliases or []),
    )


def get_champion_metric(model_name: str, metric_name: str) -> float | None:
    champion = get_model_by_alias(model_name, mlops_settings.model_champion_alias)
    if champion is None or champion.run_id is None:
        return None

    try:
        run = get_client().get_run(champion.run_id)
    except Exception:
        app_logger.exception(
            "Failed to fetch champion MLflow run",
            extra={"model_name": model_name, "run_id": champion.run_id},
        )
        raise
    metric = run.data.metrics.get(metric_name)
    return float(metric) if metric is not None else None


def should_promote(model_name: str, metrics: EvaluationMetrics) -> bool:
    from app.domains.mlops.evaluation import is_candidate_promotable

    champion_metric = get_champion_metric(model_name, mlops_settings.metric_for_promotion)
    return is_candidate_promotable(metrics, champion_metric)
