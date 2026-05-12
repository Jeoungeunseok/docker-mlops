from app.domains.mlops.config import mlops_settings
from app.domains.mlops.schemas import DriftCheckRequest, DriftCheckResult, PredictionLogPayload


def default_drift_check_request(
    limit: int = 100,
    min_samples: int | None = None,
    max_mean_error_value: float | None = None,
    metric_name: str | None = None,
    max_mean_metric_value: float | None = None,
) -> DriftCheckRequest:
    return DriftCheckRequest(
        limit=limit,
        min_samples=min_samples or mlops_settings.drift_min_samples,
        max_mean_error_value=(
            max_mean_error_value
            if max_mean_error_value is not None
            else mlops_settings.drift_max_mean_error_value
        ),
        metric_name=metric_name or mlops_settings.drift_metric_name,
        max_mean_metric_value=(
            max_mean_metric_value
            if max_mean_metric_value is not None
            else mlops_settings.drift_max_mean_metric_value
        ),
    )


def check_model_drift(
    model_name: str,
    logs: list[PredictionLogPayload],
    request: DriftCheckRequest,
) -> DriftCheckResult:
    evaluated_logs = [
        payload
        for payload in logs[: request.limit]
        if payload.actual_value is not None
        and (payload.error_value is not None or request.metric_name in payload.error_metrics)
    ]
    evaluated_samples = len(evaluated_logs)
    mean_error_value = _mean([payload.error_value for payload in evaluated_logs if payload.error_value is not None])
    mean_metric_value = _mean(
        [
            payload.error_metrics[request.metric_name]
            for payload in evaluated_logs
            if request.metric_name in payload.error_metrics
        ]
    )

    if evaluated_samples < request.min_samples:
        return DriftCheckResult(
            model_name=model_name,
            drift_detected=False,
            evaluated_samples=evaluated_samples,
            min_samples=request.min_samples,
            mean_error_value=mean_error_value,
            max_mean_error_value=request.max_mean_error_value,
            metric_name=request.metric_name,
            mean_metric_value=mean_metric_value,
            max_mean_metric_value=request.max_mean_metric_value,
            reason="Not enough actual-labeled prediction logs to evaluate drift.",
        )

    if request.max_mean_error_value is not None and mean_error_value is not None:
        if mean_error_value > request.max_mean_error_value:
            return DriftCheckResult(
                model_name=model_name,
                drift_detected=True,
                evaluated_samples=evaluated_samples,
                min_samples=request.min_samples,
                mean_error_value=mean_error_value,
                max_mean_error_value=request.max_mean_error_value,
                metric_name=request.metric_name,
                mean_metric_value=mean_metric_value,
                max_mean_metric_value=request.max_mean_metric_value,
                reason="Mean error_value exceeded threshold.",
            )

    if request.max_mean_metric_value is not None and mean_metric_value is not None:
        if mean_metric_value > request.max_mean_metric_value:
            return DriftCheckResult(
                model_name=model_name,
                drift_detected=True,
                evaluated_samples=evaluated_samples,
                min_samples=request.min_samples,
                mean_error_value=mean_error_value,
                max_mean_error_value=request.max_mean_error_value,
                metric_name=request.metric_name,
                mean_metric_value=mean_metric_value,
                max_mean_metric_value=request.max_mean_metric_value,
                reason=f"Mean {request.metric_name} exceeded threshold.",
            )

    return DriftCheckResult(
        model_name=model_name,
        drift_detected=False,
        evaluated_samples=evaluated_samples,
        min_samples=request.min_samples,
        mean_error_value=mean_error_value,
        max_mean_error_value=request.max_mean_error_value,
        metric_name=request.metric_name,
        mean_metric_value=mean_metric_value,
        max_mean_metric_value=request.max_mean_metric_value,
        reason="No drift threshold was exceeded.",
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
