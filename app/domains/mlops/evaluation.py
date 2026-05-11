from app.domains.mlops.config import mlops_settings
from app.domains.mlops.schemas import EvaluationMetrics


def is_candidate_promotable(
    candidate: EvaluationMetrics,
    champion_metric_value: float | None,
    metric_name: str | None = None,
) -> bool:
    if candidate.validation_samples < mlops_settings.min_validation_samples:
        return False

    if candidate.mape > mlops_settings.max_mape_for_promotion:
        return False

    selected_metric = metric_name or mlops_settings.metric_for_promotion
    candidate_value = getattr(candidate, selected_metric)
    if champion_metric_value is None:
        return True

    return candidate_value <= champion_metric_value
