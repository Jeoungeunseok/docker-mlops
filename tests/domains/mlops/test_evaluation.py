from app.domains.mlops.evaluation import is_candidate_promotable
from app.domains.mlops.schemas import EvaluationMetrics


def test_candidate_is_promotable_without_champion() -> None:
    metrics = EvaluationMetrics(mae=1.0, rmse=2.0, mape=10.0, validation_samples=100)

    assert is_candidate_promotable(metrics, champion_metric_value=None)


def test_candidate_is_not_promotable_when_mape_is_too_high() -> None:
    metrics = EvaluationMetrics(mae=1.0, rmse=2.0, mape=99.0, validation_samples=100)

    assert not is_candidate_promotable(metrics, champion_metric_value=None)
