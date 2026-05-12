from datetime import datetime

from app.domains.mlops.drift import check_model_drift
from app.domains.mlops.schemas import DriftCheckRequest, PredictionLogPayload


def test_check_model_drift_returns_not_enough_samples() -> None:
    request = DriftCheckRequest(min_samples=2, max_mean_error_value=10.0)

    result = check_model_drift(
        "xgboost_global",
        [_payload(error_value=12.0, actual_value=10.0)],
        request,
    )

    assert not result.drift_detected
    assert result.evaluated_samples == 1
    assert "Not enough" in result.reason


def test_check_model_drift_detects_mean_error_threshold() -> None:
    request = DriftCheckRequest(min_samples=2, max_mean_error_value=10.0)

    result = check_model_drift(
        "xgboost_global",
        [
            _payload(error_value=12.0, actual_value=10.0),
            _payload(error_value=14.0, actual_value=10.0),
        ],
        request,
    )

    assert result.drift_detected
    assert result.evaluated_samples == 2
    assert result.mean_error_value == 13.0
    assert result.reason == "Mean error_value exceeded threshold."


def test_check_model_drift_detects_named_metric_threshold() -> None:
    request = DriftCheckRequest(min_samples=2, metric_name="mape", max_mean_metric_value=15.0)

    result = check_model_drift(
        "xgboost_global",
        [
            _payload(error_metrics={"mape": 20.0}, actual_value=10.0),
            _payload(error_metrics={"mape": 18.0}, actual_value=10.0),
        ],
        request,
    )

    assert result.drift_detected
    assert result.mean_metric_value == 19.0
    assert result.reason == "Mean mape exceeded threshold."


def test_check_model_drift_ignores_logs_without_actuals() -> None:
    request = DriftCheckRequest(min_samples=1, max_mean_error_value=10.0)

    result = check_model_drift(
        "xgboost_global",
        [
            _payload(error_value=99.0, actual_value=None),
            _payload(error_value=5.0, actual_value=10.0),
        ],
        request,
    )

    assert not result.drift_detected
    assert result.evaluated_samples == 1
    assert result.mean_error_value == 5.0


def _payload(
    error_value: float | None = None,
    actual_value: object | None = 10.0,
    error_metrics: dict[str, float] | None = None,
) -> PredictionLogPayload:
    return PredictionLogPayload(
        model_name="xgboost_global",
        predicted_at=datetime(2026, 1, 1, 12, 0, 0),
        target_timestamp=datetime(2026, 1, 1, 13, 0, 0),
        predicted_value=11.0,
        actual_value=actual_value,
        error_value=error_value,
        error_metrics=error_metrics or {},
    )
