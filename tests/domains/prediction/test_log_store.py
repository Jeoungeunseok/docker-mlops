from datetime import datetime

from app.domains.mlops.schemas import PredictionLogPayload
from app.domains.prediction.log_store import InMemoryPredictionLogStore


def test_in_memory_prediction_log_store_saves_and_updates_actual_value() -> None:
    store = InMemoryPredictionLogStore()
    payload = PredictionLogPayload(
        model_name="forecast_global",
        request_id="req-1",
        predicted_at=datetime(2026, 1, 1, 12, 0, 0),
        target_timestamp=datetime(2026, 1, 1, 13, 0, 0),
        predicted_value=10.5,
    )

    store.save(payload)
    store.update_actual("req-1", actual_value=11.0, error_value=0.5, error_metrics={"mae": 0.5})

    logs = store.list_by_model("forecast_global")
    assert len(logs) == 1
    assert logs[0].actual_value == 11.0
    assert logs[0].error_value == 0.5
    assert logs[0].error_metrics == {"mae": 0.5}
