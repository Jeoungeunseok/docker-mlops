from datetime import datetime
from types import SimpleNamespace

from app.domains.mlops.schemas import PredictionLogPayload
from app.domains.prediction import log_store
from app.domains.prediction.log_store import InMemoryPredictionLogStore, PostgresPredictionLogStore


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


def test_build_prediction_log_store_defaults_to_in_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        log_store,
        "settings",
        SimpleNamespace(prediction_log_store="in_memory", app_database_url=None),
    )

    store = log_store.build_prediction_log_store()

    assert isinstance(store, InMemoryPredictionLogStore)


def test_build_prediction_log_store_supports_postgres(monkeypatch) -> None:
    monkeypatch.setattr(
        log_store,
        "settings",
        SimpleNamespace(
            prediction_log_store="postgres",
            app_database_url="postgresql://app:app_pass@localhost:5434/app",
        ),
    )

    store = log_store.build_prediction_log_store()

    assert isinstance(store, PostgresPredictionLogStore)


def test_build_prediction_log_store_requires_database_url_for_postgres(monkeypatch) -> None:
    monkeypatch.setattr(
        log_store,
        "settings",
        SimpleNamespace(prediction_log_store="postgres", app_database_url=None),
    )

    try:
        log_store.build_prediction_log_store()
    except ValueError as exc:
        assert "APP_DATABASE_URL" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
