from datetime import datetime
from types import SimpleNamespace

from app.domains.mlops import event_store
from app.domains.mlops.event_store import (
    InMemoryMlopsEventStore,
    MlopsEventRecord,
    PostgresMlopsEventStore,
    build_mlops_event_store,
)


def test_in_memory_mlops_event_store_saves_and_lists_recent_events() -> None:
    store = InMemoryMlopsEventStore()
    first = _event("training_job_failed")
    second = _event("drift_detected")

    store.save(first)
    store.save(second)

    assert store.list_recent() == [second, first]
    assert store.list_recent(event_type="training_job_failed") == [first]
    assert store.list_recent(limit=1) == [second]


def test_build_mlops_event_store_defaults_to_in_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        event_store,
        "settings",
        SimpleNamespace(mlops_event_store="in_memory", app_database_url=None),
    )

    store = build_mlops_event_store()

    assert isinstance(store, InMemoryMlopsEventStore)


def test_build_mlops_event_store_supports_postgres(monkeypatch) -> None:
    monkeypatch.setattr(
        event_store,
        "settings",
        SimpleNamespace(
            mlops_event_store="postgres",
            app_database_url="postgresql://app:app_pass@localhost:5434/app",
        ),
    )

    store = build_mlops_event_store()

    assert isinstance(store, PostgresMlopsEventStore)


def test_build_mlops_event_store_requires_database_url_for_postgres(monkeypatch) -> None:
    monkeypatch.setattr(
        event_store,
        "settings",
        SimpleNamespace(mlops_event_store="postgres", app_database_url=None),
    )

    try:
        build_mlops_event_store()
    except ValueError as exc:
        assert "APP_DATABASE_URL" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def _event(event_type: str) -> MlopsEventRecord:
    return MlopsEventRecord(
        event_id=f"{event_type}-1",
        event_type=event_type,
        severity="info",
        message="event",
        occurred_at=datetime(2026, 1, 1, 12, 0, 0),
        payload={"event_type": event_type},
    )
