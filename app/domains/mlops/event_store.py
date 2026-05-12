from threading import RLock
from typing import Protocol
from uuid import uuid4

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from app.core.config import settings
from app.domains.mlops.schemas import MlopsEventRecord


class MlopsEventStore(Protocol):
    def save(self, event: MlopsEventRecord) -> None:
        ...

    def list_recent(self, limit: int = 100, event_type: str | None = None) -> list[MlopsEventRecord]:
        ...


class InMemoryMlopsEventStore:
    def __init__(self) -> None:
        self._events: list[MlopsEventRecord] = []
        self._lock = RLock()

    def save(self, event: MlopsEventRecord) -> None:
        with self._lock:
            self._events.append(event)

    def list_recent(self, limit: int = 100, event_type: str | None = None) -> list[MlopsEventRecord]:
        with self._lock:
            filtered_events = [
                event
                for event in reversed(self._events)
                if event_type is None or event.event_type == event_type
            ]
            return filtered_events[:limit]


class PostgresMlopsEventStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._table_ready = False
        self._lock = RLock()

    def save(self, event: MlopsEventRecord) -> None:
        self._ensure_table()
        event_data = event.model_dump(mode="json")
        with psycopg2.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO mlops_events (
                        event_id,
                        event_type,
                        severity,
                        message,
                        occurred_at,
                        payload
                    )
                    VALUES (
                        %(event_id)s,
                        %(event_type)s,
                        %(severity)s,
                        %(message)s,
                        %(occurred_at)s,
                        %(payload)s
                    )
                    """,
                    {**event_data, "payload": Json(event_data["payload"])},
                )

    def list_recent(self, limit: int = 100, event_type: str | None = None) -> list[MlopsEventRecord]:
        self._ensure_table()
        query = """
            SELECT event_id, event_type, severity, message, occurred_at, payload
            FROM mlops_events
        """
        params: list[object] = []
        if event_type is not None:
            query += " WHERE event_type = %s"
            params.append(event_type)
        query += " ORDER BY occurred_at DESC LIMIT %s"
        params.append(limit)

        with psycopg2.connect(self._database_url) as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                return [MlopsEventRecord.model_validate(dict(row)) for row in cursor.fetchall()]

    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        with self._lock:
            if self._table_ready:
                return
            with psycopg2.connect(self._database_url) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS mlops_events (
                            event_id TEXT PRIMARY KEY,
                            event_type TEXT NOT NULL,
                            severity TEXT NOT NULL,
                            message TEXT NOT NULL,
                            occurred_at TIMESTAMPTZ NOT NULL,
                            payload JSONB NOT NULL DEFAULT '{}'::jsonb
                        )
                        """
                    )
                    cursor.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_mlops_events_type_occurred_at
                        ON mlops_events (event_type, occurred_at DESC)
                        """
                    )
                    cursor.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_mlops_events_occurred_at
                        ON mlops_events (occurred_at DESC)
                        """
                    )
            self._table_ready = True


def create_mlops_event_record(
    event_type: str,
    severity: str,
    message: str,
    occurred_at,
    payload: dict[str, object],
) -> MlopsEventRecord:
    return MlopsEventRecord(
        event_id=str(uuid4()),
        event_type=event_type,
        severity=severity,
        message=message,
        occurred_at=occurred_at,
        payload=payload,
    )


def build_mlops_event_store() -> MlopsEventStore:
    selected_store = settings.mlops_event_store.strip().lower()
    if selected_store == "in_memory":
        return InMemoryMlopsEventStore()
    if selected_store == "postgres":
        if settings.app_database_url is None:
            raise ValueError("APP_DATABASE_URL is required when MLOPS_EVENT_STORE=postgres")
        return PostgresMlopsEventStore(settings.app_database_url)
    raise ValueError(f"Unsupported MLOPS_EVENT_STORE: {settings.mlops_event_store}")


mlops_event_store: MlopsEventStore = build_mlops_event_store()
