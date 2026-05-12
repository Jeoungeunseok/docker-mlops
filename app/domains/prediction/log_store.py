from threading import RLock
from typing import Protocol

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from app.core.config import settings
from app.domains.mlops.schemas import PredictionLogPayload
from app.infra.migrations import run_app_database_migrations


class PredictionLogStore(Protocol):
    def save(self, payload: PredictionLogPayload) -> None:
        ...

    def update_actual(
        self,
        request_id: str,
        actual_value: object,
        error_value: float | None = None,
        error_metrics: dict[str, float] | None = None,
    ) -> None:
        ...

    def list_by_model(self, model_name: str) -> list[PredictionLogPayload]:
        ...


class InMemoryPredictionLogStore:
    def __init__(self) -> None:
        self._payloads: list[PredictionLogPayload] = []
        self._lock = RLock()

    def save(self, payload: PredictionLogPayload) -> None:
        with self._lock:
            self._payloads.append(payload)

    def update_actual(
        self,
        request_id: str,
        actual_value: object,
        error_value: float | None = None,
        error_metrics: dict[str, float] | None = None,
    ) -> None:
        with self._lock:
            for index, payload in enumerate(self._payloads):
                if payload.request_id != request_id:
                    continue
                self._payloads[index] = payload.model_copy(
                    update={
                        "actual_value": actual_value,
                        "error_value": error_value,
                        "error_metrics": error_metrics or {},
                    }
                )
                return

    def list_by_model(self, model_name: str) -> list[PredictionLogPayload]:
        with self._lock:
            return [payload for payload in self._payloads if payload.model_name == model_name]


class PostgresPredictionLogStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._table_ready = False
        self._lock = RLock()

    def save(self, payload: PredictionLogPayload) -> None:
        self._ensure_table()
        payload_data = payload.model_dump(mode="json")
        with psycopg2.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO prediction_logs (
                        model_name,
                        model_version,
                        run_id,
                        request_id,
                        target_type,
                        target_id,
                        qualifiers,
                        predicted_at,
                        target_timestamp,
                        predicted_value,
                        actual_value,
                        error_value,
                        error_metrics,
                        input_features,
                        output_metadata
                    )
                    VALUES (
                        %(model_name)s,
                        %(model_version)s,
                        %(run_id)s,
                        %(request_id)s,
                        %(target_type)s,
                        %(target_id)s,
                        %(qualifiers)s,
                        %(predicted_at)s,
                        %(target_timestamp)s,
                        %(predicted_value)s,
                        %(actual_value)s,
                        %(error_value)s,
                        %(error_metrics)s,
                        %(input_features)s,
                        %(output_metadata)s
                    )
                    """,
                    {
                        **payload_data,
                        "qualifiers": Json(payload_data["qualifiers"]),
                        "predicted_value": Json(payload_data["predicted_value"]),
                        "actual_value": Json(payload_data["actual_value"]),
                        "error_metrics": Json(payload_data["error_metrics"]),
                        "input_features": Json(payload_data["input_features"]),
                        "output_metadata": Json(payload_data["output_metadata"]),
                    },
                )

    def update_actual(
        self,
        request_id: str,
        actual_value: object,
        error_value: float | None = None,
        error_metrics: dict[str, float] | None = None,
    ) -> None:
        self._ensure_table()
        with psycopg2.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE prediction_logs
                    SET actual_value = %(actual_value)s,
                        error_value = %(error_value)s,
                        error_metrics = %(error_metrics)s
                    WHERE request_id = %(request_id)s
                    """,
                    {
                        "request_id": request_id,
                        "actual_value": Json(actual_value),
                        "error_value": error_value,
                        "error_metrics": Json(error_metrics or {}),
                    },
                )

    def list_by_model(self, model_name: str) -> list[PredictionLogPayload]:
        self._ensure_table()
        with psycopg2.connect(self._database_url) as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT
                        model_name,
                        model_version,
                        run_id,
                        request_id,
                        target_type,
                        target_id,
                        qualifiers,
                        predicted_at,
                        target_timestamp,
                        predicted_value,
                        actual_value,
                        error_value,
                        error_metrics,
                        input_features,
                        output_metadata
                    FROM prediction_logs
                    WHERE model_name = %s
                    ORDER BY predicted_at DESC
                    LIMIT 100
                    """,
                    (model_name,),
                )
                return [PredictionLogPayload.model_validate(dict(row)) for row in cursor.fetchall()]

    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        with self._lock:
            if self._table_ready:
                return
            run_app_database_migrations(self._database_url)
            self._table_ready = True


def build_prediction_log_store() -> PredictionLogStore:
    selected_store = settings.prediction_log_store.strip().lower()
    if selected_store == "in_memory":
        return InMemoryPredictionLogStore()
    if selected_store == "postgres":
        if settings.app_database_url is None:
            raise ValueError("APP_DATABASE_URL is required when PREDICTION_LOG_STORE=postgres")
        return PostgresPredictionLogStore(settings.app_database_url)
    raise ValueError(f"Unsupported PREDICTION_LOG_STORE: {settings.prediction_log_store}")


prediction_log_store: PredictionLogStore = build_prediction_log_store()
