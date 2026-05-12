from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Protocol
from uuid import uuid4

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from app.core.config import settings
from app.core.logging import app_logger
from app.core.timezone import now_in_app_timezone
from app.domains.mlops.notifications import NotificationEvent, notification_dispatcher
from app.domains.mlops.schemas import TrainingContext, TrainingJobRecord, TrainingResult


def retry_block_reason(record: TrainingJobRecord) -> str | None:
    if record.status != "failed":
        return f"Only failed training jobs can be retried. Current status is {record.status}."
    if record.attempts >= record.max_attempts:
        return f"Training job retry limit exceeded. Attempts: {record.attempts}/{record.max_attempts}."
    return None


def can_retry_training_job(record: TrainingJobRecord) -> bool:
    return retry_block_reason(record) is None


class TrainingJobStore(Protocol):
    def create(self, context: TrainingContext, max_attempts: int = 1) -> TrainingJobRecord:
        ...

    def get(self, job_id: str) -> TrainingJobRecord | None:
        ...

    def mark_running(self, job_id: str) -> TrainingJobRecord:
        ...

    def mark_succeeded(self, job_id: str, result: TrainingResult) -> TrainingJobRecord:
        ...

    def mark_failed(self, job_id: str, error_message: str) -> TrainingJobRecord:
        ...

    def reset_for_retry(self, job_id: str) -> TrainingJobRecord:
        ...


class TrainingJobExecutor(Protocol):
    def submit(self, fn: object, *args: object) -> object:
        ...


class InMemoryTrainingJobStore:
    def __init__(self) -> None:
        self._records: dict[str, TrainingJobRecord] = {}
        self._lock = RLock()

    def create(self, context: TrainingContext, max_attempts: int = 1) -> TrainingJobRecord:
        record = TrainingJobRecord(
            job_id=str(uuid4()),
            status="pending",
            context=context,
            max_attempts=max_attempts,
            created_at=now_in_app_timezone(),
        )
        with self._lock:
            self._records[record.job_id] = record
        return record

    def get(self, job_id: str) -> TrainingJobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def mark_running(self, job_id: str) -> TrainingJobRecord:
        return self._update(
            job_id,
            status="running",
            attempts=(self._required_get(job_id).attempts + 1),
            started_at=now_in_app_timezone(),
            finished_at=None,
            error_message=None,
        )

    def mark_succeeded(self, job_id: str, result: TrainingResult) -> TrainingJobRecord:
        return self._update(
            job_id,
            status="succeeded",
            result=result,
            finished_at=now_in_app_timezone(),
            error_message=None,
        )

    def mark_failed(self, job_id: str, error_message: str) -> TrainingJobRecord:
        return self._update(
            job_id,
            status="failed",
            finished_at=now_in_app_timezone(),
            error_message=error_message,
        )

    def reset_for_retry(self, job_id: str) -> TrainingJobRecord:
        return self._update(
            job_id,
            status="pending",
            started_at=None,
            finished_at=None,
            result=None,
            error_message=None,
        )

    def _update(self, job_id: str, **changes: object) -> TrainingJobRecord:
        with self._lock:
            record = self._required_get(job_id)
            updated = record.model_copy(update=changes)
            self._records[job_id] = updated
            return updated

    def _required_get(self, job_id: str) -> TrainingJobRecord:
        record = self._records.get(job_id)
        if record is None:
            raise KeyError(f"Training job was not found: {job_id}")
        return record


class PostgresTrainingJobStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._table_ready = False
        self._lock = RLock()

    def create(self, context: TrainingContext, max_attempts: int = 1) -> TrainingJobRecord:
        self._ensure_table()
        record = TrainingJobRecord(
            job_id=str(uuid4()),
            status="pending",
            context=context,
            max_attempts=max_attempts,
            created_at=now_in_app_timezone(),
        )
        record_data = record.model_dump(mode="json")
        with psycopg2.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO training_jobs (
                        job_id,
                        status,
                        context,
                        attempts,
                        max_attempts,
                        created_at,
                        started_at,
                        finished_at,
                        result,
                        error_message
                    )
                    VALUES (
                        %(job_id)s,
                        %(status)s,
                        %(context)s,
                        %(attempts)s,
                        %(max_attempts)s,
                        %(created_at)s,
                        %(started_at)s,
                        %(finished_at)s,
                        %(result)s,
                        %(error_message)s
                    )
                    """,
                    {
                        **record_data,
                        "context": Json(record_data["context"]),
                        "result": Json(record_data["result"]) if record_data["result"] is not None else None,
                    },
                )
        return record

    def get(self, job_id: str) -> TrainingJobRecord | None:
        self._ensure_table()
        with psycopg2.connect(self._database_url) as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT
                        job_id,
                        status,
                        context,
                        attempts,
                        max_attempts,
                        created_at,
                        started_at,
                        finished_at,
                        result,
                        error_message
                    FROM training_jobs
                    WHERE job_id = %s
                    """,
                    (job_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_record(row)

    def mark_running(self, job_id: str) -> TrainingJobRecord:
        return self._update(
            job_id,
            status="running",
            attempts=("increment",),
            started_at=now_in_app_timezone(),
            finished_at=None,
            error_message=None,
        )

    def mark_succeeded(self, job_id: str, result: TrainingResult) -> TrainingJobRecord:
        return self._update(
            job_id,
            status="succeeded",
            result=result.model_dump(mode="json"),
            finished_at=now_in_app_timezone(),
            error_message=None,
        )

    def mark_failed(self, job_id: str, error_message: str) -> TrainingJobRecord:
        return self._update(
            job_id,
            status="failed",
            finished_at=now_in_app_timezone(),
            error_message=error_message,
        )

    def reset_for_retry(self, job_id: str) -> TrainingJobRecord:
        return self._update(
            job_id,
            status="pending",
            started_at=None,
            finished_at=None,
            result=None,
            error_message=None,
        )

    def _update(self, job_id: str, **changes: object) -> TrainingJobRecord:
        self._ensure_table()
        assignments: list[str] = []
        params: dict[str, object] = {"job_id": job_id}
        for key, value in changes.items():
            if key == "attempts" and value == ("increment",):
                assignments.append("attempts = attempts + 1")
                continue
            assignments.append(f"{key} = %({key})s")
            if key == "result" and value is not None:
                params[key] = Json(value)
            else:
                params[key] = value

        with psycopg2.connect(self._database_url) as connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    f"""
                    UPDATE training_jobs
                    SET {", ".join(assignments)}
                    WHERE job_id = %(job_id)s
                    RETURNING
                        job_id,
                        status,
                        context,
                        attempts,
                        max_attempts,
                        created_at,
                        started_at,
                        finished_at,
                        result,
                        error_message
                    """,
                    params,
                )
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(f"Training job was not found: {job_id}")
                return self._row_to_record(row)

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
                        CREATE TABLE IF NOT EXISTS training_jobs (
                            job_id TEXT PRIMARY KEY,
                            status TEXT NOT NULL,
                            context JSONB NOT NULL,
                            attempts INTEGER NOT NULL DEFAULT 0,
                            max_attempts INTEGER NOT NULL DEFAULT 1,
                            created_at TIMESTAMPTZ NOT NULL,
                            started_at TIMESTAMPTZ,
                            finished_at TIMESTAMPTZ,
                            result JSONB,
                            error_message TEXT
                        )
                        """
                    )
                    cursor.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_training_jobs_status_created_at
                        ON training_jobs (status, created_at DESC)
                        """
                    )
            self._table_ready = True

    def _row_to_record(self, row: object) -> TrainingJobRecord:
        return TrainingJobRecord.model_validate(dict(row))


class TrainingJobRunner:
    def __init__(
        self,
        store: TrainingJobStore,
        max_workers: int = 2,
        executor: TrainingJobExecutor | None = None,
    ) -> None:
        self._store = store
        self._executor = executor or ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="training-job")

    def submit(self, context: TrainingContext, max_attempts: int = 1) -> TrainingJobRecord:
        record = self._store.create(context=context, max_attempts=max_attempts)
        self._executor.submit(self._run, record.job_id)
        return record

    def retry(self, job_id: str) -> TrainingJobRecord:
        record = self._store.get(job_id)
        if record is None:
            raise KeyError(f"Training job was not found: {job_id}")
        block_reason = retry_block_reason(record)
        if block_reason is not None:
            raise ValueError(f"{block_reason} job_id={job_id}")
        record = self._store.reset_for_retry(job_id)
        self._executor.submit(self._run, record.job_id)
        return record

    def get(self, job_id: str) -> TrainingJobRecord | None:
        return self._store.get(job_id)

    def _run(self, job_id: str) -> None:
        from app.jobs.train_model_job import train_model_job

        record = self._store.mark_running(job_id)
        app_logger.info("Async training job started", extra={"job_id": job_id, "model_type": record.context.model_type})
        try:
            result = train_model_job(record.context)
        except Exception as exc:
            failed_record = self._store.mark_failed(job_id, str(exc))
            notification_dispatcher.notify(
                NotificationEvent(
                    event_type="training_job_failed",
                    severity="error",
                    message="Async training job failed.",
                    payload={
                        "job_id": job_id,
                        "model_type": failed_record.context.model_type,
                        "attempts": failed_record.attempts,
                        "max_attempts": failed_record.max_attempts,
                        "error_message": failed_record.error_message,
                    },
                )
            )
            app_logger.exception("Async training job failed", extra={"job_id": job_id})
            return
        self._store.mark_succeeded(job_id, result)
        app_logger.info("Async training job succeeded", extra={"job_id": job_id})


def build_training_job_store() -> TrainingJobStore:
    selected_store = settings.training_job_store.strip().lower()
    if selected_store == "in_memory":
        return InMemoryTrainingJobStore()
    if selected_store == "postgres":
        if settings.app_database_url is None:
            raise ValueError("APP_DATABASE_URL is required when TRAINING_JOB_STORE=postgres")
        return PostgresTrainingJobStore(settings.app_database_url)
    raise ValueError(f"Unsupported TRAINING_JOB_STORE: {settings.training_job_store}")


training_job_store: TrainingJobStore = build_training_job_store()
training_job_runner = TrainingJobRunner(training_job_store)
