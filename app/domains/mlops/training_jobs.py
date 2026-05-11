from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from uuid import uuid4

from app.core.logging import app_logger
from app.core.timezone import now_in_app_timezone
from app.domains.mlops.schemas import TrainingContext, TrainingJobRecord, TrainingResult


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


class TrainingJobRunner:
    def __init__(self, store: InMemoryTrainingJobStore, max_workers: int = 2) -> None:
        self._store = store
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="training-job")

    def submit(self, context: TrainingContext, max_attempts: int = 1) -> TrainingJobRecord:
        record = self._store.create(context=context, max_attempts=max_attempts)
        self._executor.submit(self._run, record.job_id)
        return record

    def retry(self, job_id: str) -> TrainingJobRecord:
        record = self._store.get(job_id)
        if record is None:
            raise KeyError(f"Training job was not found: {job_id}")
        if record.status != "failed":
            raise ValueError(f"Only failed training jobs can be retried: {job_id}")
        if record.attempts >= record.max_attempts:
            raise ValueError(f"Training job retry limit exceeded: {job_id}")
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
            self._store.mark_failed(job_id, str(exc))
            app_logger.exception("Async training job failed", extra={"job_id": job_id})
            return
        self._store.mark_succeeded(job_id, result)
        app_logger.info("Async training job succeeded", extra={"job_id": job_id})


training_job_store = InMemoryTrainingJobStore()
training_job_runner = TrainingJobRunner(training_job_store)
