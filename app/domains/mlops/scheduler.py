import json
from datetime import datetime, timedelta
from threading import Event, RLock, Thread
from typing import Protocol

from pydantic import BaseModel, Field

from app.core.logging import app_logger
from app.core.timezone import now_in_app_timezone
from app.domains.mlops.config import mlops_settings
from app.domains.mlops.notifications import NotificationEvent, notification_dispatcher
from app.domains.mlops.schemas import TrainingContext, TrainingJobRecord
from app.domains.mlops.training_jobs import training_job_runner


class TrainingJobSubmitter(Protocol):
    def submit(self, context: TrainingContext, max_attempts: int = 1) -> TrainingJobRecord:
        ...


class ScheduledTrainingJob(BaseModel):
    model_type: str
    interval_seconds: int = Field(gt=0)
    train_window_hours: int = Field(gt=0)
    validation_window_hours: int = Field(gt=0)
    target_type: str = "global"
    target_id: str | None = None
    qualifiers: dict[str, str | int | float | bool] = Field(default_factory=dict)
    extra_params: dict[str, object] = Field(default_factory=dict)
    max_attempts: int = Field(default=1, ge=1)
    run_on_start: bool = False


class ScheduledTrainingJobState(BaseModel):
    model_type: str
    target_type: str
    target_id: str | None = None
    next_run_at: datetime
    last_submitted_job_id: str | None = None
    last_submitted_at: datetime | None = None
    last_error_message: str | None = None


class InProcessTrainingScheduler:
    def __init__(
        self,
        jobs: list[ScheduledTrainingJob],
        submitter: TrainingJobSubmitter,
        poll_interval_seconds: float = 30.0,
    ) -> None:
        self._jobs = jobs
        self._submitter = submitter
        self._poll_interval_seconds = poll_interval_seconds
        self._stop_event = Event()
        self._lock = RLock()
        self._thread: Thread | None = None
        self._states = [_initial_state(job) for job in jobs]

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if not self._jobs:
                app_logger.info("Scheduled retraining scheduler is disabled because no jobs are configured")
                return
            self._stop_event.clear()
            self._thread = Thread(target=self._run_loop, name="scheduled-retraining", daemon=True)
            self._thread.start()
            app_logger.info("Scheduled retraining scheduler started", extra={"job_count": len(self._jobs)})

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._stop_event.set()
        thread.join(timeout=self._poll_interval_seconds + 1)
        with self._lock:
            self._thread = None
        app_logger.info("Scheduled retraining scheduler stopped")

    def tick(self, now: datetime | None = None) -> list[ScheduledTrainingJobState]:
        current_time = now or now_in_app_timezone()
        submitted_states: list[ScheduledTrainingJobState] = []
        with self._lock:
            for index, job in enumerate(self._jobs):
                state = self._states[index]
                if current_time < state.next_run_at:
                    continue
                submitted_states.append(self._submit_job(index, job, current_time))
        return submitted_states

    def dry_run(self, now: datetime | None = None) -> list[TrainingContext]:
        current_time = now or now_in_app_timezone()
        with self._lock:
            return [
                build_scheduled_training_context(job, current_time)
                for job, state in zip(self._jobs, self._states)
                if current_time >= state.next_run_at
            ]

    def states(self) -> list[ScheduledTrainingJobState]:
        with self._lock:
            return [state.model_copy() for state in self._states]

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval_seconds):
            self.tick()

    def _submit_job(self, index: int, job: ScheduledTrainingJob, current_time: datetime) -> ScheduledTrainingJobState:
        try:
            context = build_scheduled_training_context(job, current_time)
            record = self._submitter.submit(context, max_attempts=job.max_attempts)
            self._states[index] = self._states[index].model_copy(
                update={
                    "next_run_at": current_time + timedelta(seconds=job.interval_seconds),
                    "last_submitted_job_id": record.job_id,
                    "last_submitted_at": current_time,
                    "last_error_message": None,
                }
            )
            app_logger.info(
                "Scheduled retraining job submitted",
                extra={"job_id": record.job_id, "model_type": job.model_type},
            )
            notification_dispatcher.notify(
                NotificationEvent(
                    event_type="scheduled_retraining_submitted",
                    severity="info",
                    message="Scheduled retraining job submitted.",
                    payload={
                        "job_id": record.job_id,
                        "model_type": job.model_type,
                        "target_type": job.target_type,
                        "target_id": job.target_id,
                        "next_run_at": self._states[index].next_run_at.isoformat(),
                    },
                )
            )
            return self._states[index]
        except Exception as exc:
            self._states[index] = self._states[index].model_copy(
                update={
                    "next_run_at": current_time + timedelta(seconds=job.interval_seconds),
                    "last_error_message": str(exc),
                }
            )
            notification_dispatcher.notify(
                NotificationEvent(
                    event_type="scheduled_retraining_submit_failed",
                    severity="error",
                    message="Failed to submit scheduled retraining job.",
                    payload={
                        "model_type": job.model_type,
                        "target_type": job.target_type,
                        "target_id": job.target_id,
                        "error_message": str(exc),
                        "next_run_at": self._states[index].next_run_at.isoformat(),
                    },
                )
            )
            app_logger.exception("Failed to submit scheduled retraining job", extra={"model_type": job.model_type})
            return self._states[index]


def build_scheduled_training_context(job: ScheduledTrainingJob, current_time: datetime) -> TrainingContext:
    validation_end_at = current_time
    validation_start_at = validation_end_at - timedelta(hours=job.validation_window_hours)
    train_end_at = validation_start_at
    train_start_at = train_end_at - timedelta(hours=job.train_window_hours)
    return TrainingContext(
        model_type=job.model_type,
        target_type=job.target_type,
        target_id=job.target_id,
        qualifiers=job.qualifiers,
        train_start_at=train_start_at,
        train_end_at=train_end_at,
        validation_start_at=validation_start_at,
        validation_end_at=validation_end_at,
        extra_params=job.extra_params,
    )


def build_scheduled_training_jobs(raw_jobs: str) -> list[ScheduledTrainingJob]:
    decoded = json.loads(raw_jobs)
    if not isinstance(decoded, list):
        raise ValueError("MLOPS_SCHEDULED_RETRAINING_JOBS must be a JSON array.")
    return [ScheduledTrainingJob.model_validate(item) for item in decoded]


def build_training_scheduler() -> InProcessTrainingScheduler | None:
    if not mlops_settings.enable_scheduled_retraining:
        return None
    jobs = build_scheduled_training_jobs(mlops_settings.scheduled_retraining_jobs)
    return InProcessTrainingScheduler(jobs=jobs, submitter=training_job_runner)


def _initial_state(job: ScheduledTrainingJob) -> ScheduledTrainingJobState:
    now = now_in_app_timezone()
    next_run_at = now if job.run_on_start else now + timedelta(seconds=job.interval_seconds)
    return ScheduledTrainingJobState(
        model_type=job.model_type,
        target_type=job.target_type,
        target_id=job.target_id,
        next_run_at=next_run_at,
    )
