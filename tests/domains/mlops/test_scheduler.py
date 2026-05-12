from datetime import datetime, timedelta

import pytest

from app.domains.mlops.scheduler import (
    InProcessTrainingScheduler,
    ScheduledTrainingJob,
    build_scheduled_training_context,
    build_scheduled_training_jobs,
)
from app.domains.mlops.schemas import TrainingContext, TrainingJobRecord


class CapturingSubmitter:
    def __init__(self) -> None:
        self.submissions: list[tuple[TrainingContext, int]] = []

    def submit(self, context: TrainingContext, max_attempts: int = 1) -> TrainingJobRecord:
        self.submissions.append((context, max_attempts))
        return TrainingJobRecord(
            job_id=f"job-{len(self.submissions)}",
            status="pending",
            context=context,
            created_at=datetime(2026, 1, 1),
        )


class FailingSubmitter:
    def submit(self, context: TrainingContext, max_attempts: int = 1) -> TrainingJobRecord:
        raise RuntimeError("submit failed")


def test_build_scheduled_training_context_uses_rolling_windows() -> None:
    job = ScheduledTrainingJob(
        model_type="xgboost",
        interval_seconds=3600,
        train_window_hours=24,
        validation_window_hours=6,
        target_type="store",
        target_id="42",
        qualifiers={"region": "apac"},
        extra_params={"objective": "reg:squarederror"},
    )

    context = build_scheduled_training_context(job, datetime(2026, 1, 2, 12, 0, 0))

    assert context.model_type == "xgboost"
    assert context.target_type == "store"
    assert context.target_id == "42"
    assert context.qualifiers == {"region": "apac"}
    assert context.train_start_at == datetime(2026, 1, 1, 6, 0, 0)
    assert context.train_end_at == datetime(2026, 1, 2, 6, 0, 0)
    assert context.validation_start_at == datetime(2026, 1, 2, 6, 0, 0)
    assert context.validation_end_at == datetime(2026, 1, 2, 12, 0, 0)
    assert context.extra_params == {"objective": "reg:squarederror"}


def test_build_scheduled_training_jobs_parses_json_array() -> None:
    jobs = build_scheduled_training_jobs(
        """
        [
          {
            "model_type": "gru",
            "interval_seconds": 86400,
            "train_window_hours": 168,
            "validation_window_hours": 24,
            "max_attempts": 2
          }
        ]
        """
    )

    assert len(jobs) == 1
    assert jobs[0].model_type == "gru"
    assert jobs[0].max_attempts == 2


def test_build_scheduled_training_jobs_rejects_non_array_json() -> None:
    with pytest.raises(ValueError):
        build_scheduled_training_jobs('{"model_type": "xgboost"}')


def test_scheduler_tick_submits_due_jobs_and_updates_next_run() -> None:
    submitter = CapturingSubmitter()
    job = ScheduledTrainingJob(
        model_type="xgboost",
        interval_seconds=3600,
        train_window_hours=24,
        validation_window_hours=6,
        max_attempts=3,
        run_on_start=True,
    )
    scheduler = InProcessTrainingScheduler([job], submitter=submitter)
    current_time = datetime(2026, 1, 2, 12, 0, 0)

    scheduler.tick(current_time)
    scheduler.tick(current_time + timedelta(minutes=30))

    states = scheduler.states()
    assert len(submitter.submissions) == 1
    assert submitter.submissions[0][1] == 3
    assert states[0].last_submitted_job_id == "job-1"
    assert states[0].next_run_at == current_time + timedelta(seconds=3600)


def test_scheduler_tick_notifies_when_job_is_submitted(monkeypatch) -> None:
    from app.domains.mlops import scheduler as scheduler_module

    captured = {}
    submitter = CapturingSubmitter()
    job = ScheduledTrainingJob(
        model_type="xgboost",
        interval_seconds=3600,
        train_window_hours=24,
        validation_window_hours=6,
        run_on_start=True,
    )
    scheduler = InProcessTrainingScheduler([job], submitter=submitter)
    monkeypatch.setattr(
        scheduler_module.notification_dispatcher,
        "notify",
        lambda event: captured.setdefault("event", event),
    )

    scheduler.tick(datetime(2026, 1, 2, 12, 0, 0))

    assert captured["event"].event_type == "scheduled_retraining_submitted"
    assert captured["event"].payload["job_id"] == "job-1"


def test_scheduler_tick_notifies_when_submit_fails(monkeypatch) -> None:
    from app.domains.mlops import scheduler as scheduler_module

    captured = {}
    job = ScheduledTrainingJob(
        model_type="xgboost",
        interval_seconds=3600,
        train_window_hours=24,
        validation_window_hours=6,
        run_on_start=True,
    )
    scheduler = InProcessTrainingScheduler([job], submitter=FailingSubmitter())
    monkeypatch.setattr(
        scheduler_module.notification_dispatcher,
        "notify",
        lambda event: captured.setdefault("event", event),
    )

    scheduler.tick(datetime(2026, 1, 2, 12, 0, 0))

    assert captured["event"].event_type == "scheduled_retraining_submit_failed"
    assert captured["event"].payload["error_message"] == "submit failed"
