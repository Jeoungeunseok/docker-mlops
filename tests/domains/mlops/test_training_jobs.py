from datetime import datetime
from types import SimpleNamespace

from app.domains.mlops import training_jobs
from app.domains.mlops.schemas import EvaluationMetrics, TrainingContext, TrainingResult
from app.domains.mlops.training_jobs import (
    InMemoryTrainingJobStore,
    PostgresTrainingJobStore,
    TrainingJobRunner,
    can_retry_training_job,
    retry_block_reason,
)


def test_in_memory_training_job_store_tracks_lifecycle() -> None:
    store = InMemoryTrainingJobStore()
    context = _context()
    result = TrainingResult(
        model_name="xgboost_global",
        run_id="run-1",
        model_uri="runs:/run-1/model",
        metrics=EvaluationMetrics(mae=1.0, rmse=1.0, mape=1.0, validation_samples=100),
        promoted=True,
    )

    created = store.create(context, max_attempts=2)
    running = store.mark_running(created.job_id)
    failed = store.mark_failed(created.job_id, "boom")
    retried = store.reset_for_retry(created.job_id)
    running_again = store.mark_running(created.job_id)
    succeeded = store.mark_succeeded(created.job_id, result)

    assert created.status == "pending"
    assert running.status == "running"
    assert running.attempts == 1
    assert failed.status == "failed"
    assert failed.error_message == "boom"
    assert retried.status == "pending"
    assert retried.error_message is None
    assert running_again.attempts == 2
    assert succeeded.status == "succeeded"
    assert succeeded.result == result
    assert store.get(created.job_id) == succeeded


def test_build_training_job_store_defaults_to_in_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        training_jobs,
        "settings",
        SimpleNamespace(training_job_store="in_memory", app_database_url=None),
    )

    store = training_jobs.build_training_job_store()

    assert isinstance(store, InMemoryTrainingJobStore)


def test_build_training_job_store_supports_postgres(monkeypatch) -> None:
    monkeypatch.setattr(
        training_jobs,
        "settings",
        SimpleNamespace(
            training_job_store="postgres",
            app_database_url="postgresql://app:app_pass@localhost:5434/app",
        ),
    )

    store = training_jobs.build_training_job_store()

    assert isinstance(store, PostgresTrainingJobStore)


def test_build_training_job_store_requires_database_url_for_postgres(monkeypatch) -> None:
    monkeypatch.setattr(
        training_jobs,
        "settings",
        SimpleNamespace(training_job_store="postgres", app_database_url=None),
    )

    try:
        training_jobs.build_training_job_store()
    except ValueError as exc:
        assert "APP_DATABASE_URL" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_retry_policy_allows_only_failed_jobs_under_attempt_limit() -> None:
    store = InMemoryTrainingJobStore()
    created = store.create(_context(), max_attempts=2)

    assert retry_block_reason(created) == "Only failed training jobs can be retried. Current status is pending."
    assert not can_retry_training_job(created)

    running = store.mark_running(created.job_id)
    failed = store.mark_failed(created.job_id, "boom")

    assert running.attempts == 1
    assert retry_block_reason(failed) is None
    assert can_retry_training_job(failed)

    store.reset_for_retry(created.job_id)
    store.mark_running(created.job_id)
    failed_after_limit = store.mark_failed(created.job_id, "boom again")

    assert retry_block_reason(failed_after_limit) == "Training job retry limit exceeded. Attempts: 2/2."
    assert not can_retry_training_job(failed_after_limit)


def test_training_job_runner_retries_failed_job_when_attempts_remain() -> None:
    store = InMemoryTrainingJobStore()
    submitted_job_ids: list[str] = []

    class DummyExecutor:
        def submit(self, fn, job_id: str) -> None:
            submitted_job_ids.append(job_id)

    runner = TrainingJobRunner(store, executor=DummyExecutor())
    created = store.create(_context(), max_attempts=2)
    store.mark_running(created.job_id)
    store.mark_failed(created.job_id, "boom")

    retried = runner.retry(created.job_id)

    assert retried.status == "pending"
    assert retried.attempts == 1
    assert retried.started_at is None
    assert retried.finished_at is None
    assert retried.error_message is None
    assert submitted_job_ids == [created.job_id]


def test_training_job_runner_rejects_retry_when_attempt_limit_is_reached() -> None:
    store = InMemoryTrainingJobStore()
    runner = TrainingJobRunner(store)
    created = store.create(_context(), max_attempts=1)
    store.mark_running(created.job_id)
    store.mark_failed(created.job_id, "boom")

    try:
        runner.retry(created.job_id)
    except ValueError as exc:
        assert "Attempts: 1/1" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_training_job_runner_notifies_when_job_fails(monkeypatch) -> None:
    from app.jobs import train_model_job

    captured = {}
    store = InMemoryTrainingJobStore()
    runner = TrainingJobRunner(store)
    created = store.create(_context(), max_attempts=1)

    def fake_train_model_job(context: TrainingContext) -> TrainingResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(train_model_job, "train_model_job", fake_train_model_job)
    monkeypatch.setattr(
        training_jobs.notification_dispatcher,
        "notify",
        lambda event: captured.setdefault("event", event),
    )

    runner._run(created.job_id)

    failed = store.get(created.job_id)
    assert failed is not None
    assert failed.status == "failed"
    assert captured["event"].event_type == "training_job_failed"
    assert captured["event"].payload["job_id"] == created.job_id


def test_training_job_runner_notifies_when_job_succeeds(monkeypatch) -> None:
    from app.jobs import train_model_job

    captured = {}
    store = InMemoryTrainingJobStore()
    runner = TrainingJobRunner(store)
    created = store.create(_context(), max_attempts=1)
    expected = TrainingResult(
        model_name="xgboost_global",
        run_id="run-1",
        model_uri="runs:/run-1/model",
        metrics=EvaluationMetrics(mae=1.0, rmse=1.0, mape=1.0, validation_samples=100),
        promoted=True,
    )

    def fake_train_model_job(context: TrainingContext) -> TrainingResult:
        return expected

    monkeypatch.setattr(train_model_job, "train_model_job", fake_train_model_job)
    monkeypatch.setattr(
        training_jobs.notification_dispatcher,
        "notify",
        lambda event: captured.setdefault("event", event),
    )

    runner._run(created.job_id)

    succeeded = store.get(created.job_id)
    assert succeeded is not None
    assert succeeded.status == "succeeded"
    assert captured["event"].event_type == "training_job_succeeded"
    assert captured["event"].payload["model_name"] == "xgboost_global"


def _context() -> TrainingContext:
    return TrainingContext(
        model_type="xgboost",
        train_start_at=datetime(2026, 1, 1),
        train_end_at=datetime(2026, 1, 2),
        validation_start_at=datetime(2026, 1, 3),
        validation_end_at=datetime(2026, 1, 4),
    )
