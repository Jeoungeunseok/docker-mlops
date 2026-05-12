from types import SimpleNamespace

from app.domains.mlops import notifications
from app.domains.mlops.notifications import (
    DisabledNotificationSink,
    LoggingNotificationSink,
    NotificationDispatcher,
    NotificationEvent,
    WebhookNotificationSink,
)


class CapturingSink:
    def __init__(self) -> None:
        self.events: list[NotificationEvent] = []

    def send(self, event: NotificationEvent) -> None:
        self.events.append(event)


class FailingSink:
    def send(self, event: NotificationEvent) -> None:
        raise RuntimeError("send failed")


def test_notification_dispatcher_sends_event(monkeypatch) -> None:
    sink = CapturingSink()
    dispatcher = NotificationDispatcher(sink)
    event = NotificationEvent(event_type="drift_detected", message="Drift detected.")
    captured = {}

    monkeypatch.setattr(
        notifications.mlops_event_store,
        "save",
        lambda record: captured.setdefault("record", record),
    )
    dispatcher.notify(event)

    assert sink.events == [event]
    assert captured["record"].event_type == "drift_detected"


def test_notification_dispatcher_does_not_raise_when_sink_fails() -> None:
    dispatcher = NotificationDispatcher(FailingSink())

    dispatcher.notify(NotificationEvent(event_type="training_job_failed", message="Training failed."))


def test_build_notification_sink_defaults_to_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        notifications,
        "mlops_settings",
        SimpleNamespace(notification_sink="disabled", notification_webhook_url=None),
    )

    sink = notifications.build_notification_sink()

    assert isinstance(sink, DisabledNotificationSink)


def test_build_notification_sink_supports_logging(monkeypatch) -> None:
    monkeypatch.setattr(
        notifications,
        "mlops_settings",
        SimpleNamespace(notification_sink="logging", notification_webhook_url=None),
    )

    sink = notifications.build_notification_sink()

    assert isinstance(sink, LoggingNotificationSink)


def test_build_notification_sink_supports_webhook(monkeypatch) -> None:
    monkeypatch.setattr(
        notifications,
        "mlops_settings",
        SimpleNamespace(notification_sink="webhook", notification_webhook_url="https://example.com/hook"),
    )

    sink = notifications.build_notification_sink()

    assert isinstance(sink, WebhookNotificationSink)


def test_build_notification_sink_requires_webhook_url(monkeypatch) -> None:
    monkeypatch.setattr(
        notifications,
        "mlops_settings",
        SimpleNamespace(notification_sink="webhook", notification_webhook_url=""),
    )

    try:
        notifications.build_notification_sink()
    except ValueError as exc:
        assert "MLOPS_NOTIFICATION_WEBHOOK_URL" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
