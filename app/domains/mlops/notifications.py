import json
from datetime import datetime
from typing import Protocol
from urllib import request

from pydantic import BaseModel, Field

from app.core.logging import app_logger
from app.core.timezone import now_in_app_timezone
from app.domains.mlops.config import mlops_settings


class NotificationEvent(BaseModel):
    event_type: str
    severity: str = "info"
    message: str
    occurred_at: datetime = Field(default_factory=now_in_app_timezone)
    payload: dict[str, object] = Field(default_factory=dict)


class NotificationSink(Protocol):
    def send(self, event: NotificationEvent) -> None:
        ...


class DisabledNotificationSink:
    def send(self, event: NotificationEvent) -> None:
        return


class LoggingNotificationSink:
    def send(self, event: NotificationEvent) -> None:
        app_logger.info(
            "MLOps notification event",
            extra={
                "event_type": event.event_type,
                "severity": event.severity,
                "message": event.message,
                "payload": event.payload,
            },
        )


class WebhookNotificationSink:
    def __init__(self, webhook_url: str, timeout_seconds: float = 5.0) -> None:
        self._webhook_url = webhook_url
        self._timeout_seconds = timeout_seconds

    def send(self, event: NotificationEvent) -> None:
        body = json.dumps(event.model_dump(mode="json")).encode("utf-8")
        webhook_request = request.Request(
            self._webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(webhook_request, timeout=self._timeout_seconds) as response:
            response.read()


class NotificationDispatcher:
    def __init__(self, sink: NotificationSink) -> None:
        self._sink = sink

    def notify(self, event: NotificationEvent) -> None:
        try:
            self._sink.send(event)
        except Exception:
            app_logger.exception("Failed to send MLOps notification", extra={"event_type": event.event_type})


def build_notification_sink() -> NotificationSink:
    selected_sink = mlops_settings.notification_sink.strip().lower()
    if selected_sink == "disabled":
        return DisabledNotificationSink()
    if selected_sink == "logging":
        return LoggingNotificationSink()
    if selected_sink == "webhook":
        if not mlops_settings.notification_webhook_url:
            raise ValueError("MLOPS_NOTIFICATION_WEBHOOK_URL is required when MLOPS_NOTIFICATION_SINK=webhook")
        return WebhookNotificationSink(mlops_settings.notification_webhook_url)
    raise ValueError(f"Unsupported MLOPS_NOTIFICATION_SINK: {mlops_settings.notification_sink}")


notification_dispatcher = NotificationDispatcher(build_notification_sink())
