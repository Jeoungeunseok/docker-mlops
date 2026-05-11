import json
import logging
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from app.core.config import AppSettings, settings
from app.core.timezone import app_timezone

app_logger = logging.getLogger("mlops")

_RESERVED_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, app_timezone()).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(app_settings: AppSettings | None = None) -> None:
    selected_settings = app_settings or settings
    level = getattr(logging, selected_settings.log_level.upper(), logging.INFO)
    formatter = _build_formatter(selected_settings.log_format)

    handlers: list[logging.Handler] = [_build_console_handler(formatter)]
    if selected_settings.log_to_file:
        handlers.append(_build_rotating_file_handler(selected_settings, formatter))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    for handler in handlers:
        handler.setLevel(level)
        root_logger.addHandler(handler)

    app_logger.setLevel(level)
    app_logger.propagate = True
    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)


def _build_formatter(log_format: str) -> logging.Formatter:
    if log_format.lower() == "json":
        return JsonLogFormatter()
    return logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def _build_console_handler(formatter: logging.Formatter) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    return handler


def _build_rotating_file_handler(
    app_settings: AppSettings,
    formatter: logging.Formatter,
) -> logging.Handler:
    log_path = Path(app_settings.log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        interval=1,
        backupCount=app_settings.log_retention_days,
        encoding="utf-8",
        utc=False,
    )
    handler.setFormatter(formatter)
    return handler


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
