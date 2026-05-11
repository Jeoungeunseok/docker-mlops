from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.config import settings


def app_timezone() -> ZoneInfo:
    return ZoneInfo(settings.app_timezone)


def now_in_app_timezone() -> datetime:
    return datetime.now(app_timezone())
