"""Time helpers. India has no daylight saving, so a fixed +05:30 offset is exact."""
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), "IST")


def now_ist() -> datetime:
    return datetime.now(IST)