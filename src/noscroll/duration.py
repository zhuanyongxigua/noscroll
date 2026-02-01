"""Duration parsing utilities for time window and bucket specifications."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal


# Duration pattern: number + unit (e.g., 10d, 36h, 5m, 30s)
DURATION_PATTERN = re.compile(r"^(\d+)([smhdw])$", re.IGNORECASE)

# Unit to seconds mapping
UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}


@dataclass
class Duration:
    """Represents a time duration."""

    value: int
    unit: Literal["s", "m", "h", "d", "w"]

    @property
    def total_seconds(self) -> int:
        """Get total duration in seconds."""
        return self.value * UNIT_SECONDS[self.unit]

    def to_timedelta(self) -> timedelta:
        """Convert to timedelta."""
        return timedelta(seconds=self.total_seconds)

    def __str__(self) -> str:
        return f"{self.value}{self.unit}"


def parse_duration(s: str) -> Duration:
    """
    Parse a duration string like '10d', '36h', '5m', '30s', '2w'.

    Args:
        s: Duration string

    Returns:
        Duration object

    Raises:
        ValueError: If the string is not a valid duration
    """
    match = DURATION_PATTERN.match(s.strip())
    if not match:
        raise ValueError(
            f"Invalid duration format: '{s}'. "
            "Expected format: <number><unit> where unit is s/m/h/d/w "
            "(e.g., 10d, 36h, 5m, 30s, 2w)"
        )
    value = int(match.group(1))
    unit = match.group(2).lower()
    if value <= 0:
        raise ValueError(f"Duration value must be positive, got: {value}")
    return Duration(value=value, unit=unit)  # type: ignore


def parse_rfc3339(s: str) -> datetime:
    """
    Parse an RFC3339/ISO8601 datetime string.

    Supports formats:
    - 2026-01-25T12:00:00Z
    - 2026-01-25T12:00:00+08:00
    - 2026-01-25 (treated as start of day in local timezone)

    Args:
        s: Datetime string

    Returns:
        Timezone-aware datetime

    Raises:
        ValueError: If the string is not a valid datetime
    """
    s = s.strip()

    # Try ISO format first
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    # Try date-only format (YYYY-MM-DD)
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        # Use local timezone for date-only input
        return dt.astimezone()
    except ValueError:
        pass

    raise ValueError(
        f"Invalid datetime format: '{s}'. "
        "Expected RFC3339 format (e.g., 2026-01-25T12:00:00Z) "
        "or date (e.g., 2026-01-25)"
    )


@dataclass
class TimeWindow:
    """Represents a time window with start and end."""

    start: datetime
    end: datetime

    @property
    def duration_seconds(self) -> float:
        """Get window duration in seconds."""
        return (self.end - self.start).total_seconds()

    def contains(self, dt: datetime) -> bool:
        """Check if a datetime falls within this window (inclusive start, exclusive end)."""
        return self.start <= dt < self.end


def build_time_window(
    last: str | None = None,
    from_time: str | None = None,
    to_time: str | None = None,
    default_last: str = "10d",
) -> TimeWindow:
    """
    Build a time window from CLI arguments.

    Args:
        last: Relative duration (e.g., '10d', '36h')
        from_time: Absolute start time (RFC3339)
        to_time: Absolute end time (RFC3339), defaults to now
        default_last: Default duration if no arguments provided

    Returns:
        TimeWindow object

    Raises:
        ValueError: If arguments are invalid or conflicting
    """
    now = datetime.now(timezone.utc)

    # Check for conflicting arguments
    if last and (from_time or to_time):
        raise ValueError("--last and --from/--to are mutually exclusive")

    # Case 1: Relative duration (--last)
    if last:
        duration = parse_duration(last)
        return TimeWindow(
            start=now - duration.to_timedelta(),
            end=now,
        )

    # Case 2: Absolute time range (--from/--to)
    if from_time:
        start = parse_rfc3339(from_time)
        end = parse_rfc3339(to_time) if to_time else now
        if start >= end:
            raise ValueError(f"Start time must be before end time: {start} >= {end}")
        return TimeWindow(start=start, end=end)

    # Case 3: Default
    duration = parse_duration(default_last)
    return TimeWindow(
        start=now - duration.to_timedelta(),
        end=now,
    )


@dataclass
class Bucket:
    """Represents a time bucket for splitting output."""

    duration: Duration | None = None  # For rolling window (e.g., 2h)
    calendar_unit: Literal["day", "hour"] | None = None  # For natural boundaries

    @classmethod
    def from_string(cls, s: str) -> "Bucket":
        """
        Parse a bucket specification.

        Args:
            s: Bucket string ('day', 'hour', or duration like '2h')

        Returns:
            Bucket object
        """
        s = s.strip().lower()
        if s == "day":
            return cls(calendar_unit="day")
        if s == "hour":
            return cls(calendar_unit="hour")

        # Try parsing as duration
        try:
            duration = parse_duration(s)
            return cls(duration=duration)
        except ValueError:
            raise ValueError(
                f"Invalid bucket format: '{s}'. "
                "Expected 'day', 'hour', or duration (e.g., 2h, 1d)"
            )


def split_time_window(window: TimeWindow, bucket: Bucket) -> list[TimeWindow]:
    """
    Split a time window into buckets.

    Args:
        window: The time window to split
        bucket: The bucket specification

    Returns:
        List of TimeWindow objects
    """
    buckets = []

    if bucket.calendar_unit == "day":
        # Split by natural days (local timezone)
        current = window.start.astimezone()
        # Align to start of day
        current = current.replace(hour=0, minute=0, second=0, microsecond=0)
        if current < window.start:
            current += timedelta(days=1)
        current = current.replace(hour=0, minute=0, second=0, microsecond=0)

        # Actually, we want to include partial first day
        current = window.start.astimezone().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        while current < window.end:
            next_day = current + timedelta(days=1)
            bucket_start = max(current, window.start)
            bucket_end = min(next_day, window.end)
            if bucket_start < bucket_end:
                buckets.append(TimeWindow(start=bucket_start, end=bucket_end))
            current = next_day

    elif bucket.calendar_unit == "hour":
        # Split by natural hours
        current = window.start.astimezone().replace(minute=0, second=0, microsecond=0)

        while current < window.end:
            next_hour = current + timedelta(hours=1)
            bucket_start = max(current, window.start)
            bucket_end = min(next_hour, window.end)
            if bucket_start < bucket_end:
                buckets.append(TimeWindow(start=bucket_start, end=bucket_end))
            current = next_hour

    elif bucket.duration:
        # Split by fixed duration
        delta = bucket.duration.to_timedelta()
        current = window.start

        while current < window.end:
            bucket_end = min(current + delta, window.end)
            buckets.append(TimeWindow(start=current, end=bucket_end))
            current = bucket_end

    return buckets


def format_filename(
    template: str,
    window: TimeWindow,
) -> str:
    """
    Format a filename using a template.

    Supported placeholders:
    - {start:%Y-%m-%d} - strftime format for start time
    - {end:%Y-%m-%d} - strftime format for end time

    Args:
        template: Filename template
        window: Time window

    Returns:
        Formatted filename
    """
    result = template

    # Handle {start:format} patterns
    start_pattern = re.compile(r"\{start:([^}]+)\}")
    for match in start_pattern.finditer(template):
        fmt = match.group(1)
        formatted = window.start.astimezone().strftime(fmt)
        result = result.replace(match.group(0), formatted)

    # Handle {end:format} patterns
    end_pattern = re.compile(r"\{end:([^}]+)\}")
    for match in end_pattern.finditer(template):
        fmt = match.group(1)
        formatted = window.end.astimezone().strftime(fmt)
        result = result.replace(match.group(0), formatted)

    return result
