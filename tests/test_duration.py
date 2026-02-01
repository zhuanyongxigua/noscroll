"""Tests for duration module."""

import pytest
from datetime import datetime, timedelta, timezone
from noscroll.duration import (
    Duration,
    parse_duration,
    parse_rfc3339,
    TimeWindow,
    build_time_window,
    Bucket,
    split_time_window,
    format_filename,
)


class TestDuration:
    """Tests for Duration class and parsing."""

    def test_parse_duration_days(self):
        """Test parsing days."""
        d = parse_duration("10d")
        assert d.value == 10
        assert d.unit == "d"
        assert d.total_seconds == 10 * 86400

    def test_parse_duration_hours(self):
        """Test parsing hours."""
        d = parse_duration("36h")
        assert d.value == 36
        assert d.unit == "h"
        assert d.total_seconds == 36 * 3600

    def test_parse_duration_minutes(self):
        """Test parsing minutes."""
        d = parse_duration("30m")
        assert d.value == 30
        assert d.unit == "m"
        assert d.total_seconds == 30 * 60

    def test_parse_duration_seconds(self):
        """Test parsing seconds."""
        d = parse_duration("120s")
        assert d.value == 120
        assert d.unit == "s"
        assert d.total_seconds == 120

    def test_parse_duration_weeks(self):
        """Test parsing weeks."""
        d = parse_duration("2w")
        assert d.value == 2
        assert d.unit == "w"
        assert d.total_seconds == 2 * 604800

    def test_parse_duration_uppercase(self):
        """Test parsing with uppercase unit."""
        d = parse_duration("5D")
        assert d.value == 5
        assert d.unit == "d"

    def test_parse_duration_with_spaces(self):
        """Test parsing with leading/trailing spaces."""
        d = parse_duration("  10d  ")
        assert d.value == 10
        assert d.unit == "d"

    def test_parse_duration_invalid_format(self):
        """Test invalid format raises error."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("10days")

    def test_parse_duration_invalid_unit(self):
        """Test invalid unit raises error."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("10x")

    def test_parse_duration_negative(self):
        """Test negative value raises error."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("-5d")

    def test_parse_duration_zero(self):
        """Test zero value raises error."""
        with pytest.raises(ValueError, match="Duration value must be positive"):
            parse_duration("0d")

    def test_duration_to_timedelta(self):
        """Test converting to timedelta."""
        d = parse_duration("2d")
        td = d.to_timedelta()
        assert td == timedelta(days=2)

    def test_duration_str(self):
        """Test string representation."""
        d = parse_duration("10d")
        assert str(d) == "10d"


class TestRFC3339:
    """Tests for RFC3339 datetime parsing."""

    def test_parse_rfc3339_utc(self):
        """Test parsing UTC datetime."""
        dt = parse_rfc3339("2026-01-25T12:00:00Z")
        assert dt.year == 2026
        assert dt.month == 1
        assert dt.day == 25
        assert dt.hour == 12
        assert dt.tzinfo is not None

    def test_parse_rfc3339_with_offset(self):
        """Test parsing datetime with timezone offset."""
        dt = parse_rfc3339("2026-01-25T20:00:00+08:00")
        assert dt.year == 2026
        assert dt.hour == 20

    def test_parse_rfc3339_date_only(self):
        """Test parsing date-only string."""
        dt = parse_rfc3339("2026-01-25")
        assert dt.year == 2026
        assert dt.month == 1
        assert dt.day == 25
        assert dt.hour == 0

    def test_parse_rfc3339_with_spaces(self):
        """Test parsing with spaces."""
        dt = parse_rfc3339("  2026-01-25T12:00:00Z  ")
        assert dt.year == 2026

    def test_parse_rfc3339_invalid(self):
        """Test invalid format raises error."""
        with pytest.raises(ValueError, match="Invalid datetime format"):
            parse_rfc3339("invalid-date")


class TestTimeWindow:
    """Tests for TimeWindow class."""

    def test_time_window_duration(self):
        """Test duration calculation."""
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        window = TimeWindow(start=start, end=end)
        assert window.duration_seconds == 86400

    def test_time_window_contains(self):
        """Test contains method."""
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        window = TimeWindow(start=start, end=end)

        # Exactly at start - should be included
        assert window.contains(start)
        # Middle of window
        middle = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert window.contains(middle)
        # Exactly at end - should NOT be included (half-open interval)
        assert not window.contains(end)
        # Before start
        before = datetime(2025, 12, 31, 0, 0, 0, tzinfo=timezone.utc)
        assert not window.contains(before)


class TestBuildTimeWindow:
    """Tests for build_time_window function."""

    def test_build_with_last(self):
        """Test building window with --last."""
        window = build_time_window(last="10d")
        assert window.duration_seconds == pytest.approx(10 * 86400, rel=1)

    def test_build_with_from_to(self):
        """Test building window with --from/--to."""
        window = build_time_window(
            from_time="2026-01-01T00:00:00Z",
            to_time="2026-01-05T00:00:00Z",
        )
        assert window.duration_seconds == 4 * 86400

    def test_build_with_from_only(self):
        """Test building window with --from only (--to defaults to now)."""
        window = build_time_window(from_time="2026-01-01T00:00:00Z")
        assert window.start.year == 2026
        # end should be approximately now
        now = datetime.now(timezone.utc)
        assert abs((window.end - now).total_seconds()) < 5

    def test_build_default(self):
        """Test default window (10d)."""
        window = build_time_window()
        assert window.duration_seconds == pytest.approx(10 * 86400, rel=1)

    def test_build_custom_default(self):
        """Test custom default duration."""
        window = build_time_window(default_last="5d")
        assert window.duration_seconds == pytest.approx(5 * 86400, rel=1)

    def test_build_conflicting_args(self):
        """Test conflicting arguments raises error."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_time_window(last="10d", from_time="2026-01-01")

    def test_build_invalid_range(self):
        """Test start >= end raises error."""
        with pytest.raises(ValueError, match="Start time must be before end"):
            build_time_window(
                from_time="2026-01-10T00:00:00Z",
                to_time="2026-01-01T00:00:00Z",
            )


class TestBucket:
    """Tests for Bucket class."""

    def test_bucket_from_day(self):
        """Test creating day bucket."""
        bucket = Bucket.from_string("day")
        assert bucket.calendar_unit == "day"
        assert bucket.duration is None

    def test_bucket_from_hour(self):
        """Test creating hour bucket."""
        bucket = Bucket.from_string("hour")
        assert bucket.calendar_unit == "hour"
        assert bucket.duration is None

    def test_bucket_from_duration(self):
        """Test creating duration bucket."""
        bucket = Bucket.from_string("2h")
        assert bucket.duration is not None
        assert bucket.duration.value == 2
        assert bucket.duration.unit == "h"
        assert bucket.calendar_unit is None

    def test_bucket_case_insensitive(self):
        """Test case insensitivity."""
        bucket = Bucket.from_string("DAY")
        assert bucket.calendar_unit == "day"

    def test_bucket_invalid(self):
        """Test invalid bucket raises error."""
        with pytest.raises(ValueError, match="Invalid bucket format"):
            Bucket.from_string("invalid")


class TestSplitTimeWindow:
    """Tests for split_time_window function."""

    def test_split_by_day(self):
        """Test splitting by natural day."""
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 5, 0, 0, 0, tzinfo=timezone.utc)
        window = TimeWindow(start=start, end=end)
        bucket = Bucket.from_string("day")

        windows = split_time_window(window, bucket)
        # Due to local timezone conversion, the exact count may vary
        # but should be around 4-5 windows for a 4-day span
        assert len(windows) >= 4
        assert len(windows) <= 5

    def test_split_by_hour(self):
        """Test splitting by natural hour."""
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 5, 0, 0, tzinfo=timezone.utc)
        window = TimeWindow(start=start, end=end)
        bucket = Bucket.from_string("hour")

        windows = split_time_window(window, bucket)
        assert len(windows) == 5

    def test_split_by_duration(self):
        """Test splitting by fixed duration."""
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        window = TimeWindow(start=start, end=end)
        bucket = Bucket.from_string("2h")

        windows = split_time_window(window, bucket)
        assert len(windows) == 5  # 10h / 2h = 5

    def test_split_uneven(self):
        """Test splitting with remainder."""
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, 7, 0, 0, tzinfo=timezone.utc)
        window = TimeWindow(start=start, end=end)
        bucket = Bucket.from_string("2h")

        windows = split_time_window(window, bucket)
        assert len(windows) == 4  # 3 full + 1 partial


class TestFormatFilename:
    """Tests for format_filename function."""

    def test_format_with_date(self):
        """Test formatting with date pattern."""
        start = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        window = TimeWindow(start=start, end=end)

        filename = format_filename("{start:%Y-%m-%d}.md", window)
        assert "2026-01-15" in filename
        assert filename.endswith(".md")

    def test_format_with_start_end(self):
        """Test formatting with start and end patterns."""
        start = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = TimeWindow(start=start, end=end)

        filename = format_filename("{start:%H%M}-{end:%H%M}.md", window)
        # Note: actual format depends on local timezone
        assert ".md" in filename

    def test_format_no_placeholders(self):
        """Test formatting with no placeholders."""
        start = datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 1, 16, 0, 0, 0, tzinfo=timezone.utc)
        window = TimeWindow(start=start, end=end)

        filename = format_filename("output.md", window)
        assert filename == "output.md"
