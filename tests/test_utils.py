"""Tests for utils module."""

import tempfile
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from noscroll.utils import (
    filter_by_window,
    filter_by_mode,
    format_for_llm,
    ensure_output_path,
    with_date_in_path,
    append_feed_log,
)
from noscroll.rss import FeedItem


def make_item(pub_date: str) -> FeedItem:
    """Create a test FeedItem."""
    return FeedItem(
        feed_title="Test Feed",
        feed_url="https://example.com/feed.xml",
        title="Test Item",
        link="https://example.com/item",
        pub_date=pub_date,
        summary="Test summary",
    )


def test_filter_by_window():
    """Test filter_by_window function."""
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    two_days_ago = now - timedelta(days=2)

    items = [
        make_item(now.isoformat()),
        make_item(yesterday.isoformat()),
        make_item(two_days_ago.isoformat()),
    ]

    # Filter for last 36 hours
    end_ms = int(now.timestamp() * 1000)
    start_ms = end_ms - 36 * 60 * 60 * 1000

    filtered = filter_by_window(items, start_ms, end_ms)
    # Should get items from now and yesterday (within 36 hours)
    assert len(filtered) >= 1


def test_filter_by_window_no_pub_date():
    """Test filter_by_window with items missing pub_date."""
    items = [make_item("")]
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - 24 * 60 * 60 * 1000
    
    filtered = filter_by_window(items, start_ms, end_ms)
    assert len(filtered) == 0


def test_filter_by_window_invalid_date():
    """Test filter_by_window with invalid date format."""
    items = [make_item("not-a-date")]
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - 24 * 60 * 60 * 1000
    
    filtered = filter_by_window(items, start_ms, end_ms)
    assert len(filtered) == 0


def test_filter_by_mode_all():
    """Test filter_by_mode with 'all' mode."""
    items = [make_item("2024-01-01T00:00:00Z") for _ in range(5)]
    filtered = filter_by_mode(items, "all")
    assert len(filtered) == 5


def test_filter_by_mode_24h():
    """Test filter_by_mode with '24h' mode."""
    now = datetime.now(timezone.utc)
    recent = now - timedelta(hours=12)
    old = now - timedelta(hours=48)
    
    items = [
        make_item(recent.isoformat()),
        make_item(old.isoformat()),
    ]
    filtered = filter_by_mode(items, "24h")
    assert len(filtered) == 1


def test_format_for_llm():
    """Test format_for_llm function."""
    items = [make_item("2024-01-01T00:00:00Z")]
    summaries = {"https://example.com/item": "LLM generated summary"}

    formatted = format_for_llm(items, summaries)
    assert len(formatted) == 1
    assert formatted[0]["feed"] == "Test Feed"
    assert formatted[0]["title"] == "Test Item"
    assert formatted[0]["llm_summary"] == "LLM generated summary"


def test_format_for_llm_no_summaries():
    """Test format_for_llm without summaries."""
    items = [make_item("2024-01-01T00:00:00Z")]
    
    formatted = format_for_llm(items)
    assert len(formatted) == 1
    assert formatted[0]["llm_summary"] == ""




def test_ensure_output_path():
    """Test ensure_output_path function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = ensure_output_path(tmpdir, "2024-01-15")
        assert path.suffix == ".md"
        assert "2024-01-15" in str(path)


def test_ensure_output_path_existing():
    """Test ensure_output_path with existing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the first file
        first_path = ensure_output_path(tmpdir, "2024-01-15")
        first_path.write_text("first")
        
        # Now request again - should get incremented name
        second_path = ensure_output_path(tmpdir, "2024-01-15")
        assert "2024-01-15-2" in str(second_path)




def test_with_date_in_path():
    """Test with_date_in_path function."""
    result = with_date_in_path("/output/file.md", "2024-01-15")
    assert "file-2024-01-15.md" in result


def test_with_date_in_path_empty_date():
    """Test with_date_in_path with empty date."""
    result = with_date_in_path("/output/file.md", "")
    assert result == "/output/file.md"


def test_append_feed_log():
    """Test append_feed_log function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = f"{tmpdir}/logs/feed.log"
        entries = [
            {"title": "Item 1", "link": "https://example.com/1"},
            {"title": "Item 2", "link": "https://example.com/2"},
        ]
        
        append_feed_log(log_path, entries)
        
        # Check file exists and has content
        assert Path(log_path).exists()
        content = Path(log_path).read_text()
        assert "Item 1" in content
        assert "Item 2" in content


def test_append_feed_log_empty():
    """Test append_feed_log with empty entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = f"{tmpdir}/logs/feed.log"
        
        append_feed_log(log_path, [])
        
        # File should not be created
        assert not Path(log_path).exists()


def test_append_feed_log_none_path():
    """Test append_feed_log with None path."""
    # Should not raise
    append_feed_log(None, [{"title": "Test"}])
    append_feed_log("", [{"title": "Test"}])
