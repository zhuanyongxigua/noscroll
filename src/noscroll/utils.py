"""Utility functions."""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List

from .rss import FeedItem


def filter_by_window(
    items: List[FeedItem], start_ms: int, end_ms: int
) -> List[FeedItem]:
    """Filter items by time window (milliseconds)."""
    result = []
    for item in items:
        if not item.pub_date:
            continue
        try:
            dt = datetime.fromisoformat(item.pub_date.replace("Z", "+00:00"))
            ts = int(dt.timestamp() * 1000)
            if start_ms <= ts < end_ms:
                result.append(item)
        except (ValueError, TypeError):
            pass
    return result


def filter_by_mode(items: List[FeedItem], mode: str) -> List[FeedItem]:
    """Filter items by mode ('all' or '24h')."""
    if mode == "all":
        return items
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - 24 * 60 * 60 * 1000
    return filter_by_window(items, start_ms, end_ms)


def format_for_llm(items: List[FeedItem], llm_summaries: dict = None) -> List[dict]:
    """Format items for LLM input."""
    llm_summaries = llm_summaries or {}
    return [
        {
            "feed": item.feed_title,
            "title": item.title,
            "link": item.link,
            "published_at": item.pub_date,
            "summary": item.summary,
            "llm_summary": llm_summaries.get(item.link, ""),
        }
        for item in items
    ]


def ensure_output_path(output_dir: str, date_override: str = None) -> Path:
    """Ensure output directory exists and return unique file path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    date_str = date_override or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base_name = output_path / date_str

    file_path = base_name.with_suffix(".md")
    counter = 1
    while file_path.exists():
        counter += 1
        file_path = output_path / f"{date_str}-{counter}.md"

    return file_path


def with_date_in_path(base_path: str, date_str: str) -> str:
    """Add date suffix to a file path."""
    path = Path(base_path)
    suffix = f"-{date_str}" if date_str else ""
    return str(path.parent / f"{path.stem}{suffix}{path.suffix}")


def append_feed_log(log_path: str | None, entries: List[dict]) -> None:
    """Append feed log entries to file."""
    if not log_path or not entries:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = "".join(json.dumps(entry) + "\n" for entry in entries)
    with open(path, "a", encoding="utf-8") as f:
        f.write(lines)
