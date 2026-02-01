"""Fetch utilities for RSS feeds and crawled sites."""

from __future__ import annotations

from datetime import datetime

from .rss import FeedItem


def crawled_to_feed_items(feeds: list[dict]) -> list[FeedItem]:
    """Convert crawled site feeds to FeedItem objects."""
    items = []
    for feed in feeds:
        site_name = feed.get("name", "Unknown")
        site_url = feed.get("url", "")
        for item in feed.get("items", []):
            # Parse date if available
            pub_date = item.get("date")
            if pub_date:
                try:
                    dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    # Keep the ISO format string
                except (ValueError, AttributeError):
                    pass

            # Use LLM-generated summary if available, fallback to truncated content
            summary = item.get("summary") or item.get("content", "")[:500]
            items.append(FeedItem(
                feed_title=f"[Crawled] {site_name}",
                feed_url=site_url,
                title=item.get("title", "Untitled"),
                link=item.get("url", ""),
                pub_date=pub_date or "",
                summary=summary,
            ))
    return items
