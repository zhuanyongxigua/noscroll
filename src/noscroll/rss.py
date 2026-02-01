"""RSS feed fetching module."""

import asyncio
import html
import os
import re
from email.utils import parsedate_to_datetime
from datetime import timezone
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

import feedparser
import httpx

from .opml import Feed


def get_proxy() -> str | None:
    """Get proxy URL from environment, if set."""
    return os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("ALL_PROXY")


def _strip_html(text: str) -> str:
    """Strip HTML tags and decode entities, returning plain text."""
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities
    clean = html.unescape(clean)
    # Normalize whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


@dataclass
class FeedItem:
    """Represents a single RSS feed item."""

    feed_title: str
    feed_url: str
    title: str
    link: str
    pub_date: str
    summary: str


def _parse_pub_date(entry: dict) -> str:
    """Extract publication date from feed entry, normalized to ISO format.
    
    feedparser provides *_parsed fields as time.struct_time which we prefer
    over raw strings to ensure consistent ISO8601 output.
    """
    # Priority 1: Use feedparser's already-parsed date (most reliable)
    for field in ["published_parsed", "updated_parsed", "created_parsed"]:
        parsed = entry.get(field)
        if parsed:
            try:
                # Convert struct_time to ISO format with UTC timezone
                dt = datetime(*parsed[:6])
                return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            except (TypeError, ValueError):
                pass
    
    # Priority 2: Try to parse raw string without external dependencies
    for field in ["published", "updated", "created"]:
        val = entry.get(field)
        if val:
            try:
                # RFC2822 / RFC822
                dt = parsedate_to_datetime(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except Exception:
                try:
                    # ISO 8601 (handle trailing Z)
                    iso_val = val.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(iso_val)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc).isoformat()
                except Exception:
                    pass
    
    return ""


def _parse_summary(entry: dict) -> str:
    """Extract summary/content from feed entry, stripped of HTML."""
    raw = ""
    if entry.get("summary"):
        raw = entry["summary"]
    elif entry.get("content"):
        contents = entry["content"]
        if isinstance(contents, list) and contents:
            raw = contents[0].get("value", "")
    return _strip_html(raw)


async def fetch_feed(feed: Feed, timeout: float = 15.0) -> List[FeedItem]:
    """Fetch and parse a single RSS feed."""
    # trust_env=True makes httpx use HTTP_PROXY, HTTPS_PROXY, ALL_PROXY env vars
    async with httpx.AsyncClient(timeout=timeout, trust_env=True) as client:
        headers = {
            "User-Agent": "rss-client/0.1.0 (+https://example.local)",
            "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        }
        response = await client.get(feed.xml_url, headers=headers)
        response.raise_for_status()
        content = response.text

    parsed = feedparser.parse(content)
    # feedparser.feed is a FeedParserDict, use getattr for type safety
    parsed_feed = parsed.feed
    feed_title = str(getattr(parsed_feed, "title", None) or feed.title)

    items: List[FeedItem] = []
    for entry in parsed.entries:
        items.append(
            FeedItem(
                feed_title=feed_title,
                feed_url=feed.xml_url,
                title=str(entry.get("title") or "(untitled)"),
                link=str(entry.get("link") or ""),
                pub_date=_parse_pub_date(entry),
                summary=_parse_summary(entry),
            )
        )

    return items


async def fetch_all_feeds(
    feeds: List[Feed], timeout: float = 15.0
) -> tuple[List[FeedItem], List[dict]]:
    """Fetch all feeds concurrently.

    Returns:
        Tuple of (items, failures)
    """
    tasks = [fetch_feed(feed, timeout) for feed in feeds]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    items: List[FeedItem] = []
    failures: List[dict] = []

    for feed, result in zip(feeds, results):
        if isinstance(result, Exception):
            failures.append(
                {
                    "title": feed.title,
                    "url": feed.xml_url,
                    "error": str(result),
                }
            )
        elif isinstance(result, list):
            items.extend(result)

    return items, failures
