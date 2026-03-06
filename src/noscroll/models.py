"""Domain models for NoScroll."""

from dataclasses import dataclass

@dataclass
class FeedItem:
    """Represents a single feed item (RSS, Web, HN, or X)."""
    feed_title: str
    feed_url: str
    title: str
    link: str
    pub_date: str
    summary: str

@dataclass
class Feed:
    """Represents an RSS feed configuration."""
    title: str
    xml_url: str
    html_url: str = ""
