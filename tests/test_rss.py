"""Tests for rss module."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

from noscroll.rss import (
    FeedItem,
    _parse_pub_date,
    _parse_summary,
    fetch_feed,
    fetch_all_feeds,
)
from noscroll.opml import Feed


class TestFeedItem:
    """Tests for FeedItem dataclass."""

    def test_create_feed_item(self):
        """Test creating a FeedItem."""
        item = FeedItem(
            feed_title="Test Feed",
            feed_url="https://example.com/feed.xml",
            title="Test Article",
            link="https://example.com/article",
            pub_date="2024-01-15T10:00:00Z",
            summary="Test summary",
        )
        assert item.feed_title == "Test Feed"
        assert item.title == "Test Article"
        assert item.link == "https://example.com/article"
        assert item.pub_date == "2024-01-15T10:00:00Z"
        assert item.summary == "Test summary"


class TestParsePubDate:
    """Tests for _parse_pub_date function."""

    def test_parse_published(self):
        """Test parsing 'published' field."""
        entry = {"published": "Mon, 15 Jan 2024 10:00:00 GMT"}
        result = _parse_pub_date(entry)
        assert result == "2024-01-15T10:00:00+00:00"

    def test_parse_updated(self):
        """Test parsing 'updated' field."""
        entry = {"updated": "2024-01-15T10:00:00Z"}
        result = _parse_pub_date(entry)
        assert result == "2024-01-15T10:00:00+00:00"

    def test_parse_created(self):
        """Test parsing 'created' field."""
        entry = {"created": "2024-01-15"}
        result = _parse_pub_date(entry)
        assert result == "2024-01-15T00:00:00+00:00"

    def test_parse_published_parsed(self):
        """Test parsing 'published_parsed' field."""
        entry = {"published_parsed": (2024, 1, 15, 10, 0, 0, 0, 15, 0)}
        result = _parse_pub_date(entry)
        assert "2024-01-15" in result

    def test_parse_empty(self):
        """Test parsing empty entry."""
        entry = {}
        result = _parse_pub_date(entry)
        assert result == ""

    def test_parse_invalid_parsed_date(self):
        """Test parsing invalid parsed date returns empty string."""
        entry = {"published_parsed": "invalid"}
        result = _parse_pub_date(entry)
        assert result == ""


class TestParseSummary:
    """Tests for _parse_summary function."""

    def test_parse_summary_field(self):
        """Test parsing 'summary' field."""
        entry = {"summary": "This is a summary"}
        result = _parse_summary(entry)
        assert result == "This is a summary"

    def test_parse_content_field(self):
        """Test parsing 'content' field."""
        entry = {"content": [{"value": "Content value"}]}
        result = _parse_summary(entry)
        assert result == "Content value"

    def test_parse_empty(self):
        """Test parsing empty entry."""
        entry = {}
        result = _parse_summary(entry)
        assert result == ""

    def test_parse_empty_content_list(self):
        """Test parsing empty content list."""
        entry = {"content": []}
        result = _parse_summary(entry)
        assert result == ""


class TestFetchFeed:
    """Tests for fetch_feed function."""

    @pytest.mark.asyncio
    async def test_fetch_feed_success(self):
        """Test successful feed fetch."""
        feed = Feed(title="Test Feed", xml_url="https://example.com/feed.xml")

        mock_response = MagicMock()
        mock_response.text = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Test Article</title>
      <link>https://example.com/article</link>
      <description>Test description</description>
      <pubDate>Mon, 15 Jan 2024 10:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""
        mock_response.raise_for_status = MagicMock()

        with patch("noscroll.rss.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            with patch("noscroll.rss.get_proxy", return_value=None):
                items = await fetch_feed(feed)

            assert len(items) == 1
            assert items[0].title == "Test Article"
            assert items[0].feed_title == "Test Feed"

    @pytest.mark.asyncio
    async def test_fetch_feed_http_error(self):
        """Test feed fetch with HTTP error."""
        feed = Feed(title="Test Feed", xml_url="https://example.com/feed.xml")

        with patch("noscroll.rss.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.get = AsyncMock(side_effect=Exception("Connection error"))
            mock_client.return_value = mock_instance

            with patch("noscroll.rss.get_proxy", return_value=None):
                with pytest.raises(Exception):
                    await fetch_feed(feed)


class TestFetchAllFeeds:
    """Tests for fetch_all_feeds function."""

    @pytest.mark.asyncio
    async def test_fetch_all_feeds_success(self):
        """Test fetching all feeds successfully."""
        feeds = [
            Feed(title="Feed 1", xml_url="https://example.com/feed1.xml"),
            Feed(title="Feed 2", xml_url="https://example.com/feed2.xml"),
        ]

        mock_items = [
            FeedItem(
                feed_title="Feed 1",
                feed_url="https://example.com/feed1.xml",
                title="Article 1",
                link="https://example.com/1",
                pub_date="2024-01-15",
                summary="Summary 1",
            )
        ]

        with patch("noscroll.rss.fetch_feed") as mock_fetch:
            mock_fetch.return_value = mock_items
            items, failures = await fetch_all_feeds(feeds)

            assert len(items) == 2  # One item from each feed
            assert len(failures) == 0

    @pytest.mark.asyncio
    async def test_fetch_all_feeds_partial_failure(self):
        """Test fetching all feeds with partial failure."""
        feeds = [
            Feed(title="Feed 1", xml_url="https://example.com/feed1.xml"),
            Feed(title="Feed 2", xml_url="https://example.com/feed2.xml"),
        ]

        mock_item = FeedItem(
            feed_title="Feed 1",
            feed_url="https://example.com/feed1.xml",
            title="Article 1",
            link="https://example.com/1",
            pub_date="2024-01-15",
            summary="Summary 1",
        )

        async def mock_fetch(feed, timeout=15.0):
            if "feed1" in feed.xml_url:
                return [mock_item]
            else:
                raise Exception("Network error")

        with patch("noscroll.rss.fetch_feed", side_effect=mock_fetch):
            items, failures = await fetch_all_feeds(feeds)

            assert len(items) == 1
            assert len(failures) == 1
            assert failures[0]["title"] == "Feed 2"

    @pytest.mark.asyncio
    async def test_fetch_all_feeds_empty(self):
        """Test fetching with no feeds."""
        items, failures = await fetch_all_feeds([])
        assert items == []
        assert failures == []
