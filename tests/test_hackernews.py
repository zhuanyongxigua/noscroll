"""Tests for hackernews module."""

import pytest
import datetime as dt
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock

from noscroll.hackernews import (
    parse_iso,
    to_unix_seconds,
    strip_html,
    fetch_story_item,
    extract_comments_text,
    fetch_article_content,
    fetch_window,
    fetch_top_discussed,
    hn_story_to_feed_item,
    fetch_hn_top_discussed,
    fetch_story_with_details,
)


class TestParseIso:
    """Tests for parse_iso function."""

    def test_parse_date_only(self):
        """Test parsing date-only string."""
        result = parse_iso("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.tzinfo == timezone.utc

    def test_parse_datetime(self):
        """Test parsing full datetime string."""
        result = parse_iso("2024-01-15T10:30:00")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30
        assert result.tzinfo == timezone.utc

    def test_parse_datetime_with_tz(self):
        """Test parsing datetime with timezone."""
        result = parse_iso("2024-01-15T10:30:00+00:00")
        assert result.year == 2024
        assert result.tzinfo == timezone.utc


class TestToUnixSeconds:
    """Tests for to_unix_seconds function."""

    def test_convert_datetime(self):
        """Test converting datetime to unix timestamp."""
        dt = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        result = to_unix_seconds(dt)
        # Known timestamp for 2024-01-15 00:00:00 UTC
        assert result == 1705276800

    def test_convert_specific_time(self):
        """Test converting specific time."""
        dt = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        result = to_unix_seconds(dt)
        # 12:30 is 12.5 hours = 45000 seconds after midnight
        assert result == 1705276800 + 12 * 3600 + 30 * 60


class TestStripHtml:
    """Tests for strip_html function."""

    def test_strip_basic_tags(self):
        """Test stripping basic HTML tags."""
        result = strip_html("<p>Hello <b>World</b></p>")
        assert "Hello" in result
        assert "World" in result
        assert "<" not in result

    def test_decode_entities(self):
        """Test decoding HTML entities."""
        result = strip_html("Hello &amp; World")
        assert result == "Hello & World"

    def test_empty_string(self):
        """Test empty string input."""
        result = strip_html("")
        assert result == ""

    def test_none_input(self):
        """Test None input."""
        result = strip_html(None)
        assert result == ""

    def test_normalize_whitespace(self):
        """Test normalizing whitespace."""
        result = strip_html("Hello    \n\n   World")
        assert result == "Hello World"


class TestExtractCommentsText:
    """Tests for extract_comments_text function."""

    def test_extract_single_comment(self):
        """Test extracting single comment."""
        item = {
            "children": [
                {
                    "text": "This is a comment",
                    "author": "user1",
                    "children": [],
                }
            ]
        }
        result = extract_comments_text(item)
        assert len(result) == 1
        assert "[user1]:" in result[0]
        assert "This is a comment" in result[0]

    def test_extract_nested_comments(self):
        """Test extracting nested comments."""
        item = {
            "children": [
                {
                    "text": "Parent comment",
                    "author": "user1",
                    "children": [
                        {
                            "text": "Child comment",
                            "author": "user2",
                            "children": [],
                        }
                    ],
                }
            ]
        }
        result = extract_comments_text(item)
        assert len(result) == 2
        assert any("Parent comment" in c for c in result)
        assert any("Child comment" in c for c in result)

    def test_extract_empty_children(self):
        """Test extracting with no children."""
        item = {"children": []}
        result = extract_comments_text(item)
        assert result == []

    def test_extract_with_html(self):
        """Test extracting comments with HTML."""
        item = {
            "children": [
                {
                    "text": "<p>Comment with <b>HTML</b></p>",
                    "author": "user1",
                    "children": [],
                }
            ]
        }
        result = extract_comments_text(item)
        assert len(result) == 1
        assert "<" not in result[0]

    def test_extract_anonymous(self):
        """Test extracting comment without author."""
        item = {
            "children": [
                {
                    "text": "Anonymous comment",
                    "author": None,
                    "children": [],
                }
            ]
        }
        result = extract_comments_text(item)
        assert len(result) == 1
        assert "[anonymous]:" in result[0]


class TestFetchStoryItem:
    """Tests for fetch_story_item function."""

    @pytest.mark.asyncio
    async def test_fetch_success(self):
        """Test successful story fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "123",
            "title": "Test Story",
            "children": [],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("noscroll.hackernews.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            with patch("noscroll.hackernews.get_proxy", return_value=None):
                result = await fetch_story_item("123")

            assert result is not None
            assert result["id"] == "123"
            assert result["title"] == "Test Story"

    @pytest.mark.asyncio
    async def test_fetch_error(self):
        """Test fetch with error returns None."""
        with patch("noscroll.hackernews.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.get = AsyncMock(side_effect=Exception("Network error"))
            mock_client.return_value = mock_instance

            with patch("noscroll.hackernews.get_proxy", return_value=None):
                result = await fetch_story_item("123")

            assert result is None


class TestFetchArticleContent:
    """Tests for fetch_article_content function."""

    @pytest.mark.asyncio
    async def test_fetch_article_empty_url(self):
        """Test fetching with empty URL returns empty string."""
        result = await fetch_article_content("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_fetch_article_hn_url(self):
        """Test fetching HN URL returns empty string."""
        result = await fetch_article_content("https://news.ycombinator.com/item?id=123")
        assert result == ""

    @pytest.mark.asyncio
    async def test_fetch_article_success(self):
        """Test successful article fetch."""
        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Article content here</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        with patch("noscroll.hackernews.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            with patch("noscroll.hackernews.get_proxy", return_value=None):
                result = await fetch_article_content("https://example.com/article")

            assert "Article content here" in result

    @pytest.mark.asyncio
    async def test_fetch_article_error(self):
        """Test article fetch error returns empty string."""
        with patch("noscroll.hackernews.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.get = AsyncMock(side_effect=Exception("Network error"))
            mock_client.return_value = mock_instance

            with patch("noscroll.hackernews.get_proxy", return_value=None):
                result = await fetch_article_content("https://example.com/article")

            assert result == ""

    @pytest.mark.asyncio
    async def test_fetch_article_non_html(self):
        """Test fetching non-HTML content returns empty string."""
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()

        with patch("noscroll.hackernews.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            with patch("noscroll.hackernews.get_proxy", return_value=None):
                result = await fetch_article_content("https://example.com/document.pdf")

            assert result == ""


class TestFetchWindow:
    """Tests for fetch_window function."""

    @pytest.mark.asyncio
    async def test_fetch_window_success(self):
        """Test successful window fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "hits": [{"id": "1", "title": "Story 1"}],
            "nbHits": 1,
            "nbPages": 1,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("noscroll.hackernews.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            with patch("noscroll.hackernews.get_proxy", return_value=None):
                hits, nb_hits = await fetch_window(1705276800, 1705363200)

            assert len(hits) == 1
            assert nb_hits == 1
            assert hits[0]["title"] == "Story 1"

    @pytest.mark.asyncio
    async def test_fetch_window_pagination(self):
        """Test window fetch with pagination."""
        mock_response1 = MagicMock()
        mock_response1.json.return_value = {
            "hits": [{"id": "1", "title": "Story 1"}],
            "nbHits": 2,
            "nbPages": 2,
        }
        mock_response1.raise_for_status = MagicMock()

        mock_response2 = MagicMock()
        mock_response2.json.return_value = {
            "hits": [{"id": "2", "title": "Story 2"}],
            "nbHits": 2,
            "nbPages": 2,
        }
        mock_response2.raise_for_status = MagicMock()

        with patch("noscroll.hackernews.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.get = AsyncMock(side_effect=[mock_response1, mock_response2])
            mock_client.return_value = mock_instance

            with patch("noscroll.hackernews.get_proxy", return_value=None):
                hits, nb_hits = await fetch_window(1705276800, 1705363200)

            assert len(hits) == 2
            assert nb_hits == 2


class TestFetchTopDiscussed:
    """Tests for fetch_top_discussed function."""

    @pytest.mark.asyncio
    async def test_fetch_top_discussed_basic(self):
        """Test basic top discussed fetch."""
        stories = [
            {"objectID": "1", "title": "Story 1", "num_comments": 100, "points": 50},
            {"objectID": "2", "title": "Story 2", "num_comments": 50, "points": 30},
        ]

        with patch("noscroll.hackernews.fetch_window", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (stories, 2)
            result = await fetch_top_discussed(
                start_ts=1705276800,
                end_ts=1705363200,
                top_n=10,
                min_comments=30,
            )

        assert len(result) == 2
        # Should be sorted by num_comments desc
        assert result[0]["num_comments"] == 100
        assert result[1]["num_comments"] == 50

    @pytest.mark.asyncio
    async def test_fetch_top_discussed_filters_low_comments(self):
        """Test that stories with low comments are filtered."""
        stories = [
            {"objectID": "1", "title": "Story 1", "num_comments": 100, "points": 50},
            {"objectID": "2", "title": "Story 2", "num_comments": 10, "points": 30},
        ]

        with patch("noscroll.hackernews.fetch_window", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (stories, 2)
            result = await fetch_top_discussed(
                start_ts=1705276800,
                end_ts=1705363200,
                top_n=10,
                min_comments=30,
            )

        assert len(result) == 1
        assert result[0]["num_comments"] == 100

    @pytest.mark.asyncio
    async def test_fetch_top_discussed_deduplicates(self):
        """Test that duplicate stories are deduplicated."""
        stories = [
            {"objectID": "1", "title": "Story 1", "num_comments": 100, "points": 50},
            {"objectID": "1", "title": "Story 1", "num_comments": 110, "points": 55},
        ]

        with patch("noscroll.hackernews.fetch_window", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = (stories, 2)
            result = await fetch_top_discussed(
                start_ts=1705276800,
                end_ts=1705363200,
                top_n=10,
                min_comments=30,
            )

        # Should keep the one with higher num_comments
        assert len(result) == 1
        assert result[0]["num_comments"] == 110


class TestHnStoryToFeedItem:
    """Tests for hn_story_to_feed_item function."""

    def test_convert_with_url(self):
        """Test converting story with external URL."""
        story = {
            "objectID": "123",
            "title": "Test Story",
            "url": "https://example.com/article",
            "num_comments": 50,
            "points": 100,
            "created_at": "2024-01-15T12:00:00Z",
        }
        item = hn_story_to_feed_item(story)

        assert item.title == "Test Story"
        assert item.link == "https://example.com/article"
        assert item.feed_title == "Hacker News (Top Discussed)"
        assert "Points: 100" in item.summary
        assert "Comments: 50" in item.summary
        assert "Original: https://example.com/article" in item.summary

    def test_convert_without_url(self):
        """Test converting story without external URL (Ask HN, etc)."""
        story = {
            "objectID": "123",
            "title": "Ask HN: Test",
            "url": None,
            "num_comments": 50,
            "points": 100,
            "created_at": "2024-01-15T12:00:00Z",
        }
        item = hn_story_to_feed_item(story)

        assert item.title == "Ask HN: Test"
        assert item.link == "https://news.ycombinator.com/item?id=123"
        assert "Original:" not in item.summary


class TestFetchHnTopDiscussed:
    """Tests for fetch_hn_top_discussed function."""

    @pytest.mark.asyncio
    async def test_fetch_returns_feed_items(self):
        """Test that function returns FeedItem objects."""
        stories = [
            {"objectID": "1", "title": "Story 1", "num_comments": 50, "points": 100},
        ]

        with patch("noscroll.hackernews.fetch_top_discussed", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = stories
            result = await fetch_hn_top_discussed(
                start_dt=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
                end_dt=dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc),
            )

        assert len(result) == 1
        # Should be FeedItem, not dict
        assert hasattr(result[0], 'title')
        assert hasattr(result[0], 'link')


class TestFetchStoryWithDetails:
    """Tests for fetch_story_with_details function."""

    @pytest.mark.asyncio
    async def test_fetch_with_content(self):
        """Test fetching story with article content."""
        story = {"objectID": "123", "url": "https://example.com/article"}

        with patch("noscroll.hackernews.fetch_article_content", new_callable=AsyncMock) as mock_content:
            mock_content.return_value = "Article text"
            with patch("noscroll.hackernews.fetch_story_item", new_callable=AsyncMock) as mock_item:
                mock_item.return_value = {"children": []}
                result = await fetch_story_with_details(story)

        assert result["article_content"] == "Article text"
        assert "comments" in result

    @pytest.mark.asyncio
    async def test_fetch_with_comments(self):
        """Test fetching story with comments."""
        story = {"objectID": "123", "url": ""}

        with patch("noscroll.hackernews.fetch_article_content", new_callable=AsyncMock) as mock_content:
            mock_content.return_value = ""
            with patch("noscroll.hackernews.fetch_story_item", new_callable=AsyncMock) as mock_item:
                mock_item.return_value = {
                    "children": [
                        {"text": "Comment 1", "author": "user1", "children": []},
                        {"text": "Comment 2", "author": "user2", "children": []},
                    ]
                }
                result = await fetch_story_with_details(story, fetch_content=False)

        assert len(result["comments"]) == 2
        assert any("Comment 1" in c for c in result["comments"])

    @pytest.mark.asyncio
    async def test_fetch_handles_exceptions(self):
        """Test that fetch handles exceptions gracefully."""
        story = {"objectID": "123", "url": "https://example.com"}

        with patch("noscroll.hackernews.fetch_article_content", new_callable=AsyncMock) as mock_content:
            mock_content.side_effect = Exception("Network error")
            with patch("noscroll.hackernews.fetch_story_item", new_callable=AsyncMock) as mock_item:
                mock_item.side_effect = Exception("API error")
                result = await fetch_story_with_details(story)

        # Should have empty values on error
        assert result["article_content"] == ""
        assert result["comments"] == []
