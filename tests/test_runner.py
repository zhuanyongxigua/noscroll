"""Tests for runner module."""

import json
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from noscroll.runner import (
    run_for_window,
    _fetch_rss,
    _fetch_web,
    _fetch_hn,
    _generate_output,
    _format_markdown,
    _format_json,
    _generate_llm_summary,
    _resolve_effective_subscriptions_path,
    _resolve_effective_system_prompt_path,
)
from noscroll.duration import TimeWindow
from noscroll.rss import FeedItem


def make_feed_item(title="Test Article", link="https://example.com/article", 
                   summary="Test summary", feed_title="Test Feed", 
                   feed_url="https://example.com/feed.xml", pub_date="2024-01-15") -> FeedItem:
    """Create a FeedItem for testing."""
    return FeedItem(
        feed_title=feed_title,
        feed_url=feed_url,
        title=title,
        link=link,
        pub_date=pub_date,
        summary=summary,
    )


class TestFormatJson:
    """Tests for _format_json function."""

    def test_format_empty(self):
        """Test formatting empty items list."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        result = _format_json([], window)
        parsed = json.loads(result)
        assert parsed["items"] == []
        assert "window" in parsed

    def test_format_single_item(self):
        """Test formatting single item."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        item = make_feed_item()
        result = _format_json([item], window)
        parsed = json.loads(result)
        assert len(parsed["items"]) == 1
        assert parsed["items"][0]["title"] == "Test Article"

    def test_format_multiple_items(self):
        """Test formatting multiple items."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        items = [make_feed_item(title=f"Article {i}") for i in range(3)]
        result = _format_json(items, window)
        parsed = json.loads(result)
        assert len(parsed["items"]) == 3

    def test_format_contains_window_info(self):
        """Test that output contains window information."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        result = _format_json([], window)
        parsed = json.loads(result)
        assert "window" in parsed
        assert "start" in parsed["window"]
        assert "end" in parsed["window"]


class TestGenerateOutput:
    """Tests for _generate_output function."""

    @pytest.mark.asyncio
    async def test_generate_json(self):
        """Test generating JSON output."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        item = make_feed_item()

        with patch("noscroll.runner.get_config") as mock_config:
            mock_config.return_value = MagicMock(llm_api_url="", llm_api_key="")
            result = await _generate_output([item], window, "json", False)

        parsed = json.loads(result)
        assert "items" in parsed

    @pytest.mark.asyncio
    async def test_generate_markdown_no_llm(self):
        """Test generating Markdown output without LLM configured."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        item = make_feed_item()

        with patch("noscroll.runner.get_config") as mock_config:
            mock_config.return_value = MagicMock(llm_api_url="", llm_api_key="")
            result = await _generate_output([item], window, "markdown", False)

        assert "# Digest" in result
        assert "Items collected: 1" in result


class TestFetchRSS:
    """Tests for _fetch_rss function."""

    @pytest.mark.asyncio
    async def test_fetch_rss_missing_subscriptions(self):
        """Test fetching RSS when subscriptions file is missing."""
        cfg = MagicMock()
        cfg.subscriptions_path = "/nonexistent/path.toml"

        start_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime(2024, 1, 10, tzinfo=timezone.utc).timestamp() * 1000)

        with patch("noscroll.opml.load_feeds", side_effect=FileNotFoundError()):
            result = await _fetch_rss(cfg, start_ms, end_ms, False)
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_rss_empty_feeds(self):
        """Test fetching RSS with empty feeds list."""
        cfg = MagicMock()
        cfg.subscriptions_path = "subscriptions/subscriptions.toml"

        start_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime(2024, 1, 10, tzinfo=timezone.utc).timestamp() * 1000)

        with patch("noscroll.opml.load_feeds", return_value=[]):
            result = await _fetch_rss(cfg, start_ms, end_ms, False)
            assert result == []

    @pytest.mark.asyncio
    async def test_fetch_rss_uses_builtin_fallback_when_config_missing(self):
        """Test RSS fetch falls back to built-in subscriptions path when configured path is missing."""
        cfg = MagicMock()
        cfg.subscriptions_path = "/nonexistent/path.toml"

        start_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime(2024, 1, 10, tzinfo=timezone.utc).timestamp() * 1000)

        fallback_path = Path("/tmp/builtin-subs.toml")
        with patch("noscroll.runner._resolve_effective_subscriptions_path", return_value=fallback_path):
            with patch("noscroll.opml.load_feeds", return_value=[] ) as mock_load_feeds:
                result = await _fetch_rss(cfg, start_ms, end_ms, False)
                assert result == []
                mock_load_feeds.assert_called_once_with(str(fallback_path))


class TestSubscriptionsPathResolution:
    """Tests for subscriptions path fallback resolution."""

    def test_resolve_effective_subscriptions_path_prefers_configured_path(self, tmp_path: Path):
        configured = tmp_path / "custom-subscriptions.toml"
        configured.write_text("[hackernews]\nenabled = true\n", encoding="utf-8")

        resolved = _resolve_effective_subscriptions_path(str(configured))
        assert resolved == configured

    def test_resolve_effective_subscriptions_path_returns_existing_fallback_when_config_missing(self):
        configured = "/path/that/does/not/exist/subscriptions.toml"
        resolved = _resolve_effective_subscriptions_path(configured)
        assert resolved != Path(configured)
        assert resolved.exists()


class TestSystemPromptPathResolution:
    """Tests for system prompt path fallback resolution."""

    def test_resolve_effective_system_prompt_path_prefers_configured_path(self, tmp_path: Path):
        configured = tmp_path / "custom-system-prompt.txt"
        configured.write_text("custom system prompt", encoding="utf-8")

        resolved = _resolve_effective_system_prompt_path(str(configured))
        assert resolved == configured

    def test_resolve_effective_system_prompt_path_returns_existing_fallback_when_config_missing(self):
        configured = "/path/that/does/not/exist/system.txt"
        resolved = _resolve_effective_system_prompt_path(configured)
        assert resolved != Path(configured)
        assert resolved.exists()


class TestSystemPromptFallbackUsage:
    """Tests for applying resolved system prompt fallback in LLM summarization."""

    @pytest.mark.asyncio
    async def test_generate_llm_summary_uses_fallback_system_prompt_when_config_missing(self):
        items = [make_feed_item(title="Test Item")]
        cfg = MagicMock()
        cfg.system_prompt_path = "/path/that/does/not/exist/system.txt"
        cfg.llm_api_url = "https://api.example.com/v1"
        cfg.llm_api_key = "test-key"
        cfg.llm_model = "gpt-4o-mini"
        cfg.llm_api_mode = "responses"
        cfg.llm_timeout_ms = 60000

        fallback_prompt_path = Path("/tmp/fallback-system-prompt.txt")
        fallback_prompt_content = "fallback prompt content"

        with patch("noscroll.runner._resolve_effective_system_prompt_path", return_value=fallback_prompt_path):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_text", return_value=fallback_prompt_content):
                    with patch("noscroll.llm.call_llm", new_callable=AsyncMock) as mock_call_llm:
                        mock_call_llm.return_value = "ok"
                        result = await _generate_llm_summary(items, cfg, debug=False, log_path=None)

        assert result == "ok"
        assert mock_call_llm.call_args.kwargs["system_prompt"] == fallback_prompt_content


class TestFetchWeb:
    """Tests for _fetch_web function."""

    @pytest.mark.asyncio
    async def test_fetch_web_no_crawler(self):
        """Test fetching web when crawler is not available."""
        cfg = MagicMock()
        cfg.subscriptions_path = "subscriptions/subscriptions.toml"

        start_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime(2024, 1, 10, tzinfo=timezone.utc).timestamp() * 1000)

        # The function should handle import error gracefully
        result = await _fetch_web(cfg, start_ms, end_ms, False)
        # It may return empty list if crawler is not available
        assert isinstance(result, list)


class TestFetchHN:
    """Tests for _fetch_hn function."""

    @pytest.mark.asyncio
    async def test_fetch_hn_disabled(self):
        """Test fetching HN when disabled in config."""
        cfg = MagicMock()
        cfg.subscriptions_path = "subscriptions/subscriptions.toml"

        start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_dt = datetime(2024, 1, 10, tzinfo=timezone.utc)

        # Create temp TOML with HN disabled
        toml_content = """
[hackernews]
enabled = false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            cfg.subscriptions_path = f.name

            try:
                result = await _fetch_hn(cfg, start_dt, end_dt, False)
                assert result == []
            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_fetch_hn_missing_config(self):
        """Test fetching HN with missing config uses defaults."""
        cfg = MagicMock()
        cfg.subscriptions_path = "/nonexistent/path.toml"

        start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_dt = datetime(2024, 1, 10, tzinfo=timezone.utc)

        with patch("noscroll.hackernews.fetch_hn_top_discussed", return_value=[]):
            result = await _fetch_hn(cfg, start_dt, end_dt, False)
            # Should return empty list when config doesn't exist
            assert isinstance(result, list)


class TestRunForWindow:
    """Tests for run_for_window function."""

    @pytest.mark.asyncio
    async def test_run_for_window_creates_output_file(self):
        """Test that run_for_window fails when no items are fetched."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_output.md"

            with patch("noscroll.runner.get_config") as mock_config:
                mock_cfg = MagicMock()
                mock_cfg.subscriptions_path = "/nonexistent/subs.toml"
                mock_cfg.llm_api_url = ""
                mock_cfg.llm_api_key = ""
                mock_config.return_value = mock_cfg

                with patch("noscroll.runner._fetch_rss", return_value=[]):
                    with patch("noscroll.runner._fetch_web", return_value=[]):
                        with patch("noscroll.runner._fetch_hn", return_value=[]):
                            with pytest.raises(RuntimeError, match="No content found"):
                                await run_for_window(
                                    window=window,
                                    source_types=["rss", "web", "hn"],
                                    output_path=output_path,
                                    output_format="markdown",
                                )

            assert not Path(output_path).exists()

    @pytest.mark.asyncio
    async def test_run_for_window_no_content_includes_source_breakdown(self):
        """No-content error should include requested sources and per-source item counts."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_output.md"

            with patch("noscroll.runner.get_config") as mock_config:
                mock_cfg = MagicMock()
                mock_cfg.subscriptions_path = "/nonexistent/subs.toml"
                mock_cfg.llm_api_url = ""
                mock_cfg.llm_api_key = ""
                mock_config.return_value = mock_cfg

                with patch("noscroll.runner._fetch_rss", return_value=[]):
                    with patch("noscroll.runner._fetch_web", return_value=[]):
                        with patch("noscroll.runner._fetch_hn", return_value=[]):
                            with pytest.raises(RuntimeError) as exc:
                                await run_for_window(
                                    window=window,
                                    source_types=["rss", "web", "hn"],
                                    output_path=output_path,
                                    output_format="markdown",
                                )

            message = str(exc.value)
            assert "No content found" in message
            assert "Requested sources: rss, web, hn" in message
            assert "Source fetch results:" in message
            assert "- rss: 0 items" in message
            assert "- web: 0 items" in message
            assert "- hn: 0 items" in message
            assert not Path(output_path).exists()

    @pytest.mark.asyncio
    async def test_run_for_window_rss_only(self):
        """Test running for window with RSS only."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_output.md"

            with patch("noscroll.runner.get_config") as mock_config:
                mock_cfg = MagicMock()
                mock_cfg.subscriptions_path = "/nonexistent/subs.toml"
                mock_cfg.llm_api_url = ""
                mock_cfg.llm_api_key = ""
                mock_config.return_value = mock_cfg

                with patch("noscroll.runner._fetch_rss", return_value=[]) as mock_rss:
                    with patch("noscroll.runner._fetch_web", return_value=[]) as mock_web:
                        with patch("noscroll.runner._fetch_hn", return_value=[]) as mock_hn:
                            with pytest.raises(RuntimeError, match="No content found"):
                                await run_for_window(
                                    window=window,
                                    source_types=["rss"],
                                    output_path=output_path,
                                    output_format="markdown",
                                )

                            # Only RSS should be called
                            mock_rss.assert_called_once()
                            mock_web.assert_not_called()
                            mock_hn.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_for_window_with_items(self):
        """Test running for window with actual items falls back to non-LLM markdown."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        items = [make_feed_item(title="Test RSS Item")]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_output.md"

            with patch("noscroll.runner.get_config") as mock_config:
                mock_cfg = MagicMock()
                mock_cfg.subscriptions_path = "/nonexistent/subs.toml"
                mock_cfg.llm_api_url = ""
                mock_cfg.llm_api_key = ""
                mock_config.return_value = mock_cfg

                with patch("noscroll.runner._fetch_rss", return_value=items):
                    with patch("noscroll.runner._fetch_web", return_value=[]):
                        with patch("noscroll.runner._fetch_hn", return_value=[]):
                            await run_for_window(
                                window=window,
                                source_types=["rss", "web", "hn"],
                                output_path=output_path,
                                output_format="markdown",
                            )

            assert Path(output_path).exists()
            content = Path(output_path).read_text(encoding="utf-8")
            assert "# Digest" in content
            assert "Items collected: 1" in content


class TestFormatMarkdown:
    """Tests for _format_markdown function."""

    @pytest.mark.asyncio
    async def test_format_markdown_no_llm(self):
        """Test markdown formatting when LLM is not configured."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        items = [make_feed_item(title="Test Article", summary="Test summary")]

        with patch("noscroll.runner.get_config") as mock_config:
            mock_config.return_value = MagicMock(llm_api_url="", llm_api_key="")
            result = await _format_markdown(items, window, False)

        assert "# Digest" in result
        assert "Items collected: 1" in result

    @pytest.mark.asyncio
    async def test_format_markdown_with_llm(self):
        """Test markdown formatting with LLM summary."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        items = [make_feed_item(title="Test Article")]

        with patch("noscroll.runner.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.llm_api_url = "https://api.example.com"
            mock_cfg.llm_api_key = "test-key"
            mock_cfg.llm_model = "gpt-4"
            mock_cfg.llm_api_mode = "chat"
            mock_cfg.llm_timeout_ms = 30000
            mock_cfg.system_prompt_path = "/nonexistent/prompt.txt"
            mock_config.return_value = mock_cfg

            with patch("noscroll.runner._generate_llm_summary", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = "This is the LLM summary"
                result = await _format_markdown(items, window, False)

        # LLM output is returned directly
        assert result == "This is the LLM summary"

    @pytest.mark.asyncio
    async def test_format_markdown_llm_error(self):
        """Test markdown formatting when LLM fails."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        items = [make_feed_item(title="Test Article")]

        with patch("noscroll.runner.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.llm_api_url = "https://api.example.com"
            mock_cfg.llm_api_key = "test-key"
            mock_config.return_value = mock_cfg

            with patch("noscroll.runner._generate_llm_summary", new_callable=AsyncMock) as mock_llm:
                mock_llm.side_effect = Exception("Connection failed")
                result = await _format_markdown(items, window, True)

        assert "# Digest" in result
        assert "Items collected: 1" in result


class TestFetchRSSMore:
    """Additional tests for _fetch_rss function."""

    @pytest.mark.asyncio
    async def test_fetch_rss_filters_by_window(self):
        """Test RSS items are filtered by time window."""
        cfg = MagicMock()
        cfg.subscriptions_path = "subscriptions/subscriptions.toml"

        start_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime(2024, 1, 10, tzinfo=timezone.utc).timestamp() * 1000)

        items = [make_feed_item(title="Item 1"), make_feed_item(title="Item 2")]

        with patch("noscroll.opml.load_feeds", return_value=[{"url": "http://test.com"}]):
            with patch("noscroll.rss.fetch_all_feeds", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = (items, [])
                with patch("noscroll.utils.filter_by_window", return_value=[items[0]]) as mock_filter:
                    result = await _fetch_rss(cfg, start_ms, end_ms, False)
                    mock_filter.assert_called_once()
                    assert len(result) == 1

    @pytest.mark.asyncio
    async def test_fetch_rss_handles_failures(self):
        """Test RSS fetch handles failures in debug mode."""
        cfg = MagicMock()
        cfg.subscriptions_path = "subscriptions/subscriptions.toml"

        start_ms = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime(2024, 1, 10, tzinfo=timezone.utc).timestamp() * 1000)

        failures = [{"title": "Failed Feed", "error": "Connection error"}]

        with patch("noscroll.opml.load_feeds", return_value=[{"url": "http://test.com"}]):
            with patch("noscroll.rss.fetch_all_feeds", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = ([], failures)
                with patch("noscroll.utils.filter_by_window", return_value=[]):
                    result = await _fetch_rss(cfg, start_ms, end_ms, True)
                    assert result == []


class TestRunForWindowMore:
    """Additional tests for run_for_window function."""

    @pytest.mark.asyncio
    async def test_run_for_window_json_output(self):
        """Test running for window with JSON output format."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        items = [make_feed_item(title="Test Item")]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_output.json"

            with patch("noscroll.runner.get_config") as mock_config:
                mock_cfg = MagicMock()
                mock_cfg.subscriptions_path = "/nonexistent/subs.toml"
                mock_cfg.llm_api_url = ""
                mock_cfg.llm_api_key = ""
                mock_config.return_value = mock_cfg

                with patch("noscroll.runner._fetch_rss", return_value=items):
                    with patch("noscroll.runner._fetch_web", return_value=[]):
                        with patch("noscroll.runner._fetch_hn", return_value=[]):
                            await run_for_window(
                                window=window,
                                source_types=["rss"],
                                output_path=output_path,
                                output_format="json",
                            )

            content = Path(output_path).read_text()
            parsed = json.loads(content)
            assert len(parsed["items"]) == 1
            assert parsed["items"][0]["title"] == "Test Item"

    @pytest.mark.asyncio
    async def test_run_for_window_debug_mode(self):
        """Test running for window with debug mode still raises on no-content."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/test_output.md"

            with patch("noscroll.runner.get_config") as mock_config:
                mock_cfg = MagicMock()
                mock_cfg.subscriptions_path = "/nonexistent/subs.toml"
                mock_cfg.llm_api_url = ""
                mock_cfg.llm_api_key = ""
                mock_config.return_value = mock_cfg

                with patch("noscroll.runner._fetch_rss", return_value=[]):
                    with patch("noscroll.runner._fetch_web", return_value=[]):
                        with patch("noscroll.runner._fetch_hn", return_value=[]):
                            with pytest.raises(RuntimeError, match="No content found"):
                                await run_for_window(
                                    window=window,
                                    source_types=["rss", "web", "hn"],
                                    output_path=output_path,
                                    output_format="markdown",
                                    debug=True,
                                )

            assert not Path(output_path).exists()