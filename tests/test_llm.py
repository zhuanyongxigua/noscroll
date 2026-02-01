"""Tests for llm module."""

import tempfile
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from noscroll.llm import resolve_api_url, _append_log, _truncate_text, call_llm


class TestResolveApiUrl:
    """Tests for resolve_api_url function."""

    def test_resolve_api_url_base_v1_responses(self):
        """Test URL resolution for responses mode from /v1 base."""
        url = resolve_api_url("https://api.example.com/v1", "responses")
        assert url == "https://api.example.com/v1/responses"

    def test_resolve_api_url_base_v1_chat(self):
        """Test URL resolution for chat mode from /v1 base."""
        url = resolve_api_url("https://api.example.com/v1", "chat")
        assert url == "https://api.example.com/v1/chat/completions"

    def test_resolve_api_url_base_v1_completions(self):
        """Test URL resolution for completions mode from /v1 base."""
        url = resolve_api_url("https://api.example.com/v1", "completions")
        assert url == "https://api.example.com/v1/completions"

    def test_resolve_api_url_convert_chat_to_responses(self):
        """Test URL conversion from chat/completions to responses."""
        url = resolve_api_url("https://api.example.com/v1/chat/completions", "responses")
        assert url == "https://api.example.com/v1/responses"

    def test_resolve_api_url_convert_chat_to_completions(self):
        """Test URL conversion from chat/completions to completions."""
        url = resolve_api_url("https://api.example.com/v1/chat/completions", "completions")
        assert url == "https://api.example.com/v1/completions"

    def test_resolve_api_url_convert_completions_to_chat(self):
        """Test URL conversion from /completions to /chat/completions."""
        url = resolve_api_url("https://api.example.com/v1/completions", "chat")
        assert url == "https://api.example.com/v1/chat/completions"

    def test_resolve_api_url_convert_completions_to_responses(self):
        """Test URL conversion from /completions to /responses."""
        url = resolve_api_url("https://api.example.com/v1/completions", "responses")
        assert url == "https://api.example.com/v1/responses"

    def test_resolve_api_url_trailing_slash(self):
        """Test URL with trailing slash."""
        url = resolve_api_url("https://api.example.com/v1/", "responses")
        assert url == "https://api.example.com/v1/responses"

    def test_resolve_api_url_empty(self):
        """Test empty URL."""
        url = resolve_api_url("", "responses")
        assert url == ""

    def test_resolve_api_url_unknown_mode(self):
        """Test URL with unknown mode (pass through)."""
        url = resolve_api_url("https://api.example.com/custom", "unknown")
        assert url == "https://api.example.com/custom"


class TestAppendLog:
    """Tests for _append_log function."""

    def test_append_log_none_path(self):
        """Test append log with None path does nothing."""
        _append_log(None, {"type": "request"})
        # Should not raise

    def test_append_log_empty_path(self):
        """Test append log with empty path does nothing."""
        _append_log("", {"type": "request"})
        # Should not raise

    def test_append_log_creates_file(self):
        """Test append log creates file and directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "subdir" / "llm.log")
            _append_log(log_path, {"type": "response", "data": "test"})

            assert Path(log_path).exists()
            content = Path(log_path).read_text()
            assert "response" in content
            assert "test" in content

    def test_append_log_request_adds_newline(self):
        """Test request type adds newline separator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "llm.log")
            # First entry
            _append_log(log_path, {"type": "response", "n": 1})
            # Second entry (request type)
            _append_log(log_path, {"type": "request", "n": 2})

            content = Path(log_path).read_text()
            lines = content.strip().split("\n")
            assert len(lines) >= 2
            # Should have empty line between entries
            assert content.count("\n") >= 2


class TestTruncateText:
    """Tests for _truncate_text function."""

    def test_truncate_within_limit(self):
        """Test text within limit is unchanged."""
        result = _truncate_text("hello", 10)
        assert result == "hello"

    def test_truncate_at_limit(self):
        """Test text at exact limit."""
        result = _truncate_text("hello", 5)
        assert result == "hello"

    def test_truncate_over_limit(self):
        """Test text over limit is truncated."""
        result = _truncate_text("hello world", 5)
        assert result == "hello..."

    def test_truncate_empty(self):
        """Test empty string."""
        result = _truncate_text("", 10)
        assert result == ""

    def test_truncate_none(self):
        """Test None input."""
        result = _truncate_text(None, 10)  # type: ignore[arg-type]
        assert result == ""


class TestCallLlm:
    """Tests for call_llm function."""

    @pytest.mark.asyncio
    async def test_call_llm_missing_config(self):
        """Test call_llm raises on missing config."""
        with pytest.raises(RuntimeError, match="LLM config missing"):
            await call_llm(
                api_url="",
                api_key="key",
                model="model",
                mode="chat",
                system_prompt="system",
                user_prompt="user",
                timeout_ms=5000,
            )

    @pytest.mark.asyncio
    async def test_call_llm_missing_key(self):
        """Test call_llm raises on missing key."""
        with pytest.raises(RuntimeError, match="LLM config missing"):
            await call_llm(
                api_url="https://api.example.com",
                api_key="",
                model="model",
                mode="chat",
                system_prompt="system",
                user_prompt="user",
                timeout_ms=5000,
            )

    @pytest.mark.asyncio
    async def test_call_llm_missing_model(self):
        """Test call_llm raises on missing model."""
        with pytest.raises(RuntimeError, match="LLM config missing"):
            await call_llm(
                api_url="https://api.example.com",
                api_key="key",
                model="",
                mode="chat",
                system_prompt="system",
                user_prompt="user",
                timeout_ms=5000,
            )

    @pytest.mark.asyncio
    async def test_call_llm_chat_mode_success(self):
        """Test successful call in chat mode."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = '{"choices":[{"message":{"content":"Hello!"}}]}'
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}

        with patch("noscroll.llm.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await call_llm(
                api_url="https://api.example.com/v1/chat/completions",
                api_key="test-key",
                model="gpt-4",
                mode="chat",
                system_prompt="You are helpful.",
                user_prompt="Hi!",
                timeout_ms=5000,
            )

            assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_call_llm_completions_mode_success(self):
        """Test successful call in completions mode."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = '{"choices":[{"text":"Generated text"}]}'
        mock_response.json.return_value = {"choices": [{"text": "Generated text"}]}

        with patch("noscroll.llm.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await call_llm(
                api_url="https://api.example.com/v1/completions",
                api_key="test-key",
                model="gpt-4",
                mode="completions",
                system_prompt="System",
                user_prompt="Prompt",
                timeout_ms=5000,
            )

            assert result == "Generated text"

    @pytest.mark.asyncio
    async def test_call_llm_responses_mode_output_text(self):
        """Test successful call in responses mode with output_text."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = '{"output_text":"Response text"}'
        mock_response.json.return_value = {"output_text": "Response text"}

        with patch("noscroll.llm.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await call_llm(
                api_url="https://api.example.com/v1/responses",
                api_key="test-key",
                model="gpt-4",
                mode="responses",
                system_prompt="System",
                user_prompt="Prompt",
                timeout_ms=5000,
            )

            assert result == "Response text"

    @pytest.mark.asyncio
    async def test_call_llm_responses_mode_output_array(self):
        """Test successful call in responses mode with output array."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        output_data = {
            "output": [
                {"content": [{"text": "Part 1"}, {"text": " Part 2"}]}
            ]
        }
        mock_response.text = json.dumps(output_data)
        mock_response.json.return_value = output_data

        with patch("noscroll.llm.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await call_llm(
                api_url="https://api.example.com/v1/responses",
                api_key="test-key",
                model="gpt-4",
                mode="responses",
                system_prompt="System",
                user_prompt="Prompt",
                timeout_ms=5000,
            )

            assert result == "Part 1 Part 2"

    @pytest.mark.asyncio
    async def test_call_llm_api_error(self):
        """Test call_llm raises on API error."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("noscroll.llm.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            with pytest.raises(RuntimeError, match="LLM request failed"):
                await call_llm(
                    api_url="https://api.example.com/v1/chat/completions",
                    api_key="test-key",
                    model="gpt-4",
                    mode="chat",
                    system_prompt="System",
                    user_prompt="Prompt",
                    timeout_ms=5000,
                )

    @pytest.mark.asyncio
    async def test_call_llm_json_parse_error(self):
        """Test call_llm raises on JSON parse error."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = "not json"
        mock_response.json.side_effect = json.JSONDecodeError("error", "doc", 0)

        with patch("noscroll.llm.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            with pytest.raises(RuntimeError, match="LLM response parse error"):
                await call_llm(
                    api_url="https://api.example.com/v1/chat/completions",
                    api_key="test-key",
                    model="gpt-4",
                    mode="chat",
                    system_prompt="System",
                    user_prompt="Prompt",
                    timeout_ms=5000,
                )

    @pytest.mark.asyncio
    async def test_call_llm_infers_mode_from_url(self):
        """Test call_llm infers mode from URL when mode is invalid."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = '{"output_text":"Response"}'
        mock_response.json.return_value = {"output_text": "Response"}

        with patch("noscroll.llm.httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=None)
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await call_llm(
                api_url="https://api.example.com/v1/responses",
                api_key="test-key",
                model="gpt-4",
                mode="invalid",  # Should infer 'responses' from URL
                system_prompt="System",
                user_prompt="Prompt",
                timeout_ms=5000,
            )

            assert result == "Response"

    @pytest.mark.asyncio
    async def test_call_llm_with_log_path(self):
        """Test call_llm writes to log file."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = '{"choices":[{"message":{"content":"Hi!"}}]}'
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi!"}}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "llm.log")

            with patch("noscroll.llm.httpx.AsyncClient") as mock_client:
                mock_instance = MagicMock()
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=None)
                mock_instance.post = AsyncMock(return_value=mock_response)
                mock_client.return_value = mock_instance

                await call_llm(
                    api_url="https://api.example.com/v1/chat/completions",
                    api_key="test-key",
                    model="gpt-4",
                    mode="chat",
                    system_prompt="System",
                    user_prompt="Prompt",
                    timeout_ms=5000,
                    log_path=log_path,
                )

            # Verify log file was created
            assert Path(log_path).exists()
            content = Path(log_path).read_text()
            assert "request" in content
            assert "response" in content
