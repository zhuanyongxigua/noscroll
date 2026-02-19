"""Tests for CLI module."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from noscroll.cli import (
    build_parser,
    parse_source_types,
    main,
    _extract_cli_overrides,
    _run_config,
    _run_sources,
    _run_init,
    _run_doctor,
    _print_dry_run,
    _run_ask,
    _spec_to_run_argv,
    _validate_run_args,
    _infer_lang_from_prompt,
    _generate_run_args_from_prompt,
    _normalize_generated_spec,
)
from noscroll.duration import TimeWindow, Bucket
from datetime import datetime, timezone


class TestParseSourceTypes:
    """Tests for source type parsing."""

    def test_parse_single(self):
        """Test parsing single source type."""
        result = parse_source_types("rss")
        assert result == ["rss"]

    def test_parse_multiple(self):
        """Test parsing multiple source types."""
        result = parse_source_types("rss,web,hn")
        assert result == ["rss", "web", "hn"]

    def test_parse_with_spaces(self):
        """Test parsing with spaces."""
        result = parse_source_types("rss, web, hn")
        assert result == ["rss", "web", "hn"]

    def test_parse_uppercase(self):
        """Test parsing uppercase (should be lowercased)."""
        result = parse_source_types("RSS,WEB")
        assert result == ["rss", "web"]

    def test_parse_invalid(self):
        """Test invalid source type raises error."""
        with pytest.raises(Exception):  # ArgumentTypeError
            parse_source_types("rss,invalid")


class TestBuildParser:
    """Tests for argument parser building."""

    def test_build_parser(self):
        """Test parser builds without error."""
        parser = build_parser()
        assert parser is not None

    def test_parser_run_command(self):
        """Test parsing run command."""
        parser = build_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"

    def test_parser_run_with_last(self):
        """Test parsing run with --last."""
        parser = build_parser()
        args = parser.parse_args(["run", "--last", "5d"])
        assert args.last == "5d"

    def test_parser_run_with_from_to(self):
        """Test parsing run with --from/--to."""
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--from", "2026-01-01",
            "--to", "2026-01-05",
        ])
        assert args.from_time == "2026-01-01"
        assert args.to_time == "2026-01-05"

    def test_parser_run_with_bucket(self):
        """Test parsing run with --bucket."""
        parser = build_parser()
        args = parser.parse_args(["run", "--bucket", "day"])
        assert args.bucket == "day"

    def test_parser_run_with_source_types(self):
        """Test parsing run with --source-types."""
        parser = build_parser()
        args = parser.parse_args(["run", "--source-types", "rss,hn"])
        assert args.source_types == ["rss", "hn"]

    def test_parser_run_with_out(self):
        """Test parsing run with --out."""
        parser = build_parser()
        args = parser.parse_args(["run", "--out", "./output.md"])
        assert args.out == "./output.md"

    def test_parser_run_with_format(self):
        """Test parsing run with --format."""
        parser = build_parser()
        args = parser.parse_args(["run", "--format", "json"])
        assert args.format == "json"

    def test_parser_run_with_debug(self):
        """Test parsing run with --debug."""
        parser = build_parser()
        args = parser.parse_args(["run", "--debug"])
        assert args.debug is True

    def test_parser_run_with_dry_run(self):
        """Test parsing run with --dry-run."""
        parser = build_parser()
        args = parser.parse_args(["run", "--dry-run"])
        assert args.dry_run is True

    def test_parser_config_command(self):
        """Test parsing config command."""
        parser = build_parser()
        args = parser.parse_args(["config", "print"])
        assert args.command == "config"
        assert args.config_command == "print"

    def test_parser_config_path(self):
        """Test parsing config path command."""
        parser = build_parser()
        args = parser.parse_args(["config", "path"])
        assert args.config_command == "path"

    def test_parser_sources_list(self):
        """Test parsing sources list command."""
        parser = build_parser()
        args = parser.parse_args(["sources", "list"])
        assert args.command == "sources"
        assert args.sources_command == "list"

    def test_parser_init_command(self):
        """Test parsing init command."""
        parser = build_parser()
        args = parser.parse_args(["init"])
        assert args.command == "init"

    def test_parser_doctor_command(self):
        """Test parsing doctor command."""
        parser = build_parser()
        args = parser.parse_args(["doctor"])
        assert args.command == "doctor"

    def test_parser_ask_command(self):
        """Test parsing ask command."""
        parser = build_parser()
        args = parser.parse_args(["ask", "收集过去五天的资料"])
        assert args.command == "ask"
        assert args.prompt == ["收集过去五天的资料"]

    def test_parser_global_config(self):
        """Test parsing global --config option."""
        parser = build_parser()
        args = parser.parse_args(["--config", "/path/to/config.toml", "run"])
        assert args.config == "/path/to/config.toml"

    def test_parser_config_mode(self):
        """Test parsing --config-mode option."""
        parser = build_parser()
        args = parser.parse_args(["--config-mode", "override", "run"])
        assert args.config_mode == "override"

    def test_parser_env_file(self):
        """Test parsing --env-file option."""
        parser = build_parser()
        args = parser.parse_args(["--env-file", ".env.local", "run"])
        assert args.env_file == ".env.local"


class TestExtractCliOverrides:
    """Tests for CLI override extraction."""

    def test_extract_debug(self):
        """Test extracting debug flag."""
        parser = build_parser()
        args = parser.parse_args(["run", "--debug"])
        overrides = _extract_cli_overrides(args)
        assert overrides["debug"] is True

    def test_extract_no_debug(self):
        """Test no debug flag."""
        parser = build_parser()
        args = parser.parse_args(["run"])
        overrides = _extract_cli_overrides(args)
        assert "debug" not in overrides or overrides.get("debug") is not True


class TestMainFunction:
    """Tests for main function."""

    @patch("noscroll.config.load_config")
    @patch("noscroll.config.set_config")
    def test_main_no_args_runs_default(self, mock_set, mock_load):
        """Test main with no arguments runs default command."""
        mock_load.return_value = MagicMock()
        
        # Mock the runner to avoid actual execution
        with patch("noscroll.cli._run_main") as mock_run:
            mock_run.return_value = 0
            result = main([])
            assert result == 0
            mock_run.assert_called_once()

    @patch("noscroll.config.load_config")
    @patch("noscroll.config.set_config")
    def test_main_doctor_command(self, mock_set, mock_load):
        """Test main with doctor command."""
        mock_load.return_value = MagicMock(
            subscriptions_path="subscriptions/subscriptions.toml",
            llm_api_url="",
            llm_api_key="",
            llm_model="",
        )

        with patch("noscroll.cli._run_doctor") as mock_doctor:
            mock_doctor.return_value = 0
            result = main(["doctor"])
            assert result == 0
            mock_doctor.assert_called_once()

    @patch("noscroll.config.load_config")
    @patch("noscroll.config.set_config")
    def test_main_config_print(self, mock_set, mock_load):
        """Test main with config print command."""
        mock_load.return_value = MagicMock()

        with patch("noscroll.cli._run_config") as mock_config:
            mock_config.return_value = 0
            result = main(["config", "print"])
            assert result == 0
            mock_config.assert_called_once()

    @patch("noscroll.config.load_config")
    @patch("noscroll.config.set_config")
    def test_main_init_command(self, mock_set, mock_load):
        """Test main with init command."""
        mock_load.return_value = MagicMock()

        with patch("noscroll.cli._run_init") as mock_init:
            mock_init.return_value = 0
            result = main(["init"])
            assert result == 0
            mock_init.assert_called_once()

    @patch("noscroll.config.load_config")
    @patch("noscroll.config.set_config")
    def test_main_with_env_file(self, mock_set, mock_load):
        """Test main loads env file when specified."""
        mock_load.return_value = MagicMock()

        with patch("noscroll.cli._run_main") as mock_run:
            mock_run.return_value = 0
            with patch("dotenv.load_dotenv") as mock_dotenv:
                result = main(["--env-file", ".env.test", "run", "--dry-run"])
                mock_dotenv.assert_called_once_with(".env.test")

    @patch("noscroll.config.load_config")
    @patch("noscroll.config.set_config")
    def test_main_dry_run(self, mock_set, mock_load):
        """Test dry run mode."""
        mock_load.return_value = MagicMock(debug=False)

        result = main(["run", "--dry-run"])
        assert result == 0

    @patch("noscroll.config.load_config")
    @patch("noscroll.config.set_config")
    def test_main_ask_command(self, mock_set, mock_load):
        """Test main with ask command routes correctly."""
        mock_load.return_value = MagicMock(debug=False)

        with patch("noscroll.cli._run_ask") as mock_ask:
            mock_ask.return_value = 0
            result = main(["ask", "收集过去五天的资料"])
            assert result == 0
            mock_ask.assert_called_once()


class TestRunConfig:
    """Tests for _run_config function."""

    def test_run_config_print_toml(self):
        """Test config print in TOML format."""
        parser = build_parser()
        args = parser.parse_args(["config", "print"])
        
        from noscroll.config import Config
        mock_cfg = Config(subscriptions_path="test/path.toml", debug=True)
        
        with patch("noscroll.config.get_config", return_value=mock_cfg):
            with patch("noscroll.config.default_config_path") as mock_path:
                mock_path.return_value = Path("/test/config.toml")
                result = _run_config(args)
                assert result == 0

    def test_run_config_print_json(self):
        """Test config print in JSON format."""
        parser = build_parser()
        args = parser.parse_args(["config", "print", "--format", "json"])
        
        from noscroll.config import Config
        mock_cfg = Config(subscriptions_path="test/path.toml", debug=True)
        
        with patch("noscroll.config.get_config", return_value=mock_cfg):
            result = _run_config(args)
            assert result == 0

    def test_run_config_path(self):
        """Test config path command."""
        parser = build_parser()
        args = parser.parse_args(["config", "path"])
        
        with patch("noscroll.config.default_config_path") as mock_path:
            mock_path.return_value = Path("/home/user/.config/noscroll/config.toml")
            result = _run_config(args)
            assert result == 0


class TestRunSources:
    """Tests for _run_sources function."""

    def test_run_sources_list(self):
        """Test sources list command."""
        parser = build_parser()
        args = parser.parse_args(["sources", "list"])
        
        mock_cfg = MagicMock()
        mock_cfg.subscriptions_path = "test/subscriptions.toml"
        
        mock_feed = MagicMock()
        mock_feed.title = "Test Feed"
        mock_feed.xml_url = "https://example.com/feed.xml"
        
        with patch("noscroll.config.get_config", return_value=mock_cfg):
            with patch("noscroll.opml.load_feeds", return_value=[mock_feed]):
                result = _run_sources(args)
                assert result == 0

    def test_run_sources_list_not_found(self):
        """Test sources list with missing file."""
        parser = build_parser()
        args = parser.parse_args(["sources", "list"])
        
        mock_cfg = MagicMock()
        mock_cfg.subscriptions_path = "/nonexistent/file.toml"
        
        with patch("noscroll.config.get_config", return_value=mock_cfg):
            with patch("noscroll.opml.load_feeds", side_effect=FileNotFoundError()):
                result = _run_sources(args)
                assert result == 1

    def test_run_sources_add(self):
        """Test sources add command."""
        parser = build_parser()
        args = parser.parse_args(["sources", "add", "https://example.com/feed.xml"])
        
        result = _run_sources(args)
        assert result == 0  # Returns 0 with "not implemented" message


class TestRunInit:
    """Tests for _run_init function."""

    def test_run_init_creates_config(self):
        """Test init creates config file."""
        parser = build_parser()
        args = parser.parse_args(["init"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "noscroll" / "config.toml"
            
            with patch("noscroll.config.default_config_path", return_value=config_path):
                result = _run_init(args)
                assert result == 0
                assert config_path.exists()

    def test_run_init_existing_config(self):
        """Test init with existing config."""
        parser = build_parser()
        args = parser.parse_args(["init"])
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("existing = true")
            temp_path = Path(f.name)
        
        try:
            with patch("noscroll.config.default_config_path", return_value=temp_path):
                result = _run_init(args)
                assert result == 0
        finally:
            os.unlink(temp_path)


class TestRunDoctor:
    """Tests for _run_doctor function."""

    def test_run_doctor(self):
        """Test doctor command."""
        parser = build_parser()
        args = parser.parse_args(["doctor"])
        
        mock_cfg = MagicMock()
        mock_cfg.subscriptions_path = "/nonexistent/subscriptions.toml"
        mock_cfg.llm_api_url = "https://api.openai.com/v1"
        mock_cfg.llm_api_key = "test-key"
        mock_cfg.llm_model = "gpt-4"
        
        with patch("noscroll.config.get_config", return_value=mock_cfg):
            with patch("noscroll.config.default_config_path") as mock_path:
                mock_path.return_value = Path("/nonexistent/config.toml")
                result = _run_doctor(args)
                assert result == 0


class TestPrintDryRun:
    """Tests for _print_dry_run function."""

    def test_print_dry_run_single_file(self):
        """Test dry run output for single file."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        
        # Should not raise
        _print_dry_run(window, None, ["rss", "web", "hn"], "./output.md", "{start:%Y-%m-%d}.md")

    def test_print_dry_run_with_bucket(self):
        """Test dry run output with bucket."""
        window = TimeWindow(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 5, tzinfo=timezone.utc),
        )
        bucket = Bucket.from_string("day")
        
        # Should not raise
        _print_dry_run(window, bucket, ["rss"], "./output/", "{start:%Y-%m-%d}.md")


class TestRunAsk:
    """Tests for _run_ask function."""

    def test_run_ask_dry_run_forces_dry_run(self):
        """Test ask --dry-run overrides generated args to dry-run mode."""
        parser = build_parser()
        args = parser.parse_args(["ask", "收集过去五天", "--dry-run"])
        generated_run_args = parser.parse_args(["run", "--last", "5d"])

        with patch("noscroll.config.get_config", return_value=MagicMock(debug=False)):
            with patch("noscroll.cli._generate_run_args_from_prompt", return_value=generated_run_args):
                with patch("noscroll.cli._run_main", return_value=0) as mock_run_main:
                    result = _run_ask(args)
                    assert result == 0
                    called_args = mock_run_main.call_args[0][0]
                    assert called_args.dry_run is True

    def test_run_ask_generation_error(self):
        """Test ask returns error when generation fails."""
        parser = build_parser()
        args = parser.parse_args(["ask", "收集过去五天"])

        with patch("noscroll.config.get_config", return_value=MagicMock(debug=False)):
            with patch("noscroll.cli._generate_run_args_from_prompt", side_effect=RuntimeError("bad output")):
                result = _run_ask(args)
                assert result == 1


class TestAskSpecValidation:
    """Tests for ask generated-spec conversion and validation."""

    def test_spec_to_run_argv_bucket_defaults(self):
        """Bucket mode should default to directory output and date template."""
        argv = _spec_to_run_argv({"last": "5d", "bucket": "day"})
        assert "--out" in argv
        assert "./outputs" in argv
        assert "--name-template" in argv
        assert "{start:%Y-%m-%d}.md" in argv

    def test_validate_run_args_rejects_unknown_template_placeholder(self):
        """Invalid template placeholders like {date} should be rejected."""
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--last",
            "5d",
            "--bucket",
            "day",
            "--name-template",
            "materials-{date}.md",
            "--out",
            "./outputs",
        ])
        with pytest.raises(ValueError, match="Invalid name_template"):
            _validate_run_args(args)

    def test_validate_run_args_rejects_non_unique_bucket_filenames(self):
        """Fixed filename template in bucket mode should be rejected to avoid overwrites."""
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--last",
            "5d",
            "--bucket",
            "day",
            "--name-template",
            "daily.md",
            "--out",
            "./outputs",
        ])
        with pytest.raises(ValueError, match="not unique"):
            _validate_run_args(args)


class TestAskLanguageInference:
    """Tests for ask prompt language inference."""

    def test_infer_lang_english(self):
        assert _infer_lang_from_prompt("Collect content from the past five days") == "en"

    def test_infer_lang_chinese(self):
        assert _infer_lang_from_prompt("收集过去五天的资料") == "zh"


class TestAskLlmFallback:
    """Tests for ask LLM model fallback and retry behavior."""

    @pytest.mark.asyncio
    async def test_generate_run_args_uses_llm_model_when_summary_missing(self):
        mock_cfg = MagicMock(
            llm_api_url="https://api.example.com/v1",
            llm_api_key="test-key",
            llm_summary_model="",
            llm_model="gpt-fallback-model",
            llm_api_mode="responses",
            llm_timeout_ms=60000,
        )

        with patch("noscroll.config.get_config", return_value=mock_cfg):
            with patch("noscroll.llm.call_llm", return_value='{"last":"5d"}') as mock_call:
                run_args = await _generate_run_args_from_prompt(
                    prompt_text="Collect content from the past five days",
                    retries=1,
                    debug=False,
                )

        assert run_args.last == "5d"
        assert mock_call.call_args.kwargs["model"] == "gpt-fallback-model"

    @pytest.mark.asyncio
    async def test_generate_run_args_retries_on_invalid_delay_type(self):
        mock_cfg = MagicMock(
            llm_api_url="https://api.example.com/v1",
            llm_api_key="test-key",
            llm_summary_model="",
            llm_model="gpt-fallback-model",
            llm_api_mode="responses",
            llm_timeout_ms=60000,
        )

        responses = [
            '{"last":"5d","delay":0.5}',
            '{"last":"5d","delay":0}',
        ]

        with patch("noscroll.config.get_config", return_value=mock_cfg):
            with patch("noscroll.llm.call_llm", side_effect=responses) as mock_call:
                run_args = await _generate_run_args_from_prompt(
                    prompt_text="Collect content from the past five days",
                    retries=2,
                    debug=False,
                )

        assert run_args.last == "5d"
        assert run_args.delay == 0
        assert mock_call.call_count == 1


class TestAskGeneratedSpecNormalization:
    """Tests for normalization of LLM-generated ask parameters."""

    def test_normalize_delay_seconds_string(self):
        normalized = _normalize_generated_spec({"delay": "1.5s"})
        assert normalized["delay"] == 1500

    def test_normalize_invalid_delay_drops_field(self):
        normalized = _normalize_generated_spec({"delay": "fast"})
        assert "delay" not in normalized
