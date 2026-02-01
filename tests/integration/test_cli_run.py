"""Integration tests for noscroll run command.

These tests verify CLI argument parsing and dry-run behavior.
Run with: pytest tests/integration/ -v
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from noscroll.cli import main


class TestRunDryRun:
    """Test noscroll run command in dry-run mode."""

    def test_last_5d_bucket_day_all_sources(self, capsys):
        """Test: 过去5天所有资源，每天一份"""
        result = main(["run", "--last", "5d", "--bucket", "day", "--dry-run"])
        
        assert result == 0
        captured = capsys.readouterr()
        
        # Verify output contains expected information
        assert "=== Dry Run ===" in captured.out
        assert "Source types: rss, web, hn" in captured.out
        assert "Bucket:" in captured.out
        assert "Files to generate" in captured.out
        # Should generate ~5-6 files (depending on time)
        assert ".md" in captured.out

    def test_last_5d_single_output_all_sources(self, capsys):
        """Test: 过去5天所有资源，合并成一份"""
        result = main(["run", "--last", "5d", "--out", "test_output.md", "--dry-run"])
        
        assert result == 0
        captured = capsys.readouterr()
        
        assert "=== Dry Run ===" in captured.out
        assert "Source types: rss, web, hn" in captured.out
        assert "Output file: test_output.md" in captured.out

    def test_last_5d_hn_bucket_day_chinese(self, capsys):
        """Test: 过去5天 HN，每天一份，中文输出"""
        result = main([
            "run",
            "--last", "5d",
            "--source-types", "hn",
            "--bucket", "day",
            "--lang", "zh",
            "--dry-run"
        ])
        
        assert result == 0
        captured = capsys.readouterr()
        
        assert "=== Dry Run ===" in captured.out
        assert "Source types: hn" in captured.out
        assert "Bucket:" in captured.out
        assert "Files to generate" in captured.out

    def test_last_5d_rss_single_output(self, capsys):
        """Test: 过去5天所有 RSS，合并一份"""
        result = main([
            "run",
            "--last", "5d",
            "--source-types", "rss",
            "--out", "rss_digest.md",
            "--dry-run"
        ])
        
        assert result == 0
        captured = capsys.readouterr()
        
        assert "=== Dry Run ===" in captured.out
        assert "Source types: rss" in captured.out
        assert "Output file: rss_digest.md" in captured.out


class TestRunWithLLMOptions:
    """Test noscroll run with LLM configuration options."""

    def test_serial_mode(self, capsys):
        """Test serial mode flag."""
        result = main([
            "run",
            "--last", "1d",
            "--serial",
            "--delay", "500",
            "--dry-run"
        ])
        
        assert result == 0

    def test_lang_option(self, capsys):
        """Test language option."""
        result = main([
            "run",
            "--last", "1d",
            "--lang", "ja",
            "--dry-run"
        ])
        
        assert result == 0

    def test_llm_api_url_override(self, capsys):
        """Test LLM API URL override via CLI."""
        result = main([
            "run",
            "--last", "1d",
            "--llm-api-url", "https://custom.api.com/v1",
            "--dry-run"
        ])
        
        assert result == 0


class TestRunTimeWindow:
    """Test time window parsing."""

    def test_last_duration_days(self, capsys):
        """Test --last with days."""
        result = main(["run", "--last", "10d", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Duration: 240.0 hours" in captured.out

    def test_last_duration_hours(self, capsys):
        """Test --last with hours."""
        result = main(["run", "--last", "48h", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Duration: 48.0 hours" in captured.out

    def test_last_duration_weeks(self, capsys):
        """Test --last with weeks."""
        result = main(["run", "--last", "2w", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Duration: 336.0 hours" in captured.out


class TestRunBucket:
    """Test bucket/output splitting."""

    def test_bucket_day(self, capsys):
        """Test bucket by day."""
        result = main(["run", "--last", "3d", "--bucket", "day", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Files to generate" in captured.out

    def test_bucket_hour(self, capsys):
        """Test bucket by hour."""
        result = main(["run", "--last", "6h", "--bucket", "hour", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Files to generate" in captured.out

    def test_bucket_duration(self, capsys):
        """Test bucket by custom duration."""
        result = main(["run", "--last", "12h", "--bucket", "2h", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Files to generate" in captured.out


class TestRunSourceTypes:
    """Test source type filtering."""

    def test_source_types_rss_only(self, capsys):
        """Test RSS only."""
        result = main(["run", "--last", "1d", "--source-types", "rss", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Source types: rss" in captured.out

    def test_source_types_hn_only(self, capsys):
        """Test HN only."""
        result = main(["run", "--last", "1d", "--source-types", "hn", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Source types: hn" in captured.out

    def test_source_types_web_only(self, capsys):
        """Test web only."""
        result = main(["run", "--last", "1d", "--source-types", "web", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Source types: web" in captured.out

    def test_source_types_multiple(self, capsys):
        """Test multiple source types."""
        result = main(["run", "--last", "1d", "--source-types", "rss,hn", "--dry-run"])
        assert result == 0
        captured = capsys.readouterr()
        assert "rss" in captured.out
        assert "hn" in captured.out


class TestRunOutputFormat:
    """Test output format options."""

    def test_format_markdown(self, capsys):
        """Test markdown output format."""
        result = main(["run", "--last", "1d", "--format", "markdown", "--dry-run"])
        assert result == 0

    def test_format_json(self, capsys):
        """Test JSON output format."""
        result = main(["run", "--last", "1d", "--format", "json", "--dry-run"])
        assert result == 0
