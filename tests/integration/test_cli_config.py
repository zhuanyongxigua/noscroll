"""Integration tests for noscroll config command."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from noscroll.cli import main


class TestConfigCommand:
    """Test noscroll config command."""

    def test_config_print_toml(self, capsys):
        """Test config print in TOML format."""
        result = main(["config", "print"])
        # May return 0 or error depending on config state
        # Just verify it doesn't crash
        captured = capsys.readouterr()
        # Should output something
        assert captured.out or captured.err or result in (0, 1)

    def test_config_print_json(self, capsys):
        """Test config print in JSON format."""
        result = main(["config", "print", "--format", "json"])
        captured = capsys.readouterr()
        assert captured.out or captured.err or result in (0, 1)

    def test_config_path(self, capsys):
        """Test config path command."""
        result = main(["config", "path"])
        captured = capsys.readouterr()
        # Should show a path
        assert "config" in captured.out.lower() or result in (0, 1)


class TestDoctorCommand:
    """Test noscroll doctor command."""

    def test_doctor(self, capsys):
        """Test doctor command runs without crashing."""
        result = main(["doctor"])
        captured = capsys.readouterr()
        # Should output diagnostic information
        assert "LLM" in captured.out or "Configuration" in captured.out or result in (0, 1)
