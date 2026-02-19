"""Integration tests for config path discovery and precedence."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from noscroll.cli import main


def _write_toml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parse_last_json(output: str) -> dict:
    """Parse the last JSON object from command output."""
    start = output.rfind("{")
    if start == -1:
        raise ValueError("No JSON object found in output")
    return json.loads(output[start:])


class TestConfigIsolationMatrix:
    """Isolated matrix tests for config discovery and precedence."""

    def test_env_vars_work_without_env_file(self, capsys, tmp_path):
        """No .env file is needed for process env variables to apply."""
        home_cfg = tmp_path / ".noscroll" / "config.toml"
        platform_cfg = tmp_path / ".config" / "noscroll" / "config.toml"

        with patch.dict(os.environ, {"LLM_MODEL": "env-only-model"}, clear=True):
            with patch("noscroll.config.default_config_path", return_value=home_cfg):
                with patch("noscroll.config.platform_default_config_path", return_value=platform_cfg):
                    result = main(["config", "print", "--format", "json"])

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert result == 0
        assert data["llm_model"] == "env-only-model"

    def test_home_preferred_with_warning_when_both_exist(self, capsys, tmp_path):
        """When both configs exist, home config wins and warning is emitted."""
        home_cfg = tmp_path / ".noscroll" / "config.toml"
        platform_cfg = tmp_path / ".config" / "noscroll" / "config.toml"

        _write_toml(home_cfg, 'llm_model = "home-model"\n')
        _write_toml(platform_cfg, 'llm_model = "platform-model"\n')

        with patch.dict(os.environ, {}, clear=True):
            with patch("noscroll.config.default_config_path", return_value=home_cfg):
                with patch("noscroll.config.platform_default_config_path", return_value=platform_cfg):
                    result = main(["config", "print", "--format", "json"])

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert result == 0
        assert data["llm_model"] == "home-model"
        assert "Multiple config files found" in captured.err

    def test_platform_fallback_when_home_missing(self, capsys, tmp_path):
        """Platform config is used when home config is absent."""
        home_cfg = tmp_path / ".noscroll" / "config.toml"
        platform_cfg = tmp_path / ".config" / "noscroll" / "config.toml"

        _write_toml(platform_cfg, 'llm_model = "platform-fallback-model"\n')

        with patch.dict(os.environ, {}, clear=True):
            with patch("noscroll.config.default_config_path", return_value=home_cfg):
                with patch("noscroll.config.platform_default_config_path", return_value=platform_cfg):
                    result = main(["config", "print", "--format", "json"])

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert result == 0
        assert data["llm_model"] == "platform-fallback-model"

    def test_env_config_path_overrides_default_discovery(self, capsys, tmp_path):
        """NOSCROLL_CONFIG overrides home/platform discovery."""
        home_cfg = tmp_path / ".noscroll" / "config.toml"
        platform_cfg = tmp_path / ".config" / "noscroll" / "config.toml"
        env_cfg = tmp_path / "custom" / "env-config.toml"

        _write_toml(home_cfg, 'llm_model = "home-model"\n')
        _write_toml(platform_cfg, 'llm_model = "platform-model"\n')
        _write_toml(env_cfg, 'llm_model = "env-config-model"\n')

        with patch.dict(os.environ, {"NOSCROLL_CONFIG": str(env_cfg)}, clear=True):
            with patch("noscroll.config.default_config_path", return_value=home_cfg):
                with patch("noscroll.config.platform_default_config_path", return_value=platform_cfg):
                    result = main(["config", "print", "--format", "json"])

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert result == 0
        assert data["llm_model"] == "env-config-model"
        assert "Multiple config files found" not in captured.err

    def test_cli_config_path_overrides_noscroll_config_env(self, capsys, tmp_path):
        """--config takes precedence over NOSCROLL_CONFIG."""
        env_cfg = tmp_path / "custom" / "env-config.toml"
        cli_cfg = tmp_path / "custom" / "cli-config.toml"

        _write_toml(env_cfg, 'llm_model = "env-config-model"\n')
        _write_toml(cli_cfg, 'llm_model = "cli-config-model"\n')

        with patch.dict(os.environ, {"NOSCROLL_CONFIG": str(env_cfg)}, clear=True):
            result = main(["--config", str(cli_cfg), "config", "print", "--format", "json"])

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert result == 0
        assert data["llm_model"] == "cli-config-model"

    def test_config_path_command_reports_active_path(self, capsys, tmp_path):
        """config path returns active discovered path (home preferred)."""
        home_cfg = tmp_path / ".noscroll" / "config.toml"
        platform_cfg = tmp_path / ".config" / "noscroll" / "config.toml"

        _write_toml(home_cfg, 'llm_model = "home-model"\n')
        _write_toml(platform_cfg, 'llm_model = "platform-model"\n')

        with patch.dict(os.environ, {}, clear=True):
            with patch("noscroll.config.default_config_path", return_value=home_cfg):
                with patch("noscroll.config.platform_default_config_path", return_value=platform_cfg):
                    result = main(["config", "path"])

        captured = capsys.readouterr()
        assert result == 0
        assert captured.out.strip() == str(home_cfg)

    def test_init_template_file_can_be_loaded_with_section_keys(self, capsys, tmp_path):
        """Config generated by init can be consumed after enabling section values."""
        init_cfg = tmp_path / ".noscroll" / "config.toml"
        platform_cfg = tmp_path / ".config" / "noscroll" / "config.toml"

        with patch.dict(os.environ, {}, clear=True):
            with patch("noscroll.config.default_config_path", return_value=init_cfg):
                with patch("noscroll.config.platform_default_config_path", return_value=platform_cfg):
                    init_result = main(["init"])

        assert init_result == 0
        assert init_cfg.exists()

        content = init_cfg.read_text(encoding="utf-8")
        content = content.replace(
            '# model = "gpt-4o-mini"',
            'model = "section-model"',
        )
        content = content.replace(
            '# subscriptions = "subscriptions/subscriptions.toml"  # maps to subscriptions_path',
            'subscriptions = "section/subscriptions.toml"',
        )
        content = content.replace(
            '# debug = false',
            'debug = true',
        )
        init_cfg.write_text(content, encoding="utf-8")

        with patch.dict(os.environ, {}, clear=True):
            print_result = main(["--config", str(init_cfg), "config", "print", "--format", "json"])

        captured = capsys.readouterr()
        data = _parse_last_json(captured.out)
        assert print_result == 0
        assert data["llm_model"] == "section-model"
        assert data["subscriptions_path"] == "section/subscriptions.toml"
        assert data["debug"] is True


class TestDoctorCommand:
    """Smoke test for doctor command."""

    def test_doctor_runs(self, capsys):
        """Doctor command should run and print diagnostics."""
        result = main(["doctor"])
        captured = capsys.readouterr()
        assert result in (0, 1)
        assert "LLM" in captured.out or "Configuration" in captured.out
