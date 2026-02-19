"""Tests for config module."""

import io
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from noscroll.config import (
    Config,
    default_config_path,
    resolve_config_path,
    load_config,
    get_config,
    set_config,
    _load_toml_config,
    _get_env,
)


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_values(self):
        """Test default config values."""
        cfg = Config()
        assert cfg.subscriptions_path == "subscriptions/subscriptions.toml"
        assert cfg.debug is False
        assert cfg.llm_timeout_ms == 600000
        assert cfg.llm_global_concurrency == 5

    def test_custom_values(self):
        """Test setting custom values."""
        cfg = Config(
            subscriptions_path="custom/path.toml",
            debug=True,
            llm_model="gpt-4",
        )
        assert cfg.subscriptions_path == "custom/path.toml"
        assert cfg.debug is True
        assert cfg.llm_model == "gpt-4"


class TestDefaultConfigPath:
    """Tests for default_config_path function."""

    def test_returns_path(self):
        """Test that function returns a Path."""
        path = default_config_path()
        assert isinstance(path, Path)

    def test_path_contains_noscroll(self):
        """Test that path contains noscroll."""
        path = default_config_path()
        assert ".noscroll" in str(path)

    def test_path_ends_with_toml(self):
        """Test that path ends with config.toml."""
        path = default_config_path()
        assert path.name == "config.toml"


class TestLoadTomlConfig:
    """Tests for _load_toml_config function."""

    def test_load_valid_toml(self):
        """Test loading valid TOML file."""
        toml_content = """
llm_model = "gpt-4"
debug = true
llm_timeout_ms = 300000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            temp_path = Path(f.name)

        try:
            config = _load_toml_config(temp_path)
            assert config["llm_model"] == "gpt-4"
            assert config["debug"] is True
            assert config["llm_timeout_ms"] == 300000
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file returns empty dict."""
        config = _load_toml_config(Path("/nonexistent/path.toml"))
        assert config == {}

    def test_load_sectioned_toml(self):
        """Test loading sectioned TOML from init template format."""
        toml_content = """
[llm]
api_url = "https://api.example.com/v1"
model = "gpt-4o-mini"

[paths]
subscriptions = "custom/subscriptions.toml"
output_dir = "custom_outputs"

[runtime]
debug = true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            temp_path = Path(f.name)

        try:
            config = _load_toml_config(temp_path)
            assert config["llm_api_url"] == "https://api.example.com/v1"
            assert config["llm_model"] == "gpt-4o-mini"
            assert config["subscriptions_path"] == "custom/subscriptions.toml"
            assert config["output_dir"] == "custom_outputs"
            assert config["debug"] is True
        finally:
            os.unlink(temp_path)


class TestResolveConfigPath:
    """Tests for resolve_config_path function."""

    def test_cli_path_has_highest_priority(self):
        """CLI path should override env and discovered defaults."""
        with patch.dict(os.environ, {"NOSCROLL_CONFIG": "/tmp/env.toml"}, clear=True):
            with patch("noscroll.config._discover_default_config_path", return_value=Path("/tmp/default.toml")):
                path = resolve_config_path(cli_config_path="/tmp/cli.toml")
                assert path == Path("/tmp/cli.toml")

    def test_env_path_overrides_discovered_default(self):
        """NOSCROLL_CONFIG should override discovered default paths."""
        with patch.dict(os.environ, {"NOSCROLL_CONFIG": "/tmp/env.toml"}, clear=True):
            with patch("noscroll.config._discover_default_config_path", return_value=Path("/tmp/default.toml")):
                path = resolve_config_path()
                assert path == Path("/tmp/env.toml")

    def test_resolves_to_discovered_default(self):
        """When no CLI/env path, use discovered default path."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("noscroll.config._discover_default_config_path", return_value=Path("/tmp/default.toml")):
                path = resolve_config_path()
                assert path == Path("/tmp/default.toml")


class TestGetEnv:
    """Tests for _get_env function."""

    def test_get_string_env(self):
        """Test getting string environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = _get_env("TEST_VAR", "default")
            assert result == "test_value"

    def test_get_missing_env(self):
        """Test getting missing environment variable returns default."""
        result = _get_env("NONEXISTENT_VAR_12345", "default")
        assert result == "default"

    def test_get_bool_env_true(self):
        """Test getting boolean environment variable (true)."""
        with patch.dict(os.environ, {"TEST_BOOL": "true"}):
            result = _get_env("TEST_BOOL", False)
            assert result is True

    def test_get_bool_env_1(self):
        """Test getting boolean environment variable (1)."""
        with patch.dict(os.environ, {"TEST_BOOL": "1"}):
            result = _get_env("TEST_BOOL", False)
            assert result is True

    def test_get_bool_env_false(self):
        """Test getting boolean environment variable (false)."""
        with patch.dict(os.environ, {"TEST_BOOL": "false"}):
            result = _get_env("TEST_BOOL", True)
            assert result is False

    def test_get_int_env(self):
        """Test getting integer environment variable."""
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            result = _get_env("TEST_INT", 0)
            assert result == 42

    def test_get_int_env_invalid(self):
        """Test getting invalid integer returns default."""
        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            result = _get_env("TEST_INT", 10)
            assert result == 10


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_defaults(self):
        """Test loading with defaults only."""
        # Clear environment and don't specify config path
        with patch.dict(os.environ, {}, clear=True):
            with patch("noscroll.config._discover_default_config_path", return_value=None):
                cfg = load_config()
                assert cfg.subscriptions_path == "subscriptions/subscriptions.toml"

    def test_load_from_file(self):
        """Test loading from config file."""
        toml_content = """
llm_model = "custom-model"
debug = true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            temp_path = f.name

        try:
            with patch.dict(os.environ, {}, clear=True):
                cfg = load_config(cli_config_path=temp_path)
                assert cfg.llm_model == "custom-model"
                assert cfg.debug is True
        finally:
            os.unlink(temp_path)

    def test_load_merge_mode(self):
        """Test loading with merge mode (default)."""
        toml_content = """
llm_model = "custom-model"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            temp_path = f.name

        try:
            with patch.dict(os.environ, {}, clear=True):
                cfg = load_config(cli_config_path=temp_path, config_mode="merge")
                # Custom value from file
                assert cfg.llm_model == "custom-model"
                # Default value preserved
                assert cfg.subscriptions_path == "subscriptions/subscriptions.toml"
        finally:
            os.unlink(temp_path)

    def test_load_override_mode(self):
        """Test loading with override mode."""
        toml_content = """
llm_model = "custom-model"
subscriptions_path = "custom/subs.toml"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            temp_path = f.name

        try:
            with patch.dict(os.environ, {}, clear=True):
                cfg = load_config(cli_config_path=temp_path, config_mode="override")
                assert cfg.llm_model == "custom-model"
                assert cfg.subscriptions_path == "custom/subs.toml"
        finally:
            os.unlink(temp_path)

    def test_load_from_env_path(self):
        """Test loading from NOSCROLL_CONFIG environment variable."""
        toml_content = """
llm_model = "env-model"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"NOSCROLL_CONFIG": temp_path}, clear=True):
                cfg = load_config()
                assert cfg.llm_model == "env-model"
        finally:
            os.unlink(temp_path)

    def test_env_overrides_file(self):
        """Test environment variables override config file."""
        toml_content = """
llm_model = "file-model"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"LLM_MODEL": "env-model"}, clear=True):
                cfg = load_config(cli_config_path=temp_path)
                assert cfg.llm_model == "env-model"
        finally:
            os.unlink(temp_path)

    def test_cli_overrides_env(self):
        """Test CLI overrides environment variables."""
        with patch.dict(os.environ, {"DEBUG": "false"}, clear=True):
            with patch("noscroll.config._discover_default_config_path", return_value=None):
                cfg = load_config(cli_overrides={"debug": True})
                assert cfg.debug is True

    def test_home_and_platform_both_exist_warn_and_choose_home(self):
        """When both default files exist, choose home path and emit warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            home_path = Path(tmpdir) / ".noscroll" / "config.toml"
            platform_path = Path(tmpdir) / ".config" / "noscroll" / "config.toml"
            home_path.parent.mkdir(parents=True)
            platform_path.parent.mkdir(parents=True)

            home_path.write_text('llm_model = "home-model"\n', encoding="utf-8")
            platform_path.write_text('llm_model = "platform-model"\n', encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                with patch("noscroll.config.default_config_path", return_value=home_path):
                    with patch("noscroll.config.platform_default_config_path", return_value=platform_path):
                        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                            cfg = load_config()
                            assert cfg.llm_model == "home-model"
                            assert "Multiple config files found" in stderr.getvalue()


class TestGetSetConfig:
    """Tests for get_config and set_config."""

    def test_set_and_get_config(self):
        """Test setting and getting config."""
        cfg = Config(llm_model="test-model")
        set_config(cfg)
        retrieved = get_config()
        assert retrieved.llm_model == "test-model"

    def test_get_config_lazy_load(self):
        """Test get_config lazy loads if not set."""
        # Reset global config
        import noscroll.config as config_module
        config_module._config = None

        with patch.dict(os.environ, {}, clear=True):
            with patch("noscroll.config._discover_default_config_path", return_value=None):
                cfg = get_config()
                assert cfg is not None


# Note: TestGetProxy class removed - proxy handling now uses standard env vars
# (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY) via httpx trust_env=True
