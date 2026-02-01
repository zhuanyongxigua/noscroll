"""Configuration module with 3-layer override: CLI > env > config file > defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore

from platformdirs import user_config_path


@dataclass
class Config:
    """Application configuration."""

    # Subscriptions
    subscriptions_path: str = "subscriptions/subscriptions.toml"

    # LLM settings
    llm_api_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_summary_model: str = ""
    llm_api_mode: str = ""
    llm_timeout_ms: int = 600000
    llm_global_concurrency: int = 5

    # Paths
    system_prompt_path: str = "prompts/system.txt"
    output_dir: str = "outputs"
    llm_log_path: str = "logs/llm-trace.log"
    feed_log_path: str = "logs/feed-items.log"

    # Runtime
    debug: bool = False


def default_config_path() -> Path:
    """Get the default config file path (e.g., ~/.config/noscroll/config.toml)."""
    return user_config_path("noscroll", ensure_exists=True) / "config.toml"


def _load_toml_config(path: Path) -> dict[str, Any]:
    """Load configuration from a TOML file."""
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _get_env(key: str, default: Any = None) -> Any:
    """Get environment variable with type coercion."""
    val = os.getenv(key)
    if val is None:
        return default
    # Handle boolean
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    # Handle int
    if isinstance(default, int):
        try:
            return int(val)
        except ValueError:
            return default
    return val


def load_config(
    cli_config_path: str | None = None,
    config_mode: str = "merge",
    cli_overrides: dict[str, Any] | None = None,
) -> Config:
    """
    Load configuration with 3-layer override: CLI > env > config file > defaults.

    Args:
        cli_config_path: Explicit config file path from CLI
        config_mode: 'merge' (overlay on defaults) or 'override' (replace defaults)
        cli_overrides: Additional CLI argument overrides

    Returns:
        Merged Config instance
    """
    # Determine config file path: CLI > env > default
    config_path: Path | None = None
    if cli_config_path:
        config_path = Path(cli_config_path)
    elif env_path := os.getenv("NOSCROLL_CONFIG"):
        config_path = Path(env_path)
    else:
        # Check default location
        default_path = default_config_path()
        if default_path.exists():
            config_path = default_path

    # Layer 1: Start with defaults (from dataclass) or empty config
    if config_mode == "override" and config_path and config_path.exists():
        # Override mode: ignore defaults, start fresh
        cfg = Config()
        file_config = _load_toml_config(config_path)
        for key, val in file_config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
    else:
        # Merge mode: start with defaults
        cfg = Config()
        # Layer 2: Override with config file
        if config_path and config_path.exists():
            file_config = _load_toml_config(config_path)
            for key, val in file_config.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, val)

    # Layer 3: Override with environment variables
    env_mapping = {
        "SUBSCRIPTIONS_PATH": "subscriptions_path",
        "FEEDS_PATH": "subscriptions_path",  # Legacy
        "OPML_PATH": "subscriptions_path",  # Legacy
        "LLM_API_URL": "llm_api_url",
        "LLM_API_KEY": "llm_api_key",
        "LLM_MODEL": "llm_model",
        "LLM_SUMMARY_MODEL": "llm_summary_model",
        "LLM_API_MODE": "llm_api_mode",
        "LLM_TIMEOUT_MS": "llm_timeout_ms",
        "LLM_GLOBAL_CONCURRENCY": "llm_global_concurrency",
        "SYSTEM_PROMPT_PATH": "system_prompt_path",
        "OUTPUT_DIR": "output_dir",
        "LLM_LOG_PATH": "llm_log_path",
        "FEED_LOG_PATH": "feed_log_path",
        "DEBUG": "debug",
    }

    for env_key, attr_name in env_mapping.items():
        current_val = getattr(cfg, attr_name)
        env_val = _get_env(env_key, current_val)
        if env_val != current_val or os.getenv(env_key) is not None:
            setattr(cfg, attr_name, env_val)

    # Layer 4: Override with CLI arguments
    if cli_overrides:
        for key, val in cli_overrides.items():
            if val is not None and hasattr(cfg, key):
                setattr(cfg, key, val)

    return cfg


# Global config instance (lazy loaded)
_config: Config | None = None


def get_config() -> Config:
    """Get the global config instance, loading it if necessary."""
    global _config
    if _config is None:
        # Load from .env file if present (for backward compatibility)
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        _config = load_config()
    return _config


def set_config(cfg: Config) -> None:
    """Set the global config instance (useful for testing or CLI override)."""
    global _config
    _config = cfg


# Backward compatibility: expose config as module-level variable
# Usage: from noscroll.config import config
class _ConfigProxy:
    """Proxy object for backward compatibility with `config.attr` access."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_config(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(get_config(), name, value)


config = _ConfigProxy()
