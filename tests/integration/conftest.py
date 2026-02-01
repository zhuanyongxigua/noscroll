"""Pytest configuration for integration tests."""

import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# Load .env file at the beginning
def _load_dotenv():
    """Load .env file from project root."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value

_load_dotenv()


@pytest.fixture(autouse=True)
def reset_config():
    """Reset global config before each test (reload from env vars)."""
    from noscroll.config import set_config, load_config
    
    # Force reload config from environment
    set_config(None)  # Clear the cached config
    fresh_cfg = load_config()
    set_config(fresh_cfg)
    yield
    # No need to reset after - let the next test handle it


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir
