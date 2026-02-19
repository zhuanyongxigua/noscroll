"""Configuration for the automation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


# Load environment variables from automation/.env if present
load_dotenv(Path(__file__).parent / ".env", override=False)


@dataclass
class AutomationConfig:
    """Configuration for automation runs."""
    
    # Loop settings
    max_fix_loops: int = 3

    # Stage settings
    stage: Literal["dev", "test"] = "dev"
    
    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    artifacts_dir: Path = field(default_factory=lambda: Path(__file__).parent / "artifacts")

    # Test-stage gate settings
    run_local_tests: bool = True
    publish_testpypi: bool = True
    run_install_smoke: bool = True
    testpypi_repository: str = "testpypi"
    testpypi_index_url: str = "https://test.pypi.org/simple/"
    pypi_index_url: str = "https://pypi.org/simple/"
    package_name: str = "noscroll"
    cli_command: str = "noscroll"
    sandbox_root: Path = field(default_factory=lambda: Path(__file__).parent / "artifacts" / "sandboxes")
    
    # Debug settings
    verbose: bool = True
    
    # Claude Agent SDK settings
    permission_mode: str = "acceptEdits"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

        # Stage defaults
        if self.stage == "dev":
            self.publish_testpypi = False
            self.run_install_smoke = False


def get_default_config() -> AutomationConfig:
    """Get default automation configuration."""
    return AutomationConfig()
