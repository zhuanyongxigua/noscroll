"""Configuration for the automation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from automation/.env if present
load_dotenv(Path(__file__).parent / ".env", override=False)


@dataclass
class AutomationConfig:
    """Configuration for automation runs."""
    
    # Loop settings
    max_fix_loops: int = 3
    
    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    artifacts_dir: Path = field(default_factory=lambda: Path(__file__).parent / "artifacts")
    
    # Debug settings
    verbose: bool = True
    
    # Claude Agent SDK settings
    permission_mode: str = "acceptEdits"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


def get_default_config() -> AutomationConfig:
    """Get default automation configuration."""
    return AutomationConfig()
