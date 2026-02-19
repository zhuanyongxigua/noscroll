"""Adapters for external tools (pytest, git, etc.)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tempfile
import tomllib
from datetime import datetime


@dataclass
class CommandResult:
    """Result from running an external command."""
    command: str
    return_code: int
    stdout: str
    stderr: str
    success: bool


def run_command(
    command: list[str],
    cwd: Optional[Path] = None,
    timeout: int = 300,
) -> CommandResult:
    """Run a shell command and capture output."""
    cmd_str = " ".join(command)
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(
            command=cmd_str,
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            success=result.returncode == 0,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            command=cmd_str,
            return_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            success=False,
        )
    except Exception as e:
        return CommandResult(
            command=cmd_str,
            return_code=-1,
            stdout="",
            stderr=str(e),
            success=False,
        )


def run_pytest(
    test_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
    args: Optional[list[str]] = None,
) -> CommandResult:
    """Run pytest with optional arguments."""
    command = ["python", "-m", "pytest"]
    if test_path:
        command.append(str(test_path))
    if args:
        command.extend(args)
    return run_command(command, cwd=cwd)


def run_ruff(
    path: Optional[Path] = None,
    cwd: Optional[Path] = None,
    fix: bool = False,
) -> CommandResult:
    """Run ruff linter."""
    command = ["python", "-m", "ruff", "check"]
    if fix:
        command.append("--fix")
    if path:
        command.append(str(path))
    else:
        command.append(".")
    return run_command(command, cwd=cwd)


def git_diff(cwd: Optional[Path] = None) -> CommandResult:
    """Get git diff of current changes."""
    return run_command(["git", "diff"], cwd=cwd)


def git_status(cwd: Optional[Path] = None) -> CommandResult:
    """Get git status."""
    return run_command(["git", "status", "--porcelain"], cwd=cwd)


def run_unit_tests(cwd: Optional[Path] = None) -> CommandResult:
    """Run unit tests (non-integration)."""
    return run_pytest(cwd=cwd, args=["tests/test_*.py", "-q"])


def run_integration_tests(cwd: Optional[Path] = None) -> CommandResult:
    """Run integration tests."""
    return run_pytest(cwd=cwd, args=["tests/integration", "-q"])


def build_package(cwd: Optional[Path] = None) -> CommandResult:
    """Build source and wheel artifacts."""
    return run_command(["python", "-m", "build"], cwd=cwd, timeout=600)


def upload_to_testpypi(repository: str = "testpypi", cwd: Optional[Path] = None) -> CommandResult:
    """Upload dist artifacts to TestPyPI repository alias."""
    return run_command(
        ["zsh", "-lc", f"python -m twine upload --repository {repository} dist/*"],
        cwd=cwd,
        timeout=600,
    )


def read_package_version(project_root: Path, package_name: str) -> str:
    """Read project version from pyproject.toml."""
    pyproject = project_root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"Could not resolve version for package: {package_name}")
    return version


def install_from_testpypi(
    package_name: str,
    version: str,
    testpypi_index_url: str,
    pypi_index_url: str,
    cwd: Optional[Path] = None,
) -> CommandResult:
    """Install package from TestPyPI into current Python environment."""
    spec = f"{package_name}=={version}"
    return run_command(
        [
            "python",
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-cache-dir",
            "--index-url",
            testpypi_index_url,
            "--extra-index-url",
            pypi_index_url,
            spec,
        ],
        cwd=cwd,
        timeout=600,
    )


def run_skills_install_smoke(
    cli_command: str,
    skill_name: str,
    sandbox_root: Path,
) -> CommandResult:
    """Verify external skills install behavior in isolated project directories.

    Isolation here refers to temporary project/workspace directories, not Python venv.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sandbox = Path(tempfile.mkdtemp(prefix=f"skills_smoke_{timestamp}_", dir=str(sandbox_root)))

    project_root = sandbox / "project"
    project_root.mkdir(parents=True, exist_ok=True)

    openclaw_ws = sandbox / "openclaw_workspace"
    openclaw_ws.mkdir(parents=True, exist_ok=True)

    fake_home = sandbox / "home"
    fake_home.mkdir(parents=True, exist_ok=True)

    script = " && ".join(
        [
            f"cd '{project_root}'",
            f"{cli_command} skills install {skill_name} --host claude --scope project",
            f"test -f '{project_root}/.claude/skills/{skill_name}/SKILL.md'",
            f"{cli_command} skills install {skill_name} --host codex --scope project",
            f"test -f '{project_root}/.agents/skills/{skill_name}/SKILL.md'",
            f"HOME='{fake_home}' {cli_command} skills install {skill_name} --host claude --scope user",
            f"test -f '{fake_home}/.claude/skills/{skill_name}/SKILL.md'",
            f"HOME='{fake_home}' {cli_command} skills install {skill_name} --host codex --scope user",
            f"test -f '{fake_home}/.agents/skills/{skill_name}/SKILL.md'",
            f"{cli_command} skills install {skill_name} --host openclaw --scope workspace --workdir '{openclaw_ws}'",
            f"test -f '{openclaw_ws}/skills/{skill_name}/SKILL.md'",
            f"HOME='{fake_home}' {cli_command} skills install {skill_name} --host openclaw --scope shared",
            f"test -f '{fake_home}/.openclaw/skills/{skill_name}/SKILL.md'",
            "echo SKILLS_SMOKE_OK",
        ]
    )

    return run_command(["zsh", "-lc", script], cwd=sandbox, timeout=600)
