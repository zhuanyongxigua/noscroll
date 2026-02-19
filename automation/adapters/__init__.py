"""Adapters for external tools (pytest, git, etc.)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
