"""Claude Agent SDK agents for the automation loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

from .config import AutomationConfig


@dataclass
class ExecutionResult:
    """Result from executing a task."""
    command: str
    return_code: int
    stdout: str
    stderr: str
    output_files: list[Path]
    success: bool


@dataclass
class DiagnosticReport:
    """Diagnostic report describing issues found."""
    summary: str
    phenomena: list[str]
    debug_snippets: list[str]


@dataclass
class FixResult:
    """Result from attempting to fix issues."""
    files_modified: list[str]
    changes_made: list[str]
    success: bool


def _load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / f"{name}.txt"
    return prompt_path.read_text(encoding="utf-8")


def _parse_json_response(response: str) -> dict:
    """Parse JSON from response text."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Could not parse JSON: {response[:500]}")


async def execute_task(
    instruction: str,
    output_dir: Path,
    config: AutomationConfig,
) -> ExecutionResult:
    """Execute a task based on natural language instruction."""
    
    prompt = f"""Task instruction: {instruction}

Output should be written to: {output_dir}

First output the JSON plan, then execute the noscroll command with Bash.
Remember to include --debug --serial --delay 1000 flags."""

    if config.verbose:
        print(f"[Executor] Instruction: {instruction}")
    
    system_prompt = _load_prompt("executor")
    
    cli_stderr: list[str] = []

    def _capture_stderr(text: str) -> None:
        cli_stderr.append(text)

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        cwd=str(config.project_root),
        permission_mode="acceptEdits",
        allowed_tools=["Bash", "Read", "Glob"],
        stderr=_capture_stderr,
    )
    
    response_text = ""
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
    except Exception as exc:
        if cli_stderr:
            raise Exception(f"{exc}\n\n[claude-cli stderr]\n{''.join(cli_stderr)}") from exc
        raise
    
    # Extract command info
    command = "unknown"
    try:
        plan = _parse_json_response(response_text)
        command = plan.get("command", "unknown")
    except Exception:
        pass
    
    if config.verbose:
        print(f"[Executor] Command: {command}")
    
    # Find output files
    output_files = []
    if output_dir.exists():
        if output_dir.is_dir():
            output_files = list(output_dir.glob("*.md")) + list(output_dir.glob("*.json"))
        else:
            output_files = [output_dir]
    
    success = len(output_files) > 0
    
    return ExecutionResult(
        command=command,
        return_code=0 if success else 1,
        stdout=response_text,
        stderr="".join(cli_stderr),
        output_files=output_files,
        success=success
    )


async def diagnose_failure(
    execution_result: ExecutionResult,
    eval_reason: str,
    config: AutomationConfig,
) -> DiagnosticReport:
    """Create a diagnostic report from failure."""
    
    prompt = f"""Analyze this failed NoScroll command execution:

Command: {execution_result.command}

Evaluation result: {eval_reason}

Execution output (last 3000 chars):
{execution_result.stdout[-3000:] if execution_result.stdout else "[empty]"}

Create a diagnostic report describing ONLY the observed phenomena. Do NOT suggest fixes."""

    if config.verbose:
        print(f"[Diagnostic] Analyzing failure...")
    
    system_prompt = _load_prompt("diagnostic")
    
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        cwd=str(config.project_root),
        permission_mode="default",
        allowed_tools=["Read", "Glob", "Grep"],
    )
    
    response_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
    
    try:
        result_data = _parse_json_response(response_text)
    except Exception:
        result_data = {
            "summary": "Failed to parse diagnostic response",
            "phenomena": [response_text[:1000]],
            "debug_snippets": []
        }
    
    report = DiagnosticReport(
        summary=result_data.get("summary", "Unknown issue"),
        phenomena=result_data.get("phenomena", []),
        debug_snippets=result_data.get("debug_snippets", [])
    )
    
    if config.verbose:
        print(f"[Diagnostic] Summary: {report.summary}")
    
    return report


async def apply_fix(
    diagnostic: DiagnosticReport,
    config: AutomationConfig,
) -> FixResult:
    """Attempt to fix issues based on diagnostic report."""
    
    prompt = f"""Diagnostic Report:
Summary: {diagnostic.summary}

Observed Phenomena:
{chr(10).join(f"- {p}" for p in diagnostic.phenomena)}

Debug Snippets:
{chr(10).join(diagnostic.debug_snippets)}

Please:
1. Read relevant source files to understand the issue
2. Make necessary code fixes using Edit tool
3. Summarize your changes in JSON format"""

    if config.verbose:
        print(f"[Fixer] Attempting to fix: {diagnostic.summary}")
    
    system_prompt = _load_prompt("fixer")
    
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        cwd=str(config.project_root),
        permission_mode="acceptEdits",
        allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
    )
    
    response_text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
    
    try:
        result_data = _parse_json_response(response_text)
        files_modified = result_data.get("files_modified", [])
        changes_made = result_data.get("changes_made", [])
        success = len(files_modified) > 0
    except Exception:
        files_modified = []
        changes_made = [f"Could not parse response: {response_text[:500]}"]
        success = False
    
    result = FixResult(
        files_modified=files_modified,
        changes_made=changes_made,
        success=success
    )
    
    if config.verbose:
        print(f"[Fixer] Files modified: {len(result.files_modified)}")
        for change in result.changes_made:
            print(f"  - {change}")
    
    return result
