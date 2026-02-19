"""Claude Agent SDK agents for the automation loop."""

from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

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
    session_log_path: Optional[Path] = None


@dataclass
class DiagnosticReport:
    """Diagnostic report describing issues found."""
    summary: str
    phenomena: list[str]
    debug_snippets: list[str]
    session_log_path: Optional[Path] = None


@dataclass
class FixResult:
    """Result from attempting to fix issues."""
    files_modified: list[str]
    changes_made: list[str]
    success: bool
    session_log_path: Optional[Path] = None


@dataclass
class AgentQueryResult:
    """Raw result from running an agent query with session logging."""

    response_text: str
    stderr_text: str
    session_log_path: Path


def _load_prompt(name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / f"{name}.txt"
    return prompt_path.read_text(encoding="utf-8")


def _build_session_log_path(config: AutomationConfig, agent_name: str) -> Path:
    """Create a deterministic session log path for an agent run."""
    session_dir = config.artifacts_dir / "session_logs"
    session_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return session_dir / f"{agent_name}_{stamp}.json"


def _persist_session_log(path: Path, payload: dict) -> None:
    """Write a structured session log payload."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


async def _run_agent_query(
    *,
    agent_name: str,
    prompt: str,
    system_prompt: str,
    config: AutomationConfig,
    allowed_tools: list[str],
    permission_mode: Literal["default", "acceptEdits", "plan", "bypassPermissions"],
) -> AgentQueryResult:
    """Run a Claude Agent query and capture a full session log."""

    session_log_path = _build_session_log_path(config, agent_name)
    cli_stderr: list[str] = []
    session_events: list[dict] = []

    def _capture_stderr(text: str) -> None:
        cli_stderr.append(text)
        session_events.append({"type": "stderr", "text": text})

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        cwd=str(config.project_root),
        permission_mode=permission_mode,
        allowed_tools=allowed_tools,
        stderr=_capture_stderr,
    )

    response_text = ""
    payload = {
        "agent": agent_name,
        "created_at": datetime.now().isoformat(),
        "cwd": str(config.project_root),
        "permission_mode": permission_mode,
        "allowed_tools": allowed_tools,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "events": session_events,
    }

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_text += block.text
                        session_events.append({"type": "assistant_text", "text": block.text})
                    else:
                        session_events.append(
                            {
                                "type": "assistant_block",
                                "block_type": type(block).__name__,
                            }
                        )
            else:
                session_events.append({"type": "message", "message_type": type(message).__name__})
        payload["status"] = "ok"
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = str(exc)
        payload["response_text"] = response_text
        payload["stderr"] = "".join(cli_stderr)
        _persist_session_log(session_log_path, payload)
        if cli_stderr:
            raise Exception(f"{exc}\n\n[claude-cli stderr]\n{''.join(cli_stderr)}") from exc
        raise

    payload["response_text"] = response_text
    payload["stderr"] = "".join(cli_stderr)
    _persist_session_log(session_log_path, payload)

    return AgentQueryResult(
        response_text=response_text,
        stderr_text="".join(cli_stderr),
        session_log_path=session_log_path,
    )


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
    
    query_result = await _run_agent_query(
        agent_name="executor",
        prompt=prompt,
        system_prompt=system_prompt,
        config=config,
        allowed_tools=["Bash", "Read", "Glob"],
        permission_mode="acceptEdits",
    )
    response_text = query_result.response_text
    
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
        stderr=query_result.stderr_text,
        output_files=output_files,
        success=success,
        session_log_path=query_result.session_log_path,
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
    
    query_result = await _run_agent_query(
        agent_name="diagnostic",
        prompt=prompt,
        system_prompt=system_prompt,
        config=config,
        allowed_tools=["Read", "Glob", "Grep"],
        permission_mode="default",
    )
    response_text = query_result.response_text
    
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
        debug_snippets=result_data.get("debug_snippets", []),
        session_log_path=query_result.session_log_path,
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
    
    query_result = await _run_agent_query(
        agent_name="fixer",
        prompt=prompt,
        system_prompt=system_prompt,
        config=config,
        allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
        permission_mode="acceptEdits",
    )
    response_text = query_result.response_text
    
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
        success=success,
        session_log_path=query_result.session_log_path,
    )
    
    if config.verbose:
        print(f"[Fixer] Files modified: {len(result.files_modified)}")
        for change in result.changes_made:
            print(f"  - {change}")
    
    return result
