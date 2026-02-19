"""Evaluation logic for checking task outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock


@dataclass
class EvalResult:
    """Result from evaluating task output."""
    passed: bool
    reason: str
    details: dict


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


CHECKER_SYSTEM_PROMPT = """You are an AI assistant that validates the output of NoScroll CLI commands.

Your job is to check if the output meets the expected behavior:
1. Expected output files exist and are not empty
2. Output content format is correct (markdown structure, etc.)
3. Content appears reasonable (has sections, summaries, links, etc.)

You have access to Read, Glob tools to examine files.

After examining the files, respond with ONLY a JSON object:
```json
{
    "passed": true/false,
    "reason": "brief explanation of pass/fail",
    "details": {
        "files_exist": true/false,
        "files_not_empty": true/false,
        "format_correct": true/false,
        "content_reasonable": true/false,
        "issues_found": ["list of specific issues if any"]
    }
}
```"""


async def evaluate_output(
    output_files: list[Path],
    expected_behavior: str,
    project_root: Path,
    verbose: bool = True,
) -> EvalResult:
    """Evaluate task output against expectations."""
    
    file_list = [str(f) for f in output_files]
    
    prompt = f"""Expected behavior: {expected_behavior}

Output files: {file_list}

Please examine the output files using Read tool, then provide your validation as JSON."""

    if verbose:
        print(f"[Eval] Checking {len(file_list)} files...")
    
    options = ClaudeAgentOptions(
        system_prompt=CHECKER_SYSTEM_PROMPT,
        cwd=str(project_root),
        permission_mode="default",
        allowed_tools=["Read", "Glob"],
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
            "passed": False,
            "reason": f"Could not parse response: {response_text[:500]}",
            "details": {}
        }
    
    result = EvalResult(
        passed=result_data.get("passed", False),
        reason=result_data.get("reason", "Unknown"),
        details=result_data.get("details", {})
    )
    
    if verbose:
        print(f"[Eval] Passed: {result.passed}")
        print(f"[Eval] Reason: {result.reason}")
    
    return result
