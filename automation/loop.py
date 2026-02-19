"""Main automation loop: run → test → eval → fix → repeat.

This is the core harness that orchestrates the automated iteration cycle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import AutomationConfig
from .agents import (
    execute_task,
    diagnose_failure,
    apply_fix,
    ExecutionResult,
    DiagnosticReport,
    FixResult,
)
from .evals import evaluate_output, EvalResult
from .tasks import (
    ALL_TASKS,
    BASIC_TASKS,
    EDGE_TASKS,
    COMBINED_TASKS,
    get_task_by_name,
    list_tasks,
    Task,
)
from .adapters import (
    run_unit_tests,
    run_integration_tests,
    build_package,
    upload_to_testpypi,
    read_package_version,
    install_from_testpypi,
    run_skills_install_smoke,
)


@dataclass
class LoopIteration:
    """Record of a single iteration in the fix loop."""
    iteration: int
    execution: ExecutionResult
    eval_result: EvalResult
    diagnostic: Optional[DiagnosticReport] = None
    fix: Optional[FixResult] = None


@dataclass
class LoopResult:
    """Final result of an automation run."""
    task_name: str
    instruction: str
    success: bool
    iterations: list[LoopIteration]
    total_time: float
    final_reason: str
    stopped_reason: str  # "passed", "max_loops", "fix_failed"
    gate_results: list[dict] = field(default_factory=list)


def _gate_entry(name: str, passed: bool, result: object, reason: str = "") -> dict:
    """Build a serializable gate result entry."""
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    return {
        "name": name,
        "passed": passed,
        "reason": reason,
        "stdout": stdout[-2000:] if stdout else "",
        "stderr": stderr[-2000:] if stderr else "",
    }


def _run_test_stage_gates(config: AutomationConfig) -> tuple[bool, str, list[dict]]:
    """Run test-stage gates after eval passes."""
    gate_results: list[dict] = []

    if config.run_local_tests:
        unit = run_unit_tests(cwd=config.project_root)
        unit_ok = unit.return_code == 0
        gate_results.append(
            _gate_entry("local_unit_tests", unit_ok, unit, "unit tests failed" if not unit_ok else "")
        )
        if not unit_ok:
            return False, "test gate failed: local unit tests", gate_results

        integration = run_integration_tests(cwd=config.project_root)
        integration_ok = integration.return_code == 0
        gate_results.append(
            _gate_entry(
                "local_integration_tests",
                integration_ok,
                integration,
                "integration tests failed" if not integration_ok else "",
            )
        )
        if not integration_ok:
            return False, "test gate failed: local integration tests", gate_results

    version = ""
    if config.publish_testpypi:
        build = build_package(cwd=config.project_root)
        build_ok = build.return_code == 0
        gate_results.append(_gate_entry("build_package", build_ok, build, "build failed" if not build_ok else ""))
        if not build_ok:
            return False, "test gate failed: package build", gate_results

        upload = upload_to_testpypi(repository=config.testpypi_repository, cwd=config.project_root)
        upload_ok = upload.return_code == 0
        gate_results.append(
            _gate_entry("upload_testpypi", upload_ok, upload, "TestPyPI upload failed" if not upload_ok else "")
        )
        if not upload_ok:
            return False, "test gate failed: TestPyPI upload", gate_results

    if config.run_install_smoke:
        version = read_package_version(config.project_root, config.package_name)
        install = install_from_testpypi(
            package_name=config.package_name,
            version=version,
            testpypi_index_url=config.testpypi_index_url,
            pypi_index_url=config.pypi_index_url,
            cwd=config.project_root,
        )
        install_ok = install.return_code == 0
        gate_results.append(
            _gate_entry(
                "install_from_testpypi",
                install_ok,
                install,
                f"install {config.package_name}=={version} from TestPyPI failed" if not install_ok else "",
            )
        )
        if not install_ok:
            return False, "test gate failed: install from TestPyPI", gate_results

        smoke = run_skills_install_smoke(
            cli_command=config.cli_command,
            skill_name="noscroll",
            sandbox_root=config.sandbox_root,
        )
        smoke_ok = smoke.return_code == 0
        gate_results.append(
            _gate_entry(
                "skills_install_smoke",
                smoke_ok,
                smoke,
                "skills install smoke failed in isolated directories" if not smoke_ok else "",
            )
        )
        if not smoke_ok:
            return False, "test gate failed: skills install smoke", gate_results

    return True, "all enabled test-stage gates passed", gate_results


async def run_loop(
    task_name: str,
    instruction: str,
    config: AutomationConfig,
    output_dir: Optional[Path] = None,
) -> LoopResult:
    """Run the automation loop for a single task.
    
    Loop: execute → eval → (if failed) diagnose → fix → repeat
    
    Stops when:
    - Evaluation passes
    - Max loops reached
    - Fix fails to apply
    """
    start_time = time.time()
    
    if output_dir is None:
        output_dir = config.artifacts_dir / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    iterations: list[LoopIteration] = []
    gate_results: list[dict] = []
    stopped_reason = "unknown"
    final_reason = ""
    
    if config.verbose:
        print(f"\n{'='*60}")
        print(f"[Loop] Starting: {task_name}")
        print(f"[Loop] Instruction: {instruction}")
        print(f"[Loop] Output dir: {output_dir}")
        print(f"{'='*60}")
    
    for loop_num in range(1, config.max_fix_loops + 1):
        if config.verbose:
            print(f"\n--- Iteration {loop_num}/{config.max_fix_loops} ---")
        
        # Step 1: Execute task
        if config.verbose:
            print("[Step 1] Executing task...")
        
        execution = await execute_task(instruction, output_dir, config)
        
        if config.verbose:
            print(f"  Success: {execution.success}")
            print(f"  Output files: {len(execution.output_files)}")
        
        # Step 2: Evaluate output
        if config.verbose:
            print("[Step 2] Evaluating output...")
        
        eval_result = await evaluate_output(
            execution.output_files,
            instruction,
            config.project_root,
            config.verbose,
        )
        
        # Record this iteration
        iteration = LoopIteration(
            iteration=loop_num,
            execution=execution,
            eval_result=eval_result,
        )
        
        # Check if we passed
        if eval_result.passed:
            iterations.append(iteration)
            if config.stage == "test":
                if config.verbose:
                    print("[Stage:test] Running release validation gates...")
                gates_ok, gates_reason, gate_results = _run_test_stage_gates(config)
                if gates_ok:
                    stopped_reason = "passed"
                    final_reason = f"{eval_result.reason}; {gates_reason}"
                    if config.verbose:
                        print(f"\n[Loop] SUCCESS after {loop_num} iteration(s)")
                else:
                    stopped_reason = "test_gate_failed"
                    final_reason = gates_reason
                    if config.verbose:
                        print(f"\n[Loop] FAILED test-stage gates after iteration {loop_num}")
            else:
                stopped_reason = "passed"
                final_reason = eval_result.reason
                if config.verbose:
                    print(f"\n[Loop] SUCCESS after {loop_num} iteration(s)")
            break
        
        # Step 3: If failed, diagnose
        if config.verbose:
            print("[Step 3] Diagnosing issue...")
        
        diagnostic = await diagnose_failure(execution, eval_result.reason, config)
        iteration.diagnostic = diagnostic
        
        # Step 4: Attempt fix
        if config.verbose:
            print("[Step 4] Attempting fix...")
        
        fix = await apply_fix(diagnostic, config)
        iteration.fix = fix
        
        iterations.append(iteration)
        
        # Check if fix was applied
        if not fix.success:
            if config.verbose:
                print(f"\n[Loop] Fix failed at iteration {loop_num}")
            stopped_reason = "fix_failed"
            final_reason = f"Could not apply fix: {fix.changes_made}"
            break
        
        # Check if we've reached max loops
        if loop_num == config.max_fix_loops:
            stopped_reason = "max_loops"
            final_reason = f"Reached maximum of {config.max_fix_loops} iterations"
            if config.verbose:
                print(f"\n[Loop] Max loops reached ({config.max_fix_loops})")
            break
    
    total_time = time.time() - start_time
    
    result = LoopResult(
        task_name=task_name,
        instruction=instruction,
        success=stopped_reason == "passed",
        iterations=iterations,
        total_time=total_time,
        final_reason=final_reason,
        stopped_reason=stopped_reason,
        gate_results=gate_results,
    )
    
    # Save result to artifacts
    _save_result(result, output_dir)
    
    return result


def _save_result(result: LoopResult, output_dir: Path):
    """Save loop result to JSON file."""
    result_file = output_dir / "loop_result.json"
    
    def serialize_iteration(it: LoopIteration) -> dict:
        return {
            "iteration": it.iteration,
            "execution": {
                "command": it.execution.command,
                "return_code": it.execution.return_code,
                "stdout_length": len(it.execution.stdout),
                "output_files": [str(f) for f in it.execution.output_files],
                "success": it.execution.success,
                "session_log_path": str(it.execution.session_log_path) if it.execution.session_log_path else None,
            },
            "eval": {
                "passed": it.eval_result.passed,
                "reason": it.eval_result.reason,
                "details": it.eval_result.details,
            },
            "diagnostic": {
                "summary": it.diagnostic.summary,
                "phenomena": it.diagnostic.phenomena,
                "session_log_path": str(it.diagnostic.session_log_path) if it.diagnostic.session_log_path else None,
            } if it.diagnostic else None,
            "fix": {
                "files_modified": it.fix.files_modified,
                "changes_made": it.fix.changes_made,
                "success": it.fix.success,
                "session_log_path": str(it.fix.session_log_path) if it.fix.session_log_path else None,
            } if it.fix else None,
        }
    
    result_dict = {
        "task_name": result.task_name,
        "instruction": result.instruction,
        "success": result.success,
        "total_time": result.total_time,
        "final_reason": result.final_reason,
        "stopped_reason": result.stopped_reason,
        "stage": "test" if result.gate_results else "dev",
        "gate_results": result.gate_results,
        "iterations": [serialize_iteration(it) for it in result.iterations],
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Loop] Result saved to: {result_file}")


def print_summary(results: list[LoopResult]):
    """Print a summary of results."""
    print("\n" + "=" * 60)
    print("AUTOMATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
    print(f"Success rate: {passed/len(results)*100:.1f}%")
    
    print("\nResults by task:")
    for result in results:
        status = "✓ PASS" if result.success else "✗ FAIL"
        iterations = len(result.iterations)
        print(f"  [{status}] {result.task_name} ({iterations} iteration(s), {result.total_time:.1f}s)")
        if not result.success:
            print(f"         Reason: {result.final_reason[:80]}...")
    
    print("\n" + "=" * 60)


def save_summary(results: list[LoopResult], output_dir: Path):
    """Save summary to JSON file."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tasks": len(results),
        "passed": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "results": [
            {
                "task_name": r.task_name,
                "success": r.success,
                "iterations": len(r.iterations),
                "time": r.total_time,
                "stopped_reason": r.stopped_reason,
                "final_reason": r.final_reason,
                "gate_results": r.gate_results,
            }
            for r in results
        ]
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_file}")


async def run_tasks(tasks_to_run: list[Task], config: AutomationConfig) -> list[LoopResult]:
    """Run all tasks."""
    results: list[LoopResult] = []
    
    for i, task in enumerate(tasks_to_run, 1):
        print(f"\n[{i}/{len(tasks_to_run)}] Running: {task.name}")
        
        output_dir = config.artifacts_dir / task.name
        result = await run_loop(task.name, task.instruction, config, output_dir)
        results.append(result)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NoScroll automation harness using Claude Agent SDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Task selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task",
        metavar="NAME",
        help=f"Run a specific task. Available: {', '.join(list_tasks())}",
    )
    group.add_argument(
        "--suite",
        choices=["basic", "edge", "combined"],
        help="Run a suite of tasks",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks",
    )
    group.add_argument(
        "--custom",
        metavar="INSTRUCTION",
        help="Run a custom task with natural language instruction",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List all available tasks",
    )
    
    # Options
    parser.add_argument(
        "--max-loops",
        type=int,
        default=3,
        help="Maximum number of fix iterations (default: 3)",
    )
    parser.add_argument(
        "--stage",
        choices=["dev", "test"],
        default="dev",
        help="Pipeline stage. dev=evaluate only, test=run release validation gates after eval pass",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        print("Available tasks:")
        print("\nBasic tasks:")
        for task in BASIC_TASKS:
            print(f"  {task.name}: {task.description}")
        print("\nEdge case tasks:")
        for task in EDGE_TASKS:
            print(f"  {task.name}: {task.description}")
        print("\nCombined tasks:")
        for task in COMBINED_TASKS:
            print(f"  {task.name}: {task.description}")
        return 0
    
    # Configure
    config = AutomationConfig(stage=args.stage)
    config.max_fix_loops = args.max_loops
    config.verbose = not args.quiet
    
    if args.output_dir:
        config.artifacts_dir = args.output_dir
    
    # Determine tasks to run
    tasks_to_run: list[Task] = []
    
    if args.task:
        task = get_task_by_name(args.task)
        if not task:
            print(f"Error: Unknown task '{args.task}'", file=sys.stderr)
            print(f"Available tasks: {', '.join(list_tasks())}", file=sys.stderr)
            return 1
        tasks_to_run = [task]
    
    elif args.suite:
        suite_map = {
            "basic": BASIC_TASKS,
            "edge": EDGE_TASKS,
            "combined": COMBINED_TASKS,
        }
        tasks_to_run = suite_map[args.suite]
    
    elif args.all:
        tasks_to_run = ALL_TASKS
    
    elif args.custom:
        tasks_to_run = [
            Task(
                name="custom_task",
                instruction=args.custom,
                description="Custom task from command line",
            )
        ]
    
    # Run automation
    print(f"\nRunning {len(tasks_to_run)} task(s)...")
    print(f"Max fix loops: {config.max_fix_loops}")
    print(f"Artifacts dir: {config.artifacts_dir}")
    
    results = asyncio.run(run_tasks(tasks_to_run, config))
    
    # Print and save summary
    print_summary(results)
    save_summary(results, config.artifacts_dir)
    
    # Return non-zero if any task failed
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
