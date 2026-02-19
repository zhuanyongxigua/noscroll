"""CLI entry point for NoScroll."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from argparse import Namespace
    from .duration import TimeWindow, Bucket

# Valid source types
SOURCE_TYPES = ("rss", "web", "hn")
SourceType = Literal["rss", "web", "hn"]


def _env(key: str, default: str | None = None) -> str | None:
    """Get environment variable with NOSCROLL_ prefix."""
    return os.getenv(f"NOSCROLL_{key}", default)


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable with NOSCROLL_ prefix."""
    val = os.getenv(f"NOSCROLL_{key}")
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


def _env_int(key: str, default: int) -> int:
    """Get integer environment variable with NOSCROLL_ prefix."""
    val = os.getenv(f"NOSCROLL_{key}")
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _default_parallel_mode() -> bool:
    """Resolve default parallel mode from env, defaulting to serial."""
    parallel = os.getenv("NOSCROLL_PARALLEL")
    if parallel is not None:
        return parallel.lower() in ("1", "true", "yes")

    serial = os.getenv("NOSCROLL_SERIAL")
    if serial is not None:
        return not (serial.lower() in ("1", "true", "yes"))

    return False


def parse_source_types(value: str) -> list[SourceType]:
    """Parse comma-separated source types."""
    types = [t.strip().lower() for t in value.split(",")]
    invalid = [t for t in types if t not in SOURCE_TYPES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid source type(s): {', '.join(invalid)}. "
            f"Valid types: {', '.join(SOURCE_TYPES)}"
        )
    return types  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
        prog="noscroll",
        description="Pull, don't scroll. RSS aggregator with LLM-powered summarization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  noscroll run                              # Fetch last 10 days, output to ./noscroll.md
  noscroll run --last 20d                   # Fetch last 20 days
  noscroll run --last 36h                   # Fetch last 36 hours
  noscroll run --last 5d --bucket day       # Fetch 5 days, one file per day
  noscroll run --source-types rss           # Only RSS feeds (no web/hn)
  noscroll run --source-types rss,hn        # RSS + Hacker News
  noscroll config print                     # Show effective configuration
""",
    )

    # Global config options
    p.add_argument(
        "--config",
        metavar="PATH",
        help="Path to user config file (TOML)",
    )
    p.add_argument(
        "--config-mode",
        choices=["merge", "override"],
        default="merge",
        help="Config mode: 'merge' (default) overlays on defaults, 'override' replaces defaults",
    )
    p.add_argument(
        "--env-file",
        metavar="PATH",
        help="Path to .env file for environment variables",
    )

    # Subcommands
    sub = p.add_subparsers(dest="command", help="Available commands")

    # run command (main command)
    run = sub.add_parser(
        "run",
        help="Fetch, summarize, and output digest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Time Window:
  --last <duration>     Relative: last N units (e.g., 10d, 36h, 2w)
  --from/--to           Absolute: RFC3339 timestamps or YYYY-MM-DD dates
  
  Note: --last and --from/--to are mutually exclusive.
  Default: --last 10d

Bucket (output splitting):
  --bucket day          Split by natural day boundaries (local timezone)
  --bucket hour         Split by natural hour boundaries
  --bucket <duration>   Split by fixed duration (e.g., 2h, 6h, 1d)

Examples:
  noscroll run                                    # Last 10 days → single file
  noscroll run --last 5d --bucket day             # Last 5 days → 5 files
  noscroll run --last 10h --bucket 2h             # Last 10h → 5 files (2h each)
  noscroll run --from 2026-01-20 --to 2026-01-25  # Specific date range
""",
    )

    # Time window group
    time_group = run.add_argument_group("Time Window")
    time_group.add_argument(
        "--last",
        metavar="DURATION",
        default=_env("LAST"),
        help="Relative duration (e.g., 10d, 36h, 2w). Default: 10d. Env: NOSCROLL_LAST",
    )
    time_group.add_argument(
        "--from",
        dest="from_time",
        metavar="DATETIME",
        default=_env("FROM"),
        help="Start time (RFC3339 or YYYY-MM-DD). Env: NOSCROLL_FROM",
    )
    time_group.add_argument(
        "--to",
        dest="to_time",
        metavar="DATETIME",
        default=_env("TO"),
        help="End time (RFC3339 or YYYY-MM-DD). Default: now. Env: NOSCROLL_TO",
    )

    # Bucket group
    bucket_group = run.add_argument_group("Output Splitting")
    bucket_group.add_argument(
        "--bucket",
        metavar="SPEC",
        default=_env("BUCKET"),
        help="Split output by time bucket (day, hour, or duration like 2h). Env: NOSCROLL_BUCKET",
    )
    bucket_group.add_argument(
        "--name-template",
        metavar="TEMPLATE",
        default=_env("NAME_TEMPLATE", "{start:%Y-%m-%d}.md"),
        help="Filename template when splitting. Default: {start:%%Y-%%m-%%d}.md. Env: NOSCROLL_NAME_TEMPLATE",
    )

    # Output group
    output_group = run.add_argument_group("Output")
    output_group.add_argument(
        "--out",
        metavar="PATH",
        default=_env("OUT"),
        help="Output path: file (no bucket) or directory (with bucket). Default: ./noscroll.md. Env: NOSCROLL_OUT",
    )
    output_group.add_argument(
        "--format",
        choices=["markdown", "json"],
        default=_env("FORMAT", "markdown"),
        help="Output format. Default: markdown. Env: NOSCROLL_FORMAT",
    )

    # Source filtering group
    source_group = run.add_argument_group("Source Filtering")
    source_group.add_argument(
        "--source-types",
        type=parse_source_types,
        metavar="TYPES",
        default=_env("SOURCE_TYPES"),
        help="Comma-separated source types: rss,web,hn. Default: rss,web,hn (all). Env: NOSCROLL_SOURCE_TYPES",
    )

    # LLM options
    llm_group = run.add_argument_group("LLM")
    llm_group.add_argument(
        "--llm-api-url",
        metavar="URL",
        default=os.getenv("LLM_API_URL"),
        help="LLM API URL (e.g., https://api.openai.com/v1). Env: LLM_API_URL",
    )
    llm_group.add_argument(
        "--llm-api-key",
        metavar="KEY",
        default=os.getenv("LLM_API_KEY"),
        help="LLM API key. Env: LLM_API_KEY",
    )
    llm_group.add_argument(
        "--llm-model",
        metavar="MODEL",
        default=os.getenv("LLM_MODEL"),
        help="LLM model name (e.g., gpt-4o, claude-3-opus). Env: LLM_MODEL",
    )
    llm_group.add_argument(
        "--llm-summary-model",
        metavar="MODEL",
        default=os.getenv("LLM_SUMMARY_MODEL"),
        help="Model for intermediate summaries. Env: LLM_SUMMARY_MODEL",
    )
    llm_group.add_argument(
        "--llm-api-mode",
        metavar="MODE",
        choices=["chat", "completions", "responses"],
        default=os.getenv("LLM_API_MODE"),
        help="API mode: chat, completions, or responses. Env: LLM_API_MODE",
    )
    llm_group.add_argument(
        "--llm-timeout",
        type=int,
        metavar="MS",
        default=int(os.getenv("LLM_TIMEOUT_MS", "0")) or None,
        help="LLM request timeout in milliseconds. Default: 600000. Env: LLM_TIMEOUT_MS",
    )
    llm_group.add_argument(
        "--llm-concurrency",
        type=int,
        metavar="N",
        default=int(os.getenv("LLM_GLOBAL_CONCURRENCY", "0")) or None,
        help="Max concurrent LLM requests. Default: 5. Env: LLM_GLOBAL_CONCURRENCY",
    )
    llm_group.add_argument(
        "--parallel",
        action="store_true",
        default=_default_parallel_mode(),
        help="Process LLM requests in parallel (opt-in, may trigger rate limits). Default: serial. Env: NOSCROLL_PARALLEL",
    )
    llm_group.add_argument(
        "--serial",
        dest="parallel",
        action="store_false",
        help="Process LLM requests serially (one at a time). Default behavior; kept for compatibility. Env: NOSCROLL_SERIAL",
    )
    llm_group.add_argument(
        "--delay",
        type=int,
        metavar="MS",
        default=_env_int("DELAY", 0),
        help="Delay between LLM requests in milliseconds (only in serial mode). Default: 0. Env: NOSCROLL_DELAY",
    )
    llm_group.add_argument(
        "--lang",
        metavar="LANG",
        default=_env("LANG", "en"),
        help="Output language for LLM summaries (e.g., en, zh, ja, es). Default: en. Env: NOSCROLL_LANG",
    )
    llm_group.add_argument(
        "--top-n",
        type=int,
        metavar="N",
        default=_env_int("TOP_N", 0),
        help="Keep only top N most important items in output. 0 = no limit. Default: 0. Env: NOSCROLL_TOP_N",
    )

    # Paths
    paths_group = run.add_argument_group("Paths")
    paths_group.add_argument(
        "--subscriptions",
        metavar="PATH",
        default=os.getenv("SUBSCRIPTIONS_PATH"),
        help="Path to subscriptions file. Env: SUBSCRIPTIONS_PATH",
    )
    paths_group.add_argument(
        "--system-prompt",
        metavar="PATH",
        default=os.getenv("SYSTEM_PROMPT_PATH"),
        help="Path to system prompt file. Env: SYSTEM_PROMPT_PATH",
    )
    paths_group.add_argument(
        "--llm-log",
        metavar="PATH",
        default=os.getenv("LLM_LOG_PATH"),
        help="Path for LLM trace log. Env: LLM_LOG_PATH",
    )
    paths_group.add_argument(
        "--feed-log",
        metavar="PATH",
        default=os.getenv("FEED_LOG_PATH"),
        help="Path for feed items log. Env: FEED_LOG_PATH",
    )

    # Debug options
    debug_group = run.add_argument_group("Debug")
    debug_group.add_argument(
        "--debug",
        action="store_true",
        default=_env_bool("DEBUG") or os.getenv("DEBUG", "").lower() in ("1", "true", "yes"),
        help="Enable debug mode (verbose logging). Env: NOSCROLL_DEBUG or DEBUG",
    )
    debug_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # config command
    config_cmd = sub.add_parser("config", help="Configuration management")
    config_sub = config_cmd.add_subparsers(dest="config_command")

    config_print = config_sub.add_parser("print", help="Print effective configuration")
    config_print.add_argument(
        "--format",
        choices=["toml", "json"],
        default="toml",
        help="Output format. Default: toml",
    )

    config_sub.add_parser("path", help="Show config file path")

    # sources command (for managing sources)
    sources_cmd = sub.add_parser("sources", help="Manage feed sources")
    sources_sub = sources_cmd.add_subparsers(dest="sources_command")

    sources_sub.add_parser("list", help="List configured sources")

    sources_add = sources_sub.add_parser("add", help="Add a new source")
    sources_add.add_argument("url", help="Source URL")
    sources_add.add_argument(
        "--type",
        choices=["rss", "web"],
        default="rss",
        help="Source type. Default: rss",
    )
    sources_add.add_argument("--name", help="Source name")

    # init command
    sub.add_parser("init", help="Initialize config in user config directory")

    # doctor command
    sub.add_parser("doctor", help="Check configuration and dependencies")

    # ask command (natural language)
    ask = sub.add_parser(
        "ask",
        help="Use natural language to generate and run noscroll parameters",
    )
    ask.add_argument(
        "prompt",
        nargs="+",
        help='Natural language request, e.g. "收集过去五天的资料"',
    )
    ask.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Max retries when generated parameters are invalid. Default: 2",
    )
    ask.add_argument(
        "--dry-run",
        action="store_true",
        help="Show resolved run parameters without executing",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load .env file if specified
    if getattr(args, "env_file", None):
        try:
            from dotenv import load_dotenv

            load_dotenv(args.env_file)
        except ImportError:
            print(
                "Warning: python-dotenv not installed. --env-file ignored.",
                file=sys.stderr,
            )

    # Load config with CLI overrides
    from .config import load_config, set_config

    cfg = load_config(
        cli_config_path=getattr(args, "config", None),
        config_mode=getattr(args, "config_mode", "merge"),
        cli_overrides=_extract_cli_overrides(args),
    )
    set_config(cfg)

    # Route to appropriate command
    command = args.command

    if command == "run" or command is None:
        return _run_main(args)
    elif command == "config":
        return _run_config(args)
    elif command == "sources":
        return _run_sources(args)
    elif command == "init":
        return _run_init(args)
    elif command == "doctor":
        return _run_doctor(args)
    elif command == "ask":
        return _run_ask(args)
    else:
        parser.print_help()
        return 1


def _extract_cli_overrides(args: Namespace) -> dict:
    """Extract CLI arguments that override config values."""
    overrides = {}
    
    # Map CLI argument names to config attribute names
    cli_to_config = {
        "debug": "debug",
        "llm_api_url": "llm_api_url",
        "llm_api_key": "llm_api_key",
        "llm_model": "llm_model",
        "llm_summary_model": "llm_summary_model",
        "llm_api_mode": "llm_api_mode",
        "llm_timeout": "llm_timeout_ms",
        "llm_concurrency": "llm_global_concurrency",
        "subscriptions": "subscriptions_path",
        "system_prompt": "system_prompt_path",
        "llm_log": "llm_log_path",
        "feed_log": "feed_log_path",
    }
    
    for cli_name, config_name in cli_to_config.items():
        value = getattr(args, cli_name, None)
        if value is not None:
            overrides[config_name] = value
    
    return overrides


def _run_main(args: Namespace) -> int:
    """Run the main fetch/summarize workflow."""
    import asyncio

    from .duration import (
        TimeWindow,
        Bucket,
        build_time_window,
        split_time_window,
        format_filename,
    )
    from .config import get_config
    from .runner import run_for_window

    cfg = get_config()

    # Parse time window
    try:
        window = build_time_window(
            last=getattr(args, "last", None),
            from_time=getattr(args, "from_time", None),
            to_time=getattr(args, "to_time", None),
            default_last="10d",
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Parse bucket if specified
    bucket = None
    bucket_spec = getattr(args, "bucket", None)
    if bucket_spec:
        try:
            bucket = Bucket.from_string(bucket_spec)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # Determine source types (may be string from env or list from CLI)
    source_types_raw = getattr(args, "source_types", None)
    if source_types_raw is None:
        source_types: list[str] = ["rss", "web", "hn"]
    elif isinstance(source_types_raw, str):
        # From environment variable - need to parse
        source_types = list(parse_source_types(source_types_raw))
    else:
        source_types = list(source_types_raw)

    # Determine output path
    out_path = getattr(args, "out", None) or "./noscroll.md"
    output_format = getattr(args, "format", "markdown")
    name_template = getattr(args, "name_template", "{start:%Y-%m-%d}.md")

    # Configure LLM client (serial by default; --parallel to opt in)
    parallel = getattr(args, "parallel", False)
    serial = not parallel
    delay_ms = getattr(args, "delay", 0)
    output_lang = getattr(args, "lang", "en")
    top_n = getattr(args, "top_n", 0)

    from .llm import configure_llm_client

    configure_llm_client(
        serial=serial,
        delay_ms=delay_ms,
        lang=output_lang,
        top_n=top_n,
    )
    if cfg.debug:
        print(
            f"LLM mode: serial={serial}, parallel={parallel}, delay={delay_ms}ms, lang={output_lang}, top_n={top_n}"
        )

    # Dry run mode
    if getattr(args, "dry_run", False):
        _print_dry_run(window, bucket, source_types, out_path, name_template)
        return 0

    # Split into buckets if specified
    if bucket:
        windows = split_time_window(window, bucket)
        out_dir = Path(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {len(windows)} time buckets...")
        async def _run_windows() -> None:
            for i, w in enumerate(windows, 1):
                filename = format_filename(name_template, w)
                file_path = out_dir / filename
                print(f"  [{i}/{len(windows)}] {filename}")
                try:
                    await run_for_window(
                        window=w,
                        source_types=source_types,
                        output_path=str(file_path),
                        output_format=output_format,  # type: ignore[arg-type]
                        debug=cfg.debug,
                    )
                except Exception as e:
                    print(f"    Error: {e}", file=sys.stderr)
                    if cfg.debug:
                        import traceback

                        traceback.print_exc()

        asyncio.run(_run_windows())
    else:
        # Single file output
        try:
            asyncio.run(
                run_for_window(
                    window=window,
                    source_types=source_types,
                    output_path=out_path,
                    output_format=output_format,  # type: ignore[arg-type]
                    debug=cfg.debug,
                )
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if cfg.debug:
                import traceback

                traceback.print_exc()
            return 1

    return 0


def _extract_json_object(text: str) -> dict:
    """Extract first JSON object from LLM output text."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or first >= last:
        raise ValueError("LLM output is not valid JSON object")

    snippet = stripped[first : last + 1]
    parsed = json.loads(snippet)
    if not isinstance(parsed, dict):
        raise ValueError("LLM output JSON must be an object")
    return parsed


def _infer_lang_from_prompt(prompt_text: str) -> str:
    """Infer output language code from prompt text."""
    if any("\u4e00" <= ch <= "\u9fff" for ch in prompt_text):
        return "zh"
    if any("\u3040" <= ch <= "\u30ff" for ch in prompt_text):
        return "ja"
    if any("\uac00" <= ch <= "\ud7af" for ch in prompt_text):
        return "ko"
    return "en"


def _normalize_generated_spec(spec: dict) -> dict:
    """Normalize generated spec values to CLI-friendly types."""
    normalized = dict(spec)

    # Normalize delay to integer milliseconds when possible.
    if "delay" in normalized:
        delay = normalized.get("delay")
        if isinstance(delay, int):
            pass
        elif isinstance(delay, float):
            normalized["delay"] = int(round(delay))
        elif isinstance(delay, str):
            m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*$", delay.strip(), re.IGNORECASE)
            if m:
                number = float(m.group(1))
                unit = (m.group(2) or "ms").lower()
                if unit == "s":
                    normalized["delay"] = int(round(number * 1000))
                else:
                    normalized["delay"] = int(round(number))
            else:
                normalized.pop("delay", None)
        else:
            normalized.pop("delay", None)

    # Normalize top_n to integer when possible.
    if "top_n" in normalized:
        top_n = normalized.get("top_n")
        if isinstance(top_n, int):
            pass
        elif isinstance(top_n, float):
            normalized["top_n"] = int(round(top_n))
        elif isinstance(top_n, str) and top_n.strip().isdigit():
            normalized["top_n"] = int(top_n.strip())
        else:
            normalized.pop("top_n", None)

    return normalized


def _spec_to_run_argv(spec: dict) -> list[str]:
    """Convert generated parameter spec to argv list for `noscroll run`."""
    allowed_keys = {
        "last",
        "from_time",
        "to_time",
        "bucket",
        "name_template",
        "out",
        "format",
        "source_types",
        "parallel",
        "delay",
        "lang",
        "top_n",
    }

    unknown = sorted([k for k in spec if k not in allowed_keys])
    if unknown:
        raise ValueError(f"Unsupported parameter(s): {', '.join(unknown)}")

    argv = ["run"]
    has_bucket = "bucket" in spec and bool(spec.get("bucket"))

    if "last" in spec:
        argv.extend(["--last", str(spec["last"])])
    if "from_time" in spec:
        argv.extend(["--from", str(spec["from_time"])])
    if "to_time" in spec:
        argv.extend(["--to", str(spec["to_time"])])
    if "bucket" in spec:
        argv.extend(["--bucket", str(spec["bucket"])])
    if "name_template" in spec:
        argv.extend(["--name-template", str(spec["name_template"])])
    if "out" in spec:
        argv.extend(["--out", str(spec["out"])])
    if "format" in spec:
        argv.extend(["--format", str(spec["format"])])
    if "source_types" in spec:
        source_types = spec["source_types"]
        if isinstance(source_types, list):
            source_types = ",".join(str(item) for item in source_types)
        argv.extend(["--source-types", str(source_types)])
    if "parallel" in spec:
        parallel = spec["parallel"]
        if isinstance(parallel, str):
            parallel = parallel.lower() in ("1", "true", "yes")
        argv.append("--parallel" if bool(parallel) else "--serial")
    if "delay" in spec:
        argv.extend(["--delay", str(spec["delay"])])
    if "lang" in spec:
        argv.extend(["--lang", str(spec["lang"])])
    if "top_n" in spec:
        argv.extend(["--top-n", str(spec["top_n"])])

    # Safe defaults for bucket mode to avoid invalid file/dir semantics
    if has_bucket and "out" not in spec:
        argv.extend(["--out", "./outputs"])
    if has_bucket and "name_template" not in spec:
        argv.extend(["--name-template", "{start:%Y-%m-%d}.md"])

    return argv


def _validate_run_args(run_args: Namespace) -> None:
    """Validate generated run arguments without executing workflow."""
    from .duration import (
        Bucket,
        TimeWindow,
        build_time_window,
        format_filename,
        split_time_window,
    )
    from datetime import datetime, timezone

    window = build_time_window(
        last=getattr(run_args, "last", None),
        from_time=getattr(run_args, "from_time", None),
        to_time=getattr(run_args, "to_time", None),
        default_last="10d",
    )

    bucket_spec = getattr(run_args, "bucket", None)
    if bucket_spec:
        Bucket.from_string(bucket_spec)

    source_types_raw = getattr(run_args, "source_types", None)
    if isinstance(source_types_raw, str):
        parse_source_types(source_types_raw)
    elif source_types_raw is not None:
        parse_source_types(",".join(str(item) for item in source_types_raw))

    # Cross-field validation for output semantics
    bucket_spec = getattr(run_args, "bucket", None)
    output_path = getattr(run_args, "out", None)
    output_format = getattr(run_args, "format", "markdown")

    if bucket_spec and output_path:
        suffix = Path(str(output_path)).suffix.lower()
        if suffix in {".md", ".markdown", ".json", ".txt"}:
            raise ValueError(
                "When bucket is set, --out must be a directory path, not a file path."
            )

    if bucket_spec:
        name_template = getattr(run_args, "name_template", "{start:%Y-%m-%d}.md")
        sample = format_filename(
            str(name_template),
            TimeWindow(
                start=datetime.now(timezone.utc),
                end=datetime.now(timezone.utc),
            ),
        )
        if "{" in sample or "}" in sample:
            raise ValueError(
                "Invalid name_template: only {start:...} and {end:...} placeholders are supported"
            )

        # Ensure generated filenames are unique for the actual split windows
        windows = split_time_window(window, Bucket.from_string(bucket_spec))
        filenames = [format_filename(str(name_template), w) for w in windows]
        if len(set(filenames)) != len(filenames):
            raise ValueError(
                "Invalid name_template: generated filenames are not unique for bucketed output"
            )

    if output_path and not bucket_spec:
        suffix = Path(str(output_path)).suffix.lower()
        if output_format == "markdown" and suffix == ".json":
            raise ValueError("format=markdown cannot use .json output file")
        if output_format == "json" and suffix in {".md", ".markdown"}:
            raise ValueError("format=json cannot use .md output file")


async def _generate_run_args_from_prompt(
    prompt_text: str,
    retries: int,
    debug: bool = False,
) -> Namespace:
    """Generate and validate run args from natural language prompt via LLM."""
    from .config import get_config
    from .llm import call_llm

    cfg = get_config()
    model = cfg.llm_summary_model or cfg.llm_model
    local_now = datetime.now().astimezone().isoformat()

    if not cfg.llm_api_url or not cfg.llm_api_key or not model:
        raise RuntimeError(
            "LLM is not configured. Please set LLM_API_URL, LLM_API_KEY, and LLM_SUMMARY_MODEL (or LLM_MODEL)."
        )

    parser = build_parser()

    def _raise_parse_error(message: str) -> None:
        raise ValueError(message)

    parser.error = _raise_parse_error  # type: ignore[method-assign]
    parser.exit = lambda status=0, message=None: _raise_parse_error(  # type: ignore[method-assign]
        message or f"argparse exit status={status}"
    )

    feedback = ""

    for attempt in range(1, max(1, retries) + 2):
        system_prompt = (
            "You convert a natural language request into NoScroll run parameters. "
            "Return ONLY a JSON object, no markdown, no explanations. "
            "Allowed keys: last, from_time, to_time, bucket, name_template, out, format, source_types, parallel, delay, lang, top_n. "
            "Rules: use source_types as array or comma-separated string containing only rss/web/hn; format must be markdown or json; "
            "if the request is under-specified, choose safe defaults compatible with noscroll run. "
            "Important defaults: prefer format=markdown; do NOT set bucket unless user explicitly asks to split output; "
            "do NOT set out unless user explicitly asks for output path. "
            "For relative requests like 'past/last N days', use `last` and do NOT invent fixed historical from_time/to_time dates. "
            "For bucketed output, name_template must generate unique filenames per bucket (avoid fixed filenames)."
        )
        user_prompt = (
            f"Current local datetime: {local_now}\n"
            f"Request: {prompt_text}"
        )
        if feedback:
            user_prompt += f"\n\nPrevious output was invalid: {feedback}\nPlease regenerate a valid JSON object."

        response = await call_llm(
            api_url=cfg.llm_api_url,
            api_key=cfg.llm_api_key,
            model=model,
            mode=cfg.llm_api_mode,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_ms=cfg.llm_timeout_ms,
            tag="nl_ask",
        )

        try:
            spec = _extract_json_object(response)
            spec = _normalize_generated_spec(spec)
            if not spec.get("lang"):
                spec["lang"] = _infer_lang_from_prompt(prompt_text)
            run_argv = _spec_to_run_argv(spec)
            with contextlib.redirect_stderr(io.StringIO()):
                run_args = parser.parse_args(run_argv)
            _validate_run_args(run_args)
            if debug:
                print(f"ask attempt {attempt}: resolved args={run_argv}")
            return run_args
        except (Exception, SystemExit) as e:
            feedback = str(e)
            if debug:
                print(f"ask attempt {attempt} invalid: {feedback}", file=sys.stderr)

    raise RuntimeError(
        f"Failed to generate valid run parameters after {max(1, retries) + 1} attempts: {feedback}"
    )


def _run_ask(args: Namespace) -> int:
    """Run NoScroll using a natural language prompt."""
    import asyncio

    from .config import get_config

    cfg = get_config()
    prompt_text = " ".join(getattr(args, "prompt", [])).strip()
    if not prompt_text:
        print("Error: empty prompt", file=sys.stderr)
        return 1

    retries = max(0, int(getattr(args, "retries", 2)))

    try:
        run_args = asyncio.run(
            _generate_run_args_from_prompt(
                prompt_text=prompt_text,
                retries=retries,
                debug=cfg.debug,
            )
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            'Tip: try a more explicit prompt, e.g. "收集过去五天的 Hacker News，用中文输出"',
            file=sys.stderr,
        )
        return 1

    if getattr(args, "dry_run", False):
        run_args.dry_run = True

    return _run_main(run_args)


def _print_dry_run(
    window: "TimeWindow",
    bucket: "Bucket | None",
    source_types: list[str],
    out_path: str,
    name_template: str,
) -> None:
    """Print dry run information."""
    from .duration import split_time_window, format_filename

    print("=== Dry Run ===")
    print(f"Time window: {window.start.isoformat()} to {window.end.isoformat()}")
    print(f"Duration: {window.duration_seconds / 3600:.1f} hours")
    print(f"Source types: {', '.join(source_types)}")

    if bucket:
        windows = split_time_window(window, bucket)
        print(f"Bucket: {bucket}")
        print(f"Output directory: {out_path}")
        print(f"Files to generate ({len(windows)}):")
        for w in windows:
            filename = format_filename(name_template, w)
            print(f"  - {filename}")
    else:
        print(f"Output file: {out_path}")


def _run_config(args: Namespace) -> int:
    """Run config subcommands."""
    from .config import get_config, default_config_path, resolve_config_path

    config_command = getattr(args, "config_command", None)

    if config_command == "print":
        cfg = get_config()
        output_format = getattr(args, "format", "toml")

        if output_format == "json":
            import json
            from dataclasses import asdict

            print(json.dumps(asdict(cfg), indent=2))
        else:
            # TOML format
            from dataclasses import fields

            for f in fields(cfg):
                value = getattr(cfg, f.name)
                if isinstance(value, str):
                    print(f'{f.name} = "{value}"')
                elif isinstance(value, bool):
                    print(f"{f.name} = {str(value).lower()}")
                elif value is None:
                    print(f"# {f.name} = ")
                else:
                    print(f"{f.name} = {value}")
        return 0

    elif config_command == "path":
        active_path = resolve_config_path()
        print(active_path or default_config_path())
        return 0

    else:
        print("Usage: noscroll config <print|path>", file=sys.stderr)
        return 1


def _run_sources(args: Namespace) -> int:
    """Run sources subcommands."""
    sources_command = getattr(args, "sources_command", None)

    if sources_command == "list":
        from .config import get_config
        from .opml import load_feeds

        cfg = get_config()
        try:
            feeds = load_feeds(cfg.subscriptions_path)
            print(f"Sources from {cfg.subscriptions_path}:")
            for i, feed in enumerate(feeds, 1):
                print(f"  {i}. [{feed.title}] {feed.xml_url}")
            print(f"\nTotal: {len(feeds)} sources")
        except FileNotFoundError:
            print(f"Subscriptions file not found: {cfg.subscriptions_path}")
            return 1
        return 0

    elif sources_command == "add":
        print("Source addition not yet implemented (edit subscriptions.toml manually)")
        return 0

    else:
        print("Usage: noscroll sources <list|add>", file=sys.stderr)
        return 1


def _run_init(args: Namespace) -> int:
    """Initialize config in user config directory."""
    from .config import default_config_path

    config_path = default_config_path()

    if config_path.exists():
        print(f"Config already exists: {config_path}")
        return 0

    # Write default config
    default_config = '''# NoScroll Configuration
# See: https://github.com/your/noscroll

[run]
last = "10d"
source_types = ["rss", "web", "hn"]
format = "markdown"
out = "./noscroll.md"

[llm]
# api_url = "https://api.openai.com/v1"
# api_key = ""  # Or use LLM_API_KEY env var
# model = "gpt-4o-mini"
# summary_model = "gpt-4o-mini"
# api_mode = "responses"
# timeout_ms = 600000
# global_concurrency = 5

[paths]
# subscriptions = "subscriptions/subscriptions.toml"  # maps to subscriptions_path
# system_prompt = "prompts/system.txt"                # maps to system_prompt_path
# output_dir = "outputs"
# llm_log = "logs/llm-trace.log"
# feed_log = "logs/feed-items.log"

[runtime]
# debug = false
'''

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(default_config)
    print(f"Created config: {config_path}")
    return 0


def _run_doctor(args: Namespace) -> int:
    """Check configuration and dependencies."""
    from .config import get_config, default_config_path, resolve_config_path

    print("=== NoScroll Doctor ===\n")

    # Check config
    print("Configuration:")
    config_path = resolve_config_path()
    if config_path and config_path.exists():
        print(f"  ✓ Config file: {config_path}")
    else:
        print(
            f"  ○ Config file: {default_config_path()} "
            "(not found, using defaults/env)"
        )

    cfg = get_config()

    # Check subscriptions
    print("\nSubscriptions:")
    subs_path = Path(cfg.subscriptions_path)
    if subs_path.exists():
        print(f"  ✓ Subscriptions: {subs_path}")
        try:
            from .opml import load_feeds

            feeds = load_feeds(str(subs_path))
            print(f"    {len(feeds)} feeds configured")
        except Exception as e:
            print(f"    ✗ Error loading: {e}")
    else:
        print(f"  ✗ Subscriptions not found: {subs_path}")

    # Check LLM
    print("\nLLM Configuration:")
    if cfg.llm_api_url:
        print(f"  ✓ API URL: {cfg.llm_api_url}")
    else:
        print("  ✗ API URL not configured (set LLM_API_URL)")

    if cfg.llm_api_key:
        print("  ✓ API key: configured")
    else:
        print("  ✗ API key not configured (set LLM_API_KEY)")

    if cfg.llm_model:
        print(f"  ✓ Model: {cfg.llm_model}")
    else:
        print("  ○ Model: not set (will use provider default)")

    # Check optional dependencies
    print("\nOptional Dependencies:")

    try:
        import crawl4ai  # type: ignore[import-not-found]  # noqa

        print("  ✓ crawl4ai: installed (web crawling enabled)")
    except ImportError:
        print("  ○ crawl4ai: not installed (pip install crawl4ai)")

    try:
        from dotenv import load_dotenv  # noqa

        print("  ✓ python-dotenv: installed")
    except ImportError:
        print("  ○ python-dotenv: not installed (pip install python-dotenv)")

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
