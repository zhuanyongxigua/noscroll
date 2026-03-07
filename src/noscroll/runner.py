"""Runner module - executes the main workflow for a time window."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Literal

if TYPE_CHECKING:
    from .duration import TimeWindow

from .config import get_config
from .models import FeedItem
from .sources.providers import (
    fetch_hn_items,
    fetch_rss_items,
    fetch_web_items,
    fetch_x_items,
)
from .utils import append_feed_log, with_date_in_path


def _resolve_effective_subscriptions_path(configured_path: str) -> Path:
    """Resolve subscriptions config path with built-in fallback.

    Priority:
    1) configured path
    2) packaged built-in subscriptions
    3) repository subscriptions file (development)
    """
    configured = Path(configured_path)
    if configured.exists():
        return configured

    module_dir = Path(__file__).resolve().parent
    packaged_builtin = module_dir / "_builtin_subscriptions.toml"
    if packaged_builtin.exists():
        return packaged_builtin

    repo_builtin = module_dir.parent.parent / "subscriptions" / "subscriptions.toml"
    if repo_builtin.exists():
        return repo_builtin

    return configured


def _resolve_effective_system_prompt_path(configured_path: str) -> Path:
    """Resolve system prompt path with built-in fallback.

    Priority:
    1) configured path
    2) packaged built-in system prompt
    3) repository prompts/system.txt (development)
    """
    configured = Path(configured_path)
    if configured.exists():
        return configured

    module_dir = Path(__file__).resolve().parent
    packaged_builtin = module_dir / "_builtin_system_prompt.txt"
    if packaged_builtin.exists():
        return packaged_builtin

    repo_builtin = module_dir.parent.parent / "prompts" / "system.txt"
    if repo_builtin.exists():
        return repo_builtin

    return configured


async def run_for_window(
    window: "TimeWindow",
    source_types: list[str],
    output_path: str,
    output_format: Literal["markdown", "json"] = "markdown",
    debug: bool = False,
    refresh_x: bool = False,
) -> None:
    """
    Run the fetch/summarize workflow for a single time window.

    Args:
        window: Time window to fetch
        source_types: List of source types to include ('rss', 'web', 'hn', 'x')
        output_path: Output file path
        output_format: Output format ('markdown' or 'json')
        debug: Enable debug logging
        refresh_x: Force re-fetch X data, ignoring cache
    """
    cfg = get_config()
    all_items: list[FeedItem] = []

    # Convert window to milliseconds for filtering
    start_ms = int(window.start.timestamp() * 1000)
    end_ms = int(window.end.timestamp() * 1000)

    if debug:
        print(f"  Window: {window.start.isoformat()} to {window.end.isoformat()}")
        print(f"  Source types: {', '.join(source_types)}")

    # 1-3. Fetch sources in parallel (rss/web/hn are independent)
    fetch_jobs: list[tuple[str, Awaitable[list[FeedItem]]]] = []
    source_item_counts: dict[str, int] = {source: 0 for source in source_types}
    source_errors: dict[str, str] = {}

    if "rss" in source_types:
        fetch_jobs.append(("rss", fetch_rss_items(cfg, start_ms, end_ms, debug)))
    if "web" in source_types:
        fetch_jobs.append(("web", fetch_web_items(cfg, start_ms, end_ms, debug)))
    if "hn" in source_types:
        fetch_jobs.append(("hn", fetch_hn_items(cfg, window.start, window.end, debug)))
    if "x" in source_types:
        fetch_jobs.append(("x", fetch_x_items(cfg, window.start, window.end, debug, refresh=refresh_x)))

    if fetch_jobs:
        results = await asyncio.gather(
            *(job for _, job in fetch_jobs),
            return_exceptions=True,
        )

        for (source, _), result in zip(fetch_jobs, results):
            if isinstance(result, BaseException):
                source_errors[source] = str(result)
                if debug:
                    print(f"  {source.upper()} fetch error: {result}")
                continue

            source_items = result
            source_item_counts[source] = len(source_items)
            all_items.extend(source_items)

            if debug:
                if source == "rss":
                    print(f"  RSS items: {len(source_items)}")
                elif source == "web":
                    print(f"  Web items: {len(source_items)}")
                elif source == "hn":
                    print(f"  HN items: {len(source_items)}")
                elif source == "x":
                    print(f"  X items: {len(source_items)}")

    if debug:
        print(f"  Total items: {len(all_items)}")

    # Log raw feed items when debug mode is enabled
    if debug and all_items:
        target_date = window.start.strftime("%Y-%m-%d")
        feed_log_path = with_date_in_path(cfg.feed_log_path, target_date)
        feed_log_entries = [
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "source_type": _get_source_type(item),
                "feed": item.feed_title,
                "title": item.title,
                "link": item.link,
                "published_at": item.pub_date,
                "summary": item.summary,
            }
            for item in all_items
        ]
        append_feed_log(feed_log_path, feed_log_entries)
        print(f"  Feed log written: {feed_log_path}")

    if not all_items:
        requested_sources = ", ".join(source_types) if source_types else "(none)"
        source_lines = []
        for source in source_types:
            if source in source_errors:
                err = source_errors[source].replace("\n", " ")[:200]
                source_lines.append(f"- {source}: fetch error ({err})")
            else:
                source_lines.append(f"- {source}: {source_item_counts.get(source, 0)} items")
        details = "\n".join(
            [
                "No content found.",
                "",
                "No items were fetched from the selected sources in this time window.",
                "This usually means source data is empty for the window, source config is invalid, or network fetch failed.",
                "",
                f"Time window: {window.start.isoformat()} to {window.end.isoformat()}",
                f"Requested sources: {requested_sources}",
                "",
                "Source fetch results:",
                *source_lines,
                "",
                "Recommended checks:",
                "- Verify source configuration is valid (RSS URLs, enabled sources, HN settings).",
                "- Verify network access to source endpoints and LLM API (DNS/proxy/firewall/rate limits).",
                "- Re-run with --debug to see per-source fetch details and errors.",
                "- Expand or shift time window (--last / --from / --to) to confirm data exists.",
            ]
        )
        raise RuntimeError(details)

    # 4. Summarize and format output
    output = await _generate_output(
        items=all_items,
        window=window,
        output_format=output_format,
        debug=debug,
    )

    # 5. Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(output, encoding="utf-8")
    print(f"  Written: {output_path} ({len(all_items)} items)")


def _get_source_type(item: FeedItem) -> str:
    """Determine source type from feed item."""
    feed_title = item.feed_title.lower()
    if "hacker news" in feed_title or feed_title.startswith("hn:"):
        return "hn"
    elif "crawled" in feed_title or "web:" in feed_title:
        return "web"
    elif feed_title.startswith("[x]"):
        return "x"
    return "rss"


async def _generate_output(
    items: list[FeedItem],
    window: "TimeWindow",
    output_format: Literal["markdown", "json"],
    debug: bool,
) -> str:
    """Generate formatted output from items."""
    cfg = get_config()
    
    # Get log_path for LLM logging when debug is enabled
    log_path = None
    if debug:
        target_date = window.start.strftime("%Y-%m-%d")
        log_path = with_date_in_path(cfg.llm_log_path, target_date)

    if output_format == "json":
        return _format_json(items, window)

    # Markdown format
    return await _format_markdown(items, window, debug, log_path)


def _format_json(items: list[FeedItem], window: "TimeWindow") -> str:
    """Format items as JSON."""
    output = {
        "window": {
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": [
            {
                "feed_title": item.feed_title,
                "feed_url": item.feed_url,
                "title": item.title,
                "link": item.link,
                "pub_date": item.pub_date,
                "summary": item.summary,
            }
            for item in items
        ],
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


async def _format_markdown(
    items: list[FeedItem],
    window: "TimeWindow",
    debug: bool,
    log_path: str | None = None,
) -> str:
    """Format items as Markdown using LLM summarization."""
    cfg = get_config()

    # Check if LLM is configured
    use_llm = bool(cfg.llm_api_url and cfg.llm_api_key)
    if debug:
        print(f"  LLM config: url={'SET' if cfg.llm_api_url else 'NOT SET'}, key={'SET' if cfg.llm_api_key else 'NOT SET'}, use_llm={use_llm}")

    if not use_llm:
        print("Warning: LLM not configured. Falling back to non-LLM markdown output.", file=sys.stderr)
        return _format_markdown_fallback(items)

    try:
        summary = await _generate_llm_summary(items, cfg, debug, log_path)
        if debug and log_path:
            print(f"  LLM log written: {log_path}")
        return summary
    except Exception as e:
        print(f"Warning: LLM summary failed, falling back to non-LLM markdown output: {e}", file=sys.stderr)
        return _format_markdown_fallback(items)


def _format_markdown_fallback(items: list[FeedItem]) -> str:
    """Format a simple markdown digest without LLM assistance."""
    lines: list[str] = ["# Digest", "", f"Items collected: {len(items)}", ""]
    for idx, item in enumerate(items, start=1):
        title = item.title.strip() or "(untitled)"
        link = item.link.strip()
        pub_date = item.pub_date.strip() if item.pub_date else ""
        summary = (item.summary or "").strip()

        if link:
            lines.append(f"{idx}. [{title}]({link})")
        else:
            lines.append(f"{idx}. {title}")

        meta_parts = [p for p in [item.feed_title, pub_date] if p]
        if meta_parts:
            lines.append(f"   - {' | '.join(meta_parts)}")
        if summary:
            lines.append(f"   - {summary}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


async def _generate_llm_summary(
    items: list[FeedItem],
    cfg,
    debug: bool,
    log_path: str | None = None,
) -> str:
    """Generate LLM summary of items."""
    from .llm import call_llm
    from .utils import format_for_llm

    # Format items for LLM
    content = format_for_llm(items)

    # Load system prompt with fallback
    system_prompt_path = _resolve_effective_system_prompt_path(cfg.system_prompt_path)
    if system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = """You are a news digest assistant. Summarize the following feed items into a concise digest.
Focus on:
- Key topics and trends
- Important announcements
- Interesting technical content
Be concise but informative. Use bullet points for clarity."""

    # Language injection is handled by LLM client (via --lang option)

    # Pass structured JSON payload so prompt rules for url/source_type can be applied reliably
    if isinstance(content, list):
        user_payload = {
            "items": content,
        }
        user_prompt = json.dumps(user_payload, ensure_ascii=False, indent=2)
    else:
        user_prompt = str(content)

    response = await call_llm(
        api_url=cfg.llm_api_url,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model or "gpt-4o-mini",
        mode=cfg.llm_api_mode,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout_ms=cfg.llm_timeout_ms,
        log_path=log_path,
    )

    return response.strip()

