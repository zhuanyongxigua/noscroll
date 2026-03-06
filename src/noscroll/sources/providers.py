"""Source provider helpers for runner orchestration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..config import Config
from ..models import FeedItem
from ..utils import filter_by_window


def resolve_effective_subscriptions_path(configured_path: str) -> Path:
    """Resolve subscriptions path with packaged and repo fallbacks."""
    configured = Path(configured_path)
    if configured.exists():
        return configured

    sources_dir = Path(__file__).resolve().parent
    package_dir = sources_dir.parent

    packaged_builtin = package_dir / "_builtin_subscriptions.toml"
    if packaged_builtin.exists():
        return packaged_builtin

    repo_builtin = package_dir.parent.parent / "subscriptions" / "subscriptions.toml"
    if repo_builtin.exists():
        return repo_builtin

    return configured


async def fetch_rss_items(cfg: Config, start_ms: int, end_ms: int, debug: bool) -> list[FeedItem]:
    """Fetch RSS items for a time window."""
    from .opml import load_feeds
    from .rss import fetch_all_feeds

    effective_subscriptions_path = resolve_effective_subscriptions_path(cfg.subscriptions_path)

    try:
        feeds = load_feeds(str(effective_subscriptions_path))
    except FileNotFoundError:
        if debug:
            print(f"  Subscriptions not found: {cfg.subscriptions_path}")
        return []

    if not feeds:
        return []

    items, failures = await fetch_all_feeds(feeds)
    if failures and debug:
        for failure in failures:
            print(f"  RSS fetch failed: {failure.get('title', 'unknown')}")

    return filter_by_window(items, start_ms, end_ms)


async def fetch_web_items(cfg: Config, start_ms: int, end_ms: int, debug: bool) -> list[FeedItem]:
    """Fetch crawled web items for a time window."""
    try:
        from .crawler import crawl_all_sites
    except ImportError:
        if debug:
            print("  Crawler not available (crawl4ai not installed)")
        return []

    from .fetch import crawled_to_feed_items

    effective_subscriptions_path = resolve_effective_subscriptions_path(cfg.subscriptions_path)

    try:
        crawled_feeds = await crawl_all_sites(
            config_path=str(effective_subscriptions_path),
            output_dir="crawled",
        )
    except Exception as error:
        if debug:
            print(f"  Crawler error: {error}")
        return []

    items = crawled_to_feed_items(crawled_feeds)
    return filter_by_window(items, start_ms, end_ms)


async def fetch_hn_items(cfg: Config, start_dt: datetime, end_dt: datetime, debug: bool) -> list[FeedItem]:
    """Fetch Hacker News items for a time window."""
    try:
        from .hackernews import fetch_hn_top_discussed
    except ImportError:
        if debug:
            print("  HN module not available")
        return []

    top_n = 30
    min_comments = 30
    window_hours = 6

    try:
        import tomllib

        subs_path = resolve_effective_subscriptions_path(cfg.subscriptions_path)
        if subs_path.exists():
            subs_config = tomllib.loads(subs_path.read_text(encoding="utf-8"))
            hn_config = subs_config.get("hackernews", {})
            if isinstance(hn_config, dict) and not hn_config.get("enabled", True):
                return []
            if isinstance(hn_config, dict):
                top_n = int(hn_config.get("top_n", top_n))
                min_comments = int(hn_config.get("min_comments", min_comments))
                window_hours = int(hn_config.get("window_hours", window_hours))
    except Exception as error:
        if debug:
            print(f"  HN config parse warning: {error}")

    try:
        return await fetch_hn_top_discussed(
            start_dt=start_dt,
            end_dt=end_dt,
            top_n=top_n,
            min_comments=min_comments,
            window_hours=window_hours,
        )
    except Exception as error:
        if debug:
            print(f"  HN fetch error: {error}")
        return []


async def fetch_x_items(cfg: Config, start_dt: datetime, end_dt: datetime, debug: bool) -> list[FeedItem]:
    """Fetch X items for a time window."""
    from .x import fetch_x_users

    try:
        import tomllib

        subs_path = resolve_effective_subscriptions_path(cfg.subscriptions_path)
        if not subs_path.exists():
            return []

        subs_config = tomllib.loads(subs_path.read_text(encoding="utf-8"))
        x_defaults = subs_config.get("x_defaults", {})
        if isinstance(x_defaults, dict) and not x_defaults.get("enabled", True):
            if debug:
                print("  X fetching is disabled in config")
            return []

        x_users = subs_config.get("x_users", [])
        if not isinstance(x_users, list):
            x_users = [x_users]

        usernames = [str(item.get("username")) for item in x_users if isinstance(item, dict) and item.get("username")]
        if not usernames:
            if debug:
                print("  No X users configured")
            return []

        return await fetch_x_users(
            usernames=usernames,
            bearer_token=cfg.x_bearer_token,
            start_dt=start_dt,
            end_dt=end_dt,
            debug=debug,
        )
    except Exception as error:
        if debug:
            print(f"  X fetch config or execution error: {error}")
        return []
