"""Hacker News top discussed stories fetcher using Algolia API."""

import asyncio
import datetime as dt
import html
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import httpx

from .rss import FeedItem

ALGOLIA_ENDPOINT = "https://hn.algolia.com/api/v1/search_by_date"
ALGOLIA_ITEM_ENDPOINT = "https://hn.algolia.com/api/v1/items"


def get_proxy() -> str | None:
    """Get proxy URL from environment, if set."""
    return os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("ALL_PROXY")


def parse_iso(s: str) -> dt.datetime:
    """Parse ISO format date string to datetime (UTC).

    Accept: 2026-01-20 or 2026-01-20T12:30:00 (assume UTC if no tz)
    """
    if "T" not in s:
        d = dt.datetime.fromisoformat(s)
        return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc)
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def to_unix_seconds(d: dt.datetime) -> int:
    """Convert datetime to unix timestamp."""
    return int(d.timestamp())


def strip_html(text: str | None) -> str:
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    # Decode HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def fetch_story_item(
    item_id: str,
    timeout: float = 30.0,
) -> Optional[dict]:
    """Fetch a single HN item with all its children (comments).
    
    Returns:
        Item dict with nested children, or None on error
    """
    url = f"{ALGOLIA_ITEM_ENDPOINT}/{item_id}"
    
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except Exception:
        return None


def extract_comments_text(item: dict, max_depth: int = 10) -> List[str]:
    """Recursively extract comment texts from HN item.
    
    Args:
        item: HN item dict with children
        max_depth: Maximum recursion depth
        
    Returns:
        List of comment texts (author: text format)
    """
    comments = []
    
    def _extract(node: dict, depth: int = 0):
        if depth > max_depth:
            return
        
        # Get comment text
        text = strip_html(node.get("text") or "")
        author = node.get("author") or "anonymous"
        if text:
            comments.append(f"[{author}]: {text}")
        
        # Recurse into children
        for child in node.get("children") or []:
            _extract(child, depth + 1)
    
    for child in item.get("children") or []:
        _extract(child)
    
    return comments


async def fetch_article_content(
    url: str,
    timeout: float = 15.0,
    max_length: int = 10000,
) -> str:
    """Fetch article content from URL.
    
    Returns:
        Extracted text content or empty string on error
    """
    if not url or "news.ycombinator.com" in url:
        return ""
    
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=True, follow_redirects=True) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; HNBot/1.0)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return ""
            
            html_content = response.text
            
            # Simple text extraction (remove scripts, styles, tags)
            text = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = strip_html(text)
            
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            return text
    except Exception:
        return ""


async def fetch_window(
    start_ts: int,
    end_ts: int,
    hits_per_page: int = 1000,
    timeout: float = 30.0,
) -> Tuple[List[dict], int]:
    """Fetch stories from a time window.

    Returns:
        Tuple of (hits, nb_hits_reported)
    """
    params = {
        "tags": "story",
        "numericFilters": f"created_at_i>={start_ts},created_at_i<{end_ts}",
        "hitsPerPage": hits_per_page,
        "page": 0,
    }

    async with httpx.AsyncClient(timeout=timeout, trust_env=True) as client:
        response = await client.get(ALGOLIA_ENDPOINT, params=params)
        response.raise_for_status()
        data = response.json()
        hits = data.get("hits", [])
        nb_hits = int(data.get("nbHits", 0))
        nb_pages = int(data.get("nbPages", 0))

        # Paginate if API says there are more pages
        all_hits = list(hits)
        for page in range(1, nb_pages):
            params["page"] = page
            response = await client.get(ALGOLIA_ENDPOINT, params=params)
            response.raise_for_status()
            data = response.json()
            all_hits.extend(data.get("hits", []))

    return all_hits, nb_hits


async def fetch_top_discussed(
    start_ts: int,
    end_ts: int,
    top_n: int = 30,
    min_comments: int = 30,
    window_seconds: int = 6 * 3600,
) -> List[dict]:
    """Fetch top discussed stories from HN within a time range.

    Args:
        start_ts: Start timestamp (unix seconds)
        end_ts: End timestamp (unix seconds)
        top_n: Number of top stories to return
        min_comments: Minimum comment count threshold
        window_seconds: Initial window size for splitting requests

    Returns:
        List of story dicts sorted by num_comments desc
    """
    results: Dict[str, dict] = {}
    cur = start_ts

    while cur < end_ts:
        nxt = min(cur + window_seconds, end_ts)
        hits, nb_hits = await fetch_window(cur, nxt)

        # If this window is too dense, shrink it and retry (simple backoff)
        if nb_hits >= 950 and window_seconds > 60 * 10:  # 10 minutes min window
            window_seconds = max(window_seconds // 2, 60 * 10)
            continue

        for h in hits:
            oid = str(h.get("objectID", ""))
            if not oid:
                continue
            num_comments = int(h.get("num_comments") or 0)
            if num_comments < min_comments:
                continue
            # Deduplicate by objectID; keep max num_comments
            if oid not in results or num_comments > int(
                results[oid].get("num_comments") or 0
            ):
                results[oid] = h

        cur = nxt

    # Sort by num_comments desc, then points desc
    items = list(results.values())
    items.sort(
        key=lambda x: (int(x.get("num_comments") or 0), int(x.get("points") or 0)),
        reverse=True,
    )
    return items[:top_n]


def hn_story_to_feed_item(story: dict) -> FeedItem:
    """Convert HN story dict to FeedItem."""
    object_id = story.get("objectID", "")
    title = (story.get("title") or "").strip()
    url = story.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
    hn_url = f"https://news.ycombinator.com/item?id={object_id}"
    num_comments = int(story.get("num_comments") or 0)
    points = int(story.get("points") or 0)
    created_at = story.get("created_at", "")

    # Build summary with HN discussion link and stats
    summary = f"Points: {points} | Comments: {num_comments}\nHN Discussion: {hn_url}"
    if url != hn_url:
        summary = f"Original: {url}\n{summary}"

    return FeedItem(
        feed_title="Hacker News (Top Discussed)",
        feed_url="https://news.ycombinator.com/",
        title=title,
        link=url,
        pub_date=created_at,
        summary=summary,
    )


async def fetch_hn_top_discussed(
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    top_n: int = 30,
    min_comments: int = 30,
    window_hours: int = 6,
) -> List[FeedItem]:
    """Fetch top discussed HN stories as FeedItems.

    Args:
        start_dt: Start datetime (UTC)
        end_dt: End datetime (UTC)
        top_n: Number of top stories to return
        min_comments: Minimum comment count threshold
        window_hours: Initial window size in hours

    Returns:
        List of FeedItem objects
    """
    start_ts = to_unix_seconds(start_dt)
    end_ts = to_unix_seconds(end_dt)

    stories = await fetch_top_discussed(
        start_ts=start_ts,
        end_ts=end_ts,
        top_n=top_n,
        min_comments=min_comments,
        window_seconds=window_hours * 3600,
    )

    return [hn_story_to_feed_item(s) for s in stories]


async def fetch_story_with_details(
    story: dict,
    fetch_content: bool = True,
    max_comments: int = 100,
) -> dict:
    """Fetch story with article content and comments.
    
    Args:
        story: Basic story dict from search API
        fetch_content: Whether to fetch article content
        max_comments: Maximum number of comments to include
        
    Returns:
        Enhanced story dict with 'article_content' and 'comments' keys
    """
    object_id = story.get("objectID", "")
    url = story.get("url", "")
    
    # Fetch article content and comments in parallel
    async def empty_coro():
        return ""
    
    tasks = []
    if fetch_content and url:
        tasks.append(fetch_article_content(url))
    else:
        tasks.append(empty_coro())
    
    tasks.append(fetch_story_item(object_id))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)

    article_content = results[0] if not isinstance(results[0], Exception) else ""
    item_data = results[1] if not isinstance(results[1], Exception) else None

    # Extract comments
    comments = []
    if isinstance(item_data, dict):
        comments = extract_comments_text(item_data)[:max_comments]
    
    return {
        **story,
        "article_content": article_content,
        "comments": comments,
    }


async def summarize_hn_story(
    story: dict,
    llm_config: dict,
) -> str:
    """Summarize a single HN story using LLM.
    
    Args:
        story: Story dict with article_content and comments
        llm_config: Dict with api_url, api_key, model, mode, timeout_ms
        
    Returns:
        LLM-generated summary
    """
    from .llm import call_llm
    
    title = story.get("title", "")
    url = story.get("url", "")
    points = story.get("points", 0)
    num_comments = story.get("num_comments", 0)
    article_content = story.get("article_content", "")
    comments = story.get("comments", [])
    
    # Build context for LLM
    context_parts = [
        f"Title: {title}",
        f"URL: {url}",
        f"Points: {points}, Comments: {num_comments}",
    ]
    
    if article_content:
        # Truncate article content
        truncated = article_content[:3000] + "..." if len(article_content) > 3000 else article_content
        context_parts.append(f"\n--- Article Content ---\n{truncated}")
    
    if comments:
        # Include top comments
        top_comments = comments[:50]
        context_parts.append(f"\n--- Top Comments ({len(top_comments)} of {len(comments)}) ---")
        context_parts.extend(top_comments)
    
    user_prompt = "\n".join(context_parts)
    
    system_prompt = """You are a tech news summarization expert. Based on the given Hacker News post, its article content, and community comments, provide a concise summary including:

1. **Overview**: Summarize what this article/post is about in 1-2 sentences
2. **Key Discussion Points**: What are the main topics being discussed in the comments? (2-4 points)
3. **Community Sentiment**: Is the overall reaction positive, negative, neutral, or skeptical? Any notable disagreements?

Keep the summary under 200 words. Remain objective and neutral. Use Markdown format."""

    try:
        response = await call_llm(
            api_url=llm_config["api_url"],
            api_key=llm_config["api_key"],
            model=llm_config["model"],
            mode=llm_config.get("mode", ""),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_ms=llm_config.get("timeout_ms", 60000),
            log_path=llm_config.get("log_path"),
        )
        return response or "(No summary generated)"
    except Exception as e:
        return f"(Summary error: {e})"


async def fetch_and_summarize_stories(
    stories: List[dict],
    llm_config: dict,
    concurrency: int = 3,
    fetch_content: bool = True,
    max_comments: int = 100,
    progress_callback: Optional[Callable[[int, int, dict], None]] = None,
) -> List[dict]:
    """Fetch details and summarize multiple stories.
    
    Args:
        stories: List of story dicts from search API
        llm_config: LLM configuration dict
        concurrency: Max concurrent LLM calls
        fetch_content: Whether to fetch article content
        max_comments: Maximum comments per story
        progress_callback: Optional callback(index, total, story) for progress
        
    Returns:
        List of stories with 'summary' key added
    """
    semaphore = asyncio.Semaphore(concurrency)
    
    async def process_story(index: int, story: dict) -> dict:
        async with semaphore:
            if progress_callback:
                progress_callback(index + 1, len(stories), story)
            
            # Fetch details
            detailed = await fetch_story_with_details(
                story,
                fetch_content=fetch_content,
                max_comments=max_comments,
            )
            
            # Summarize
            summary = await summarize_hn_story(detailed, llm_config)
            
            return {**detailed, "summary": summary}
    
    tasks = [process_story(i, s) for i, s in enumerate(stories)]
    return await asyncio.gather(*tasks)


async def hn_cli(args) -> int:
    """CLI entry point for HN command (called from cli.py)."""
    import sys
    from .config import get_config
    
    cfg = get_config()
    
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    top_n = getattr(args, "top", 30)
    min_comments = getattr(args, "min_comments", 30)
    output_path = getattr(args, "output", None)
    
    # Default to last 24 hours if not specified
    if not start or not end:
        end_dt = dt.datetime.now(dt.timezone.utc)
        start_dt = end_dt - dt.timedelta(days=1)
    else:
        start_dt = parse_iso(start)
        end_dt = parse_iso(end)
    
    if end_dt <= start_dt:
        print("Error: end must be > start", file=sys.stderr)
        return 2
    
    # Fetch top stories
    stories = await fetch_top_discussed(
        start_ts=to_unix_seconds(start_dt),
        end_ts=to_unix_seconds(end_dt),
        top_n=top_n,
        min_comments=min_comments,
    )
    
    print(f"Found {len(stories)} top discussed stories\n", file=sys.stderr)
    
    if cfg.llm_api_url and cfg.llm_api_key:
        # Summarize with LLM
        llm_config = {
            "api_url": cfg.llm_api_url,
            "api_key": cfg.llm_api_key,
            "model": cfg.llm_summary_model or cfg.llm_model,
            "mode": cfg.llm_api_mode,
            "timeout_ms": cfg.llm_timeout_ms,
        }
        
        def progress(idx, total, story):
            print(f"[{idx}/{total}] Summarizing: {story.get('title', '')[:60]}...", file=sys.stderr)
        
        summarized = await fetch_and_summarize_stories(
            stories,
            llm_config,
            concurrency=3,
            progress_callback=progress,
        )
        
        # Generate markdown output
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")
        
        lines = [
            f"# Hacker News Top Discussed",
            f"**Period:** {start_str} to {end_str}",
            f"**Stories:** {len(summarized)}",
            "",
        ]
        
        for i, story in enumerate(summarized, 1):
            title = story.get("title", "")
            url = story.get("url", "")
            hn_url = f"https://news.ycombinator.com/item?id={story.get('objectID', '')}"
            points = story.get("points", 0)
            num_comments = story.get("num_comments", 0)
            summary = story.get("summary", "")
            
            lines.append(f"## {i}. {title}")
            lines.append("")
            if url:
                lines.append(f"🔗 [Article]({url}) | [HN Discussion]({hn_url}) | {points} points | {num_comments} comments")
            else:
                lines.append(f"🔗 [HN Discussion]({hn_url}) | {points} points | {num_comments} comments")
            lines.append("")
            lines.append(summary)
            lines.append("")
            lines.append("---")
            lines.append("")
        
        output = "\n".join(lines)
        
        # Write output
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"\nSaved to {output_path}", file=sys.stderr)
        else:
            from pathlib import Path
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            default_path = output_dir / f"hn-{start_str}-to-{end_str}.md"
            with open(default_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"\nSaved to {default_path}", file=sys.stderr)
    else:
        # Simple output without summarization
        for i, story in enumerate(stories, 1):
            title = (story.get("title") or "").strip()
            url = story.get("url") or f"https://news.ycombinator.com/item?id={story.get('objectID')}"
            hn_url = f"https://news.ycombinator.com/item?id={story.get('objectID')}"
            num_comments = int(story.get("num_comments") or 0)
            points = int(story.get("points") or 0)
            
            print(f"{i:02d}. {title}")
            print(f"    Points: {points} | Comments: {num_comments}")
            if url != hn_url:
                print(f"    Article: {url}")
            print(f"    HN: {hn_url}")
            print()
    
    return 0


def main():
    """CLI entry point for HN top discussed fetcher."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Fetch top discussed Hacker News stories"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start time (ISO format, e.g. 2026-01-20 or 2026-01-20T00:00:00+00:00)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End time (ISO format)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top stories to return (default: 30)",
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=30,
        help="Minimum comment count threshold (default: 30)",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=6,
        help="Initial window size in hours; auto-shrinks if too dense (default: 6)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Fetch article content and comments, then summarize with LLM",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: stdout, or outputs/hn-YYYY-MM-DD.md with --summarize)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent LLM calls when summarizing (default: 3)",
    )
    args = parser.parse_args()

    start_dt = parse_iso(args.start)
    end_dt = parse_iso(args.end)
    if end_dt <= start_dt:
        print("Error: end must be > start", file=sys.stderr)
        sys.exit(2)

    async def run():
        # Fetch top stories
        stories = await fetch_top_discussed(
            start_ts=to_unix_seconds(start_dt),
            end_ts=to_unix_seconds(end_dt),
            top_n=args.top,
            min_comments=args.min_comments,
            window_seconds=args.window_hours * 3600,
        )
        
        print(f"Found {len(stories)} top discussed stories from {args.start} to {args.end}\n", file=sys.stderr)
        
        if args.summarize:
            # Load LLM config
            from .config import get_config
            cfg = get_config()

            if not cfg.llm_api_url or not cfg.llm_api_key:
                print("Error: LLM_API_URL and LLM_API_KEY required for --summarize", file=sys.stderr)
                sys.exit(1)
            
            llm_config = {
                "api_url": cfg.llm_api_url,
                "api_key": cfg.llm_api_key,
                "model": cfg.llm_summary_model or cfg.llm_model,
                "mode": cfg.llm_api_mode,
                "timeout_ms": cfg.llm_timeout_ms,
            }
            
            def progress(idx, total, story):
                print(f"[{idx}/{total}] Summarizing: {story.get('title', '')[:60]}...", file=sys.stderr)
            
            summarized = await fetch_and_summarize_stories(
                stories,
                llm_config,
                concurrency=args.concurrency,
                progress_callback=progress,
            )
            
            # Generate markdown output
            lines = [
                f"# Hacker News Top Discussed",
                f"**Period:** {args.start} to {args.end}",
                f"**Stories:** {len(summarized)}",
                "",
            ]
            
            for i, story in enumerate(summarized, 1):
                title = story.get("title", "")
                url = story.get("url", "")
                hn_url = f"https://news.ycombinator.com/item?id={story.get('objectID', '')}"
                points = story.get("points", 0)
                num_comments = story.get("num_comments", 0)
                summary = story.get("summary", "")
                
                lines.append(f"## {i}. {title}")
                lines.append("")
                if url:
                    lines.append(f"🔗 [Article]({url}) | [HN Discussion]({hn_url}) | {points} points | {num_comments} comments")
                else:
                    lines.append(f"🔗 [HN Discussion]({hn_url}) | {points} points | {num_comments} comments")
                lines.append("")
                lines.append(summary)
                lines.append("")
                lines.append("---")
                lines.append("")
            
            output = "\n".join(lines)
            
            # Write output
            if args.output:
                output_path = args.output
            else:
                from pathlib import Path
                output_dir = Path("outputs")
                output_dir.mkdir(exist_ok=True)
                output_path = output_dir / f"hn-{args.start}-to-{args.end}.md"
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"\nSaved to {output_path}", file=sys.stderr)
            
        else:
            # Simple output without summarization
            for i, story in enumerate(stories, 1):
                title = (story.get("title") or "").strip()
                url = story.get("url") or f"https://news.ycombinator.com/item?id={story.get('objectID')}"
                hn_url = f"https://news.ycombinator.com/item?id={story.get('objectID')}"
                num_comments = int(story.get("num_comments") or 0)
                points = int(story.get("points") or 0)
                
                print(f"{i:02d}. {title}")
                print(f"    Points: {points} | Comments: {num_comments}")
                if url != hn_url:
                    print(f"    Article: {url}")
                print(f"    HN: {hn_url}")
                print()
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
