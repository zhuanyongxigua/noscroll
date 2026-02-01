"""Blog crawler using Crawl4AI and LLM-based link extraction.

This module crawls websites that don't provide RSS feeds by:
1. Fetching the blog index page with Crawl4AI
2. Using LLM to extract article links from the page
3. Fetching article content as Markdown
4. Using LLM to summarize each article
5. Generating a feed.json compatible with the RSS aggregator
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

# Crawl4AI for content extraction
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from .config import get_config
from .llm import call_llm


class Post(BaseModel):
    """Represents a discovered blog post."""

    title: str = Field(..., description="Post title")
    url: str = Field(..., description="Absolute URL of the post")
    date: Optional[str] = Field(
        None, description="Publish date if visible (ISO-8601 preferred)"
    )


def _slugify(url: str) -> str:
    """Convert URL to a safe filename slug."""
    path = re.sub(r"https?://", "", url)
    path = re.sub(r"[?#].*$", "", path)
    tail = path.rstrip("/").split("/")[-1]
    tail = re.sub(r"[^a-zA-Z0-9._-]+", "-", tail).strip("-")
    if not tail:
        tail = str(abs(hash(url)))
    return tail[:80]


def _get_browser_config() -> BrowserConfig:
    """Create BrowserConfig with proxy support from environment."""
    import os
    # crawl4ai/playwright uses standard proxy env vars
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("ALL_PROXY")
    if proxy:
        print(f"  Using proxy: {proxy}")
        return BrowserConfig(
            headless=True,
            proxy_config={"server": proxy},
        )
    return BrowserConfig(headless=True)

def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string to datetime object."""
    if not date_str:
        return None
    
    # Try various formats
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    
    # Try dateutil as fallback
    try:
        from dateutil import parser
        dt = parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except:
        pass
    
    return None


def filter_posts_by_days(posts: List[Post], days: Optional[int]) -> List[Post]:
    """Filter posts to only include those within the last N days."""
    if days is None:
        return posts
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    
    for post in posts:
        post_date = _parse_date(post.date)
        if post_date is None:
            # Include posts without dates (can't determine age)
            filtered.append(post)
        elif post_date >= cutoff:
            filtered.append(post)
    
    return filtered


async def discover_posts_with_llm(
    start_url: str,
    max_posts: int = 30,
    explore: bool = False,
) -> List[Post]:
    """
    Fetch a blog index page and use LLM to extract article links.
    
    If explore=True, the LLM will first try to find the blog section,
    then extract individual post links.

    Args:
        start_url: The starting URL (may be homepage, about page, etc.)
        max_posts: Maximum number of posts to extract
        explore: If True, explore to find blog section first

    Returns:
        List of Post objects
    """
    cfg = get_config()
    # Fetch the page with Crawl4AI
    browser_conf = _get_browser_config()
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
    )

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(start_url, config=run_conf)

    if not result.success:
        raise RuntimeError(f"Failed to fetch {start_url}")

    # Get the markdown
    md_obj = getattr(result, "markdown", None)
    markdown = ""
    if md_obj:
        markdown = (
            getattr(md_obj, "fit_markdown", None)
            or getattr(md_obj, "raw_markdown", None)
            or str(md_obj)
        )

    domain = re.match(r"https?://[^/]+", start_url)
    domain_str = domain.group() if domain else start_url

    # If explore mode, first try to find the blog section
    if explore:
        explore_prompt = """You are a web navigation assistant. Analyze this webpage and find the blog/articles section.

Return a JSON object with this structure:
{
  "has_blog_list": true/false,
  "blog_url": "https://example.com/blog" or null,
  "posts": [{"title": "...", "url": "...", "date": "..."}] or []
}

Rules:
- has_blog_list: true if this page already shows a list of blog posts/articles
- blog_url: if this is NOT a blog list page, provide the URL to the blog/articles section (look for links like "Blog", "Articles", "Posts", "News", "Writing", etc.)
- posts: if has_blog_list is true, extract the blog posts here
- Use absolute URLs
- Return ONLY the JSON object"""

        explore_user = f"""Analyze this webpage and find the blog section.

URL: {start_url}

Page content:
{markdown[:8000]}"""

        explore_response = await call_llm(
            api_url=cfg.llm_api_url,
            api_key=cfg.llm_api_key,
            model=cfg.llm_model,
            mode=cfg.llm_api_mode,
            system_prompt=explore_prompt,
            user_prompt=explore_user,
            timeout_ms=cfg.llm_timeout_ms,
        )

        try:
            json_match = re.search(r'\{[\s\S]*\}', explore_response)
            if json_match:
                explore_data = json.loads(json_match.group())
            else:
                explore_data = json.loads(explore_response)

            # If page has blog list, extract posts directly
            if explore_data.get("has_blog_list") and explore_data.get("posts"):
                posts = []
                for item in explore_data["posts"][:max_posts]:
                    url = item.get("url", "")
                    if url.startswith("/"):
                        url = domain_str + url
                    if url:
                        posts.append(Post(
                            title=item.get("title", "Untitled"),
                            url=url,
                            date=item.get("date"),
                        ))
                if posts:
                    return posts

            # If found a blog URL, recursively crawl it
            blog_url = explore_data.get("blog_url")
            if blog_url:
                if blog_url.startswith("/"):
                    blog_url = domain_str + blog_url
                if blog_url != start_url:
                    print(f"    Found blog section: {blog_url}")
                    # Recursively crawl the blog page (without explore to avoid infinite loop)
                    return await discover_posts_with_llm(blog_url, max_posts, explore=False)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Explore mode failed, falling back to direct extraction: {e}")
            # Fall through to normal extraction

    # Normal extraction: Use LLM to extract links directly
    system_prompt = """You are a web scraping assistant. Extract blog post links from the given page content.

Return ONLY a valid JSON array of objects with this exact structure:
[
  {"title": "Post Title", "url": "https://example.com/post-url", "date": "2024-01-15"},
  ...
]

Rules:
- Extract actual blog post/article links, not navigation or category links
- Use absolute URLs (include https://domain.com prefix)
- For dates, use ISO format YYYY-MM-DD if visible, otherwise use null
- Maximum {max_posts} posts
- Return ONLY the JSON array, no other text"""

    user_prompt = f"""Extract blog post links from this page.

URL: {start_url}

Page content (markdown):
{markdown[:8000]}

Return a JSON array of blog posts (max {max_posts} posts)."""

    response = await call_llm(
        api_url=cfg.llm_api_url,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        mode=cfg.llm_api_mode,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        timeout_ms=cfg.llm_timeout_ms,
    )

    # Parse JSON from response
    try:
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            posts_data = json.loads(json_match.group())
        else:
            posts_data = json.loads(response)
    except json.JSONDecodeError as e:
        print(f"  Failed to parse LLM response as JSON: {e}")
        print(f"  Response: {response[:500]}")
        return []

    posts = []
    for item in posts_data[:max_posts]:
        url = item.get("url", "")
        # Make URL absolute if relative
        if url.startswith("/"):
            url = domain_str + url

        if url:
            posts.append(Post(
                title=item.get("title", "Untitled"),
                url=url,
                date=item.get("date"),
            ))

    return posts


async def summarize_article(
    title: str,
    url: str,
    content: str,
    summary_model: str,
) -> str:
    """Use LLM to summarize a single article."""
    cfg = get_config()
    system_prompt = """You are a technical content summarizer. 
Summarize the article content concisely, capturing the key points and insights.
Be factual and avoid speculation. Output in the same language as the input."""

    user_prompt = f"""Summarize this article in 2-4 sentences.

Title: {title}
URL: {url}

Content:
{content[:6000]}

Summary:"""

    try:
        response = await call_llm(
            api_url=cfg.llm_api_url,
            api_key=cfg.llm_api_key,
            model=summary_model,
            mode=cfg.llm_api_mode,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_ms=cfg.llm_timeout_ms,
        )
        return response.strip()
    except Exception as e:
        print(f"    Summary error for {title}: {e}")
        return content[:500]  # Fallback to truncated content


async def fetch_and_summarize_posts(
    posts: List[Post],
    out_dir: str,
    summary_model: str,
    concurrency: int = 3,
) -> List[dict]:
    """
    Fetch article content, summarize with LLM, and save.

    Args:
        posts: List of Post objects with URLs to fetch
        out_dir: Directory to save markdown files
        summary_model: LLM model for summarization
        concurrency: Max concurrent LLM calls

    Returns:
        List of feed items with metadata and summaries
    """
    os.makedirs(out_dir, exist_ok=True)

    browser_conf = _get_browser_config()
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.4, threshold_type="fixed")
    )
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
    )

    urls = [p.url for p in posts]
    items: List[dict] = []
    semaphore = asyncio.Semaphore(concurrency)

    # First, fetch all pages
    print(f"    Fetching {len(urls)} pages...")
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        results = await crawler.arun_many(urls, config=run_conf)

    # Then summarize with LLM (with concurrency limit)
    async def process_post(post: Post, result) -> Optional[dict]:
        success = getattr(result, "success", True)
        if not success:
            print(f"    Failed to fetch: {post.url}")
            return None

        md_obj = getattr(result, "markdown", None)
        if md_obj is None:
            markdown = ""
        else:
            markdown = (
                getattr(md_obj, "fit_markdown", None)
                or getattr(md_obj, "raw_markdown", None)
                or str(md_obj)
            )

        # Save raw markdown
        slug = _slugify(post.url)
        md_path = os.path.join(out_dir, f"{slug}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        # Summarize with LLM
        async with semaphore:
            print(f"    Summarizing: {post.title[:50]}...")
            summary = await summarize_article(
                title=post.title,
                url=post.url,
                content=markdown,
                summary_model=summary_model,
            )

        return {
            "title": post.title,
            "date": post.date,
            "url": post.url,
            "markdown_path": md_path,
            "summary": summary,
            "content": markdown[:2000],  # Keep truncated content for reference
        }

    # Process all posts concurrently (LLM calls limited by semaphore)
    tasks = [process_post(p, r) for p, r in zip(posts, results)]
    results = await asyncio.gather(*tasks)
    items = [r for r in results if r is not None]

    return items


async def crawl_site(
    name: str,
    url: str,
    max_posts: int = 30,
    days: Optional[int] = None,
    output_dir: str = "crawled",
    explore: bool = False,
) -> dict:
    """
    Crawl a single site and generate feed data.

    Args:
        name: Site name for logging
        url: Site URL (may be homepage, blog index, etc.)
        max_posts: Maximum posts to collect
        days: Only include posts from the last N days (None = all)
        output_dir: Base output directory
        explore: If True, explore the site to find blog section first

    Returns:
        Feed dictionary with items
    """
    print(f"Crawling {name} ({url}){'  [explore mode]' if explore else ''}...")

    cfg = get_config()
    try:
        # Step 1: Discover posts using LLM
        posts = await discover_posts_with_llm(url, max_posts=max_posts, explore=explore)
        print(f"  Discovered {len(posts)} posts")

        if not posts:
            raise RuntimeError("No posts found")

        # Step 2: Filter by date if days is specified
        if days is not None:
            posts = filter_posts_by_days(posts, days)
            print(f"  Filtered to {len(posts)} posts from last {days} days")

        if not posts:
            print(f"  No posts within the last {days} days")
            return {
                "name": name,
                "url": url,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "items": [],
            }

        # Step 3: Fetch and summarize posts
        site_dir = os.path.join(output_dir, _slugify(url))
        posts_dir = os.path.join(site_dir, "posts")
        summary_model = cfg.llm_summary_model or cfg.llm_model

        items = await fetch_and_summarize_posts(
            posts, posts_dir, summary_model, concurrency=3
        )
        print(f"  Processed {len(items)} posts successfully")

        feed = {
            "name": name,
            "url": url,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "days_filter": days,
            "items": items,
        }

        # Save individual site feed
        feed_path = os.path.join(site_dir, "feed.json")
        os.makedirs(site_dir, exist_ok=True)
        with open(feed_path, "w", encoding="utf-8") as f:
            json.dump(feed, f, ensure_ascii=False, indent=2)

        print(f"  Saved feed to {feed_path}")
        return feed

    except Exception as e:
        print(f"  Error crawling {name}: {e}")
        return {
            "name": name,
            "url": url,
            "error": str(e),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "items": [],
        }


def load_sites_config(config_path: str = "subscriptions/subscriptions.toml") -> dict:
    """Load sites configuration from TOML file."""
    import tomllib

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    # Support both unified config (with sites/sites_defaults keys) and standalone format
    if "sites" in data and isinstance(data.get("sites"), list):
        # Could be either format, check for sites_defaults to distinguish
        defaults = data.get("sites_defaults", data.get("defaults", {}))
        return {"sites": data["sites"], "defaults": defaults}
    
    return data


async def crawl_all_sites(
    config_path: str = "subscriptions/subscriptions.toml",
    output_dir: str = "crawled",
    days: Optional[int] = None,
) -> List[dict]:
    """
    Crawl all enabled sites from configuration.

    Args:
        config_path: Path to subscriptions.toml
        output_dir: Base output directory
        days: Only include posts from the last N days (None = all)

    Returns:
        List of feed dictionaries
    """
    sites_config = load_sites_config(config_path)
    defaults = sites_config.get("defaults", {})
    sites = sites_config.get("sites", [])

    feeds = []
    for site in sites:
        if not site.get("enabled", True):
            continue

        # explore can be set per-site or in defaults
        explore = site.get("explore", defaults.get("explore", False))

        feed = await crawl_site(
            name=site["name"],
            url=site["url"],
            max_posts=site.get("max_posts", defaults.get("max_posts", 30)),
            days=days,
            output_dir=output_dir,
            explore=explore,
        )
        feeds.append(feed)

    # Save combined feed
    combined_feed = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days_filter": days,
        "sites": feeds,
    }
    combined_path = os.path.join(output_dir, "all_feeds.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_feed, f, ensure_ascii=False, indent=2)

    print(f"\nSaved combined feed to {combined_path}")
    return feeds


async def crawl_cli(args) -> int:
    """CLI entry point for crawl command (called from cli.py)."""
    from .config import get_config
    cfg = get_config()
    
    url = getattr(args, "url", None)
    name = getattr(args, "name", "CLI")
    max_posts = getattr(args, "max_posts", 30)
    output_dir = getattr(args, "output", "crawled")
    days = getattr(args, "days", None)
    explore = getattr(args, "explore", False)
    config_path = cfg.subscriptions_path

    if url:
        # Single URL mode
        await crawl_site(
            name=name,
            url=url,
            max_posts=max_posts,
            days=days,
            output_dir=output_dir,
            explore=explore,
        )
    else:
        # Config file mode
        await crawl_all_sites(
            config_path=config_path,
            output_dir=output_dir,
            days=days,
        )
    return 0


def main():
    """CLI entry point for site crawler (standalone)."""
    import argparse

    parser = argparse.ArgumentParser(description="Crawl sites without RSS feeds")
    parser.add_argument(
        "--config",
        default="subscriptions/subscriptions.toml",
        help="Path to subscriptions configuration file",
    )
    parser.add_argument(
        "--output",
        default="crawled",
        help="Output directory for crawled content",
    )
    parser.add_argument(
        "--url",
        help="Crawl a single URL instead of using config file",
    )
    parser.add_argument(
        "--name",
        default="CLI",
        help="Name for single URL crawl",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=30,
        help="Maximum posts to collect per site",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Only include posts from the last N days",
    )
    args = parser.parse_args()

    if args.url:
        # Single URL mode
        asyncio.run(
            crawl_site(
                name=args.name,
                url=args.url,
                max_posts=args.max_posts,
                days=args.days,
                output_dir=args.output,
            )
        )
    else:
        # Config file mode
        asyncio.run(
            crawl_all_sites(
                config_path=args.config,
                output_dir=args.output,
                days=args.days,
            )
        )


if __name__ == "__main__":
    main()
