"""X (Twitter) source adapter."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from ..models import FeedItem
from ..utils import filter_by_window


async def _fetch_user_id(
    client: httpx.AsyncClient,
    username: str,
    bearer_token: str,
    debug: bool,
) -> str | None:
    """Resolve X user id from username."""
    url = f"https://api.twitter.com/2/users/by/username/{username}"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            user = data.get("data")
            if isinstance(user, dict):
                user_id = user.get("id")
                if isinstance(user_id, str) and user_id:
                    return user_id
    except Exception as error:
        if debug:
            print(f"  X user lookup failed @{username}: {error}")
    return None


async def _fetch_user_posts(
    client: httpx.AsyncClient,
    user_id: str,
    username: str,
    bearer_token: str,
    start_time: datetime | None,
    debug: bool,
) -> list[FeedItem]:
    """Fetch recent posts for one X user."""
    url = f"https://api.twitter.com/2/users/{user_id}/tweets"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params: dict[str, Any] = {
        "tweet.fields": "created_at,text",
        "max_results": 100,
        "exclude": "retweets,replies",
    }

    if start_time:
        start_utc = start_time.astimezone(timezone.utc)
        params["start_time"] = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        payload = response.json()
    except Exception as error:
        if debug:
            print(f"  X timeline fetch failed @{username}: {error}")
        return []

    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(data, list):
        return []

    items: list[FeedItem] = []
    for tweet in data:
        if not isinstance(tweet, dict):
            continue
        text = str(tweet.get("text") or "")
        tweet_id = str(tweet.get("id") or "")
        created_at = str(tweet.get("created_at") or "")
        first_line = text.split("\n")[0] if text else "(empty post)"
        title = first_line[:100] + ("..." if len(first_line) > 100 else "")
        link = f"https://x.com/{username}/status/{tweet_id}" if tweet_id else f"https://x.com/{username}"

        items.append(
            FeedItem(
                feed_title=f"[x] @{username}",
                feed_url=f"https://x.com/{username}",
                title=title,
                link=link,
                pub_date=created_at,
                summary=text,
            )
        )
    return items


async def fetch_x_users(
    usernames: list[str],
    bearer_token: str,
    start_dt: datetime,
    end_dt: datetime,
    debug: bool = False,
) -> list[FeedItem]:
    """Fetch posts from multiple X usernames within a window."""
    if not bearer_token:
        if debug:
            print("  X_BEARER_TOKEN is not set; skipping X source")
        return []

    normalized_usernames = [name.strip().lstrip("@") for name in usernames if name and name.strip()]
    if not normalized_usernames:
        return []

    async with httpx.AsyncClient(timeout=30.0, trust_env=True) as client:
        user_pairs: list[tuple[str, str]] = []
        for username in normalized_usernames:
            user_id = await _fetch_user_id(client, username, bearer_token, debug)
            if user_id:
                user_pairs.append((username, user_id))

        tasks = [
            _fetch_user_posts(client, user_id, username, bearer_token, start_dt, debug)
            for username, user_id in user_pairs
        ]
        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    items: list[FeedItem] = []
    for result in results:
        if isinstance(result, Exception):
            if debug:
                print(f"  X fetch task error: {result}")
            continue
        items.extend(filter_by_window(result, start_ms, end_ms))
    return items
