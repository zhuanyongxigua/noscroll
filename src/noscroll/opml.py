"""Feed loading module - supports OPML and TOML formats."""

from __future__ import annotations

import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Feed:
    """Represents an RSS feed."""

    title: str
    xml_url: str
    html_url: str = ""


def _collect_outlines(element: ET.Element, feeds: List[Feed]) -> None:
    """Recursively collect feed outlines from OPML."""
    for outline in element.findall("outline"):
        xml_url = outline.get("xmlUrl")
        if xml_url:
            feeds.append(
                Feed(
                    title=outline.get("title") or outline.get("text") or xml_url,
                    xml_url=xml_url,
                    html_url=outline.get("htmlUrl") or "",
                )
            )
        # Recurse into nested outlines
        _collect_outlines(outline, feeds)


def load_opml_feeds(opml_path: str) -> List[Feed]:
    """Load RSS feeds from an OPML file."""
    tree = ET.parse(opml_path)
    root = tree.getroot()

    feeds: List[Feed] = []
    body = root.find("body")
    if body is not None:
        _collect_outlines(body, feeds)

    return feeds


def load_toml_feeds(toml_path: str) -> List[Feed]:
    """Load RSS feeds from a TOML file."""
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    feeds: List[Feed] = []
    feeds_list = data.get("feeds", [])

    for item in feeds_list:
        if isinstance(item, dict) and item.get("feed_url"):
            feeds.append(
                Feed(
                    title=item.get("name", item["feed_url"]),
                    xml_url=item["feed_url"],
                )
            )

    return feeds


def load_feeds(path: str) -> List[Feed]:
    """Load RSS feeds from either OPML or TOML file based on extension."""
    p = Path(path)
    if p.suffix.lower() == ".toml":
        return load_toml_feeds(path)
    else:
        return load_opml_feeds(path)
