"""Tests for opml module."""

import tempfile
import os
from noscroll.opml import load_opml_feeds, load_toml_feeds, load_feeds, Feed


def test_load_opml_feeds_basic():
    """Test basic OPML parsing."""
    opml_content = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="1.0">
  <head>
    <title>Test Feeds</title>
  </head>
  <body>
    <outline text="Feed 1" title="Feed 1" type="rss" xmlUrl="https://example.com/feed1.xml" htmlUrl="https://example.com/"/>
    <outline text="Feed 2" title="Feed 2" type="rss" xmlUrl="https://example.com/feed2.xml" htmlUrl="https://example.com/page"/>
  </body>
</opml>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(opml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_opml_feeds(temp_path)
        assert len(feeds) == 2
        assert feeds[0].title == "Feed 1"
        assert feeds[0].xml_url == "https://example.com/feed1.xml"
        assert feeds[1].title == "Feed 2"
    finally:
        os.unlink(temp_path)


def test_load_opml_feeds_nested():
    """Test nested OPML outlines."""
    opml_content = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="1.0">
  <body>
    <outline text="Category">
      <outline text="Nested Feed" xmlUrl="https://example.com/nested.xml"/>
    </outline>
    <outline text="Top Feed" xmlUrl="https://example.com/top.xml"/>
  </body>
</opml>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(opml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_opml_feeds(temp_path)
        assert len(feeds) == 2
        urls = {f.xml_url for f in feeds}
        assert "https://example.com/nested.xml" in urls
        assert "https://example.com/top.xml" in urls
    finally:
        os.unlink(temp_path)


def test_load_opml_feeds_empty():
    """Test empty OPML."""
    opml_content = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="1.0">
  <body></body>
</opml>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(opml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_opml_feeds(temp_path)
        assert len(feeds) == 0
    finally:
        os.unlink(temp_path)


def test_load_toml_feeds_basic():
    """Test basic TOML feed loading."""
    toml_content = """
[[feeds]]
name = "Feed 1"
feed_url = "https://example.com/feed1.xml"

[[feeds]]
name = "Feed 2"
feed_url = "https://example.com/feed2.xml"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_toml_feeds(temp_path)
        assert len(feeds) == 2
        assert feeds[0].title == "Feed 1"
        assert feeds[0].xml_url == "https://example.com/feed1.xml"
        assert feeds[1].title == "Feed 2"
    finally:
        os.unlink(temp_path)


def test_load_toml_feeds_without_name():
    """Test TOML feeds without name (should use feed_url as title)."""
    toml_content = """
[[feeds]]
feed_url = "https://example.com/feed.xml"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_toml_feeds(temp_path)
        assert len(feeds) == 1
        assert feeds[0].title == "https://example.com/feed.xml"
    finally:
        os.unlink(temp_path)


def test_load_toml_feeds_empty():
    """Test empty TOML feeds list."""
    toml_content = """
[config]
some_setting = true
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_toml_feeds(temp_path)
        assert len(feeds) == 0
    finally:
        os.unlink(temp_path)


def test_load_toml_feeds_invalid_entries():
    """Test TOML feeds with invalid entries (missing feed_url)."""
    toml_content = """
[[feeds]]
name = "Valid Feed"
feed_url = "https://example.com/feed.xml"

[[feeds]]
name = "Invalid Feed"
# Missing feed_url
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_toml_feeds(temp_path)
        assert len(feeds) == 1  # Only valid entry
        assert feeds[0].title == "Valid Feed"
    finally:
        os.unlink(temp_path)


def test_load_feeds_toml_extension():
    """Test load_feeds with .toml extension."""
    toml_content = """
[[feeds]]
name = "Test Feed"
feed_url = "https://example.com/feed.xml"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_feeds(temp_path)
        assert len(feeds) == 1
        assert feeds[0].title == "Test Feed"
    finally:
        os.unlink(temp_path)


def test_load_feeds_xml_extension():
    """Test load_feeds with .xml extension (OPML)."""
    opml_content = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="1.0">
  <body>
    <outline text="Test Feed" xmlUrl="https://example.com/feed.xml"/>
  </body>
</opml>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(opml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_feeds(temp_path)
        assert len(feeds) == 1
        assert feeds[0].xml_url == "https://example.com/feed.xml"
    finally:
        os.unlink(temp_path)


def test_load_feeds_opml_extension():
    """Test load_feeds with .opml extension."""
    opml_content = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="1.0">
  <body>
    <outline text="Test Feed" xmlUrl="https://example.com/feed.xml"/>
  </body>
</opml>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".opml", delete=False) as f:
        f.write(opml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_feeds(temp_path)
        assert len(feeds) == 1
    finally:
        os.unlink(temp_path)


def test_feed_dataclass():
    """Test Feed dataclass."""
    feed = Feed(title="Test", xml_url="https://example.com/feed.xml")
    assert feed.title == "Test"
    assert feed.xml_url == "https://example.com/feed.xml"
    assert feed.html_url == ""

    feed_with_html = Feed(
        title="Test", 
        xml_url="https://example.com/feed.xml",
        html_url="https://example.com/"
    )
    assert feed_with_html.html_url == "https://example.com/"


def test_load_opml_feeds_fallback_text():
    """Test OPML parsing with text attribute fallback (no title)."""
    opml_content = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="1.0">
  <body>
    <outline text="Text Only" xmlUrl="https://example.com/feed.xml"/>
  </body>
</opml>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(opml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_opml_feeds(temp_path)
        assert len(feeds) == 1
        assert feeds[0].title == "Text Only"
    finally:
        os.unlink(temp_path)


def test_load_opml_feeds_url_fallback():
    """Test OPML parsing with URL as title fallback."""
    opml_content = """<?xml version="1.0" encoding="UTF-8"?>
<opml version="1.0">
  <body>
    <outline xmlUrl="https://example.com/feed.xml"/>
  </body>
</opml>"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(opml_content)
        f.flush()
        temp_path = f.name

    try:
        feeds = load_opml_feeds(temp_path)
        assert len(feeds) == 1
        assert feeds[0].title == "https://example.com/feed.xml"
    finally:
        os.unlink(temp_path)
