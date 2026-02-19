# NoScroll - Pull, don't scroll
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CLI](https://img.shields.io/badge/interface-CLI-black.svg)](https://github.com/zhuanyongxigua/noscroll)
[![Sources](https://img.shields.io/badge/sources-RSS%20%7C%20Web%20%7C%20HN-orange.svg)](https://github.com/zhuanyongxigua/noscroll)

## What is NoScroll
NoScroll is a Python CLI that pulls information from RSS feeds, web pages, and Hacker News, then uses an LLM to summarize and rank the most useful items.

It is designed for a pull-based reading workflow: define sources once, run on schedule, read only the high-signal digest.

## Installation

```bash
pipx install noscroll
```

or:

```bash
uv tool install noscroll
uvx noscroll --help
```

## Quick Start

```bash
noscroll init
noscroll sources add https://hnrss.org/frontpage
noscroll sources add https://example.com/feed.xml
noscroll run --once --top 12 --format markdown --out ./digest.md
```

## Configuration
Config loading priority:

1. `--config <path>`
2. `NOSCROLL_CONFIG`
3. Default OS path

Default locations:

- Linux: `~/.config/noscroll/config.toml`
- macOS: `~/Library/Application Support/noscroll/config.toml`
- Windows: `%AppData%\noscroll\config.toml`

## CLI

```bash
noscroll init
noscroll sources add <url> [--type rss|web] [--name "..."]
noscroll sources list
noscroll run [--once] [--top N] [--format markdown|json] [--out PATH]
noscroll doctor
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License
MIT. See [LICENSE](LICENSE).
