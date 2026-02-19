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

If you need `web` source crawling support, install with crawler extras:

```bash
pipx install "noscroll[crawler]"
```

## Skills Installation

Install the built-in `noscroll` skill to your target host:

```bash
# Claude Code (project scope)
noscroll skills install noscroll --host claude --scope project

# Codex (project scope)
noscroll skills install noscroll --host codex --scope project

# Claude Code (user scope)
noscroll skills install noscroll --host claude --scope user

# Codex (user scope)
noscroll skills install noscroll --host codex --scope user

# OpenClaw (workspace scope)
noscroll skills install noscroll --host openclaw --scope workspace --workdir /path/to/workspace

# OpenClaw (shared scope)
noscroll skills install noscroll --host openclaw --scope shared
```

## Ask Command

Use natural language directly:

```bash
noscroll --env-file .env ask "Collect content from the past five days, one file per day"
```

This will generate daily digest files in `outputs/`.

Example generated text:

```markdown
## AI (3)
1) Off Grid: Running text/image/vision models offline on mobile | Value: 4/5 | Type: Practice
- Conclusion: This open-source project demonstrates on-device multimodal inference on smartphones, with strong privacy and offline usability.
- Why it matters: On-device AI can reduce privacy risk and cloud inference cost, and is a good fit for offline-first products.
- Evidence links: https://github.com/alichherawalla/off-grid-mobile

## Other News (2)
4) uBlock rule: hide YouTube Shorts with one click | Value: 4/5 | Domain: Tech

## Life & Health (2)
6) AI avatars for rural healthcare support | Value: 3/5 | Domain: Health
```

## Configuration

You can provide a config file in these places:

1. CLI argument: `--config /path/to/config.toml`
2. Environment variable: `NOSCROLL_CONFIG=/path/to/config.toml`
3. Default path: `~/.noscroll/config.toml`

Example `config.toml`:

```toml
[llm]
api_url = "https://api.openai.com/v1"
api_key = "your-api-key"
model = "gpt-4o-mini"

[paths]
subscriptions = "subscriptions/subscriptions.toml"
output_dir = "outputs"

[runtime]
debug = false
```

Create a starter config:

```bash
noscroll init
```

## License
MIT. See [LICENSE](LICENSE).
