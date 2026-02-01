# NoScroll

**Pull, don't scroll.**  
NoScroll proactively pulls information from **RSS**, **web pages (crawler)**, and **Hacker News**, then uses **AI summarization + ranking** to produce a small number of high-signal digests.

> The goal: stop drowning in feeds and timelines. Define your sources once, pull on your schedule, and read the *best* results.

---

## What NoScroll does

1. **Ingest**: Fetch content from RSS feeds, URLs, and Hacker News (frontpage/newest/search-style feeds).
2. **Extract**: Normalize content into a consistent "document" format (title, url, text, metadata).
3. **Distill**: Use an LLM to summarize, de-duplicate, and score items for "signal".
4. **Deliver**: Output a ranked digest (Markdown/JSON) to stdout, file, or integrations (optional).

---

## Key features

- **Proactive information intake** (pull model; no infinite scrolling)
- **Multi-source**: RSS, web crawl, Hacker News
- **AI distillation**: summaries, key claims, and "why it matters"
- **Ranking**: produce a short Top-N instead of a firehose
- **Config-first**: reproducible runs from a single config file
- **Bring-your-own-LLM**: OpenAI-compatible endpoints, local models, or vendor-specific providers via adapters
- **Safety controls**: explicit allowlists, fetch limits, and no "LLM-driven browsing" by default

---

## Install

### Recommended (end users)

```bash
pipx install noscroll
```

### Alternative (fast tool runner)

```bash
uv tool install noscroll
# or run without installing:
uvx noscroll --help
```

### From source (contributors)

```bash
git clone https://github.com/<you>/noscroll.git
cd noscroll
pip install -e ".[dev]"
noscroll --help
```

---

## Quickstart

### 1) Initialize a config

```bash
noscroll init
# writes a starter config to your user config directory
```

### 2) Add sources

```bash
noscroll sources add https://hnrss.org/frontpage
noscroll sources add https://news.ycombinator.com/rss
noscroll sources add https://example.com/feed.xml
noscroll sources add https://example.com/blog/ --type web
```

### 3) Run once

```bash
noscroll run --once --top 12 --format markdown
```

### 4) Write to a file

```bash
noscroll run --once --top 12 --format markdown --out ./digest.md
```

---

## Configuration

NoScroll is designed to be **config-first**. The CLI reads config using this priority:

1. `--config <path>`
2. `NOSCROLL_CONFIG`
3. Default user config path (OS-specific)

Typical default locations:

- **Linux**: `~/.config/noscroll/config.toml`
- **macOS**: `~/Library/Application Support/noscroll/config.toml`
- **Windows**: `%AppData%\noscroll\config.toml`

### Example `config.toml`

```toml
[app]
top_n = 10
format = "markdown"        # markdown | json
out = "stdout"             # stdout | /path/to/file
cache_dir = "auto"
dedupe = true

[fetch]
user_agent = "NoScroll/0.x (+https://github.com/<you>/noscroll)"
timeout_seconds = 20
max_items_per_source = 50
allowlist_domains = ["news.ycombinator.com", "hnrss.org", "example.com"]

[llm]
provider = "openai_compatible"    # openai_compatible | local | <custom>
base_url = "http://localhost:11434/v1"
api_key_env = "NOSCROLL_LLM_API_KEY"
model = "gpt-4.1-mini"            # example; choose what you run
temperature = 0.2

[prompt]
style = "concise"
include_citations = true          # include source URLs in the digest
focus = ["engineering", "ai-tools", "security"]

[[sources]]
name = "HN Frontpage (hnrss)"
type = "rss"
url = "https://hnrss.org/frontpage"

[[sources]]
name = "HN Official RSS"
type = "rss"
url = "https://news.ycombinator.com/rss"

[[sources]]
name = "Example Blog RSS"
type = "rss"
url = "https://example.com/feed.xml"

[[sources]]
name = "Example Blog (crawl)"
type = "web"
url = "https://example.com/blog/"
crawl_depth = 2
include_patterns = ["*/blog/*"]
exclude_patterns = ["*/tag/*", "*/page/*"]
```

---

## CLI (design contract)

NoScroll aims for a stable, scriptable CLI:

```bash
noscroll init
noscroll sources add <url> [--type rss|web] [--name "..."]
noscroll sources list
noscroll run [--once] [--top N] [--format markdown|json] [--out PATH]
noscroll doctor
```

If you want a workflow that runs periodically, pair `noscroll run --once` with:

- **cron / systemd timers** (Linux)
- **launchd** (macOS)
- **Task Scheduler** (Windows)

---

## Output format

### Markdown (default)

- A short Top-N list
- For each item: title, link, 3–7 bullet summary, "why it matters", tags (optional)
- Optional: citations/URLs preserved for traceability

### JSON

Machine-friendly structure for downstream pipelines (notifications, dashboards, search, archival).

---

## Security notes

NoScroll is designed to reduce exposure to prompt injection and "agentic browsing" risks:

- Fetches only from configured sources (optionally domain-allowlisted)
- Applies size limits and parsing safeguards
- Does not allow the LLM to browse the web by default
- Keeps secrets in env vars (e.g., `NOSCROLL_LLM_API_KEY`)

---

## Roadmap (high-level)

- **Source adapters**: more HN modes, Atom, GitHub releases, newsletters
- **Delivery adapters**: email, Slack/Discord, Notion/Obsidian export
- **Better ranking**: topic profiles, novelty detection, recency weighting
- **UI client**: optional cross-platform client (separate package/app)

---

## Development

### Setup

```bash
pip install -e ".[dev]"
pre-commit install
pytest
```

### Project structure (recommended)

- `src/noscroll/` — core library + CLI
- `src/noscroll/providers/` — LLM adapters
- `src/noscroll/sources/` — RSS/web/HN ingest
- `tests/` — unit/integration tests

---

## Contributing

PRs and issues are welcome.

1. Start with an issue describing the use case and expected behavior.
2. Keep changes small and well-tested.
3. Add docs/examples when you introduce new CLI flags or config fields.

(If this project grows, we will add `CONTRIBUTING.md`, a code of conduct, and a security policy.)

---

## Changelog and versioning

- **Changelog format**: [Keep a Changelog](https://keepachangelog.com/)
- **Versioning**: [Semantic Versioning](https://semver.org/)

---

## License

MIT. See [LICENSE](LICENSE).

---

## Acknowledgements

- The RSS ecosystem and the "pull model" philosophy
- Hacker News RSS feeds (official + community variants)
- The Python packaging ecosystem that makes CLI distribution straightforward
