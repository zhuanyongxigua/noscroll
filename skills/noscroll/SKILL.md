---
name: noscroll
description: Use this skill when the task is RSS/Web/HN collection and digest generation with NoScroll CLI; do not use it for generic crawling, login automation, or repository code modification.
---

# noscroll

This Skill gives agents an executable operating guide for NoScroll, including:
- collecting content by time window (RSS / Web / Hacker News)
- generating either a single digest file or bucketed outputs
- using natural language via `ask` to generate parameters and run

Non-goals:
- generic website crawling beyond NoScroll source capabilities
- login/session automation for gated websites
- editing source code in repositories as part of content collection

## When to Use

- user asks to collect content from the past N days/hours
- user wants one file per day/hour
- user requests specific source types only (e.g., RSS-only, HN-only)
- user wants a dry-run preview before execution

## Core Commands

### 1) Standard Run (default choice)

```bash
noscroll run --last 5d
```

### 2) Split by Day (one file per day)

```bash
noscroll run --last 5d --bucket day --out ./outputs --name-template "{start:%Y-%m-%d}.md"
```

### 3) Restrict Source Types

```bash
noscroll run --last 5d --source-types rss,hn
```

Allowed values: `rss`, `web`, `hn` (comma-separated combinations supported).

### 4) Preview Without Writing Files

```bash
noscroll run --last 5d --bucket day --dry-run
```

### 5) Natural-Language Driven

```bash
noscroll --env-file .env ask "Collect content from the past five days, one file per day"
```

Preview only (no actual execution):

```bash
noscroll --env-file .env ask "Collect content from the past five days, one file per day" --dry-run
```

## Prerequisites / Sanity Checks

Before using this skill in a new environment, run:

```bash
noscroll --version
noscroll run --help
noscroll ask --help
```

If config or subscriptions are required, verify they exist before execution:

```bash
test -f subscriptions/subscriptions.toml || echo "Missing subscriptions/subscriptions.toml"
test -f .env || echo "Missing .env (optional, but required for ask in most setups)"
```

Quick bootstrap (if missing):

```bash
noscroll init
mkdir -p subscriptions && touch subscriptions/subscriptions.toml
```

## Recommended Agent Workflow

1. Decide whether `--env-file .env` is needed (usually yes for LLM scenarios).
2. If user intent is natural language or underspecified, use `ask --dry-run` first.
3. If time window and output shape are explicit, use `run` directly.
4. Start with `--dry-run` for confirmation, then execute real command only after user confirmation.
5. For day/hour splitting, always set all three:
	 - `--bucket`
	 - directory-style `--out` (not a file path)
	 - `--name-template` that produces unique filenames

## Ask Semantics (Fixed)

- `ask` resolves natural-language intent into NoScroll run parameters.
- `ask --dry-run` prints the resolved plan/parameters and does not execute collection.
- `ask` without `--dry-run` proceeds to execute using resolved parameters and writes output files.
- Preferred reproducible workflow:
  1) run `ask ... --dry-run`
  2) convert/confirm exact `run ...` flags
  3) execute `run ...`

## Quick Mapping (Intent → Flags)

- “past five days” → `--last 5d`
- “from X to Y” → `--from X --to Y`
- “one file per day” → `--bucket day`
- “one file per hour” → `--bucket hour`
- “write to a directory” → `--out ./outputs`
- “only RSS/HN” → `--source-types rss,hn`
- “preview only” → `--dry-run`

## Output Rules

- without `--bucket`: single output file (default like `./noscroll.md`)
- with `--bucket`: multiple files under output directory, based on `--name-template`
- `--format` supports `markdown|json` (default: `markdown`)

## Time Window and Timezone Rules

- `--last`: relative duration (examples: `5d`, `36h`, `2w`).
- `--from` / `--to`: RFC3339/ISO-8601 timestamp or `YYYY-MM-DD` date.
- If only date is provided, boundaries are interpreted by NoScroll time parsing rules.
- Day/hour bucket splitting follows natural local-time boundaries.
- When handling cross-day tasks, explicitly confirm timezone assumptions with the user when relevant.

## Supported Environment Variables

You can load variables from an env file via:

```bash
noscroll --env-file .env run ...
```

or

```bash
noscroll --env-file .env ask "..."
```

### Config Path Resolution

- `NOSCROLL_CONFIG`: explicit config file path

Keep only commonly needed variables in this file:

- `LLM_API_URL`, `LLM_API_KEY`, `LLM_MODEL` (minimum for most `ask` scenarios)
- `NOSCROLL_CONFIG`
- `SUBSCRIPTIONS_PATH`
- `NOSCROLL_OUT`, `NOSCROLL_FORMAT`, `NOSCROLL_SOURCE_TYPES`
- `NOSCROLL_DEBUG` / `DEBUG`

For the complete environment-variable mapping, see `references/env.md` in this skill directory.

## Guardrails / DO NOT

- Do not invent flags; check `noscroll run --help` / `noscroll ask --help` when unsure.
- Default to `--dry-run` first; execute real writes only after explicit user approval.
- Do not overwrite existing outputs silently; if overwrite is needed, explain impact and ask first.
- Do not assume unsupported data sources outside RSS/Web/HN.

## Common Errors and Fixes

- Error: `When bucket is set, --out must be a directory path`
	- Fix: change `--out` from a file path like `xxx.md` to a directory like `./outputs`

- Error: LLM is not configured (for `ask` usage)
	- Fix: provide `LLM_API_URL`, `LLM_API_KEY`, and `LLM_MODEL` or `LLM_SUMMARY_MODEL`

- Error: invalid source type
	- Fix: use only `rss`, `web`, `hn`

## Example Templates

### Past 3 Days, Single File

```bash
noscroll run --last 3d --out ./noscroll.md
```

### Past 7 Days, Daily Files, HN Only

```bash
noscroll run --last 7d --bucket day --source-types hn --out ./outputs --name-template "{start:%Y-%m-%d}.md"
```

### Natural-Language Request (auto-parameterized)

```bash
noscroll --env-file .env ask "Collect Hacker News from the past seven days, output in English, one file per day"
```
