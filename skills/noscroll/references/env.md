# NoScroll Environment Variables Reference

This document contains the full environment-variable mapping for NoScroll.

## Config Path

- `NOSCROLL_CONFIG`: explicit config file path

## Run-Time Window and Output (`NOSCROLL_*`)

- `NOSCROLL_LAST` → `--last`
- `NOSCROLL_FROM` → `--from`
- `NOSCROLL_TO` → `--to`
- `NOSCROLL_BUCKET` → `--bucket`
- `NOSCROLL_NAME_TEMPLATE` → `--name-template`
- `NOSCROLL_OUT` → `--out`
- `NOSCROLL_FORMAT` → `--format`
- `NOSCROLL_SOURCE_TYPES` → `--source-types`

## LLM and Runtime Tuning

- `LLM_API_URL` → `--llm-api-url`
- `LLM_API_KEY` → `--llm-api-key`
- `LLM_MODEL` → `--llm-model`
- `LLM_SUMMARY_MODEL` → `--llm-summary-model`
- `LLM_API_MODE` → `--llm-api-mode`
- `LLM_TIMEOUT_MS` → `--llm-timeout`
- `LLM_GLOBAL_CONCURRENCY` → `--llm-concurrency`
- `NOSCROLL_PARALLEL` / `NOSCROLL_SERIAL` → parallel/serial mode default
- `NOSCROLL_DELAY` → `--delay`
- `NOSCROLL_LANG` → `--lang`
- `NOSCROLL_TOP_N` → `--top-n`

## Paths

- `SUBSCRIPTIONS_PATH` → `--subscriptions`
- `SYSTEM_PROMPT_PATH` → `--system-prompt`
- `LLM_LOG_PATH` → `--llm-log`
- `FEED_LOG_PATH` → `--feed-log`
- `OUTPUT_DIR` → config-level default output directory

## Debug

- `NOSCROLL_DEBUG` or `DEBUG` → `--debug`

## Legacy Compatibility

- `FEEDS_PATH` (legacy alias of `SUBSCRIPTIONS_PATH`)
- `OPML_PATH` (legacy alias of `SUBSCRIPTIONS_PATH`)
