# NoScroll Automation Harness

Automated **run → test → eval → fix** loop using Claude Agent SDK.

This is an **agent harness** layer, separate from business logic in `src/`.

## Directory Structure

```
automation/
├── __init__.py          # Package entry
├── __main__.py          # python -m automation
├── loop.py              # Main loop: run → test → eval → fix → repeat
├── config.py            # Configuration
├── agents.py            # Claude Agent SDK agents (executor, diagnostic, fixer)
├── tasks/               # Task/scenario definitions
│   └── __init__.py      # Predefined tasks
├── evals/               # Evaluation logic
│   └── __init__.py      # Output validation
├── prompts/             # System prompts (file-based for easy editing)
│   ├── executor.txt     # Executor agent prompt
│   ├── diagnostic.txt   # Diagnostic agent prompt
│   └── fixer.txt        # Fixer agent prompt
├── adapters/            # External tool adapters
│   └── __init__.py      # pytest, ruff, git adapters
└── artifacts/           # Loop outputs: logs, diffs, reports
```

## The Loop

```
┌─────────────────────────────────────────────────────────────┐
│                     Start Task                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Execute: Translate instruction → Run noscroll command    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Evaluate: Check output files, validate content           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    Passed?    │
                    └───────────────┘
                      │           │
                  Yes │           │ No
                      ▼           ▼
              ┌─────────────┐  ┌─────────────────────────────────┐
              │   SUCCESS   │  │  3. Diagnose: Analyze failure    │
              │   (break)   │  │     (describe phenomena only)    │
              └─────────────┘  └─────────────────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────────────────┐
                               │  4. Fix: Apply code changes      │
                               └─────────────────────────────────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │  Max loops?   │
                                  └───────────────┘
                                    │           │
                                Yes │           │ No
                                    ▼           ▼
                              ┌─────────┐   (back to 1)
                              │  FAIL   │
                              └─────────┘
```

## Stop Conditions

1. **Success**: Evaluation passes ✓
2. **Max loops**: Reached iteration limit (default: 3)
3. **Fix failed**: Fixer couldn't apply changes

## Installation

```bash
# Claude Agent SDK requires Claude Code CLI
pip install claude-agent-sdk
```

## Usage

```bash
# List available tasks
python -m automation --list

# Run a specific task
python -m automation --task basic_run_5d

# Run a suite
python -m automation --suite basic

# Custom task
python -m automation --custom "运行 noscroll，获取过去 3 天的 HN 内容"

# Options
python -m automation --task basic_run_5d --max-loops 5
python -m automation --task basic_run_5d --quiet
```

## Artifacts

Each run produces artifacts in `automation/artifacts/<task_name>/`:

- `loop_result.json` - Full iteration history
- Output files from the noscroll command

## Design Principles

1. **Separation of concerns**: This is a harness layer, not business logic
2. **File-based prompts**: Easy to edit and version control
3. **Adapters for tools**: Clean interface to pytest, git, etc.
4. **Artifacts tracking**: Every run produces traceable outputs

## Why Not in `src/`?

The `src/` layout convention keeps only publishable package code in `src/`.
Automation tools, scripts, and harnesses belong at the project root level:

```
project/
├── src/noscroll/     # Business logic (pip installable)
├── tests/            # Unit/integration tests
└── automation/       # Agent harness (development tool)
```

This keeps import paths clean and separates runtime code from development tooling.
