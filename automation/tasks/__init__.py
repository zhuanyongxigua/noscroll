"""Predefined tasks for automation.

Each task is a natural language instruction describing what to test/run.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    """A task for automation."""
    name: str
    instruction: str
    description: str
    expected_files: Optional[list[str]] = None


# Basic functionality tasks
BASIC_TASKS = [
    Task(
        name="basic_run_5d",
        instruction="运行 noscroll，获取过去 5 天的内容，输出到一个文件",
        description="Basic 5-day run with single output file",
    ),
    Task(
        name="bucket_day_5d",
        instruction="运行 noscroll，获取过去 5 天的内容，按天分桶输出，每天一个文件",
        description="5-day run with daily bucket splitting",
    ),
    Task(
        name="hn_only",
        instruction="运行 noscroll，只获取 Hacker News 的内容，过去 3 天",
        description="Hacker News only for 3 days",
    ),
    Task(
        name="rss_only",
        instruction="运行 noscroll，只获取 RSS 订阅源的内容，过去 5 天",
        description="RSS feeds only for 5 days",
    ),
    Task(
        name="chinese_output",
        instruction="运行 noscroll，获取过去 3 天的内容，输出语言设置为中文",
        description="3-day run with Chinese output",
    ),
]

# Edge case tasks
EDGE_TASKS = [
    Task(
        name="short_window_1h",
        instruction="运行 noscroll，获取过去 1 小时的内容",
        description="Very short time window (1 hour)",
    ),
    Task(
        name="long_window_30d",
        instruction="运行 noscroll，获取过去 30 天的内容",
        description="Long time window (30 days)",
    ),
    Task(
        name="bucket_hour",
        instruction="运行 noscroll，获取过去 12 小时的内容，按小时分桶",
        description="Hourly bucket splitting",
    ),
]

# Combined tasks
COMBINED_TASKS = [
    Task(
        name="hn_chinese_daily",
        instruction="运行 noscroll，只获取 Hacker News，过去 5 天，中文输出，按天分桶",
        description="HN + Chinese + daily buckets",
    ),
    Task(
        name="all_sources_serial",
        instruction="运行 noscroll，获取所有来源（RSS、Web、HN），过去 3 天，使用串行模式避免速率限制",
        description="All sources with serial mode",
    ),
]

# All tasks
ALL_TASKS = BASIC_TASKS + EDGE_TASKS + COMBINED_TASKS


def get_task_by_name(name: str) -> Optional[Task]:
    """Get a task by name."""
    for task in ALL_TASKS:
        if task.name == name:
            return task
    return None


def list_tasks() -> list[str]:
    """List all task names."""
    return [task.name for task in ALL_TASKS]
