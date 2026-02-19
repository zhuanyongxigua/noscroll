# NoScroll - 主动拉取，而不是被动刷流

[English](README.md) | **中文**

NoScroll 是一个 Python CLI 工具，用于从 RSS、网页和 Hacker News 抓取内容，并使用 LLM 进行摘要与排序，输出高信号日报。

## 安装

```bash
pipx install noscroll
```

如果需要 `web` 抓取能力，请安装 crawler 扩展：

```bash
pipx install "noscroll[crawler]"
```

## 最小权限配置（沙箱）

NoScroll 需要联网访问 RSS 与 Hacker News。

### Codex（`~/.codex/config.toml` 或项目内 `.codex/config.toml`）

```toml
# workspace-write 默认无网络。
# 抓取时需要打开 network_access。
sandbox_mode = "workspace-write" # read-only | workspace-write | danger-full-access

[sandbox_workspace_write]
network_access = true
```

### Claude Code（项目内 `.claude/settings.json` 或 `~/.claude/settings.json`）

```json
{
  "permissions": {
    "allow": ["Bash(noscroll *)"]
  },
  "sandbox": {
    "enabled": true,
    "network": {
      "allowedDomains": [
        "hn.algolia.com",
        "<YOUR_RSS_DOMAIN_1>",
        "<YOUR_RSS_DOMAIN_2>"
      ]
    }
  }
}
```

### OpenClaw（`~/.openclaw/openclaw.json`）

```json
{
  "tools": {
    "allow": ["exec", "read"]
  }
}
```

## 说明

- 完整功能与命令说明请优先参考英文文档：[README.md](README.md)
- 中文文档会持续补充。
