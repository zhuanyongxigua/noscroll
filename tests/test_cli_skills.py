"""Tests for skills install CLI behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import noscroll.cli as cli


def _write_skill(skills_root: Path, name: str, content: str = "example") -> None:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} description\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    (skill_dir / "assets").mkdir(parents=True, exist_ok=True)
    (skill_dir / "assets" / "sample.txt").write_text(content, encoding="utf-8")


class TestSkillsInstall:
    def test_install_to_empty_project_targets(self, tmp_path: Path):
        parser = cli.build_parser()
        source_root = tmp_path / "source_skills"
        source_root.mkdir()
        _write_skill(source_root, "mytool")

        scenarios = [
            ("claude", [".claude/skills/mytool"]),
            ("codex", [".agents/skills/mytool"]),
            (
                "both",
                [
                    ".claude/skills/mytool",
                    ".agents/skills/mytool",
                ],
            ),
        ]

        for i, (host, expected_rel_paths) in enumerate(scenarios):
            project_root = tmp_path / f"project-{i}"
            project_root.mkdir()

            args = parser.parse_args([
                "skills",
                "install",
                "mytool",
                "--host",
                host,
                "--scope",
                "project",
            ])
            with patch("noscroll.cli._resolve_skills_source_root", return_value=source_root):
                with patch("noscroll.cli._resolve_project_root", return_value=project_root):
                    result = cli._run_skills_install(args)
            assert result == 0

            expected_paths = [project_root / rel_path for rel_path in expected_rel_paths]
            for path in expected_paths:
                assert path.exists()
                assert (path / "SKILL.md").exists()

    def test_install_with_all(self, tmp_path: Path):
        parser = cli.build_parser()
        source_root = tmp_path / "source_skills"
        source_root.mkdir()
        _write_skill(source_root, "mytool")
        _write_skill(source_root, "secondtool")

        project_root = tmp_path / "project"
        project_root.mkdir()

        args = parser.parse_args([
            "skills",
            "install",
            "--all",
            "--host",
            "both",
            "--scope",
            "project",
        ])

        with patch("noscroll.cli._resolve_skills_source_root", return_value=source_root):
            with patch("noscroll.cli._resolve_project_root", return_value=project_root):
                result = cli._run_skills_install(args)

        assert result == 0
        assert (project_root / ".claude" / "skills" / "mytool" / "SKILL.md").exists()
        assert (project_root / ".claude" / "skills" / "secondtool" / "SKILL.md").exists()
        assert (project_root / ".agents" / "skills" / "mytool" / "SKILL.md").exists()
        assert (project_root / ".agents" / "skills" / "secondtool" / "SKILL.md").exists()

    def test_existing_target_without_force_returns_2(self, tmp_path: Path):
        parser = cli.build_parser()
        source_root = tmp_path / "source_skills"
        source_root.mkdir()
        _write_skill(source_root, "mytool", content="new")

        project_root = tmp_path / "project"
        existing = project_root / ".claude" / "skills" / "mytool"
        existing.mkdir(parents=True)
        (existing / "SKILL.md").write_text(
            "---\nname: mytool\ndescription: old\n---\n",
            encoding="utf-8",
        )

        args = parser.parse_args([
            "skills",
            "install",
            "mytool",
            "--host",
            "claude",
            "--scope",
            "project",
        ])

        with patch("noscroll.cli._resolve_skills_source_root", return_value=source_root):
            with patch("noscroll.cli._resolve_project_root", return_value=project_root):
                result = cli._run_skills_install(args)

        assert result == 2

    def test_force_creates_backup_and_overwrites(self, tmp_path: Path):
        parser = cli.build_parser()
        source_root = tmp_path / "source_skills"
        source_root.mkdir()
        _write_skill(source_root, "mytool", content="new-content")

        project_root = tmp_path / "project"
        existing = project_root / ".claude" / "skills" / "mytool"
        existing.mkdir(parents=True)
        (existing / "SKILL.md").write_text(
            "---\nname: mytool\ndescription: old\n---\n",
            encoding="utf-8",
        )
        (existing / "assets").mkdir(parents=True, exist_ok=True)
        (existing / "assets" / "sample.txt").write_text("old-content", encoding="utf-8")

        args = parser.parse_args([
            "skills",
            "install",
            "mytool",
            "--host",
            "claude",
            "--scope",
            "project",
            "--force",
        ])

        with patch("noscroll.cli._resolve_skills_source_root", return_value=source_root):
            with patch("noscroll.cli._resolve_project_root", return_value=project_root):
                result = cli._run_skills_install(args)

        assert result == 0
        installed_file = project_root / ".claude" / "skills" / "mytool" / "assets" / "sample.txt"
        assert installed_file.read_text(encoding="utf-8") == "new-content"

        backups = list((project_root / ".claude" / "skills").glob("mytool.bak.*"))
        assert backups
        backup_file = backups[0] / "assets" / "sample.txt"
        assert backup_file.read_text(encoding="utf-8") == "old-content"

    def test_dry_run_prints_targets_and_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
        parser = cli.build_parser()
        source_root = tmp_path / "source_skills"
        source_root.mkdir()
        _write_skill(source_root, "mytool")

        project_root = tmp_path / "project"
        project_root.mkdir()

        args = parser.parse_args([
            "skills",
            "install",
            "mytool",
            "--host",
            "both",
            "--scope",
            "project",
            "--dry-run",
        ])

        with patch("noscroll.cli._resolve_skills_source_root", return_value=source_root):
            with patch("noscroll.cli._resolve_project_root", return_value=project_root):
                result = cli._run_skills_install(args)

        assert result == 0
        out = capsys.readouterr().out
        assert str(project_root / ".claude" / "skills" / "mytool") in out
        assert str(project_root / ".agents" / "skills" / "mytool") in out
        assert "SKILL.md" in out
        assert "assets/sample.txt" in out

    def test_missing_skill_returns_3(self, tmp_path: Path):
        parser = cli.build_parser()
        source_root = tmp_path / "source_skills"
        source_root.mkdir()

        project_root = tmp_path / "project"
        project_root.mkdir()

        args = parser.parse_args([
            "skills",
            "install",
            "missing",
            "--host",
            "claude",
            "--scope",
            "project",
        ])

        with patch("noscroll.cli._resolve_skills_source_root", return_value=source_root):
            with patch("noscroll.cli._resolve_project_root", return_value=project_root):
                result = cli._run_skills_install(args)

        assert result == 3

    def test_openclaw_workspace_install_with_workdir(self, tmp_path: Path):
        parser = cli.build_parser()
        source_root = tmp_path / "source_skills"
        source_root.mkdir()
        _write_skill(source_root, "mytool")

        workspace_root = tmp_path / "openclaw-workspace"
        workspace_root.mkdir()

        args = parser.parse_args([
            "skills",
            "install",
            "mytool",
            "--host",
            "openclaw",
            "--scope",
            "workspace",
            "--workdir",
            str(workspace_root),
        ])

        with patch("noscroll.cli._resolve_skills_source_root", return_value=source_root):
            result = cli._run_skills_install(args)

        assert result == 0
        assert (workspace_root / "skills" / "mytool" / "SKILL.md").exists()

    def test_openclaw_workspace_defaults_to_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        parser = cli.build_parser()
        source_root = tmp_path / "source_skills"
        source_root.mkdir()
        _write_skill(source_root, "mytool")

        workspace_root = tmp_path / "cwd-workspace"
        workspace_root.mkdir()
        monkeypatch.chdir(workspace_root)

        args = parser.parse_args([
            "skills",
            "install",
            "mytool",
            "--host",
            "openclaw",
            "--scope",
            "workspace",
        ])

        with patch("noscroll.cli._resolve_skills_source_root", return_value=source_root):
            result = cli._run_skills_install(args)

        assert result == 0
        assert (workspace_root / "skills" / "mytool" / "SKILL.md").exists()

    def test_openclaw_shared_install(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        parser = cli.build_parser()
        source_root = tmp_path / "source_skills"
        source_root.mkdir()
        _write_skill(source_root, "mytool")

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        args = parser.parse_args([
            "skills",
            "install",
            "mytool",
            "--host",
            "openclaw",
            "--scope",
            "shared",
        ])

        with patch("noscroll.cli._resolve_skills_source_root", return_value=source_root):
            result = cli._run_skills_install(args)

        assert result == 0
        assert (fake_home / ".openclaw" / "skills" / "mytool" / "SKILL.md").exists()

    def test_openclaw_invalid_scope_returns_1(self, tmp_path: Path):
        parser = cli.build_parser()
        source_root = tmp_path / "source_skills"
        source_root.mkdir()
        _write_skill(source_root, "mytool")

        args = parser.parse_args([
            "skills",
            "install",
            "mytool",
            "--host",
            "openclaw",
            "--scope",
            "project",
        ])

        with patch("noscroll.cli._resolve_skills_source_root", return_value=source_root):
            result = cli._run_skills_install(args)

        assert result == 1

    def test_claude_workspace_scope_returns_1(self, tmp_path: Path):
        parser = cli.build_parser()
        source_root = tmp_path / "source_skills"
        source_root.mkdir()
        _write_skill(source_root, "mytool")

        project_root = tmp_path / "project"
        project_root.mkdir()

        args = parser.parse_args([
            "skills",
            "install",
            "mytool",
            "--host",
            "claude",
            "--scope",
            "workspace",
        ])

        with patch("noscroll.cli._resolve_skills_source_root", return_value=source_root):
            with patch("noscroll.cli._resolve_project_root", return_value=project_root):
                result = cli._run_skills_install(args)

        assert result == 1


class TestProjectRootResolution:
    def test_non_git_project_uses_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        mocked = MagicMock(returncode=1, stdout="", stderr="not a git repo")
        with patch("subprocess.run", return_value=mocked):
            root = cli._resolve_project_root()
        assert root == tmp_path
