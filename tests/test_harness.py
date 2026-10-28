"""The harness sync's local half: which files sync, what an install writes and records, and which install a
command means."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reef_client.harness import (
    INSTALL_FILE,
    TOKEN_PLACEHOLDER,
    HarnessInstall,
    HarnessSyncError,
    file_pattern,
    install_directory,
    read_tree,
    split_command,
)

SYNCED = ["claude/CLAUDE.md", "claude/skills/{name}/SKILL.md", "claude/memory/{name}.md"]


def test_only_the_files_the_templates_name_are_read(tmp_path: Path) -> None:
    (tmp_path / "claude" / "skills" / "fix").mkdir(parents=True)
    (tmp_path / "claude" / "memory").mkdir()
    (tmp_path / "claude" / "CLAUDE.md").write_text("Be brief.\n")
    (tmp_path / "claude" / "skills" / "fix" / "SKILL.md").write_text("skill\n")
    (tmp_path / "claude" / "memory" / "MEMORY.md").write_text("- a note\n")
    (tmp_path / "claude" / "settings.json").write_text('{"env": {}}\n')
    (tmp_path / "claude" / "skills" / "linked").mkdir()
    (tmp_path / "claude" / "skills" / "linked" / "SKILL.md").symlink_to(tmp_path / "claude" / "settings.json")
    assert read_tree(tmp_path, file_pattern(SYNCED)) == {
        "claude/CLAUDE.md": "Be brief.\n",
        "claude/memory/MEMORY.md": "- a note\n",
        "claude/skills/fix/SKILL.md": "skill\n",
    }


def test_an_install_writes_the_token_keeps_the_command_and_sees_an_edit(tmp_path: Path) -> None:
    install = HarnessInstall.create(tmp_path / "s", "http://reef.invalid", "s", token="secret")
    install.record["files"] = {"claude/skills/old/SKILL.md": "x"}
    (tmp_path / "s" / "claude" / "skills" / "old").mkdir(parents=True)
    (tmp_path / "s" / "claude" / "skills" / "old" / "SKILL.md").write_text("old\n")
    files = {"claude/CLAUDE.md": "Be brief.\n", "claude/settings.json": f'{{"key": "{TOKEN_PLACEHOLDER}"}}\n'}
    run = {"binary": "claude", "args": ["--settings", "x"], "env": {}}
    install.install({"release_id": "r1", "files": files, "run": run, "synced": SYNCED})
    assert json.loads((tmp_path / "s" / "claude" / "settings.json").read_text()) == {"key": "secret"}
    # The previous install's file is gone; the record, which holds the token, is the user's alone.
    assert not (tmp_path / "s" / "claude" / "skills" / "old" / "SKILL.md").exists()
    saved = json.loads((tmp_path / "s" / INSTALL_FILE).read_text())
    assert saved["release_id"] == "r1" and saved["run"] == run and saved["token"] == "secret"
    assert (tmp_path / "s" / INSTALL_FILE).stat().st_mode & 0o077 == 0
    assert install.changed() == []
    (tmp_path / "s" / "claude" / "CLAUDE.md").write_text("Briefer.\n")
    (tmp_path / "s" / "claude" / "memory").mkdir()
    (tmp_path / "s" / "claude" / "memory" / "MEMORY.md").write_text("- learned\n")
    (tmp_path / "s" / "claude" / "settings.json").write_text("{}\n")
    # A config file is Reef's: only the files that sync count as changed.
    assert install.changed() == ["claude/CLAUDE.md", "claude/memory/MEMORY.md"]


def test_a_command_without_a_scenario_means_the_only_install(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    with pytest.raises(HarnessSyncError, match="none is installed"):
        install_directory(None)
    one = tmp_path / "reef-client" / "harness" / "code-repair"
    one.mkdir(parents=True)
    (one / INSTALL_FILE).write_text("{}")
    assert install_directory(None) == one
    (tmp_path / "reef-client" / "harness" / "other").mkdir()
    (tmp_path / "reef-client" / "harness" / "other" / INSTALL_FILE).write_text("{}")
    with pytest.raises(HarnessSyncError, match="installed: code-repair, other"):
        install_directory(None)


def test_the_arguments_after_the_separator_are_the_agents() -> None:
    assert split_command(["run", "s", "--", "--resume", "--", "x"]) == (["run", "s"], ["--resume", "--", "x"])
    assert split_command(["status"]) == (["status"], [])
