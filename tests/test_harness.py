"""The harness sync's local half: which files are the harness's, and what an install writes and records."""

from __future__ import annotations

import json
from pathlib import Path

from reef_client.harness import (
    INSTALL_FILE,
    TOKEN_PLACEHOLDER,
    HarnessInstall,
    file_pattern,
    read_tree,
    split_command,
)

TEMPLATES = ["pi-agent/AGENTS.md", "pi-agent/skills/{name}/SKILL.md"]


def test_only_the_files_the_templates_name_are_read(tmp_path: Path) -> None:
    root = tmp_path / "agent"
    (root / "skills" / "fix").mkdir(parents=True)
    (root / "sessions").mkdir()
    (root / "AGENTS.md").write_text("Be brief.\n")
    (root / "skills" / "fix" / "SKILL.md").write_text("skill\n")
    (root / "auth.json").write_text('{"key": "a credential"}\n')
    (root / "sessions" / "one.jsonl").write_text("{}\n")
    (root / "skills" / "linked").mkdir()
    (root / "skills" / "linked" / "SKILL.md").symlink_to(root / "auth.json")
    files = read_tree({"pi-agent": root}, file_pattern(TEMPLATES))
    assert files == {"pi-agent/AGENTS.md": "Be brief.\n", "pi-agent/skills/fix/SKILL.md": "skill\n"}


def test_an_install_writes_the_token_records_the_release_and_sees_an_edit(tmp_path: Path) -> None:
    record = {
        "reef_url": "http://reef.invalid",
        "scenario": "s",
        "adapter": "pi",
        "binary": "pi",
        "roots": {"pi-agent": {"variable": "PI_CODING_AGENT_DIR", "default": "~/.pi/agent"}},
        "files_templates": TEMPLATES,
        "release_id": None,
        "files": {"pi-agent/skills/old/SKILL.md": "x"},
    }
    install = HarnessInstall(tmp_path, record, token="secret")
    (tmp_path / "pi-agent" / "skills" / "old").mkdir(parents=True)
    (tmp_path / "pi-agent" / "skills" / "old" / "SKILL.md").write_text("old\n")
    files = {"pi-agent/AGENTS.md": "Be brief.\n", "pi-agent/models.json": f'{{"apiKey": "{TOKEN_PLACEHOLDER}"}}\n'}
    install.install({"release_id": "r1", "files": files})
    assert json.loads((tmp_path / "pi-agent" / "models.json").read_text()) == {"apiKey": "secret"}
    # The previous install's file is gone; the record names the release and each file.
    assert not (tmp_path / "pi-agent" / "skills" / "old" / "SKILL.md").exists()
    saved = json.loads((tmp_path / INSTALL_FILE).read_text())
    assert saved["release_id"] == "r1" and sorted(saved["files"]) == sorted(files)
    assert install.changed() == []
    (tmp_path / "pi-agent" / "AGENTS.md").write_text("Briefer.\n")
    (tmp_path / "pi-agent" / "models.json").write_text("{}\n")
    # A config file is Reef's: only the harness files count as changed.
    assert install.changed() == ["pi-agent/AGENTS.md"]


def test_a_command_after_the_separator_is_the_agents() -> None:
    assert split_command(["run", "s", "--dir", "d", "--", "pi", "--", "x"]) == (
        ["run", "s", "--dir", "d"],
        True,
        ["pi", "--", "x"],
    )
    assert split_command(["status", "s"]) == (["status", "s"], False, [])
