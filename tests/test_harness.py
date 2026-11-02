"""The harness sync's local half: which files sync, what an install writes and records, which install a command
means, and how connect installs the agent and routes its command through Reef."""

from __future__ import annotations

import fcntl
import json
import os
import re
from pathlib import Path

import pytest

from reef_client.harness import (
    INSTALL_FILE,
    SESSION_LOCK,
    TOKEN_PLACEHOLDER,
    HarnessInstall,
    HarnessSyncError,
    commands_root,
    file_pattern,
    find_agent,
    install_agent,
    install_directory,
    locked,
    read_tree,
    route_through_reef,
    split_command,
    unroute,
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


def test_a_sync_writes_no_file_for_the_files_it_sent_and_none_under_a_session(tmp_path: Path, monkeypatch) -> None:
    install = HarnessInstall.create(tmp_path / "s", "http://reef.invalid", "s", token=None)
    base = {"release_id": "r1", "files": {"claude/CLAUDE.md": "Be brief.\n"}, "run": {}, "synced": SYNCED}
    heads = [base, {**base, "release_id": "r2", "files": {"claude/CLAUDE.md": "Briefer.\n"}}]
    sent = {"changed": True, "merged": False, "release_id": "r2", "parent_release_id": "r1"}
    monkeypatch.setattr(install, "head", lambda: heads[-1])
    monkeypatch.setattr(install, "import_files", lambda files, base_release_id: sent)
    install.install(base)
    rules = tmp_path / "s" / "claude" / "CLAUDE.md"
    rules.write_text("Briefer.\n")
    with locked(tmp_path / "s" / SESSION_LOCK, fcntl.LOCK_SH):
        # The head is the files just sent: the record moves, and nothing is written or reported as installed.
        assert install.sync() == "sent claude/CLAUDE.md as release r2"
        assert install.record["release_id"] == "r2" and install.changed() == []
        # A step's release would write a file under the running session, so it waits.
        heads.append({**base, "release_id": "r3", "files": {"claude/CLAUDE.md": "Briefer, always.\n"}})
        assert install.sync() == "release r3 installs at the next run: an agent session runs"
        assert rules.read_text() == "Briefer.\n" and install.record["release_id"] == "r2"
        with pytest.raises(HarnessSyncError, match="discard after it exits"):
            install.sync(discard=True)
    assert install.sync() == "installed release r3 (was r2)"
    assert rules.read_text() == "Briefer, always.\n"


def test_the_agents_register_commands_run_once_per_release_until_disconnect_undoes_them(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    log = tmp_path / "codex.log"
    agent = tmp_path / "codex"
    agent.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\n[ "$1" != fail ]\n')
    agent.chmod(0o755)
    run = {"binary": "codex", "args": [], "env": {}, "register": [["plugin", "add", "reef@reef"]]}
    run["unregister"] = [["plugin", "remove", "reef@reef"]]
    install = HarnessInstall.create(tmp_path / "s", "http://reef.invalid", "s", token=None)
    install.install({"release_id": "r1", "files": {}, "run": run, "synced": []})
    install.record["agent"] = str(agent)
    install.register()
    install.register()
    install.install({"release_id": "r2", "files": {}, "run": run, "synced": []})
    install.register()
    assert log.read_text() == "plugin add reef@reef\n" * 2
    # A command that fails names itself, and the next run tries the release again.
    install.install({"release_id": "r3", "files": {}, "run": {**run, "register": [["fail"]]}, "synced": []})
    with pytest.raises(HarnessSyncError, match="codex fail failed"):
        install.register()
    assert install.record["registered"] == "r2"
    assert install.disconnect() == "removed the install of s; your own configuration is as it was"
    assert log.read_text().endswith("fail\nplugin remove reef@reef\n")


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


def test_the_agent_command_runs_through_reef_until_disconnect(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.delenv("ZDOTDIR", raising=False)
    profile = tmp_path / "home" / ".zshrc"
    profile.parent.mkdir()
    profile.write_text("export EDITOR=vim\n")
    real = tmp_path / "vendor"
    real.mkdir()
    (real / "claude").write_text("#!/bin/sh\n")
    (real / "claude").chmod(0o755)
    assert route_through_reef("claude", "code-repair", tmp_path / "install") == (None, profile)
    assert route_through_reef("claude", "code-repair", tmp_path / "install") == ("code-repair", profile)
    command = commands_root() / "claude"
    assert "harness run code-repair --dir" in command.read_text() and command.stat().st_mode & 0o111
    # The startup file gains one block, ahead on PATH, however often connect runs.
    assert profile.read_text().count("reef-client/bin") == 1 and profile.read_text().startswith("export EDITOR")
    # The client's own command never stands in for the agent it runs.
    monkeypatch.setenv("PATH", f"{commands_root()}{os.pathsep}{real}")
    assert find_agent("claude") == str(real / "claude")
    assert not unroute("claude", "another") and command.exists()
    assert unroute("claude", "code-repair") and not commands_root().exists()
    assert profile.read_text() == "export EDITOR=vim\n"


def test_a_missing_agent_installs_its_pinned_npm_package(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    pinned = {"kind": "npm", "package": "@acme/agent", "version": "1.2.3", "binary_path": "node_modules/.bin/agent"}
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(HarnessSyncError, match=re.escape("needs npm: install Node.js, or @acme/agent@1.2.3")):
        install_agent("agent", pinned)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # A stand-in npm: it records its arguments and makes the binary under the prefix it was given.
    (bin_dir / "npm").write_text(
        '#!/bin/sh\necho "$@" > "$3/args"\nmkdir -p "$3/node_modules/.bin" && touch "$3/node_modules/.bin/agent"\n'
    )
    (bin_dir / "npm").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin")
    executable = install_agent("agent", pinned)
    prefix = tmp_path / "data" / "reef-client" / "agents" / "agent"
    assert executable == str(prefix / "node_modules" / ".bin" / "agent")
    arguments = ["install", "--prefix", str(prefix), "--no-fund", "--no-audit", "--loglevel=error", "@acme/agent@1.2.3"]
    assert (prefix / "args").read_text().split() == arguments
