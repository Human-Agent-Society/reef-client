"""Run an agent on the harness Reef evolves, over the person's own configuration, and keep the two in sync.

    reef-client harness connect SCENARIO --reef URL
    reef-client harness run [SCENARIO] [-- AGENT ARGS]
    reef-client harness sync [SCENARIO] [--discard]
    reef-client harness status [SCENARIO]
    reef-client harness disconnect [SCENARIO] [--discard]

``connect`` installs the scenario's head in a directory that this client owns,
as the adapter's layer: the agent loads the person's own configuration as it
always does, and the release over it, with its model calls going to Reef. The
person's own configuration is never read or written. ``run`` syncs, runs the
agent with the layer's arguments and variables, and syncs again when the agent
exits, so every session starts on the newest release and what the session
changed in the layer, such as Claude Code's memory, reaches Reef before its
next step. ``sync`` sends the files changed since the install and installs the
head; Reef merges the changes onto its head as each part's ``merge`` says.
``--discard`` drops the changes instead. ``status`` shows where the install
stands, and ``disconnect`` removes it.

The client knows no adapter: the head, in the client form, carries the files,
the command and the templates of the files that sync. The token comes from
``REEF_TOKEN`` on ``connect``, and the install keeps it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import urllib.parse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef_client.client import ReefClient, ReefClientError

#: The record of an install, inside the install directory.
INSTALL_FILE = ".reef-install.json"
#: The key that Reef writes in the files and variables of a client install, which the client replaces with its token.
TOKEN_PLACEHOLDER = "REEF_TOKEN_PLACEHOLDER"
#: The key an install gets when the client has no token: some agents refuse to start without one.
NO_TOKEN_KEY = "reef"
#: Seconds a sync around a run waits for Reef, so a Reef that does not answer delays a session little.
RUN_SYNC_TIMEOUT_S = 15.0


class HarnessSyncError(RuntimeError):
    """A sync that cannot go on, with what the person should do next."""


def installs_root() -> Path:
    """Where this client keeps its installs: ``$XDG_DATA_HOME/reef-client/harness``."""
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data_home) / "reef-client" / "harness"


def install_directory(scenario: str | None) -> Path:
    """The install of ``scenario``, or of the only scenario installed when ``scenario`` is ``None``."""
    if scenario is not None:
        return installs_root() / scenario
    found = sorted(path.parent.name for path in installs_root().glob(f"*/{INSTALL_FILE}"))
    if len(found) != 1:
        listed = f"installed: {', '.join(found)}" if found else "none is installed"
        raise HarnessSyncError(f"name the scenario; {listed}")
    return installs_root() / found[0]


def file_pattern(templates: Sequence[str]) -> re.Pattern[str]:
    """One pattern that matches the render paths of an adapter's harness files: ``{name}`` is one path segment."""
    parts = [re.escape(template).replace(re.escape("{name}"), "[^/]+") for template in templates]
    return re.compile("|".join(f"(?:{part})" for part in parts) or "(?!)")


def required(manifest: Mapping[str, Any]) -> list[str]:
    """One line per item a release ``requires`` of this machine, with the sentence it asks the person; the client
    runs no check: a person reads a check before running it."""
    return [
        f"the release requires {item.get('kind')} {item.get('name')}: {item.get('prompt') or 'no prompt given'}"
        for item in manifest.get("requires") or []
    ]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_tree(directory: Path, pattern: re.Pattern[str]) -> dict[str, str]:
    """The harness files under ``directory``, by render path: each file whose relative path matches ``pattern``."""
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_file() and not path.is_symlink() and pattern.fullmatch(relative):
            try:
                files[relative] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
    return files


class HarnessInstall:
    """One scenario's harness installed on this machine as a layer, and the Reef it syncs with.

    Values: ``directory`` holds the installed files at their render paths and
    ``INSTALL_FILE``, the record: the Reef URL, the scenario, the token, the
    installed release, the digest of each installed file, the command that
    runs the agent (``run``) and the templates of the files that sync
    (``synced``). ``client`` reaches the Reef the record names.
    Purpose: ``connect`` makes one (``create``); the other commands read it
    back (``open``).
    """

    def __init__(self, directory: Path, record: dict[str, Any], *, timeout_s: float = 60.0) -> None:
        self.directory = directory
        self.record = record
        self.client = ReefClient(record["reef_url"], token=record.get("token"), timeout_s=timeout_s)

    @classmethod
    def open(cls, directory: Path, *, timeout_s: float = 60.0) -> HarnessInstall:
        path = directory / INSTALL_FILE
        if not path.is_file():
            raise HarnessSyncError(f"nothing is installed at {directory}: run reef-client harness connect first")
        return cls(directory, json.loads(path.read_text(encoding="utf-8")), timeout_s=timeout_s)

    @classmethod
    def create(cls, directory: Path, reef_url: str, scenario: str, token: str | None) -> HarnessInstall:
        record = {"reef_url": reef_url.rstrip("/"), "scenario": scenario, "token": token, "release_id": None}
        return cls(directory, {**record, "files": {}, "run": None, "synced": []})

    def changed(self) -> list[str]:
        """The harness files of the install that differ from the installed release: changed, added or removed."""
        pattern = file_pattern(self.record["synced"])
        installed = {path: checksum for path, checksum in self.record["files"].items() if pattern.fullmatch(path)}
        current = {path: digest(text) for path, text in read_tree(self.directory, pattern).items()}
        return sorted(path for path in installed.keys() | current.keys() if installed.get(path) != current.get(path))

    # -- the routes ------------------------------------------------------------------------------------------------

    def head(self) -> dict[str, Any]:
        """The head release in the client form: the layer's files under this install, and the command."""
        query = urllib.parse.urlencode(
            {"form": "client", "service_url": self.record["reef_url"], "root": str(self.directory.resolve())}
        )
        return self.client.get(f"/reef/harness?{query}", extra_headers={"x-reef-scenario": self.record["scenario"]})

    def import_files(self, files: Mapping[str, str], base_release_id: str) -> dict[str, Any]:
        """``POST /reef/harness/import``; HarnessSyncError for a refusal the person resolves: a file that fails
        Reef's checks (HTTP 400), a file that the head changed too, or a step that is running (HTTP 409)."""
        try:
            answer, _ = self.client.post(
                "/reef/harness/import",
                self.record["scenario"],
                {"files": dict(files), "base_release_id": base_release_id},
            )
        except ReefClientError as exc:
            if exc.status in (400, 409):
                raise HarnessSyncError(exc.body) from exc
            raise
        return answer

    # -- the commands ----------------------------------------------------------------------------------------------

    def install(self, manifest: Mapping[str, Any]) -> None:
        """Write ``manifest``'s files with this client's key, remove the files of the previous install it lacks, and
        write the record, readable by this user alone: it holds the token."""
        key = self.record.get("token") or NO_TOKEN_KEY
        files = {path: text.replace(TOKEN_PLACEHOLDER, key) for path, text in manifest["files"].items()}
        self.directory.mkdir(parents=True, exist_ok=True)
        self.directory.chmod(0o700)
        for path in self.record["files"]:
            if path not in files:
                (self.directory / path).unlink(missing_ok=True)
        for path, text in files.items():
            target = self.directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        self.record.update(
            release_id=manifest["release_id"],
            files={path: digest(text) for path, text in files.items()},
            run=manifest["run"],
            synced=manifest["synced"],
        )
        record = self.directory / INSTALL_FILE
        record.write_text(json.dumps(self.record, indent=2) + "\n", encoding="utf-8")
        record.chmod(0o600)

    def connect(self) -> str:
        """Create the scenario when it is new and install its head; a report with the command to run."""
        self.client.post("/reef/scenarios", self.record["scenario"], {"name": self.record["scenario"]})
        manifest = self.head()
        self.install(manifest)
        scenario = self.record["scenario"]
        return "\n".join(
            [
                f"installed release {manifest['release_id']} of {scenario} at {self.directory}",
                *required(manifest),
                f"run your agent with: reef-client harness run {scenario}",
                f"or make it the default: alias {manifest['run']['binary']}='reef-client harness run {scenario} --'",
            ]
        )

    def sync(self, *, discard: bool = False) -> str:
        """Send the changed files, then install the head; a report.

        Reef merges the changes onto its head, so the head after the import holds them. HarnessSyncError when Reef
        refuses them; the changed files then stay as they are. ``discard`` sends nothing and installs the head
        over the changes.
        """
        previous = self.record["release_id"]
        changed = [] if discard else self.changed()
        lines = []
        if changed:
            files = read_tree(self.directory, file_pattern(self.record["synced"]))
            answer = self.import_files(files, previous)
            if answer["changed"]:
                onto = f", merged onto release {answer['parent_release_id']}" if answer["merged"] else ""
                lines.append(f"sent {', '.join(changed)} as release {answer['release_id']}{onto}")
        manifest = self.head()
        if manifest["release_id"] == previous and not changed and not discard:
            return f"release {previous} is installed and is the head"
        # The import reads back in the form Reef renders it, so the install is the release again.
        self.install(manifest)
        lines.append(f"installed release {manifest['release_id']} (was {previous})")
        return "\n".join([*lines, *required(manifest)])

    def status(self) -> str:
        head = self.head()["release_id"]
        changed = self.changed()
        return "\n".join(
            [
                f"scenario {self.record['scenario']} on {self.record['reef_url']}: {self.record['run']['binary']}",
                f"installed release {self.record['release_id']} at {self.directory}",
                "the head" if head == self.record["release_id"] else f"the head is release {head}: sync to install it",
                f"changed, not sent: {', '.join(changed)}" if changed else "no file changed since the install",
            ]
        )

    def disconnect(self, *, discard: bool = False) -> str:
        """Remove the install; HarnessSyncError while it holds changes not sent, unless ``discard``."""
        changed = self.changed()
        if changed and not discard:
            raise HarnessSyncError(f"not sent yet: {', '.join(changed)}. Sync them, or disconnect with --discard")
        shutil.rmtree(self.directory)
        return f"removed the install of {self.record['scenario']}; your own configuration is as it was"

    def try_sync(self) -> None:
        """``sync``, telling the person on stderr why a sync that could not finish keeps the files as they are."""
        kept = f"kept your files on release {self.record['release_id']}"
        try:
            print(self.sync(), file=sys.stderr)
        except HarnessSyncError as exc:
            print(f"{kept}: {exc}", file=sys.stderr)
        except (ReefClientError, OSError) as exc:
            print(f"{kept}: Reef did not answer ({exc}); the next run syncs", file=sys.stderr)

    def run(self, agent_args: Sequence[str]) -> int:
        """Sync, run the agent on the layer with ``agent_args`` added, and sync again when it exits; its exit status.

        A sync that cannot finish keeps the files, and the agent runs on them. While the agent runs, this process
        ignores Ctrl-C, which the agent handles, so the sync after it still runs.
        """
        self.try_sync()
        command = self.record["run"]
        key = self.record.get("token") or NO_TOKEN_KEY
        layer_env = {name: value.replace(TOKEN_PLACEHOLDER, key) for name, value in command["env"].items()}
        interrupt = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            argv = [command["binary"], *command["args"], *agent_args]
            status = subprocess.run(argv, env={**os.environ, **layer_env}, check=False).returncode
        finally:
            signal.signal(signal.SIGINT, interrupt)
        self.try_sync()
        return status


def split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    """The arguments before the first ``--``, and the agent's arguments after it."""
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


def main(argv: Sequence[str] | None = None) -> None:
    ours, agent_args = split_command(list(sys.argv[1:] if argv is None else argv))
    parser = argparse.ArgumentParser(prog="reef-client harness", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    connect = commands.add_parser("connect", help="install the scenario's head as a layer over your own config")
    connect.add_argument("scenario")
    connect.add_argument("--reef", required=True, help="the Reef URL, such as http://127.0.0.1:8900")
    for name, help in (
        ("run", "sync, run the agent, and sync again when it exits; arguments after -- go to the agent"),
        ("sync", "send the changed files and install the head"),
        ("status", "show the installed release, the head and the changed files"),
        ("disconnect", "remove the install"),
    ):
        command = commands.add_parser(name, help=help)
        command.add_argument("scenario", nargs="?", help="the scenario; optional when only one is installed")
        if name in ("sync", "disconnect"):
            command.add_argument("--discard", action="store_true", help="drop the changed files")
    for command in commands.choices.values():
        command.add_argument("--dir", type=Path, help="the install directory")
    arguments = parser.parse_args(ours)
    if agent_args and arguments.command != "run":
        parser.error("only run takes the agent's arguments after --")
    try:
        directory = arguments.dir or install_directory(arguments.scenario)
        if arguments.command == "connect":
            token = os.environ.get("REEF_TOKEN") or None
            print(HarnessInstall.create(directory, arguments.reef, arguments.scenario, token).connect())
            return
        timeout_s = RUN_SYNC_TIMEOUT_S if arguments.command == "run" else 60.0
        install = HarnessInstall.open(directory, timeout_s=timeout_s)
        if arguments.command == "run":
            raise SystemExit(install.run(agent_args))
        if arguments.command == "sync":
            print(install.sync(discard=arguments.discard))
        elif arguments.command == "status":
            print(install.status())
        else:
            print(install.disconnect(discard=arguments.discard))
    except HarnessSyncError as exc:
        raise SystemExit(f"reef-client harness: {exc}") from exc


__all__ = ["INSTALL_FILE", "TOKEN_PLACEHOLDER", "HarnessInstall", "HarnessSyncError", "installs_root", "main"]
