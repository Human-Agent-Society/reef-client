"""Sync an agent's harness with Reef, and run the agent on the release Reef serves.

    reef-client harness connect SCENARIO --reef URL --adapter pi
    reef-client harness run SCENARIO [-- pi ...]
    reef-client harness pull SCENARIO [--force]
    reef-client harness push SCENARIO
    reef-client harness status SCENARIO

``connect`` reads the harness where the agent keeps it (``~/.pi/agent``, or
where its variable points) and imports it into a scenario that has not evolved
yet. Then it installs the scenario's head in a directory that this client
owns, bound to Reef: the agent sends its model calls to Reef, which records
them as the scenario's traffic. The person's own directory is read and never
written.

``run`` installs a newer release when the installed files are unchanged, then
runs the agent with its root variables at the install, so every session starts
on the newest release. ``pull`` installs the head and refuses to drop changed
files unless forced. ``push`` publishes the changed files as a new release and
installs it. ``status`` shows the installed release, the head, and the files
changed since the install.

The client knows no adapter: ``GET /reef/harness/adapters`` says where each
root lives and which files are the harness's. The token comes from
``REEF_TOKEN``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.parse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reef_client.client import ReefClient, ReefClientError

#: The record of an install, inside the install directory.
INSTALL_FILE = ".reef-install.json"
#: The key that Reef writes in the files of a client install, which the client replaces with its token.
TOKEN_PLACEHOLDER = "REEF_TOKEN_PLACEHOLDER"
#: The key an install gets when the client has no token: some agents refuse to start without one.
NO_TOKEN_KEY = "reef"


class HarnessSyncError(RuntimeError):
    """A sync that cannot go on, with what the person should do next."""


def install_root(scenario: str) -> Path:
    """Where this client installs ``scenario``'s harness: ``$XDG_DATA_HOME/reef-client/harness/<scenario>``."""
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data_home) / "reef-client" / "harness" / scenario


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


def read_tree(roots: Mapping[str, Path], pattern: re.Pattern[str]) -> dict[str, str]:
    """The harness files under ``roots``, by render path: each root's files whose ``<root>/<path>`` matches."""
    files: dict[str, str] = {}
    for name, directory in roots.items():
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            relative = f"{name}/{path.relative_to(directory).as_posix()}"
            if path.is_file() and not path.is_symlink() and pattern.fullmatch(relative):
                try:
                    files[relative] = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
    return files


class HarnessInstall:
    """One scenario's harness installed on this machine, and the Reef it syncs with.

    Values: ``directory`` holds the installed files at their render paths
    and ``INSTALL_FILE``, the record: the Reef URL, the scenario, the
    adapter's binary, roots and file templates, the installed release and
    the digest of each installed file. ``client`` reaches the Reef the record
    names.
    Purpose: the ``reef-client harness`` commands make one from the record
    (``open``), or from the adapter listing on ``connect`` (``create``).
    """

    def __init__(self, directory: Path, record: dict[str, Any], token: str | None) -> None:
        self.directory = directory
        self.record = record
        self.token = token
        self.client = ReefClient(record["reef_url"], token=token, timeout_s=300.0)
        self.pattern = file_pattern(record["files_templates"])

    @classmethod
    def open(cls, directory: Path, token: str | None) -> HarnessInstall:
        path = directory / INSTALL_FILE
        if not path.is_file():
            raise HarnessSyncError(f"nothing is installed at {directory}: run reef-client harness connect first")
        return cls(directory, json.loads(path.read_text(encoding="utf-8")), token)

    @classmethod
    def create(cls, directory: Path, reef_url: str, scenario: str, adapter: str, token: str | None) -> HarnessInstall:
        client = ReefClient(reef_url, token=token, timeout_s=300.0)
        adapters = {entry["name"]: entry for entry in client.get("/reef/harness/adapters")["adapters"]}
        entry = adapters.get(adapter)
        if entry is None:
            raise HarnessSyncError(f"this Reef has no adapter {adapter!r}; it has {', '.join(sorted(adapters))}")
        if not entry["roots"]:
            raise HarnessSyncError(f"adapter {adapter!r} does not sync with a person's machine yet")
        record = {
            "reef_url": reef_url.rstrip("/"),
            "scenario": scenario,
            "adapter": adapter,
            "binary": entry["binary"],
            "roots": entry["roots"],
            "files_templates": entry["files"],
            "release_id": None,
            "files": {},
        }
        return cls(directory, record, token)

    # -- what is where ---------------------------------------------------------------------------------------------

    def person_roots(self) -> dict[str, Path]:
        """Where the agent keeps each root on this machine: its variable when set, else its default."""
        return {
            name: Path(os.environ.get(root["variable"]) or os.path.expanduser(root["default"]))
            for name, root in self.record["roots"].items()
        }

    def install_roots(self) -> dict[str, Path]:
        return {name: self.directory / name for name in self.record["roots"]}

    def changed(self) -> list[str]:
        """The harness files of the install that differ from the installed release: changed, added or removed."""
        installed = {path: checksum for path, checksum in self.record["files"].items() if self.pattern.fullmatch(path)}
        current = {path: digest(text) for path, text in read_tree(self.install_roots(), self.pattern).items()}
        return sorted(path for path in installed.keys() | current.keys() if installed.get(path) != current.get(path))

    # -- the routes ------------------------------------------------------------------------------------------------

    def head(self) -> dict[str, Any]:
        """The head release rendered for this install: model calls bound to this Reef's route of the scenario."""
        query = urllib.parse.urlencode({"form": "client", "service_url": self.record["reef_url"]})
        return self.client.get(f"/reef/harness?{query}", extra_headers={"x-reef-scenario": self.record["scenario"]})

    def import_files(self, files: Mapping[str, str], base_release_id: str | None) -> dict[str, Any]:
        """``POST /reef/harness/import``; HarnessSyncError for a refusal the person resolves: a file that fails
        Reef's checks (HTTP 400), or a head the install is not on (HTTP 409)."""
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
        """Write ``manifest``'s files, remove the files of the previous install it lacks, and write the record."""
        key = self.token or NO_TOKEN_KEY
        files = {path: text.replace(TOKEN_PLACEHOLDER, key) for path, text in manifest["files"].items()}
        for path in self.record["files"]:
            if path not in files:
                (self.directory / path).unlink(missing_ok=True)
        for path, text in files.items():
            target = self.directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        self.record["release_id"] = manifest["release_id"]
        self.record["files"] = {path: digest(text) for path, text in files.items()}
        (self.directory / INSTALL_FILE).write_text(json.dumps(self.record, indent=2) + "\n", encoding="utf-8")

    def connect(self) -> str:
        """Import the person's harness into a scenario that has not evolved, then install the head; a report."""
        self.client.post("/reef/scenarios", self.record["scenario"], {"name": self.record["scenario"]})
        mine = read_tree(self.person_roots(), self.pattern)
        lines = []
        if not mine:
            lines.append("found no harness files on this machine; installing the scenario's head")
        else:
            try:
                answer = self.import_files(mine, None)
                lines.append(f"imported {len(mine)} files from {', '.join(map(str, self.person_roots().values()))}")
                if answer["ignored"]:
                    lines.append(f"ignored: {', '.join(answer['ignored'])}")
            except HarnessSyncError as exc:
                lines.append(f"did not import your harness: {exc}")
        manifest = self.head()
        self.install(manifest)
        lines.append(f"installed release {self.record['release_id']} at {self.directory}")
        lines.extend(required(manifest))
        lines.append(f"run your agent with: reef-client harness run {self.record['scenario']}")
        return "\n".join(lines)

    def pull(self, *, force: bool = False) -> str:
        manifest = self.head()
        if manifest["release_id"] == self.record["release_id"]:
            return f"release {manifest['release_id']} is installed and is the head"
        changed = self.changed()
        if changed and not force:
            raise HarnessSyncError(
                f"files changed since release {self.record['release_id']}: {', '.join(changed)}. "
                f"Push them with reef-client harness push {self.record['scenario']}, or drop them with pull --force"
            )
        previous = self.record["release_id"]
        self.install(manifest)
        return "\n".join([f"installed release {manifest['release_id']} (was {previous})", *required(manifest)])

    def push(self) -> str:
        changed = self.changed()
        if not changed:
            return f"nothing changed since release {self.record['release_id']}"
        answer = self.import_files(read_tree(self.install_roots(), self.pattern), self.record["release_id"])
        # The release reads back in the form Reef renders it, so the install is the release again.
        self.install(self.head())
        return f"published {', '.join(changed)} as release {answer['release_id']}, and installed it"

    def status(self) -> str:
        head = self.head()["release_id"]
        changed = self.changed()
        lines = [
            f"scenario {self.record['scenario']} on {self.record['reef_url']} ({self.record['adapter']})",
            f"installed release {self.record['release_id']} at {self.directory}",
            "the head" if head == self.record["release_id"] else f"the head is release {head}: pull it",
        ]
        lines.append(f"changed: {', '.join(changed)}" if changed else "no file changed since the install")
        return "\n".join(lines)

    def run(self, command: Sequence[str]) -> None:
        """Pull a newer release when nothing changed, then replace this process with ``command`` on the install."""
        try:
            print(self.pull(), file=sys.stderr)
        except HarnessSyncError as exc:
            print(f"kept release {self.record['release_id']}: {exc}", file=sys.stderr)
        except (ReefClientError, OSError) as exc:
            print(f"kept release {self.record['release_id']}: Reef did not answer ({exc})", file=sys.stderr)
        environment = dict(os.environ)
        for name, root in self.record["roots"].items():
            environment[root["variable"]] = str(self.directory / name)
        argv = list(command) or [self.record["binary"]]
        os.execvpe(argv[0], argv, environment)


def split_command(argv: list[str]) -> tuple[list[str], bool, list[str]]:
    """The arguments before the first ``--``, whether there is one, and the agent command after it."""
    if "--" not in argv:
        return argv, False, []
    index = argv.index("--")
    return argv[:index], True, argv[index + 1 :]


def main(argv: Sequence[str] | None = None) -> None:
    arguments_before, separator, agent = split_command(list(sys.argv[1:] if argv is None else argv))
    parser = argparse.ArgumentParser(prog="reef-client harness", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    connect = commands.add_parser("connect", help="import your harness and install the scenario's head")
    connect.add_argument("scenario")
    connect.add_argument("--reef", required=True, help="the Reef URL, such as http://127.0.0.1:8900")
    connect.add_argument("--adapter", required=True, help="the agent: pi, claude, codex or opencode")
    for name, help in (
        ("run", "run the agent on the newest release; a command after -- replaces the agent's own"),
        ("pull", "install the head"),
        ("push", "publish the changed files as a release"),
        ("status", "show the installed release, the head and the changed files"),
    ):
        command = commands.add_parser(name, help=help)
        command.add_argument("scenario")
        if name == "pull":
            command.add_argument("--force", action="store_true", help="drop the changed files")
    for command in commands.choices.values():
        command.add_argument("--dir", type=Path, help="the install directory")
    arguments = parser.parse_args(arguments_before)
    if separator and arguments.command != "run":
        parser.error("only run takes a command after --")
    token = os.environ.get("REEF_TOKEN") or None
    directory = arguments.dir or install_root(arguments.scenario)
    try:
        if arguments.command == "connect":
            directory.mkdir(parents=True, exist_ok=True)
            install = HarnessInstall.create(directory, arguments.reef, arguments.scenario, arguments.adapter, token)
            print(install.connect())
            return
        install = HarnessInstall.open(directory, token)
        if arguments.command == "run":
            install.run(agent)
        elif arguments.command == "pull":
            print(install.pull(force=arguments.force))
        elif arguments.command == "push":
            print(install.push())
        else:
            print(install.status())
    except HarnessSyncError as exc:
        raise SystemExit(f"reef-client harness: {exc}") from exc


__all__ = ["INSTALL_FILE", "TOKEN_PLACEHOLDER", "HarnessInstall", "HarnessSyncError", "install_root", "main"]
