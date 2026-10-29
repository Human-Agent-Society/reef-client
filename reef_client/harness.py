"""Run an agent on the harness Reef evolves, over the person's own configuration, and keep the two in sync.

    reef-client harness connect [SCENARIO] [--reef URL]
    reef-client harness run [SCENARIO] [-- AGENT ARGS]
    reef-client harness sync [SCENARIO] [--discard]
    reef-client harness status [SCENARIO]
    reef-client harness disconnect [SCENARIO] [--discard]

``connect`` sets a machine up in one command. It asks only for the harness,
when Reef serves more than one, and the token, when Reef refuses the call
without one (else ``REEF_TOKEN``); the Reef URL is ``--reef`` or
``REEF_URL``. It installs the harness's head in a directory that this client
owns, as the adapter's layer: the agent loads the person's own configuration
as it always does, and the release over it, with its model calls going to
Reef. This client never reads or writes the person's own configuration. An
agent that loads a part from no path a flag names gets the install through
its own commands, which the release names (``register``): Codex's ``plugin``
commands add the install's skills as a plugin to the person's ``~/.codex``.
``connect`` and each ``run`` run them once per installed release, while no
other session runs on the install, and ``disconnect`` undoes them. It installs
the agent, at the version Reef evaluates with, when the machine lacks it, and
makes the agent's own command, such as ``claude``, run through Reef: a command
of that name in this client's ``bin`` directory, put ahead on ``PATH`` in the
shell's startup file. ``run`` syncs, runs the
agent with the layer's arguments and variables, and when the agent exits sends
what the session changed in the layer, such as Claude Code's memory, from a
process of its own: the terminal is back at once. Reef refuses the send while a
step runs, and the change then waits for the next ``run`` or ``sync``. ``sync`` sends the files changed since the install and
installs the head; Reef merges the changes onto its head as each part's
``merge`` says. An install writes only the files that differ, so a head that is
the files just sent writes none, and it writes none while an agent session runs
on the install. ``--discard`` drops the changes instead. ``status`` shows where
the install stands, and ``disconnect`` removes it and the agent command it added.

The client knows no adapter: the head, in the client form, carries the files,
the command and the templates of the files that sync. The token comes from
``REEF_TOKEN`` on ``connect``, and the install keeps it.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import getpass
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import urllib.parse
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from reef_client.client import ReefClient, ReefClientError

#: The record of an install, inside the install directory.
INSTALL_FILE = ".reef-install.json"
#: The key that Reef writes in the files and variables of a client install, which the client replaces with its token.
TOKEN_PLACEHOLDER = "REEF_TOKEN_PLACEHOLDER"
#: The key an install gets when the client has no token: some agents refuse to start without one.
NO_TOKEN_KEY = "reef"
#: Seconds the sync before a run waits for each answer, so a Reef that does not answer delays a session little.
RUN_SYNC_TIMEOUT_S = 2.0
#: The lock that each run holds, shared, while its agent runs; an install that writes files holds it alone.
SESSION_LOCK = ".reef-session.lock"
#: The lock that a sync holds, so two syncs of one install never run at once.
SYNC_LOCK = ".reef-sync.lock"
#: The report of the sync that a run starts in the background when its agent exits; the next run prints it.
SYNC_LOG = ".reef-sync.log"
#: The lines around what ``connect`` adds to the shell's startup file, so ``disconnect`` finds it again.
PROFILE_BLOCK = ("# >>> reef-client >>>", "# <<< reef-client <<<")
#: The line of an agent command that names the scenario it runs.
COMMAND_SCENARIO = re.compile(r"^# reef-client scenario: (.+)$", re.M)


class HarnessSyncError(RuntimeError):
    """A sync that cannot go on, with what the person should do next."""


def data_root() -> Path:
    """Where this client keeps its files: ``$XDG_DATA_HOME/reef-client``."""
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data_home) / "reef-client"


def installs_root() -> Path:
    """Where this client keeps its installs: ``data_root()/harness``."""
    return data_root() / "harness"


def commands_root() -> Path:
    """Where this client keeps the agent commands that run through Reef: ``data_root()/bin``, ahead on ``PATH``."""
    return data_root() / "bin"


def agents_root() -> Path:
    """Where this client installs an agent that the machine lacks: ``data_root()/agents/<binary>``."""
    return data_root() / "agents"


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


@contextlib.contextmanager
def locked(path: Path, operation: int) -> Iterator[bool]:
    """Hold ``fcntl.flock(operation)`` on the file ``path`` for the block; whether it is held: false only when
    ``operation`` has ``LOCK_NB`` and another process holds a lock that conflicts. Closing the file releases it."""
    with path.open("a") as file:
        try:
            fcntl.flock(file, operation)
            held = True
        except BlockingIOError:
            held = False
        yield held


def ask(question: str, *, secret: bool = False) -> str:
    """The person's typed answer to ``question``; HarnessSyncError without a terminal to ask in."""
    if not sys.stdin.isatty():
        raise HarnessSyncError(f"{question} (no terminal to ask in: pass it as an option or a variable)")
    return (getpass.getpass(f"{question} ") if secret else input(f"{question} ")).strip()


def find_agent(binary: str) -> str | None:
    """The agent's own executable on ``PATH``, skipping this client's agent commands, which would run Reef again."""
    commands = commands_root().resolve()
    entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    return shutil.which(binary, path=os.pathsep.join(entry for entry in entries if Path(entry).resolve() != commands))


def install_agent(binary: str, install: Mapping[str, Any]) -> str:
    """Install ``install``, the agent's pinned vendor install, under ``agents_root()``; the installed executable.

    HarnessSyncError when the install is not an npm package, npm is missing, or npm fails.
    """
    package = f"{install.get('package')}@{install.get('version')}"
    if install.get("kind") != "npm":
        raise HarnessSyncError(f"{binary} is not installed: install {package} yourself, then connect again")
    npm = shutil.which("npm")
    if npm is None:
        raise HarnessSyncError(
            f"{binary} is not installed, and installing it needs npm: install Node.js, or {package}"
        )
    prefix = agents_root() / binary
    prefix.mkdir(parents=True, exist_ok=True)
    arguments = [npm, "install", "--prefix", str(prefix), "--no-fund", "--no-audit", "--loglevel=error", package]
    done = subprocess.run(arguments, capture_output=True, text=True, check=False)
    executable = prefix / str(install.get("binary_path"))
    if done.returncode != 0 or not executable.is_file():
        raise HarnessSyncError(f"npm could not install {package}: {done.stderr.strip()[-400:]}")
    return str(executable)


def installed_agent(binary: str, install: Mapping[str, Any] | None) -> str | None:
    """The executable that an earlier ``install_agent`` left for ``binary``, or ``None``."""
    if install is None:
        return None
    executable = agents_root() / binary / str(install.get("binary_path"))
    return str(executable) if executable.is_file() else None


def shell_profile() -> Path | None:
    """The startup file of the person's shell, by ``$SHELL``: zsh, bash or fish; ``None`` for another shell."""
    shell = Path(os.environ.get("SHELL", "")).name
    home = Path.home()
    if shell == "zsh":
        return Path(os.environ.get("ZDOTDIR") or home) / ".zshrc"
    if shell == "bash":
        # macOS terminals start login shells, which read .bash_profile instead of .bashrc.
        return home / (".bash_profile" if sys.platform == "darwin" else ".bashrc")
    if shell == "fish":
        return home / ".config" / "fish" / "config.fish"
    return None


def path_line(profile: Path) -> str:
    """The line of ``profile`` that puts ``commands_root()`` ahead on ``PATH``."""
    if profile.name == "config.fish":
        return f"fish_add_path --prepend {shlex.quote(str(commands_root()))}"
    return f'export PATH={shlex.quote(str(commands_root()))}:"$PATH"'


def command_scenario(command: Path) -> str | None:
    """The scenario an agent command of this client runs, or ``None`` when ``command`` is not one."""
    if not command.is_file():
        return None
    match = COMMAND_SCENARIO.search(command.read_text(encoding="utf-8", errors="replace"))
    return None if match is None else match.group(1)


def route_through_reef(binary: str, scenario: str, directory: Path) -> tuple[str | None, Path | None]:
    """Make ``binary`` run through Reef's ``scenario``, installed at ``directory``: an agent command of that name in
    ``commands_root()``, and ``commands_root()`` ahead on ``PATH`` in the shell's startup file.

    Returns the scenario the command ran before, if any, and the startup file, ``None`` when the shell is not one
    this client knows.
    """
    command = commands_root() / binary
    previous = command_scenario(command)
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text(
        "#!/bin/sh\n"
        f"# reef-client scenario: {scenario}\n"
        f"# Runs {binary} through Reef. Undo with: reef-client harness disconnect {scenario}\n"
        f"exec {shlex.quote(sys.executable)} -m reef_client harness run {shlex.quote(scenario)}"
        f' --dir {shlex.quote(str(directory.resolve()))} -- "$@"\n',
        encoding="utf-8",
    )
    command.chmod(0o755)
    profile = shell_profile()
    if profile is not None:
        text = profile.read_text(encoding="utf-8") if profile.is_file() else ""
        if PROFILE_BLOCK[0] not in text:
            block = "\n".join([PROFILE_BLOCK[0], path_line(profile), PROFILE_BLOCK[1]])
            profile.parent.mkdir(parents=True, exist_ok=True)
            profile.write_text(f"{text.rstrip()}\n\n{block}\n" if text.strip() else f"{block}\n", encoding="utf-8")
    return previous, profile


def unroute(binary: str, scenario: str) -> bool:
    """Remove the agent command ``binary`` when it runs ``scenario``, and the startup file's block once no command
    is left; whether a command was removed."""
    command = commands_root() / binary
    if command_scenario(command) != scenario:
        return False
    command.unlink()
    if not any(commands_root().iterdir()):
        commands_root().rmdir()
        profile = shell_profile()
        if profile is not None and profile.is_file():
            text = profile.read_text(encoding="utf-8")
            start, end = text.find(PROFILE_BLOCK[0]), text.find(PROFILE_BLOCK[1])
            if 0 <= start < end:
                kept = text[:start].rstrip("\n") + text[end + len(PROFILE_BLOCK[1]) :]
                profile.write_text(kept.rstrip("\n") + "\n" if kept.strip() else "", encoding="utf-8")
    return True


class HarnessInstall:
    """One scenario's harness installed on this machine as a layer, and the Reef it syncs with.

    Values: ``directory`` holds the installed files at their render paths and
    ``INSTALL_FILE``, the record: the Reef URL, the scenario, the token, the
    installed release, the digest of each installed file, the command that
    runs the agent and the agent's commands that record and remove the
    install (``run``), the templates of the files that sync (``synced``),
    once ``connect`` found it, the agent's own executable (``agent``), and
    the release whose files the agent's ``register`` commands last recorded
    (``registered``). The directory also holds ``SESSION_LOCK``, ``SYNC_LOCK`` and
    ``SYNC_LOG``. ``client`` reaches the Reef the record names.
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

    def install_files(self, manifest: Mapping[str, Any]) -> dict[str, str]:
        """``manifest``'s files, by path under the install, with this client's key in place of the placeholder."""
        key = self.record.get("token") or NO_TOKEN_KEY
        return {path: text.replace(TOKEN_PLACEHOLDER, key) for path, text in manifest["files"].items()}

    def differing(self, manifest: Mapping[str, Any]) -> list[str]:
        """The paths that installing ``manifest`` writes or removes, sorted: each of its files that the install lacks
        or holds with other text, and each file of the previous install that it lacks and that is still there."""
        files = self.install_files(manifest)
        paths = [path for path in self.record["files"] if path not in files and (self.directory / path).exists()]
        for path, text in files.items():
            target = self.directory / path
            if not target.is_file() or target.read_text(encoding="utf-8", errors="replace") != text:
                paths.append(path)
        return sorted(paths)

    def install(self, manifest: Mapping[str, Any]) -> list[str]:
        """Write the ``differing`` paths of ``manifest`` and record it; those paths.

        Installing the release that the files already are writes no file and only moves the record.
        """
        files = self.install_files(manifest)
        paths = self.differing(manifest)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.directory.chmod(0o700)
        for path in paths:
            target = self.directory / path
            if path in files:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(files[path], encoding="utf-8")
            else:
                target.unlink(missing_ok=True)
        self.record.update(
            release_id=manifest["release_id"],
            files={path: digest(text) for path, text in files.items()},
            run=manifest["run"],
            synced=manifest["synced"],
        )
        self.save()
        return paths

    def save(self) -> None:
        """Write the record, readable by this user alone: it holds the token."""
        record = self.directory / INSTALL_FILE
        record.write_text(json.dumps(self.record, indent=2) + "\n", encoding="utf-8")
        record.chmod(0o600)

    def agent(self) -> str:
        """The agent's own executable, never this client's command of the same name, which would run Reef again;
        HarnessSyncError when the machine has none."""
        agent = self.record.get("agent")
        if agent is None or not Path(agent).is_file():
            agent = find_agent(self.record["run"]["binary"])
        if agent is None:
            raise HarnessSyncError(f"{self.record['run']['binary']} is not installed: connect again to install it")
        return agent

    def register(self) -> None:
        """Run the agent's ``register`` commands on the installed release, once per release, so the agent loads
        what it reads from no path a flag names; HarnessSyncError naming a command that failed."""
        commands = self.record["run"].get("register") or []
        if not commands or self.record.get("registered") == self.record["release_id"]:
            return
        agent = self.agent()
        for command in commands:
            done = subprocess.run([agent, *command], capture_output=True, text=True, check=False)
            if done.returncode != 0:
                raise HarnessSyncError(
                    f"{self.record['run']['binary']} {shlex.join(command)} failed: {done.stderr.strip()[-400:]}"
                )
        self.record["registered"] = self.record["release_id"]
        self.save()

    def unregister(self) -> list[str]:
        """Run the agent's ``unregister`` commands, each even when one before it failed; a line per command that
        did not finish, with the command for the person to run."""
        if self.record.get("registered") is None:
            return []
        commands = self.record["run"].get("unregister") or []
        binary = self.record["run"]["binary"]
        try:
            agent = self.agent()
        except HarnessSyncError:
            return [
                f"{binary} is not installed; once it is, run {binary} {shlex.join(command)}" for command in commands
            ]
        return [
            f"{binary} {shlex.join(command)} did not finish; run it yourself if it is needed"
            for command in commands
            if subprocess.run([agent, *command], capture_output=True, check=False).returncode != 0
        ]

    def connect(self) -> dict[str, Any]:
        """Create the scenario when it is new and install its head; the head's manifest."""
        self.client.post("/reef/scenarios", self.record["scenario"], {"name": self.record["scenario"]})
        manifest = self.head()
        self.install(manifest)
        return manifest

    def sync(self, *, discard: bool = False) -> str:
        """Send the changed files, then install the head; a report.

        Reef merges the changes onto its head, so the head after the import holds them. HarnessSyncError when Reef
        refuses them; the changed files then stay as they are. A head that differs from the files waits for the next
        sync while an agent session runs on the install, so no file changes under the agent. ``discard`` sends
        nothing and installs the head over the changes; HarnessSyncError while a session runs.
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
        with locked(self.directory / SESSION_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB) as alone:
            if not alone and self.differing(manifest):
                if discard:
                    raise HarnessSyncError("an agent session runs on this install: discard after it exits")
                lines.append(f"release {manifest['release_id']} installs at the next run: an agent session runs")
                return "\n".join(lines)
            written = self.install(manifest)
        # A head that is the files just sent writes none: the sent line says it all.
        if written or not lines:
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
        unfinished = self.unregister()
        shutil.rmtree(self.directory)
        scenario = self.record["scenario"]
        lines = [f"removed the install of {scenario}; your own configuration is as it was", *unfinished]
        binary = (self.record.get("run") or {}).get("binary")
        if binary and unroute(binary, scenario):
            lines.append(f"{binary} runs on its own again in a new terminal")
        return "\n".join(lines)

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
        """Sync, run the agent on the layer with ``agent_args`` added, and when it exits send what the session changed
        from a process of its own; the agent's exit status.

        The sync before the agent waits ``RUN_SYNC_TIMEOUT_S`` for each answer, and is skipped while the send after
        an earlier session still runs; a sync that cannot finish keeps the files, and the agent runs on them. The send
        after the agent writes its report to ``SYNC_LOG``, which the next run prints, so the terminal is back at once.
        While the agent runs, this process holds ``SESSION_LOCK`` shared, so no sync writes a file under the agent,
        and ignores Ctrl-C, which the agent handles.
        """
        log = self.directory / SYNC_LOG
        with locked(self.directory / SYNC_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB) as free:
            if free:
                report = log.read_text(encoding="utf-8").strip() if log.is_file() else ""
                log.unlink(missing_ok=True)
                if report:
                    print(f"after the last session: {report}", file=sys.stderr)
                self.try_sync()
            else:
                print(
                    f"the last session's changes are still on their way to Reef: this session starts on release "
                    f"{self.record['release_id']}",
                    file=sys.stderr,
                )
        command = self.record["run"]
        agent = self.agent()
        # The agent's register commands change what its other sessions read too, so they wait for none to run.
        with locked(self.directory / SESSION_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB) as alone:
            if alone:
                try:
                    self.register()
                except HarnessSyncError as exc:
                    print(f"{exc}; the agent starts without what those commands load", file=sys.stderr)
        key = self.record.get("token") or NO_TOKEN_KEY
        layer_env = {name: value.replace(TOKEN_PLACEHOLDER, key) for name, value in command["env"].items()}
        interrupt = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            argv = [agent, *command["args"], *agent_args]
            with locked(self.directory / SESSION_LOCK, fcntl.LOCK_SH):
                status = subprocess.run(argv, env={**os.environ, **layer_env}, check=False).returncode
        finally:
            signal.signal(signal.SIGINT, interrupt)
        # A send that ended during the session moved the record on disk.
        changed = HarnessInstall.open(self.directory).changed()
        if changed:
            with log.open("w", encoding="utf-8") as report_file:
                subprocess.Popen(
                    [sys.executable, "-m", "reef_client", "harness", "sync", "--dir", str(self.directory.resolve())],
                    stdin=subprocess.DEVNULL,
                    stdout=report_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            print(f"sending {', '.join(changed)} to Reef in the background", file=sys.stderr)
        return status


def choose_harness(rows: Sequence[Mapping[str, Any]]) -> str:
    """The scenario of the harness the person picks from Reef's ``rows``: the only one without asking, else by its
    number, its agent or its name; a name of their own when Reef has none yet."""
    if len(rows) == 1:
        return str(rows[0]["scenario"])
    if not rows:
        return ask("Reef has no harness yet. Name one:") or "default"
    for number, row in enumerate(rows, start=1):
        print(f"  {number}. {row.get('adapter') or 'harness'}  {row.get('scenario')}")
    answer = ask("Harness:")
    if answer.isdigit() and 1 <= int(answer) <= len(rows):
        return str(rows[int(answer) - 1]["scenario"])
    matches = [row for row in rows if answer in (row.get("scenario"), row.get("adapter"))]
    if len(matches) != 1:
        raise HarnessSyncError(f"pick a harness by its number: {answer!r} names {len(matches)} of them")
    return str(matches[0]["scenario"])


def connect(scenario: str | None, reef_url: str | None, directory: Path | None) -> str:
    """Set this machine up for ``scenario``'s harness: install its head as a layer, install the agent when the
    machine lacks it, and make the agent's own command run through Reef; what the person must still know.

    It asks only for the harness, when Reef serves more than one, and the token, when Reef refuses the call without
    one; the URL is asked for only when neither ``reef_url`` nor ``REEF_URL`` gives it.
    """
    reef_url = reef_url or os.environ.get("REEF_URL") or ask("Reef URL:")
    token = os.environ.get("REEF_TOKEN") or None
    try:
        rows = ReefClient(reef_url, token=token).get("/reef/scenarios")["scenarios"]
    except ReefClientError as exc:
        if exc.status != 401:
            raise
        token = ask("Reef token:", secret=True) or None
        rows = ReefClient(reef_url, token=token).get("/reef/scenarios")["scenarios"]
    scenario = scenario or choose_harness(rows)
    install = HarnessInstall.create(directory or installs_root() / scenario, reef_url, scenario, token)
    manifest = install.connect()
    binary, pinned = manifest["run"]["binary"], manifest["run"].get("install")
    agent = find_agent(binary) or installed_agent(binary, pinned)
    if agent is None and pinned is None:
        raise HarnessSyncError(f"{binary} is not installed: install it, then connect again")
    if agent is None:
        # npm takes a while, so the person learns why the command waits.
        print(f"installing {binary} {pinned['version']}...", file=sys.stderr)
        agent = install_agent(binary, pinned)
    install.record["agent"] = agent
    install.save()
    install.register()
    previous, profile = route_through_reef(binary, scenario, install.directory)
    lines = required(manifest)
    if previous not in (None, scenario):
        lines.append(f"{binary} now runs {scenario} instead of {previous}")
    if profile is None:
        lines.append(f"put {commands_root()} first on PATH: reef-client knows no startup file of your shell")
    lines.append(f"Done. Open a new terminal and run {binary}.")
    return "\n".join(lines)


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
    connect_command = commands.add_parser("connect", help="set this machine up: the layer, the agent and its command")
    connect_command.add_argument("scenario", nargs="?", help="the harness's scenario; asked for when Reef has several")
    connect_command.add_argument("--reef", help="the Reef URL, such as http://127.0.0.1:8900; else REEF_URL")
    for name, help in (
        ("run", "sync, run the agent, and send its changes when it exits; arguments after -- go to the agent"),
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
        if arguments.command == "connect":
            print(connect(arguments.scenario, arguments.reef, arguments.dir))
            return
        directory = arguments.dir or install_directory(arguments.scenario)
        timeout_s = RUN_SYNC_TIMEOUT_S if arguments.command == "run" else 60.0
        install = HarnessInstall.open(directory, timeout_s=timeout_s)
        if arguments.command == "run":
            raise SystemExit(install.run(agent_args))
        if arguments.command == "sync":
            with locked(directory / SYNC_LOCK, fcntl.LOCK_EX):
                print(install.sync(discard=arguments.discard))
        elif arguments.command == "status":
            print(install.status())
        else:
            print(install.disconnect(discard=arguments.discard))
    except HarnessSyncError as exc:
        raise SystemExit(f"reef-client harness: {exc}") from exc
    except (ReefClientError, OSError) as exc:
        # The send after a run reports to a file the next run prints: a line, not a traceback.
        raise SystemExit(f"reef-client harness: {exc}") from exc


__all__ = [
    "INSTALL_FILE",
    "TOKEN_PLACEHOLDER",
    "HarnessInstall",
    "HarnessSyncError",
    "commands_root",
    "connect",
    "find_agent",
    "install_agent",
    "installs_root",
    "main",
    "route_through_reef",
    "unroute",
]
