"""The workbench: shell commands and scripts agents run in their own workspaces, in a sandbox.

Agents use it to check things: build or inspect a structure, test a script on a sample file,
grep an output, run pytest. It runs on any node, login nodes included, without the
researcher's approval for each command, because the sandbox (``sandbox.py``) keeps a command
to its workspace: no network, no scheduler, no DFT code, nothing written elsewhere, and
nothing hidden readable. DFT runs only as an examined calculation, which the researcher
approves.

What a command may use depends on where CEDAR runs:

- on a login node, shared with other users: a few cores per command, a few commands at once
  across all agents, and a shorter time limit;
- on an interactive node, inside the researcher's Slurm allocation: the allocation's cores
  not taken by other commands, and never past the allocation's end.

Every command is recorded, whatever its outcome: what ran, where, with which resources, how
it ended, what it printed (stored by hash), and the content hash of every workspace file it
created or changed. That record is context, never evidence.
"""

import hashlib
import os
import shlex
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from cedar.contracts.base import Actor, ActorKind, RelativePath, new_id
from cedar.contracts.events import EventDraft, WorkbenchRunRecorded
from cedar.contracts.workbench import WorkbenchOutcome, WorkbenchRun
from cedar.engine.sandbox import (
    SandboxView,
    bwrap_command,
    bwrap_path,
    guarded_python,
    run_process,
    sandbox_environment,
)
from cedar.hpc.node import CorePool, ExecutionHost, HostWatch
from cedar.hpc.profile import LocalExecution
from cedar.paths import workspace_path
from cedar.state.ledger import ResearchLedger
from cedar.store.artifacts import ArtifactStore

WORKBENCH = Actor(kind=ActorKind.WORKBENCH, id="workbench")
OUTPUT_DIRECTORY = "workbench-output"
"""In the workspace: what each command printed, one file per run."""
_NOT_TRACKED = (OUTPUT_DIRECTORY, ".home", ".tmp")
SHOWN_CHARS = 12_000
"""How much of what a command printed goes back to the agent; the rest is in the saved file."""


@dataclass(frozen=True)
class WorkbenchReport:
    run_id: str | None
    """None when the command did not run; ``problems`` says why."""
    run: WorkbenchRun | None
    problems: tuple[str, ...] = ()
    shown_output: str = ""
    output_path: Path | None = None


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    files = {}
    for directory, subdirectories, names in os.walk(root):
        here = Path(directory)
        if here == root:
            subdirectories[:] = [d for d in subdirectories if d not in _NOT_TRACKED]
        for name in names:
            path = here / name
            try:
                info = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                files[path.relative_to(root).as_posix()] = (info.st_size, info.st_mtime_ns)
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_RELATIVE = TypeAdapter(RelativePath)


def _recordable(name: str) -> bool:
    try:
        _RELATIVE.validate_python(name)
    except ValidationError:
        return False
    return True


def _shown(path: Path) -> str:
    """The start and end of what a command printed, without reading all of a large output."""
    size = path.stat().st_size
    half = SHOWN_CHARS // 2
    with path.open("rb") as stream:
        if size <= SHOWN_CHARS:
            return stream.read().decode(errors="replace")
        head = stream.read(half).decode(errors="replace")
        stream.seek(size - half)
        tail = stream.read().decode(errors="replace")
    return f"{head}\n…[{size - SHOWN_CHARS} bytes omitted; the whole output is in {path}]…\n{tail}"


class Workbench:
    """One per dispatcher: the machine's cores and where CEDAR runs, shared by every agent's commands."""

    def __init__(self, ledger: ResearchLedger, artifacts: ArtifactStore, *,
                 host: Callable[[], ExecutionHost] | None = None, pool: CorePool | None = None,
                 bwrap: Callable[[], str | None] = bwrap_path, actor: Actor = WORKBENCH) -> None:
        self._ledger = ledger
        self._artifacts = artifacts
        self.host = host or HostWatch()
        self.pool = pool or CorePool()
        self._bwrap = bwrap
        self._actor = actor

    @property
    def sandboxed(self) -> bool:
        """Whether commands run in bubblewrap; without it, only guarded Python scripts run."""
        return self._bwrap() is not None

    def limits(self, local: LocalExecution, host: ExecutionHost | None = None) -> tuple[int, int, int | None]:
        """(most cores per command, most minutes per command, most commands at once) where CEDAR runs now."""
        host = host or self.host()
        if host.interactive:
            minutes = local.node_max_command_minutes
            left = host.seconds_left()
            if left is not None:
                minutes = min(minutes, int((left - 60) // 60))
            return min(host.cores, self.pool.size), minutes, None
        return local.login_cores_per_command, local.login_max_command_minutes, local.login_max_commands

    def run(self, investigation_id: str, hypothesis_id: str, agent_id: str, view: SandboxView, *, command: str,
            purpose: str, local: LocalExecution, directory: str = ".", minutes: int | None = None,
            cores: int | None = None, should_stop: Callable[[], bool] | None = None,
            code_paths: Sequence[Path] = ()) -> WorkbenchReport:
        """Run ``command`` with bash in the agent's workspace; ``code_paths`` are extra Python environments' bin
        directories put on the search path."""
        if not self._ledger.load(investigation_id).workbench_enabled:
            return WorkbenchReport(None, None, ("the researcher has turned the workbench off for this investigation "
                                                "(/workbench on turns it back on)",))
        workspace = view.workspace.resolve()
        cwd = workspace_path(workspace, directory)
        if not cwd.is_dir():
            return WorkbenchReport(None, None, (f"{directory} is not a directory in your workspace",))
        host = self.host()
        most_cores, most_minutes, most_running = self.limits(local, host)
        if most_minutes < 1:
            return WorkbenchReport(None, None, (f"the Slurm allocation {host.allocation} ends within a minute; "
                                                "nothing more can run here",))
        problems = []
        wanted_minutes = minutes or local.default_command_minutes
        if wanted_minutes > most_minutes:
            problems.append(f"the time limit was lowered from {wanted_minutes} to {most_minutes} minutes, the most "
                            f"a command may take on this {'node' if host.interactive else 'login node'}")
        time_limit = min(wanted_minutes, most_minutes) * 60
        wanted_cores = cores or 1
        if wanted_cores > most_cores:
            problems.append(f"{most_cores} core(s) instead of {wanted_cores}: the most a command gets on this "
                            + ("interactive node" if host.interactive else "login node, shared with other users"))
        count = min(wanted_cores, most_cores)

        bwrap = self._bwrap()
        env = sandbox_environment(count, paths=code_paths, workspace=workspace)
        if bwrap is not None:
            sandbox, argv = "bwrap", bwrap_command(bwrap, command, view, cwd=cwd)
        else:
            words = shlex.split(command)
            if len(words) < 2 or Path(words[0]).name not in ("python", "python3") or not words[1].endswith(".py"):
                return WorkbenchReport(None, None, ("this machine has no bubblewrap, so the workbench runs only "
                                                    "Python scripts in your workspace: 'python <script>.py "
                                                    "[arguments]', and they cannot start programs",))
            script = workspace_path(workspace, str(cwd / words[1]))
            if not script.is_file():
                return WorkbenchReport(None, None, (f"{words[1]} is not a file in your workspace",))
            sandbox = "python-guard"
            argv, env = guarded_python(script, words[2:], view, cwd=cwd, env=env)

        cpus = self.pool.acquire(count, most_running=most_running)
        if cpus is None:
            return WorkbenchReport(None, None, ("the machine's cores are taken by other commands; try again in a "
                                                "few minutes",))
        run_id = new_id("wb")
        output = workspace / OUTPUT_DIRECTORY / f"{run_id}.txt"
        output.parent.mkdir(exist_ok=True)
        try:
            before = _snapshot(workspace)
            result = run_process(argv, cwd=cwd, env=env, output=output, time_limit=time_limit, cpus=cpus,
                                 should_stop=should_stop)
        finally:
            self.pool.release(cpus)
        after = _snapshot(workspace)
        changed = {}
        for name, signature in after.items():
            if before.get(name) == signature:
                continue
            if _recordable(name):
                changed[name] = _sha256(workspace / name)
            else:
                problems.append(f"{name!r} changed but its name cannot be recorded")
        deleted = tuple(sorted(n for n in before if n not in after and _recordable(n)))
        if result.timed_out:
            outcome = WorkbenchOutcome.TIMED_OUT
            problems.append(f"the command did not finish within {time_limit // 60} minutes and was stopped")
        elif result.stopped:
            outcome = WorkbenchOutcome.STOPPED
            problems.append("the command was stopped: your work was halted")
        else:
            outcome = WorkbenchOutcome.COMPLETED if result.exit_code == 0 else WorkbenchOutcome.FAILED
        stored = self._artifacts.put_file(output, media_type="text/plain")
        relative = cwd.relative_to(workspace).as_posix()
        run = WorkbenchRun(
            command=command, purpose=purpose, directory=relative, sandbox=sandbox, host=host.hostname,
            allocation=host.allocation, cores=count, time_limit_seconds=time_limit, outcome=outcome,
            exit_code=result.exit_code, duration_seconds=round(result.seconds, 3), output=stored, changed=changed,
            deleted=deleted, problems=tuple(problems))
        self._ledger.record(investigation_id, [EventDraft(actor=self._actor, payload=WorkbenchRunRecorded(
            run_id=run_id, hypothesis_id=hypothesis_id, agent_id=agent_id, run=run))])
        return WorkbenchReport(run_id, run, tuple(problems), _shown(output), output)


@dataclass
class AgentBench:
    """One agent's use of the workbench: its workspace and what its commands may see."""

    workbench: Workbench
    investigation_id: str
    hypothesis_id: str
    agent_id: str
    view: Callable[[], SandboxView]
    """Asked anew for each command, since the agent's own run directories grow."""
    local: LocalExecution
    code_paths: Sequence[Path] = ()
    should_stop: Callable[[], bool] | None = None

    def run(self, command: str, purpose: str, *, directory: str = ".", minutes: int | None = None,
            cores: int | None = None) -> WorkbenchReport:
        return self.workbench.run(self.investigation_id, self.hypothesis_id, self.agent_id, self.view(),
                                  command=command, purpose=purpose, local=self.local, directory=directory,
                                  minutes=minutes, cores=cores, should_stop=self.should_stop,
                                  code_paths=self.code_paths)

    def where(self) -> str:
        host = self.workbench.host()
        most_cores, most_minutes, most_running = self.workbench.limits(self.local, host)
        shared = f"; at most {most_running} commands at once across all agents" if most_running else ""
        sandbox = "any program" if self.workbench.sandboxed else "Python scripts only (no bubblewrap)"
        return (f"CEDAR runs on the {host.describe()}: commands get up to {most_cores} core(s) and "
                f"{most_minutes} minutes{shared}; {sandbox}.")
