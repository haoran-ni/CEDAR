"""Deleting what CEDAR stored: ``cedar reset`` and ``cedar purge``.

``reset`` deletes every investigation: the research data directory's contents (the event log, stored files, the
agents' workspaces and conversations, the notebooks and the dispatcher's files) and CEDAR's records of its local
runs. It keeps the setup. ``purge`` deletes the setup too: the compute profiles and their approvals, ``models.toml``,
the setup agent's history and the stored API keys. Run directories, where the calculations ran and left their
outputs, go only when asked. Only what CEDAR writes is deleted, each by its name: a directory that also holds
anything else keeps it, and stays.

Neither deletes while something could still write there, or run on with nothing recording it: a dispatcher working
on the data directory, or a calculation CEDAR handed to a scheduler that has not ended. Whether one has ended is
asked of its scheduler, not read from the record, which stops following the jobs of a closed investigation.
"""

import os
import shutil
import sqlite3
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from cedar.contracts.status import SCHEDULER_TERMINAL_STATES, CalculationState
from cedar.dispatch import ACTIVITY, HEARTBEAT, HOST, LOCK, LOG, PID, dispatcher_host, dispatcher_running
from cedar.engine.jobs import job_name_for
from cedar.hpc.local import RECORDS
from cedar.hpc.profile import ProfileStore
from cedar.hpc.slurm import SchedulerBackend, SchedulerError, default_lookback
from cedar.secrets import PROVIDER_ENVIRONMENT_VARIABLES
from cedar.workspace import Workspace

DATA = ("events.sqlite", "events.sqlite-wal", "events.sqlite-shm", "events.sqlite-journal", "artifacts", "analyses",
        "workspaces", "transcripts", "notebooks", ACTIVITY, LOCK, HEARTBEAT, PID, HOST, LOG)
"""What CEDAR writes in the research data directory."""
SETUP = ("credentials.toml", "models.toml", "profiles", "setup_runs.jsonl", "transcripts")
"""What CEDAR writes in the configuration directory."""
_AT_SCHEDULER = frozenset({CalculationState.SUBMISSION_PENDING, CalculationState.SUBMITTED, CalculationState.QUEUED,
                           CalculationState.RUNNING})
_ASKING_FAILED = (SchedulerError, ValueError, OSError, subprocess.TimeoutExpired)
CANCEL_WAIT_SECONDS = 120
"""How long cancelled jobs have to end before nothing is deleted."""


@dataclass(frozen=True)
class _Calc:
    investigation_id: str
    calculation_id: str
    execution: str
    state: CalculationState
    job_id: str | None
    workdir: str | None
    profile: str | None
    runs_root: Path | None
    """Where the investigation's profile runs calculations; None if that profile cannot be read."""


@dataclass(frozen=True)
class Job:
    """A calculation CEDAR handed to a scheduler that has not ended, as the scheduler sees it now."""

    investigation_id: str
    calculation_id: str
    execution: str
    job_id: str
    state: str
    runs_root: Path | None

    def __str__(self) -> str:
        kind = "local run" if self.execution == "local" else "Slurm job"
        return f"{self.investigation_id} {self.calculation_id}: {kind} {self.job_id}, {self.state}"


class Cleanup:
    """What ``reset`` deletes, or with ``setup`` what ``purge`` deletes, read from the record before anything is."""

    def __init__(self, data: Path, config: Path, *, setup: bool, runs: bool, slurm: SchedulerBackend,
                 local: Callable[[Path], SchedulerBackend], environ: Mapping[str, str] = os.environ) -> None:
        """``local`` gives the local backend of a profile's runs root."""
        self.data, self.config, self.setup, self.runs = Path(data), Path(config), setup, runs
        self._slurm, self._local, self._environ = slurm, local, environ
        self._workspace = Workspace.open(self.data) if (self.data / "events.sqlite").exists() else None
        self.investigations: list[str] = []
        self.unreadable: list[str] = []
        """Why an investigation's record cannot be read; its jobs and run directories are then unknown."""
        self._calcs: list[_Calc] = []
        self._read()

    def _read(self) -> None:
        if self._workspace is None:
            return
        profiles = ProfileStore(self.config / "profiles")
        try:
            self.investigations = self._workspace.store.investigation_ids()
        except sqlite3.DatabaseError as exc:
            self.unreadable.append(f"the event log {self.data / 'events.sqlite'} cannot be read ({exc})")
            return
        for investigation_id in self.investigations:
            try:
                state = self._workspace.ledger.load(investigation_id)
            except Exception as exc:  # a log this version of CEDAR cannot replay is still deleted
                self.unreadable.append(f"the record of {investigation_id} cannot be read ({exc})")
                continue
            name = state.spec.constraints.compute_profile
            try:
                runs_root = Path(profiles.load(name).runs_root) if name else None
            except (OSError, ValueError, ValidationError):
                runs_root = None
            self._calcs += [_Calc(investigation_id, c.calculation_id, c.task.execution, c.state, c.scheduler_job_id,
                                  None if c.registered else c.workdir, name, runs_root)
                            for c in state.calculations.values()]

    # -- what goes, and what stays ---------------------------------------------------------------

    def _data_entries(self) -> list[Path]:
        if not self.data.is_dir():
            return []
        partial = [p for name in (HEARTBEAT, PID, HOST) for p in self.data.glob(f".{name}.*.partial")]
        return [self.data / name for name in DATA if _exists(self.data / name)] + partial

    def _setup_entries(self) -> list[Path]:
        return [self.config / name for name in SETUP if _exists(self.config / name)] if self.config.is_dir() else []

    def _local_records(self) -> dict[Path, list[Path]]:
        found: dict[Path, list[Path]] = {}
        for calc in self._calcs:
            if calc.execution == "local" and calc.job_id and calc.runs_root is not None:
                records = calc.runs_root / RECORDS
                found.setdefault(records, []).extend(records.glob(f"{calc.job_id}.*"))
        return {records: paths for records, paths in found.items() if paths}

    def _run_directories(self) -> dict[Path, list[Path]]:
        found: dict[Path, list[Path]] = {}
        for calc in self._calcs:
            if calc.workdir and calc.runs_root is not None:
                path = calc.runs_root / calc.workdir
                if _exists(path) and path.resolve().is_relative_to(calc.runs_root.resolve()):
                    found.setdefault(calc.runs_root, []).append(path)
        return found

    def targets(self) -> list[Path]:
        """Everything that would be deleted, in the order it is."""
        paths = [p for group in self._local_records().values() for p in group]
        if self.runs:
            paths += [p for group in self._run_directories().values() for p in group]
        paths += self._data_entries()
        if self.setup:
            paths += self._setup_entries()
        return paths

    def describe(self) -> list[str]:
        """What would be deleted, one line per place."""
        lines = []
        entries = self._data_entries()
        if entries:
            count = len(self.investigations)
            named = f"{count} investigation{_s(count)} ({', '.join(self.investigations)})" if count else \
                "no investigations"
            lines.append(f"{self.data} ({_size(entries)}): {named}, with the event log, stored files, the agents' "
                         "workspaces and conversations, the notebooks and the dispatcher's files")
        for records, paths in self._local_records().items():
            count = len({p.name.split(".")[0] for p in paths})
            lines.append(f"{records}: CEDAR's records of {count} local run{_s(count)}")
        if self.runs:
            for root, paths in self._run_directories().items():
                lines.append(f"{root} ({_size(paths)}): {len(paths)} run director{_ies(len(paths))}, with the "
                             "calculations' inputs and outputs")
        entries = self._setup_entries()
        if self.setup and entries:
            profiles = ProfileStore(self.config / "profiles").names()
            names = {p.name for p in entries}
            parts = [f"the compute profile{_s(len(profiles))} {', '.join(profiles)} and approvals" if profiles else "",
                     "models.toml" if "models.toml" in names else "",
                     "the setup agent's history" if names & {"setup_runs.jsonl", "transcripts"} else "",
                     "the stored API keys" if "credentials.toml" in names else ""]
            parts = [p for p in parts if p] or ["CEDAR's configuration"]
            lines.append(f"{self.config} ({_size(entries)}): " + (", ".join(parts[:-1]) + " and " if parts[1:] else "")
                         + parts[-1])
        return lines

    def kept(self) -> list[str]:
        """What stays that the researcher might expect to go, one line each."""
        lines = []
        if not self.setup and self._setup_entries():
            lines.append(f"your setup in {self.config}: compute profiles, models.toml and API keys (`cedar purge` "
                         "deletes it too)")
        if not self.runs:
            for root, paths in self._run_directories().items():
                lines.append(f"{len(paths)} run director{_ies(len(paths))} under {root} ({_size(paths)}); --runs "
                             "deletes them too")
        elif self.unreadable:
            lines.append("the run directories of investigations whose record cannot be read")
        lost = sorted({c.profile for c in self._calcs if c.runs_root is None and c.profile and c.workdir})
        if lost:
            lines.append(f"the run directories and local-run records under compute profile{_s(len(lost))} "
                         f"{', '.join(lost)}, which cannot be read to say where they are")
        return lines

    def environment_keys(self) -> list[str]:
        """The API-key variables set in the environment, which no deletion reaches."""
        return [v for v in PROVIDER_ENVIRONMENT_VARIABLES.values() if self._environ.get(v)]

    # -- what keeps it from deleting ---------------------------------------------------------------

    def dispatcher(self) -> str | None:
        """Why no deletion can start while a dispatcher works here, or None."""
        if self._workspace is None or not dispatcher_running(self._workspace):
            return None
        host = dispatcher_host(self._workspace)
        return (f"a dispatcher is working on {self.data}" + (f" (on {host})" if host else "") + ". Stop it with "
                "`cedar dispatch --stop` on that machine, or wait until it stops by itself once it has nothing to do.")

    def live(self) -> tuple[list[Job], list[str]]:
        """The calculations CEDAR handed to a scheduler that it has not ended, and why it could not tell for
        others."""
        jobs = []
        problems = [f"{reason}, so neither can whether its calculations still run" for reason in self.unreadable]
        for calc in self._calcs:
            if calc.state not in _AT_SCHEDULER:
                continue
            label = f"{calc.investigation_id} {calc.calculation_id}"
            backend = self._backend(calc.execution, calc.runs_root)
            if backend is None:
                problems.append(f"{label}: a local run, and its compute profile {calc.profile}, which says where "
                                "such runs keep their records, cannot be read")
                continue
            try:
                if calc.job_id is None:  # submitted, or about to be, with its id not yet recorded
                    seen = backend.find_by_name(job_name_for(calc.calculation_id), default_lookback())
                else:
                    seen = list(backend.query([calc.job_id]).values())
            except _ASKING_FAILED as exc:
                problems.append(f"{label}: cannot ask its scheduler ({exc})")
                continue
            jobs += [Job(calc.investigation_id, calc.calculation_id, calc.execution, o.job_id,
                         o.state.value + (f" ({o.raw_state})" if o.raw_state != o.state.value else ""),
                         calc.runs_root)
                     for o in seen if o.state not in SCHEDULER_TERMINAL_STATES]
        return jobs, problems

    def cancel(self, jobs: list[Job]) -> list[str]:
        """Ask each job's scheduler to end it; what could not be asked."""
        problems = []
        for job in jobs:
            try:
                self._backend(job.execution, job.runs_root).cancel(job.job_id)
            except _ASKING_FAILED as exc:
                problems.append(f"{job}: {exc}")
        return problems

    def wait(self, timeout: float = CANCEL_WAIT_SECONDS, sleep: Callable[[float], None] = time.sleep,
             ) -> tuple[list[Job], list[str]]:
        """Wait up to ``timeout`` seconds for cancelled jobs to end; what has not, as ``live`` says."""
        deadline = time.monotonic() + timeout
        while True:
            jobs, problems = self.live()
            if not (jobs or problems) or time.monotonic() >= deadline:
                return jobs, problems
            sleep(2)

    def _backend(self, execution: str, runs_root: Path | None) -> SchedulerBackend | None:
        if execution != "local":
            return self._slurm
        return self._local(runs_root) if runs_root is not None else None

    # -- deleting ---------------------------------------------------------------------------------

    def delete(self) -> list[str]:
        """Delete every target, then the directories CEDAR made that it leaves empty; what could not be deleted."""
        emptied = [*self._local_records()]
        if self.runs:
            emptied += [p.parent for group in self._run_directories().values() for p in group]
        problems = []
        for path in self.targets():
            try:
                _remove(path)
            except OSError as exc:
                problems.append(f"{path}: {exc.strerror or exc}")
        for directory in [*dict.fromkeys(emptied), self.data, *([self.config] if self.setup else [])]:
            try:
                directory.rmdir()
            except OSError:
                pass  # gone already, or it holds something else
        return problems

    def left_over(self) -> list[str]:
        """The directories CEDAR keeps because they hold files it did not write."""
        places = [self.data, *([self.config] if self.setup else [])]
        return [f"{place}, which also holds {', '.join(sorted(p.name for p in place.iterdir()))}"
                for place in places if place.is_dir()]


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove(path: Path) -> None:
    """Delete a file, a symbolic link (not what it points to) or a whole directory, even one with read-only
    directories inside, as an agent's workspace or a run directory may have."""
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
        return

    def writable_and_again(function, name, error):
        if not isinstance(error, PermissionError):
            raise error
        os.chmod(os.path.dirname(name), 0o700)
        function(name)

    shutil.rmtree(path, onexc=writable_and_again)


def _size(paths: list[Path]) -> str:
    total = 0
    for path in paths:
        files = [str(path)] if path.is_symlink() or not path.is_dir() else \
            [os.path.join(directory, name) for directory, _, names in os.walk(path) for name in names]
        for name in files:
            try:
                total += os.lstat(name).st_size
            except OSError:
                pass  # gone while we looked
    for unit in ("bytes", "kB", "MB"):
        if total < 1000:
            return f"{total} {unit}" if unit == "bytes" else f"{total:.1f} {unit}"
        total /= 1000
    return f"{total:.1f} GB"


def _s(count: int) -> str:
    return "" if count == 1 else "s"


def _ies(count: int) -> str:
    return "y" if count == 1 else "ies"
