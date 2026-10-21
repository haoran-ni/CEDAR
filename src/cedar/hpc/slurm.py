"""Talking to Slurm. Commands are argument lists, never shell strings.

Job state comes from ``squeue`` for the user's live jobs and from ``sacct`` for the rest.
A job neither command reports is UNKNOWN, never assumed finished.
"""

import getpass
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from cedar.contracts.status import SchedulerState
from cedar.secrets import scrubbed_environment

_STATE_MAP = {
    "PENDING": SchedulerState.PENDING,
    "CONFIGURING": SchedulerState.PENDING,
    "REQUEUED": SchedulerState.PENDING,
    "REQUEUE_HOLD": SchedulerState.PENDING,
    "REQUEUE_FED": SchedulerState.PENDING,
    "RUNNING": SchedulerState.RUNNING,
    "COMPLETING": SchedulerState.RUNNING,
    "COMPLETED": SchedulerState.COMPLETED,
    "FAILED": SchedulerState.FAILED,
    "TIMEOUT": SchedulerState.TIMEOUT,
    "CANCELLED": SchedulerState.CANCELLED,
    "OUT_OF_MEMORY": SchedulerState.OUT_OF_MEMORY,
    "NODE_FAIL": SchedulerState.NODE_FAIL,
    "PREEMPTED": SchedulerState.PREEMPTED,
    "BOOT_FAIL": SchedulerState.BOOT_FAIL,
    "DEADLINE": SchedulerState.DEADLINE,
}


def map_state(raw: str) -> SchedulerState:
    """Slurm state text (e.g. ``CANCELLED by 1234``) to a scheduler state; UNKNOWN if unsure."""
    words = raw.split()
    return _STATE_MAP.get(words[0].rstrip("+"), SchedulerState.UNKNOWN) if words else SchedulerState.UNKNOWN


@dataclass(frozen=True)
class SchedulerObservation:
    job_id: str
    state: SchedulerState
    raw_state: str


class SchedulerBackend(Protocol):
    def submit(self, script: Path, workdir: Path, job_name: str) -> str: ...
    def query(self, job_ids: Sequence[str]) -> dict[str, SchedulerObservation]: ...
    def find_by_name(self, job_name: str, since: datetime) -> list[SchedulerObservation]: ...
    def cancel(self, job_id: str) -> None: ...
    def log_file_names(self, job_id: str) -> list[str]:
        """Names of the scheduler's own log files for a job, relative to its run directory."""
        ...


class SchedulerError(RuntimeError):
    pass


Runner = Callable[..., subprocess.CompletedProcess]


class SlurmBackend:
    def __init__(self, *, user: str | None = None, runner: Runner = subprocess.run, timeout: float = 60.0,
                 environ: Mapping[str, str] = os.environ) -> None:
        self._user = user or getpass.getuser()
        self._run = runner
        self._timeout = timeout
        self._environ = environ

    def _call(self, argv: list[str]) -> str:
        # sbatch exports the caller's environment into the job: never pass credentials along.
        result = self._run(argv, capture_output=True, text=True, timeout=self._timeout, check=False,
                           env=scrubbed_environment(self._environ))
        if result.returncode != 0:
            raise SchedulerError(f"{argv[0]} failed ({result.returncode}): {result.stderr.strip()}")
        return result.stdout

    def submit(self, script: Path, workdir: Path, job_name: str) -> str:
        output = self._call(["sbatch", "--parsable", f"--job-name={job_name}", f"--chdir={workdir}", str(script)])
        job_id = output.strip().split(";")[0]
        if not job_id.isdigit():
            raise SchedulerError(f"unexpected sbatch output: {output!r}")
        return job_id

    def query(self, job_ids: Sequence[str]) -> dict[str, SchedulerObservation]:
        if not job_ids:
            return {}
        wanted = set(job_ids)
        found = {o.job_id: o for o in self._squeue(["-u", self._user]) if o.job_id in wanted}
        missing = sorted(wanted - set(found))
        if missing:
            for o in self._sacct(["-j", ",".join(missing)]):
                if o.job_id in wanted:
                    found.setdefault(o.job_id, o)
        return found

    def find_by_name(self, job_name: str, since: datetime) -> list[SchedulerObservation]:
        live = self._squeue(["-u", self._user, "-n", job_name])
        known = {o.job_id for o in live}
        start = since.astimezone().strftime("%Y-%m-%dT%H:%M:%S")
        past = self._sacct(["-u", self._user, f"--name={job_name}", "-S", start])
        return live + [o for o in past if o.job_id not in known]

    def cancel(self, job_id: str) -> None:
        if not job_id.isdigit():
            raise ValueError(f"not a Slurm job id: {job_id!r}")
        self._call(["scancel", job_id])

    def log_file_names(self, job_id: str) -> list[str]:
        return [f"slurm-{job_id}.out"]  # Slurm's default when the script sets no --output

    def _squeue(self, selection: list[str]) -> list[SchedulerObservation]:
        output = self._call(["squeue", "-h", *selection, "-o", "%i|%T"])
        return [self._observation(line) for line in output.splitlines() if line.strip()]

    def _sacct(self, selection: list[str]) -> list[SchedulerObservation]:
        output = self._call(["sacct", "-n", "-P", "-X", *selection, "-o", "JobID,State"])
        return [self._observation(line) for line in output.splitlines() if line.strip()]

    @staticmethod
    def _observation(line: str) -> SchedulerObservation:
        job_id, _, raw = line.strip().partition("|")
        return SchedulerObservation(job_id, map_state(raw), raw)


def default_lookback() -> datetime:
    return datetime.now().astimezone() - timedelta(days=14)
