"""Running examined local scripts on the interactive node CEDAR runs on, as a scheduler would.

A local calculation's examined ``local.sh`` runs here, inside the researcher's Slurm
allocation, instead of being submitted to Slurm. The backend has the scheduler adapter's
interface, so the job manager (journaled submission, reconciliation by job name), the job
monitor and the run checks work as they do for Slurm, and the researcher's authorization is
still needed before anything starts.

- ``submit`` records the job under ``<runs_root>/.cedar-local/`` before starting it, and
  starts it once the allocation's cores are free; until then it is PENDING.
- Each run is a detached wrapper (``local_runner.py``) that starts the script pinned to its
  cores, stops it at its time limit and records its exit code. It outlives the dispatcher.
- ``query`` reads those records: RUNNING while the wrapper lives, then COMPLETED, FAILED,
  TIMEOUT or CANCELLED. A run whose allocation ended before it recorded an exit code FAILED;
  a run on another node whose allocation still runs is UNKNOWN, never assumed finished.

Nothing runs on a login node: ``submit`` refuses there.
"""

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

from cedar.contracts.status import SchedulerState
from cedar.hpc.jobscript import local_resources
from cedar.hpc.node import WORKBENCH_CORES, CorePool, ExecutionHost
from cedar.hpc.slurm import SchedulerError, SchedulerObservation
from cedar.secrets import scrubbed_environment

RUNNER = Path(__file__).with_name("local_runner.py")
RECORDS = ".cedar-local"
MARGIN_SECONDS = 5 * 60
"""A local run must end this long before its allocation does."""
LOGIN_NODE = ("CEDAR is running on a login node ({host}), and DFT cannot run locally on login nodes; start CEDAR on "
              "an interactive node to run local calculations")


def _write(path: Path, data: dict) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data))
    os.replace(temporary, path)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    """Whether a local runner with this process id still runs on this machine."""
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return RUNNER.name.encode() in command


Runner = Callable[..., subprocess.CompletedProcess]
START_SECONDS = 60
"""A run started this long ago that has not recorded its start never did."""


def allocation_running(allocation: str, runner: Runner = subprocess.run) -> bool | None:
    """Whether Slurm reports the allocation as running; None if it cannot tell."""
    try:
        result = runner(["squeue", "-h", "-j", allocation, "-o", "%T"], capture_output=True, text=True, timeout=30,
                        check=False, env=scrubbed_environment())
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return False if "Invalid job id" in result.stderr else None
    return result.stdout.strip() == "RUNNING"


class LocalBackend:
    def __init__(self, runs_root: str | Path, *, host: Callable[[], ExecutionHost], pool: CorePool | None = None,
                 environ: Mapping[str, str] = os.environ,
                 allocation_state: Callable[[str], bool | None] = allocation_running) -> None:
        """``pool`` hands out this machine's cores; runs started by an earlier dispatcher here keep theirs."""
        self._records = Path(runs_root) / RECORDS
        self._host = host
        self._pool = pool
        self._environ = environ
        self._allocation_state = allocation_state
        self._started: dict[str, subprocess.Popen] = {}
        self._held: dict[str, tuple[int, ...]] = {}
        self._adopt()

    # -- the scheduler interface ---------------------------------------------------------------

    def submit(self, script: Path, workdir: Path, job_name: str) -> str:
        host = self._host()
        problem = self.unavailable(host)
        if problem:
            raise SchedulerError(problem)
        resources = local_resources(Path(script).read_text())
        if resources is None:
            raise SchedulerError(f"{script} states no task count and time limit")
        if resources.ntasks > host.dft_cores:
            raise SchedulerError(f"{resources.ntasks} tasks exceed the {host.dft_cores} cores local DFT may use here: "
                                 f"the allocation has {host.cores}, and {WORKBENCH_CORES} are kept for the agents' "
                                 "commands")
        left = host.seconds_left()
        if left is not None and resources.walltime_minutes * 60 > left - MARGIN_SECONDS:
            raise SchedulerError(f"{resources.walltime_minutes} minutes would run past the end of allocation "
                                 f"{host.allocation}, {left / 60:.0f} minutes from now")
        self._records.mkdir(parents=True, exist_ok=True)
        job_id = f"local-{uuid.uuid4().hex[:16]}"
        _write(self._records / f"{job_id}.json", {
            "job_id": job_id, "job_name": job_name, "workdir": str(workdir), "script": str(script),
            "host": host.hostname, "allocation": host.allocation, "ntasks": resources.ntasks,
            "minutes": resources.walltime_minutes, "submitted_at": time.time(), "cpus": None})
        self._start_waiting()
        return job_id

    def query(self, job_ids: Sequence[str]) -> dict[str, SchedulerObservation]:
        self._reap()
        self._start_waiting()
        found = {}
        for job_id in job_ids:
            record = _read(self._records / f"{job_id}.json")
            if record is not None:
                state, raw = self._state(record)
                found[job_id] = SchedulerObservation(job_id, state, raw)
        return found

    def find_by_name(self, job_name: str, since: datetime) -> list[SchedulerObservation]:
        found = []
        for record in self._all():
            if record["job_name"] == job_name and record["submitted_at"] >= since.timestamp():
                state, raw = self._state(record)
                found.append(SchedulerObservation(record["job_id"], state, raw))
        return found

    def cancel(self, job_id: str) -> None:
        record = _read(self._records / f"{job_id}.json")
        if record is None:
            raise SchedulerError(f"no local run {job_id}")
        base = self._records / job_id
        _write(base.with_suffix(".cancel"), {"cancelled_at": time.time()})
        started = _read(base.with_suffix(".started"))
        if started is None:
            if record["cpus"] is None:  # not started yet: it never will
                _write(base.with_suffix(".exit"), {"exit_code": None, "timed_out": False, "ended_at": time.time()})
            return
        if started["host"] != self._host().hostname:
            raise SchedulerError(f"{job_id} runs on {started['host']}; cancel it from CEDAR there")
        try:
            os.killpg(started["child_pgid"], signal.SIGTERM)
        except ProcessLookupError:
            pass

    def log_file_names(self, job_id: str) -> list[str]:
        return [f"{job_id}.log"]

    def where(self, job_id: str) -> dict[str, str | None]:
        """Where a local run ran, for its record."""
        record = _read(self._records / f"{job_id}.json") or {}
        started = _read(self._records / f"{job_id}.started") or {}
        return {"execution": "local", "host": started.get("host") or record.get("host"),
                "allocation": record.get("allocation")}

    # -- local runs ---------------------------------------------------------------------------

    @staticmethod
    def unavailable(host: ExecutionHost) -> str | None:
        """Why no local run can start here, or None."""
        return None if host.interactive else LOGIN_NODE.format(host=host.hostname)

    def _all(self) -> list[dict]:
        if not self._records.is_dir():
            return []
        records = (_read(path) for path in self._records.glob("local-*.json"))
        return sorted((r for r in records if r is not None), key=lambda r: r["submitted_at"])

    def _ended(self, job_id: str) -> dict | None:
        return _read(self._records / f"{job_id}.exit")

    def _state(self, record: dict) -> tuple[SchedulerState, str]:
        job_id = record["job_id"]
        ended = self._ended(job_id)
        cancelled = (self._records / f"{job_id}.cancel").exists()
        if ended is not None:
            if cancelled:
                return SchedulerState.CANCELLED, "CANCELLED"
            if ended["timed_out"]:
                return SchedulerState.TIMEOUT, f"TIMEOUT after {record['minutes']} minutes"
            code = ended["exit_code"]
            return (SchedulerState.COMPLETED, "COMPLETED") if code == 0 else (SchedulerState.FAILED, f"exit code {code}")
        started = _read(self._records / f"{job_id}.started")
        host = self._host()
        here = (started or record)["host"] == host.hostname
        if here and started is not None:
            if _alive(started["wrapper_pid"]):
                return SchedulerState.RUNNING, "RUNNING"
            return SchedulerState.FAILED, "the run ended without recording an exit code"
        if here and record["allocation"] == host.allocation:
            if record["cpus"] is None:
                return SchedulerState.PENDING, "waiting for free cores"
            if time.time() - record["launched_at"] < START_SECONDS:
                return SchedulerState.RUNNING, "starting"
            return SchedulerState.FAILED, "the run did not start"
        running = self._allocation_state(record["allocation"]) if record["allocation"] else False
        if running is False:
            return SchedulerState.FAILED, f"allocation {record['allocation']} ended before the run finished"
        return SchedulerState.UNKNOWN, f"on {record['host']}, in allocation {record['allocation']}"

    def _start_waiting(self) -> None:
        host = self._host()
        if not host.interactive:
            return
        for record in self._all():
            if record["cpus"] is not None or record["host"] != host.hostname or self._ended(record["job_id"]):
                continue
            if record["allocation"] != host.allocation:
                continue
            cpus = self._pool.take(record["ntasks"], keep=WORKBENCH_CORES) if self._pool is not None else ()
            if cpus is None:
                return  # first come, first served
            record["cpus"], record["launched_at"] = list(cpus), time.time()
            path = self._records / f"{record['job_id']}.json"
            _write(path, record)
            self._started[record["job_id"]] = subprocess.Popen(
                [sys.executable, "-I", str(RUNNER), str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True, env=scrubbed_environment(self._environ))
            self._held[record["job_id"]] = tuple(cpus)

    def _reap(self) -> None:
        for job_id, process in list(self._started.items()):
            if process.poll() is not None:
                del self._started[job_id]
        for job_id in list(self._held):
            if job_id not in self._started and (self._ended(job_id) or not self._running_here(job_id)):
                cpus = self._held.pop(job_id)
                if self._pool is not None:
                    self._pool.give(cpus)

    def _running_here(self, job_id: str) -> bool:
        started = _read(self._records / f"{job_id}.started")
        return started is not None and started["host"] == self._host().hostname and _alive(started["wrapper_pid"])

    def _adopt(self) -> None:
        """Keep the cores of runs an earlier dispatcher started on this machine, while they run."""
        if self._pool is None:
            return
        for record in self._all():
            job_id = record["job_id"]
            if record["cpus"] is not None and not self._ended(job_id) and self._running_here(job_id):
                cpus = self._pool.take(record["ntasks"])
                if cpus is not None:
                    self._held[job_id] = cpus
