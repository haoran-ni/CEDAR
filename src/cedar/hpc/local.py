"""Running examined local scripts on the interactive node CEDAR runs on, as a scheduler would.

A local calculation's examined ``local.sh`` runs here, inside the researcher's Slurm
allocation, instead of being submitted to Slurm. The backend has the scheduler adapter's
interface, so the job manager (journaled submission, reconciliation by job name), the job
monitor and the run checks work as they do for Slurm, and the researcher's authorization is
still needed before anything starts.

Each run is its own Slurm job step in the allocation, of exactly its task count. Slurm fences
the step to its own CPUs with a cgroup, which every MPI library and code respects, however it
starts its processes: tested on an interactive node, Intel MPI's ``mpirun`` escaped a
``taskset`` pin by starting its ranks through ``srun``, Open MPI ignored the pin and bound its
ranks anew, and two runs on the same cores did not slow down twofold but all but stopped,
since MPI ranks spin while they wait. Inside the step the script runs without the step's
``SLURM_*`` variables, so its ``mpirun`` starts the ranks itself, inside the fence.

- ``submit`` records the job under ``<runs_root>/.cedar-local/`` and starts it once enough of
  the cores local DFT may use are free (CEDAR's core pool counts them; Slurm would otherwise
  hold the step back while the run's time passed). Until then it is PENDING, waiting for free
  cores, first come, first served. A run that waited until its time limit no longer fits in
  the allocation fails without starting.
- Each run is a detached wrapper (``local_runner.py``) that starts the step, stops it at its
  time limit, counted from when the step started, and records its exit code. It outlives the
  dispatcher.
- ``query`` reads those records: PENDING until the step has started (waiting for cores, then
  for Slurm), RUNNING on the CPUs Slurm gave it, then COMPLETED, FAILED, TIMEOUT or CANCELLED;
  each with a note saying so. A run whose allocation ended before it recorded an exit code
  FAILED; a run on another node whose allocation still runs is UNKNOWN, never assumed finished.

Nothing runs on a login node: ``submit`` refuses there.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

from cedar.contracts.status import SchedulerState
from cedar.hpc.jobscript import format_walltime, local_resources
from cedar.hpc.node import WORKBENCH_CORES, CorePool, ExecutionHost
from cedar.hpc.slurm import SchedulerError, SchedulerObservation
from cedar.secrets import scrubbed_environment

RUNNER = Path(__file__).with_name("local_runner.py")
RECORDS = ".cedar-local"
MARGIN_SECONDS = 5 * 60
"""A local run must end this long before its allocation does."""
STEP_SLACK_MINUTES = 5
"""Slurm's own time limit for a step exceeds the run's by this much: it stops the step only if the runner did not."""
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


def cpu_list(cpus: Sequence[int]) -> str:
    """CPUs as Linux lists them, e.g. ``0-59,64``."""
    parts, ordered = [], sorted(cpus)
    start = previous = None
    for cpu in [*ordered, None]:
        if cpu is not None and previous is not None and cpu == previous + 1:
            previous = cpu
            continue
        if start is not None:
            parts.append(f"{start}-{previous}" if previous != start else f"{start}")
        start = previous = cpu
    return ",".join(parts)


def _srun_reason(log: Path) -> str | None:
    """What ``srun`` last said about the step, e.g. ``Requested nodes are busy``, from the run's log."""
    try:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 8192))
            lines = stream.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    said = next((line for line in reversed(lines) if line.startswith("srun:")), None)
    if said is None:
        return None
    reason = re.search(r"\(([^()]+)\)\s*$", said)
    return reason.group(1) if reason else said.removeprefix("srun:").strip()


Runner = Callable[..., subprocess.CompletedProcess]
START_SECONDS = 60
"""A run started this long ago that has not recorded its start never did; Slurm has as long to start its step
before the run's note says that Slurm makes it wait."""


class StepLauncher(Protocol):
    """How a local run gets its own CPUs: the command that starts the run's inner runner in a step of its size."""

    def unavailable(self) -> str | None:
        """Why no step can start here, or None."""
        ...

    def launch(self, record: Mapping) -> list[str]:
        """The command, followed by the inner runner's, that starts a step for the run ``record`` describes."""
        ...

    def stop(self) -> list[str]:
        """The command, followed by a step id, that stops a step; empty if stopping the launcher is enough."""
        ...

    def environment(self, environ: Mapping[str, str]) -> dict[str, str]:
        """The environment the launcher starts in."""
        ...


class SlurmSteps:
    """Each local run as its own Slurm step in the allocation, on this node, of exactly its task count.

    ``--exact`` gives the step only the CPUs it asks for, and ``--gres=none`` none of the allocation's GPUs, which
    a step otherwise takes, so that a second step would wait for them. ``--mpi=none``: the script's ``mpirun``
    starts the ranks, not ``srun``. The launcher starts without the researcher's shell's ``SLURM_*`` variables
    (its CPU binding, task count), which describe the shell's step, not the run's."""

    def unavailable(self) -> str | None:
        return None if shutil.which("srun") else "srun is not on this machine's PATH, so no Slurm step can start"

    def launch(self, record: Mapping) -> list[str]:
        node = [f"--nodelist={record['node']}"] if record.get("node") else []
        return ["srun", f"--jobid={record['allocation']}", "--nodes=1", *node, "--exact", "--gres=none",
                "--mpi=none", "--ntasks=1", f"--cpus-per-task={record['ntasks']}",
                f"--time={format_walltime(record['minutes'] + STEP_SLACK_MINUTES)}",
                f"--job-name={record['job_name']}"]

    def stop(self) -> list[str]:
        return ["scancel"]

    def environment(self, environ: Mapping[str, str]) -> dict[str, str]:
        return {name: value for name, value in environ.items()
                if not name.startswith("SLURM_") or name == "SLURM_CONF"}


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
                 steps: StepLauncher | None = None, environ: Mapping[str, str] = os.environ,
                 allocation_state: Callable[[str], bool | None] = allocation_running,
                 runner: Runner = subprocess.run) -> None:
        """``pool`` hands out this machine's cores; only a backend that has it starts runs, and runs started by an
        earlier dispatcher here keep theirs."""
        self._records = Path(runs_root) / RECORDS
        self._host = host
        self._pool = pool
        self._steps = steps or SlurmSteps()
        self._environ = environ
        self._allocation_state = allocation_state
        self._runner = runner
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
            "host": host.hostname, "node": host.node, "allocation": host.allocation, "ntasks": resources.ntasks,
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
            return  # a runner about to start sees the cancellation first
        if started["host"] != self._host().hostname:
            raise SchedulerError(f"{job_id} runs on {started['host']}; cancel it from CEDAR there")
        step = (_read(base.with_suffix(".inside")) or {}).get("step")
        if step and record.get("stop"):
            try:
                self._runner([*record["stop"], step], capture_output=True, text=True, timeout=60, check=False,
                             env=scrubbed_environment(self._environ))
            except (OSError, subprocess.TimeoutExpired):
                pass
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
        inside = _read(self._records / f"{job_id}.inside") or {}
        place = {"execution": "local", "host": started.get("host") or record.get("host"),
                 "allocation": record.get("allocation")}
        if inside.get("cpus"):
            place["cpus"] = cpu_list(inside["cpus"])
        if inside.get("step"):
            place["slurm_step"] = inside["step"]
        return place

    # -- local runs ---------------------------------------------------------------------------

    def unavailable(self, host: ExecutionHost) -> str | None:
        """Why no local run can start here, or None."""
        if not host.interactive:
            return LOGIN_NODE.format(host=host.hostname)
        return self._steps.unavailable()

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
        inside = _read(self._records / f"{job_id}.inside")
        if ended is not None:
            if cancelled:
                return SchedulerState.CANCELLED, "CANCELLED"
            if ended.get("reason"):
                return SchedulerState.FAILED, ended["reason"]
            if ended["timed_out"]:
                return SchedulerState.TIMEOUT, f"stopped at its {record['minutes']}-minute limit"
            code = ended["exit_code"]
            if code == 0:
                return SchedulerState.COMPLETED, "COMPLETED"
            if inside is None:
                said = _srun_reason(Path(record["workdir"]) / f"{job_id}.log")
                return SchedulerState.FAILED, "its Slurm step did not start" + (f": {said}" if said else "")
            return SchedulerState.FAILED, f"exit code {code}"
        started = _read(self._records / f"{job_id}.started")
        host = self._host()
        here = (started or record)["host"] == host.hostname
        if here and started is not None:
            if not _alive(started["wrapper_pid"]):
                return SchedulerState.FAILED, "the run ended without recording an exit code"
            if inside is not None:
                return SchedulerState.RUNNING, f"running on CPUs {cpu_list(inside['cpus'])}"
            if time.time() - started["started_at"] < START_SECONDS:
                return SchedulerState.PENDING, "starting its Slurm step"
            said = _srun_reason(Path(record["workdir"]) / f"{job_id}.log")
            return SchedulerState.PENDING, "waiting for Slurm to start its step" + (f" ({said})" if said else "")
        if here and record["allocation"] == host.allocation:
            if record["cpus"] is None:
                n = record["ntasks"]
                return SchedulerState.PENDING, f"waiting for {n} free core{'' if n == 1 else 's'}"
            if time.time() - record["launched_at"] < START_SECONDS:
                return SchedulerState.PENDING, "starting its Slurm step"
            return SchedulerState.FAILED, "the run did not start"
        running = self._allocation_state(record["allocation"]) if record["allocation"] else False
        if running is False:
            return SchedulerState.FAILED, f"allocation {record['allocation']} ended before the run finished"
        return SchedulerState.UNKNOWN, f"on {record['host']}, in allocation {record['allocation']}"

    def _start_waiting(self) -> None:
        host = self._host()
        if self._pool is None or not host.interactive or self._steps.unavailable():
            return
        for record in self._all():
            if record["cpus"] is not None or record["host"] != host.hostname or self._ended(record["job_id"]):
                continue
            if record["allocation"] != host.allocation:
                continue
            path = self._records / f"{record['job_id']}.json"
            left = host.seconds_left()
            if left is not None and record["minutes"] * 60 > left - MARGIN_SECONDS:
                _write(path.with_suffix(".exit"), {
                    "exit_code": None, "timed_out": False, "ended_at": time.time(),
                    "reason": f"it waited for free cores until its {record['minutes']}-minute limit no longer fit in "
                              f"allocation {host.allocation}, which ends {left / 60:.0f} minutes from now; it never "
                              "started"})
                continue
            cpus = self._pool.take(record["ntasks"], keep=WORKBENCH_CORES)
            if cpus is None:
                return  # first come, first served
            record.update(cpus=list(cpus), launched_at=time.time())
            record.update(launch=self._steps.launch(record), stop=self._steps.stop())
            _write(path, record)
            self._started[record["job_id"]] = subprocess.Popen(
                [sys.executable, "-I", str(RUNNER), str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True,
                env=self._steps.environment(scrubbed_environment(self._environ)))
            self._held[record["job_id"]] = tuple(cpus)

    def _reap(self) -> None:
        for job_id, process in list(self._started.items()):
            if process.poll() is not None:
                del self._started[job_id]
        for job_id in list(self._held):
            live = job_id in self._started or self._running_here(job_id) or self._launching(job_id)
            if self._ended(job_id) or not live:
                cpus = self._held.pop(job_id)
                if self._pool is not None:
                    self._pool.give(cpus)
                continue
            inside = _read(self._records / f"{job_id}.inside")
            if inside is not None and self._pool is not None and set(inside["cpus"]) != set(self._held[job_id]):
                self._pool.rebook(self._held[job_id], inside["cpus"])
                self._held[job_id] = tuple(inside["cpus"])

    def _running_here(self, job_id: str) -> bool:
        started = _read(self._records / f"{job_id}.started")
        return started is not None and started["host"] == self._host().hostname and _alive(started["wrapper_pid"])

    def _launching(self, job_id: str) -> bool:
        """Whether a runner was started here so recently that it may not have recorded its start yet."""
        record = _read(self._records / f"{job_id}.json") or {}
        return (record.get("host") == self._host().hostname and not (self._records / f"{job_id}.started").exists()
                and time.time() - record.get("launched_at", 0) < START_SECONDS)

    def _adopt(self) -> None:
        """Keep the cores of runs an earlier dispatcher started on this machine, while they run: the CPUs their
        steps got, or the ones booked for a step not started yet."""
        if self._pool is None:
            return
        for record in self._all():
            job_id = record["job_id"]
            if record["cpus"] is None or self._ended(job_id):
                continue
            if self._running_here(job_id) or self._launching(job_id):
                inside = _read(self._records / f"{job_id}.inside")
                cpus = tuple(inside["cpus"] if inside is not None else record["cpus"])
                self._pool.claim(cpus)
                self._held[job_id] = cpus
