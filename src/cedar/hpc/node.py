"""Where CEDAR is running: a login node, or a compute node inside a running Slurm allocation.

CEDAR counts as being on an interactive node only when ``SLURM_JOB_ID`` is set, that job is
running, and this machine is one of its nodes. Then the allocation's cores on this node
(``SLURM_CPUS_ON_NODE``) and its remaining time bound what may run here. Anything else,
including a shell from ``salloc`` that stayed on the login node, is a login node, where DFT
never runs. Unknown means a login node: nothing is assumed about an allocation it cannot see.
"""

import os
import re
import socket
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from cedar.secrets import scrubbed_environment

Runner = Callable[..., subprocess.CompletedProcess]
WORKBENCH_CORES = 4
"""Cores of the allocation that local DFT never uses: they stay free for the agents' workbench commands."""
WAIT_SECONDS = 120.0
"""How long a workbench command waits for cores or a slot before it is refused as busy."""


@dataclass(frozen=True)
class ExecutionHost:
    hostname: str
    allocation: str | None = None
    """The Slurm job whose allocation CEDAR runs in, when this machine is one of its nodes."""
    cores: int = 1
    """Cores CEDAR may use here: the allocation's on this node, or the machine's own."""
    ends_at: datetime | None = None
    """When the allocation ends, if it has a time limit."""
    reason: str = ""
    """Why this is, or is not, an interactive node."""
    node: str | None = None
    """This machine's name in the allocation's node list, which Slurm steps name."""

    @property
    def interactive(self) -> bool:
        return self.allocation is not None

    @property
    def dft_cores(self) -> int:
        """Cores a local DFT run may use here: the allocation's, less the ones kept for the workbench."""
        return max(self.cores - WORKBENCH_CORES, 0) if self.interactive else 0

    def seconds_left(self, now: datetime | None = None) -> float | None:
        if self.ends_at is None:
            return None
        return (self.ends_at - (now or datetime.now(timezone.utc))).total_seconds()

    def describe(self) -> str:
        if self.interactive:
            left = self.seconds_left()
            until = f", {left / 60:.0f} minutes left" if left is not None else ""
            return f"interactive node {self.hostname} in Slurm allocation {self.allocation} ({self.cores} cores{until})"
        return f"login node {self.hostname} ({self.reason})"


def expand_nodelist(text: str) -> list[str]:
    """Slurm's compact node lists, e.g. ``trace[01-03,07],gpu5``, as host names."""
    names = []
    for match in re.finditer(r"([^,\[]+)(?:\[([^\]]*)\])?", text):
        prefix, ranges = match.group(1), match.group(2)
        if ranges is None:
            names.append(prefix)
            continue
        for part in ranges.split(","):
            start, _, end = part.partition("-")
            if not end:
                names.append(prefix + start)
                continue
            width = len(start)
            names += [f"{prefix}{n:0{width}d}" for n in range(int(start), int(end) + 1)]
    return [n for n in names if n]


def parse_time_left(text: str) -> timedelta | None:
    """``squeue -o %L`` (``[days-]hours:minutes:seconds``, or ``UNLIMITED``) as a duration; None if unlimited."""
    match = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", text.strip())
    if match is None:
        return None
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def detect_host(environ: Mapping[str, str] = os.environ, *, runner: Runner = subprocess.run,
                hostname: Callable[[], str] = socket.gethostname,
                now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> ExecutionHost:
    name = hostname()
    short = name.split(".")[0]
    own_cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    job = environ.get("SLURM_JOB_ID")
    if not job:
        return ExecutionHost(name, cores=own_cores, reason="not inside a Slurm allocation")
    nodes = expand_nodelist(environ.get("SLURM_JOB_NODELIST", environ.get("SLURM_NODELIST", "")))
    if short not in nodes and name not in nodes:
        where = ", ".join(nodes) or "other nodes"
        return ExecutionHost(name, cores=own_cores, reason=f"Slurm job {job} runs on {where}, not here")
    try:
        result = runner(["squeue", "-h", "-j", job, "-o", "%T|%L"], capture_output=True, text=True, timeout=30,
                        check=False, env=scrubbed_environment(environ))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ExecutionHost(name, cores=own_cores, reason=f"could not ask Slurm about job {job}: {exc}")
    state, _, left = result.stdout.strip().partition("|")
    if result.returncode != 0 or state != "RUNNING":
        return ExecutionHost(name, cores=own_cores, reason=f"Slurm does not report job {job} as running")
    remaining = parse_time_left(left)
    try:
        cores = int(environ.get("SLURM_CPUS_ON_NODE", "")) or own_cores
    except ValueError:
        cores = own_cores
    return ExecutionHost(name, allocation=job, cores=min(cores, own_cores),
                         ends_at=now() + remaining if remaining is not None else None,
                         reason=f"this machine is a node of running Slurm job {job}",
                         node=short if short in nodes else name)


def local_dft(host: ExecutionHost) -> str:
    """Whether calculations can run locally where CEDAR runs, in words for the agents and the researcher."""
    if host.interactive and host.dft_cores == 0:
        return (f"not possible: CEDAR runs on the {host.describe()}, and {WORKBENCH_CORES} cores are always kept for "
                "the agents' commands, which leaves none for DFT. The researcher can ask for a larger allocation; "
                "otherwise calculations run through Slurm")
    if host.interactive:
        return (f"possible: CEDAR runs on the {host.describe()}; a local run may use up to {host.dft_cores} cores, "
                f"since {WORKBENCH_CORES} are kept for the agents' commands. Local runs share those cores: each runs "
                "as its own Slurm step on cores no other run uses, and one that does not fit waits, first come, first "
                "served, until enough are free")
    return (f"not possible: CEDAR runs on the {host.describe()}, and DFT cannot run locally on login nodes. The "
            "researcher can start CEDAR on an interactive node (inside a Slurm allocation) to run calculations "
            "locally; otherwise they run through Slurm")


class HostWatch:
    """The current host, asked again at most every ``ttl`` seconds; an allocation can end at any time."""

    def __init__(self, detect: Callable[[], ExecutionHost] = detect_host, ttl: float = 60.0) -> None:
        self._detect = detect
        self._ttl = ttl
        self._host: ExecutionHost | None = None
        self._at = 0.0

    def __call__(self) -> ExecutionHost:
        if self._host is None or time.monotonic() - self._at > self._ttl:
            self._host, self._at = self._detect(), time.monotonic()
        return self._host


class CorePool:
    """The CPUs of this machine (of the allocation, on an interactive node) that workbench commands and local runs
    use, handed out so that no two share one.

    Commands get the highest free CPUs. A local run books the lowest, the ones Slurm gives a new step first, and
    once its step has started holds the CPUs Slurm actually gave it (``rebook``). A CPU held twice, because a step
    landed on a command's CPU, stays taken until both have let go."""

    def __init__(self, cpus: Sequence[int] | None = None) -> None:
        self._cpus = sorted(cpus if cpus is not None else os.sched_getaffinity(0))
        self._holders: Counter[int] = Counter()
        self.size = len(self._cpus)
        self._running = 0
        self._changed = threading.Condition()

    def _free(self) -> list[int]:
        return [c for c in self._cpus if not self._holders[c]]

    def _hold(self, cpus: Sequence[int]) -> tuple[int, ...]:
        """Hold ``cpus``; the ones another holder had already."""
        shared = tuple(c for c in cpus if self._holders[c])
        self._holders.update(cpus)
        return shared

    def _let_go(self, cpus: Sequence[int]) -> None:
        for c in cpus:
            if self._holders[c] > 0:
                self._holders[c] -= 1

    def acquire(self, count: int, *, most_running: int | None = None, timeout: float = WAIT_SECONDS
                ) -> tuple[int, ...] | None:
        """``count`` CPUs, once that many are free and fewer than ``most_running`` commands hold some; None if
        that does not happen within ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        with self._changed:
            while len(self._free()) < count or (most_running is not None and self._running >= most_running):
                left = deadline - time.monotonic()
                if left <= 0 or not self._changed.wait(left):
                    if len(self._free()) < count or (most_running is not None and self._running >= most_running):
                        return None
            taken = tuple(self._free()[-count:])
            self._hold(taken)
            self._running += 1
            return taken

    def release(self, cpus: Sequence[int]) -> None:
        with self._changed:
            self._let_go(cpus)
            self._running -= 1
            self._changed.notify_all()

    def take(self, count: int, keep: int = 0) -> tuple[int, ...] | None:
        """``count`` CPUs for a local run, if that many are free now with ``keep`` more left over; a run is not a
        command."""
        with self._changed:
            free = self._free()
            if len(free) - count < keep or len(free) < count:
                return None
            taken = tuple(free[:count])
            self._hold(taken)
            return taken

    def claim(self, cpus: Sequence[int]) -> tuple[int, ...]:
        """Hold exactly these CPUs, a running step's, free or not; the ones another holder has too."""
        with self._changed:
            return self._hold(cpus)

    def rebook(self, booked: Sequence[int], actual: Sequence[int]) -> tuple[int, ...]:
        """Exchange the CPUs ``take`` booked for a run for the ones its step got; the ones another holder has too."""
        with self._changed:
            self._let_go(booked)
            shared = self._hold(actual)
            self._changed.notify_all()
            return shared

    def give(self, cpus: Sequence[int]) -> None:
        """Return what ``take``, ``claim`` or ``rebook`` handed out."""
        with self._changed:
            self._let_go(cpus)
            self._changed.notify_all()
