"""Read-only questions about the scheduler, for working out how to run on a system.

Each query is a fixed command with validated arguments, run without a shell and with
credentials removed from its environment. Nothing here changes the system.
"""

import getpass
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta

from sim2real.secrets import scrubbed_environment

MAX_OUTPUT = 20_000
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SCHEDULER_COMMANDS = {"slurm": ("sbatch", "squeue", "sacct", "sinfo"), "pbs": ("qsub", "qstat"),
                      "lsf": ("bsub", "bjobs")}


@dataclass(frozen=True)
class QueryResult:
    command: str
    output: str
    truncated: bool = False


Runner = Callable[..., subprocess.CompletedProcess]


class SchedulerQueries:
    QUERIES = {
        "detect_scheduler": "Which batch schedulers' commands are installed.",
        "slurm_partitions": "Slurm partitions: availability, time limit, nodes, CPUs and memory per node.",
        "slurm_partition_detail": "Full Slurm settings of one partition (argument: partition name).",
        "slurm_node_cpus": "Allocated/idle/other/total CPUs per node in one partition (argument: partition name).",
        "slurm_associations": "The user's Slurm accounts, partitions and QOS.",
        "slurm_recent_jobs": "The user's jobs from the last 30 days: name, partition, CPUs, elapsed time, state.",
    }

    def __init__(self, *, user: str | None = None, runner: Runner = subprocess.run,
                 environ: Mapping[str, str] = os.environ, which: Callable[[str], str | None] = shutil.which,
                 today: Callable[[], date] = date.today) -> None:
        self._user = user or getpass.getuser()
        self._run = runner
        self._environ = environ
        self._which = which
        self._today = today

    def run(self, query: str, argument: str | None = None) -> QueryResult:
        if query not in self.QUERIES:
            raise ValueError(f"unknown query {query!r}; available: {', '.join(self.QUERIES)}")
        needs_argument = query in ("slurm_partition_detail", "slurm_node_cpus")
        if needs_argument and (argument is None or not _NAME.match(argument)):
            raise ValueError(f"{query} needs a partition name (letters, digits, '_', '.', '-')")
        if not needs_argument and argument is not None:
            raise ValueError(f"{query} takes no argument")
        if query == "detect_scheduler":
            lines = [f"{scheduler}: " + ", ".join(f"{c}={self._which(c) or 'not found'}" for c in commands)
                     for scheduler, commands in SCHEDULER_COMMANDS.items()]
            return QueryResult("which " + " ".join(c for cs in SCHEDULER_COMMANDS.values() for c in cs),
                               "\n".join(lines))
        argv = {
            "slurm_partitions": ["sinfo", "-h", "-o", "%P|%a|%l|%D|%c|%m"],
            "slurm_partition_detail": ["scontrol", "show", "partition", argument or ""],
            "slurm_node_cpus": ["sinfo", "-p", argument or "", "-h", "-o", "%n %C"],
            "slurm_associations": ["sacctmgr", "-nP", "show", "assoc", f"user={self._user}",
                                   "format=account,partition,qos"],
            "slurm_recent_jobs": ["sacct", "-u", self._user, "-n", "-P", "-X", "-S",
                                  (self._today() - timedelta(days=30)).isoformat(),
                                  "-o", "JobID,JobName,Partition,AllocCPUS,Elapsed,State"],
        }[query]
        command = " ".join(argv)
        try:
            result = self._run(argv, capture_output=True, text=True, timeout=60, check=False,
                               env=scrubbed_environment(self._environ))
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return QueryResult(command, f"could not run: {exc}")
        output = result.stdout if result.returncode == 0 else f"exit {result.returncode}: {result.stderr.strip()}"
        if len(output) > MAX_OUTPUT:
            return QueryResult(command, output[:MAX_OUTPUT], truncated=True)
        return QueryResult(command, output)
