"""The workbench's sandbox: what a command an agent runs can see, change and reach.

With bubblewrap (``bwrap``), a command runs in new namespaces:

- the whole filesystem is visible read-only, as the researcher sees it, except hidden paths
  (secrets, CEDAR's own records, other hypotheses' run directories, the DFT codes' binaries
  and the MPI launchers), which appear empty; paths the agent may read inside a hidden
  directory, such as its own run directories, are shown again, read-only;
- only one directory, the agent's workspace, is writable, besides a private ``/tmp`` (where
  the machine's own ``/tmp`` entries are visible, read-only);
- there is no network, so no web, ssh or scheduler (Slurm commands cannot reach their
  controller);
- it has its own process tree, which dies with the command's time limit.

Without bubblewrap, only a Python script can run, under the analysis guard's audit hook
(reads limited the same way, writes only in the workspace, no programs, no sockets). That
guard stops mistakes, not code written to evade it.

Nothing here limits memory or disk; the researcher asked for none. Cores are limited by
pinning the command to the CPUs it was given.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from cedar.engine.analysis import GUARD
from cedar.secrets import scrubbed_environment

MPI_AND_SCHEDULER_PROGRAMS = ("mpirun", "mpiexec", "mpiexec.hydra", "orterun", "srun", "sbatch", "salloc", "scancel")
"""Hidden in the sandbox: DFT runs only as an examined calculation."""
HOME_DIRECTORY = ".home"
"""The command's $HOME, inside the workspace; caches written there persist between commands."""


@cache
def bwrap_path() -> str | None:
    """The bubblewrap program, if it is installed and can make the namespaces the sandbox needs here."""
    path = shutil.which("bwrap")
    if path is None:
        return None
    try:
        result = subprocess.run([path, "--die-with-parent", "--unshare-all", "--ro-bind", "/", "/", "true"],
                                capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return path if result.returncode == 0 else None


@dataclass(frozen=True)
class SandboxView:
    """What one agent's commands see."""

    workspace: Path
    """The only writable directory, besides a private /tmp."""
    hidden: tuple[Path, ...] = ()
    """Shown empty: directories as an empty directory, files as an empty file."""
    shown: tuple[Path, ...] = ()
    """Shown again, read-only, although inside a hidden directory."""


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int | None
    timed_out: bool
    stopped: bool
    seconds: float


def program_paths(names: Sequence[str] = MPI_AND_SCHEDULER_PROGRAMS, path: str | None = None) -> list[Path]:
    """Where the named programs are installed on this machine's search path."""
    found = []
    for name in names:
        located = shutil.which(name, path=path)
        if located is not None:
            found += [Path(located), Path(located).resolve()]
    return list(dict.fromkeys(found))


def bwrap_arguments(view: SandboxView, *, cwd: Path) -> list[str]:
    """The bubblewrap options for ``view``; hidden and shown paths that do not exist are skipped."""
    args = ["--die-with-parent", "--new-session", "--unshare-all", "--ro-bind", "/", "/", "--dev", "/dev",
            "--proc", "/proc", "--tmpfs", "/dev/shm", "--tmpfs", "/tmp"]
    # The private /tmp shows what the machine's /tmp holds, read-only; bind sources are the machine's paths.
    try:
        entries = sorted(Path("/tmp").iterdir())
    except OSError:
        entries = []
    for entry in entries:
        if entry.is_symlink():
            args += ["--symlink", os.readlink(entry), str(entry)]
        elif entry.is_dir() or entry.is_file():
            args += ["--ro-bind", str(entry), str(entry)]
    for path in view.hidden:
        if path.is_dir() and not path.is_symlink():
            args += ["--tmpfs", str(path)]
        elif path.exists():
            args += ["--ro-bind", "/dev/null", str(path)]
    for path in view.shown:
        if path.exists():
            args += ["--ro-bind", str(path), str(path)]
    workspace = view.workspace.resolve()
    args += ["--bind", str(workspace), str(workspace), "--chdir", str(cwd)]
    return args


def sandbox_environment(cores: int, *, paths: Sequence[Path] = (), workspace: Path,
                        environ: Mapping[str, str] = os.environ) -> dict[str, str]:
    """A clean environment: CEDAR's Python first on the search path, no credentials, threads matching the cores."""
    search = [*(str(p) for p in paths), str(Path(sys.executable).parent), "/usr/local/bin", "/usr/bin", "/bin"]
    threads = str(cores)
    home = workspace / HOME_DIRECTORY
    home.mkdir(exist_ok=True)
    env = {"PATH": os.pathsep.join(dict.fromkeys(search)), "HOME": str(home), "TMPDIR": "/tmp", "LANG": "C.UTF-8",
           "LC_ALL": "C.UTF-8", "MPLBACKEND": "Agg", "MPLCONFIGDIR": f"{home}/.matplotlib",
           "PYTHONDONTWRITEBYTECODE": "1", "CEDAR_WORKSPACE": str(workspace), "OMP_NUM_THREADS": threads,
           "OPENBLAS_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "NUMEXPR_NUM_THREADS": threads}
    for name in ("USER", "LOGNAME"):
        if name in environ:
            env[name] = environ[name]
    return scrubbed_environment(env)


def run_process(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], output: Path, time_limit: float,
                cpus: Sequence[int] = (), should_stop: Callable[[], bool] | None = None) -> SandboxResult:
    """Run ``argv`` in its own session, pinned to ``cpus``, writing what it prints to ``output``; kill its whole
    process group at the time limit or once ``should_stop`` says so."""
    command = list(argv)
    taskset = shutil.which("taskset")
    if cpus and taskset is not None:
        command = [taskset, "-c", ",".join(str(c) for c in cpus), *command]
    started = time.monotonic()
    with open(output, "wb") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=dict(env), stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        if cpus and taskset is None:
            try:
                os.sched_setaffinity(process.pid, set(cpus))
            except OSError:
                pass
        timed_out = stopped = False
        while True:
            try:
                code = process.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() - started > time_limit:
                timed_out = True
            elif should_stop is not None and should_stop():
                stopped = True
            else:
                continue
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = None
            break
    return SandboxResult(code, timed_out, stopped, time.monotonic() - started)


def bwrap_command(bwrap: str, command: str, view: SandboxView, *, cwd: Path) -> list[str]:
    shell = shutil.which("bash") or "/bin/sh"
    return [bwrap, *bwrap_arguments(view, cwd=cwd), shell, "-c", command]


def guarded_python(script: Path, args: Sequence[str], view: SandboxView, *, cwd: Path,
                   env: Mapping[str, str]) -> tuple[list[str], dict[str, str]]:
    """The command and environment that run a Python script under the analysis guard, for machines without
    bubblewrap: reads anywhere but the hidden paths, writes only in the workspace, no programs, no sockets."""
    workspace = view.workspace.resolve()
    guard_env = {**env, "TMPDIR": str(workspace / ".tmp"),
                 "CEDAR_ANALYSIS_ROOT": str(workspace), "CEDAR_READ_ROOTS": "/",
                 "CEDAR_DENY_ROOTS": os.pathsep.join(str(p) for p in view.hidden),
                 "CEDAR_ALLOW_ROOTS": os.pathsep.join(str(p) for p in view.shown),
                 "CEDAR_SCRIPT": str(script.resolve().relative_to(workspace)),
                 "CEDAR_CWD": str(cwd.resolve().relative_to(workspace)), "CEDAR_ARGV": json.dumps(list(args)),
                 "CEDAR_LIMITS": "{}"}
    (workspace / ".tmp").mkdir(exist_ok=True)
    return [sys.executable, "-I", "-B", str(GUARD)], guard_env
