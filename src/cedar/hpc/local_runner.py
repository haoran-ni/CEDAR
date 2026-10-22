"""Runs one examined local script and records how it ended; started by ``LocalBackend`` as
``python -I <this file> <job record>``, detached, so it outlives the dispatcher that started it.

It starts the record's step launcher (``srun`` for a Slurm step of the run's size) in its own
session, and records its process ids next to the record (``<job>.started``). The launcher runs
this file again inside the step (``--inside``), which records the CPUs the step was given
(``<job>.inside``) and then runs the script, without the step's ``SLURM_*`` variables so that
its ``mpirun`` starts the ranks itself, inside the step's CPUs, instead of asking Slurm for
another step. The time limit counts from when the step started: a run Slurm makes wait loses
none of it. At the limit, or once the run is cancelled (``<job>.cancel``), the runner stops the
step (``scancel``) and the launcher's process group, politely and then by force. It then
records the exit code (``<job>.exit``).

Standard library only: this file runs before anything else in its process.
"""

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time

GRACE_SECONDS = 30
POLL_SECONDS = 1.0


def _write(path: str, data: dict) -> None:
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w") as stream:
        json.dump(data, stream)
    os.replace(temporary, path)


def _read(path: str) -> dict | None:
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def _interpreter(script: str) -> list[str]:
    with open(script) as stream:
        first = stream.readline().strip()
    return shlex.split(first[2:]) if first.startswith("#!") else ["/bin/sh"]


def inside(record_path: str) -> None:
    """In the step: record its CPUs, then become the script, without the step's ``SLURM_*`` variables."""
    with open(record_path) as stream:
        record = json.load(stream)
    job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
    _write(f"{record_path.removesuffix('.json')}.inside", {
        "cpus": sorted(os.sched_getaffinity(0)), "step": f"{job}.{step}" if job and step else None,
        "host": socket.gethostname(), "started_at": time.time()})
    environment = {name: value for name, value in os.environ.items() if not name.startswith("SLURM_")}
    argv = [*_interpreter(record["script"]), record["script"]]
    os.chdir(record["workdir"])
    os.execvpe(argv[0], argv, environment)


def _stop(child: subprocess.Popen, record: dict, base: str) -> int:
    """Stop the step and the launcher; the launcher's exit code."""
    step = (_read(f"{base}.inside") or {}).get("step")
    if step and record.get("stop"):
        try:
            subprocess.run([*record["stop"], step], stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return child.wait(timeout=GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        return child.wait()


def main(record_path: str) -> int:
    with open(record_path) as stream:
        record = json.load(stream)
    base = record_path.removesuffix(".json")
    if os.path.exists(f"{base}.cancel"):  # cancelled before it started
        _write(f"{base}.exit", {"exit_code": None, "timed_out": False, "ended_at": time.time()})
        return 0
    argv = [*record.get("launch", ()), sys.executable, "-I", os.path.abspath(__file__), "--inside", record_path]
    with open(os.path.join(record["workdir"], f"{record['job_id']}.log"), "ab") as log:
        child = subprocess.Popen(argv, cwd=record["workdir"], stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        _write(f"{base}.started", {"wrapper_pid": os.getpid(), "child_pgid": child.pid,
                                   "host": socket.gethostname(), "started_at": time.time()})
        deadline, timed_out = None, False
        while True:
            try:
                code = child.wait(timeout=POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pass
            if deadline is None:
                began = _read(f"{base}.inside")
                if began is not None:
                    deadline = began["started_at"] + record["minutes"] * 60
            if deadline is not None and time.time() >= deadline:
                timed_out = True
                code = _stop(child, record, base)
                break
            if os.path.exists(f"{base}.cancel"):  # also one that came before the backend saw the run start
                code = _stop(child, record, base)
                break
    _write(f"{base}.exit", {"exit_code": code, "timed_out": timed_out, "ended_at": time.time()})
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "--inside":
        inside(sys.argv[2])
    else:
        sys.exit(main(sys.argv[1]))
