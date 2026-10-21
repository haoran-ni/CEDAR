"""Runs one examined local script and records how it ended; started by ``LocalBackend`` as
``python -I <this file> <job record>``, detached, so it outlives the dispatcher that started it.

It starts the script in its own session, pinned to the CPUs the record names, and records its
process ids next to the record (``<job>.started``). At the record's time limit it stops the
script's whole process group, politely and then by force. It then records the exit code
(``<job>.exit``). A cancellation kills the script's process group from outside; the exit is
recorded the same way.

Standard library only: this file runs before anything else in its process.
"""

import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time

GRACE_SECONDS = 30


def _write(path: str, data: dict) -> None:
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w") as stream:
        json.dump(data, stream)
    os.replace(temporary, path)


def _interpreter(script: str) -> list[str]:
    with open(script) as stream:
        first = stream.readline().strip()
    return shlex.split(first[2:]) if first.startswith("#!") else ["/bin/sh"]


def main(record_path: str) -> int:
    with open(record_path) as stream:
        record = json.load(stream)
    base = record_path.removesuffix(".json")
    argv = [*_interpreter(record["script"]), record["script"]]
    taskset = shutil.which("taskset")
    if record.get("cpus") and taskset is not None:
        argv = [taskset, "-c", ",".join(str(c) for c in record["cpus"]), *argv]
    with open(os.path.join(record["workdir"], f"{record['job_id']}.log"), "ab") as log:
        child = subprocess.Popen(argv, cwd=record["workdir"], stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        _write(f"{base}.started", {"wrapper_pid": os.getpid(), "child_pgid": child.pid,
                                   "host": socket.gethostname(), "started_at": time.time()})
        timed_out = False
        try:
            code = child.wait(timeout=record["minutes"] * 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    _write(f"{base}.exit", {"exit_code": code, "timed_out": timed_out, "ended_at": time.time()})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
