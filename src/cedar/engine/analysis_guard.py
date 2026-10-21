"""Runs one analysis script under guards; started by ``AnalysisRunner`` as ``python -I -B <this file>``.

Before the script starts, this sets CPU, memory and file-size limits and installs an audit
hook that refuses to read anything but the working directory and the read-only roots it is
given (the Python installation), or anything in a denied root, to change anything outside the
working directory, to start processes or to open sockets. Audit hooks cannot be removed once
installed, but they are not a security boundary against code written to evade them; they stop
mistakes and naive misuse. The workbench uses the same guard for Python scripts where the
machine has no bubblewrap.

Standard library only: this file runs in the child process before anything else.
"""

import json
import os
import resource
import runpy
import sys

ROOT = os.path.realpath(os.environ["CEDAR_ANALYSIS_ROOT"])
READ_ROOTS = tuple(os.path.realpath(p) for p in os.environ.get("CEDAR_READ_ROOTS", "").split(os.pathsep) if p)
DENY_ROOTS = tuple(os.path.realpath(p) for p in os.environ.get("CEDAR_DENY_ROOTS", "").split(os.pathsep) if p)
ALLOW_ROOTS = tuple(os.path.realpath(p) for p in os.environ.get("CEDAR_ALLOW_ROOTS", "").split(os.pathsep) if p)
"""Readable although inside a denied root."""
LIMITS = json.loads(os.environ.get("CEDAR_LIMITS", "{}"))
SCRIPT = os.environ.get("CEDAR_SCRIPT", "analysis.py")
"""The script to run, relative to the root directory."""
ARGV = json.loads(os.environ.get("CEDAR_ARGV", "[]"))
CWD = os.environ.get("CEDAR_CWD", ".")
"""Where the script runs, relative to the root directory."""

REFUSED_EVENTS = frozenset({
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
    "os.kill", "os.killpg", "pty.spawn", "socket.__new__", "socket.connect", "socket.bind", "webbrowser.open",
})
CHANGE_EVENTS = frozenset({
    "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.link", "os.symlink",
    "os.truncate", "os.utime", "shutil.rmtree", "shutil.move", "shutil.copytree",
})
LIST_EVENTS = frozenset({"os.listdir", "os.scandir", "glob.glob"})
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _inside(path: str, base: str) -> bool:
    return path == base or path.startswith(base if base.endswith(os.sep) else base + os.sep)


def _real(path) -> str:
    return os.path.realpath(os.fsdecode(path))


def _readable(path: str) -> bool:
    if _inside(path, ROOT) or any(_inside(path, root) for root in ALLOW_ROOTS):
        return True
    return not any(_inside(path, root) for root in DENY_ROOTS) and any(_inside(path, root) for root in READ_ROOTS)


def _hook(event: str, args: tuple) -> None:
    if event in REFUSED_EVENTS:
        raise PermissionError(f"analysis scripts may not use {event}: they cannot start programs or use the network")
    if event == "open":
        path, mode, flags = args
        if path is None or isinstance(path, int):
            return
        target = _real(path)
        writing = bool(mode and any(c in mode for c in "wax+")) or bool((flags or 0) & _WRITE_FLAGS)
        if _inside(target, ROOT) or (not writing and _readable(target)):
            return
        what = "write" if writing else "read"
        raise PermissionError(f"analysis scripts may not {what} {target}: read the staged inputs under inputs/ "
                              "and write only in the working directory")
    if event in CHANGE_EVENTS:
        for arg in args:
            if isinstance(arg, (str, bytes, os.PathLike)) and not _inside(_real(arg), ROOT):
                raise PermissionError(f"analysis scripts may not change {_real(arg)}")
    elif event in LIST_EVENTS:
        arg = args[0] if args and args[0] is not None else "."
        if not isinstance(arg, int) and not _readable(_real(arg)):
            raise PermissionError(f"analysis scripts may not list {_real(arg)}")


def _limit(kind: int, value) -> None:
    if value is None:
        return
    _, hard = resource.getrlimit(kind)
    value = int(value) if hard == resource.RLIM_INFINITY else min(int(value), hard)
    resource.setrlimit(kind, (value, value))


def main() -> None:
    _limit(resource.RLIMIT_CPU, LIMITS.get("cpu_seconds"))
    _limit(resource.RLIMIT_AS, LIMITS.get("memory_bytes"))
    _limit(resource.RLIMIT_FSIZE, LIMITS.get("file_bytes"))
    _limit(resource.RLIMIT_CORE, 0)
    script = os.path.join(ROOT, SCRIPT)
    os.chdir(os.path.join(ROOT, CWD))
    sys.argv = [script, *ARGV]
    sys.addaudithook(_hook)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
