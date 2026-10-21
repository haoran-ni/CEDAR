"""Running the analysis scripts agents write, on the login node, under guards.

An agent writes a Python script to read or compute quantities from finished runs. The
runner:

1. examines the script and refuses one that imports process, network or foreign-code
   modules, calls ``os.system`` and the like, or uses ``eval``/``exec``;
2. stages the requested files of each run, by content hash, read-only under
   ``inputs/<calculation_id>/`` in a fresh working directory;
3. runs the script with this environment's Python and a clean environment (no
   credentials), under CPU, memory and file-size limits, a wall-clock timeout, and an
   audit hook that refuses to read anything but the staged inputs and the Python
   installation, to write outside the working directory, to start processes or to open
   sockets (``analysis_guard.py``);
4. stores what the script printed, the files it wrote and the results it reported in
   ``results.json``, and records the run.

Scripts cannot start programs, so they never run DFT. The guards stop mistakes and naive
misuse; they are not a security boundary against code written to evade them.
"""

import ast
import contextlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import metadata
from pathlib import Path

from pydantic import ValidationError

from sim2real.contracts.analysis import AnalysisInput, AnalysisOutcome, AnalysisResult, AnalysisRun
from sim2real.contracts.base import Actor, ActorKind, ArtifactRef, Quantity, new_id
from sim2real.contracts.evidence import QualityCheck
from sim2real.contracts.events import AnalysisRecorded, EventDraft
from sim2real.errors import RuleViolation
from sim2real.state.ledger import ResearchLedger
from sim2real.state.projection import CalculationRecord
from sim2real.store.artifacts import ArtifactStore

ANALYSIS_RUNNER = Actor(kind=ActorKind.ANALYSIS_RUNNER, id="analysis-runner")
SCRIPT_FILE = "analysis.py"
RESULTS_FILE = "results.json"
OUTPUT_FILE = "output.txt"
GUARD = Path(__file__).with_name("analysis_guard.py")
_PRIVATE = (".home", ".tmp")

FORBIDDEN_MODULES = frozenset({
    "subprocess", "socket", "ssl", "http", "urllib", "urllib3", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib",
    "xmlrpc", "requests", "httpx", "aiohttp", "paramiko", "fabric", "multiprocessing", "concurrent", "asyncio",
    "ctypes", "cffi", "pty", "pexpect", "sh", "plumbum", "webbrowser", "importlib", "runpy", "code", "codeop",
})
"""Modules that start processes, open network connections, load foreign code or import dynamically."""
FORBIDDEN_OS_CALLS = frozenset({
    "system", "popen", "fork", "forkpty", "kill", "killpg", "posix_spawn", "posix_spawnp", "startfile",
    "setuid", "setgid", "seteuid", "setegid", "setreuid", "setregid", "chroot",
})
FORBIDDEN_BUILTINS = frozenset({"eval", "exec", "compile", "__import__", "breakpoint"})
RESULTS_FORMAT = ('{"results": {"<name>": {"value": number, "unit": text, "description": text, "checks": {...}}}, '
                  '"checks": {"<name>": {"passed": true or false, "detail": text}}}')
"""Top-level checks apply to every result; a result's own checks apply to it alone."""
_CHECK_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}")
MAX_SCRIPT_BYTES = 200_000
MAX_PRODUCTS = 200

_SIGNALS = {signal.SIGXCPU: "the CPU time limit", signal.SIGXFSZ: "the file size limit",
            signal.SIGKILL: "a kill signal (often the memory limit)", signal.SIGSEGV: "a crash (segmentation fault)"}


@dataclass(frozen=True)
class AnalysisLimits:
    cpu_seconds: int = 600
    wall_seconds: int = 900
    memory_bytes: int = 4 * 1024 ** 3
    file_bytes: int = 200 * 1024 ** 2
    """Largest file a script may write; also the largest product stored."""
    input_bytes: int = 2 * 1024 ** 3
    """Total size of the files staged for one analysis."""
    shown_chars: int = 12_000
    """How much of the script's printed output goes back to the agent."""


@dataclass(frozen=True)
class AnalysisReport:
    analysis_id: str | None
    """None when the script was refused or could not be staged; nothing ran and nothing was recorded."""
    run: AnalysisRun | None
    problems: tuple[str, ...]
    shown_output: str = ""


def _forbidden_os_call(name: str) -> bool:
    return name in FORBIDDEN_OS_CALLS or name.startswith(("exec", "spawn"))


def examine_script(text: str) -> list[str]:
    """Reasons to refuse a script before it runs; empty when it may run."""
    if len(text.encode()) > MAX_SCRIPT_BYTES:
        return [f"the script is larger than {MAX_SCRIPT_BYTES} bytes"]
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [f"line {exc.lineno}: {exc.msg}"]
    os_names = {alias.asname or "os" for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names if alias.name == "os"}
    problems = []
    for node in ast.walk(tree):
        where = f"line {getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Import):
            problems += [f"{where}: imports {a.name}" for a in node.names if a.name.split(".")[0] in FORBIDDEN_MODULES]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] in FORBIDDEN_MODULES:
                problems.append(f"{where}: imports from {module}")
            elif module == "os":
                problems += [f"{where}: imports os.{a.name}" for a in node.names if _forbidden_os_call(a.name)]
        elif (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in os_names
              and _forbidden_os_call(node.attr)):
            problems.append(f"{where}: uses os.{node.attr}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_BUILTINS:
            problems.append(f"{where}: uses {node.id}")
    return [f"{p} (not allowed in analysis scripts)" for p in problems]


@cache
def environment_description() -> str:
    versions = []
    for package in ("numpy", "scipy", "ase", "matplotlib"):
        try:
            versions.append(f"{package} {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            continue
    return f"{platform.python_implementation()} {platform.python_version()}; " + ", ".join(versions)


def default_read_roots() -> tuple[Path, ...]:
    """What scripts may read besides their inputs: the Python installation, sim2real's own parsers, system files."""
    import sim2real

    candidates = {sys.prefix, sys.base_prefix, sys.exec_prefix, str(Path(sim2real.__file__).parent),
                  "/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev"}
    roots = {Path(p) for p in candidates} | {Path(os.path.realpath(p)) for p in candidates}
    return tuple(sorted(p for p in roots if p.exists()))


def _remove_tree(path: Path) -> None:
    def retry(function, target, _exc):
        os.chmod(os.path.dirname(target), 0o700)
        if os.path.isdir(target) and not os.path.islink(target):
            os.chmod(target, 0o700)
        function(target)

    if path.exists():
        shutil.rmtree(path, onexc=retry)


class AnalysisRunner:
    def __init__(self, ledger: ResearchLedger, artifacts: ArtifactStore, runs_root: str | Path, work_root: str | Path,
                 *, limits: AnalysisLimits = AnalysisLimits(), read_roots: Sequence[Path] | None = None,
                 actor: Actor = ANALYSIS_RUNNER, slots: threading.Semaphore | None = None) -> None:
        """``slots``, shared between runners, limits how many scripts run at once on the login node."""
        self._ledger = ledger
        self._artifacts = artifacts
        self._runs_root = Path(runs_root)
        self._work_root = Path(work_root)
        self.limits = limits
        self._read_roots = tuple(read_roots) if read_roots is not None else default_read_roots()
        self._actor = actor
        self._slots = slots

    def run(self, investigation_id: str, hypothesis_id: str, *, purpose: str, script: str,
            inputs: Mapping[str, Sequence[str] | None]) -> AnalysisReport:
        """Run ``script`` on the named files of each calculation (default: its stored outputs and inputs)."""
        problems = examine_script(script)
        if problems:
            return AnalysisReport(None, None, tuple(problems))
        state = self._ledger.load(investigation_id)
        if hypothesis_id not in state.hypotheses:
            raise RuleViolation(f"unknown hypothesis {hypothesis_id}")
        if not inputs:
            return AnalysisReport(None, None, ("name at least one calculation whose files the script reads",))
        plan: dict[str, tuple[CalculationRecord, list[str]]] = {}
        for calculation_id, names in inputs.items():
            calc = state.calculations.get(calculation_id)
            if calc is None:
                problems.append(f"unknown calculation {calculation_id}")
            elif hypothesis_id not in calc.hypothesis_ids:
                problems.append(f"{calculation_id} is not one of {hypothesis_id}'s calculations")
            elif calc.run_check is None:
                problems.append(f"the run of {calculation_id} has not been checked yet ({calc.state.value})")
            else:
                plan[calculation_id] = (calc, list(names) if names else
                                        sorted({*calc.run_check.outputs, *calc.examination.files}))
        if problems:
            return AnalysisReport(None, None, tuple(problems))

        analysis_id = new_id("an")
        stage = self._work_root / analysis_id
        (stage / "inputs").mkdir(parents=True)
        try:
            staged, problems = self._stage(stage, plan)
            if problems:
                return AnalysisReport(None, None, tuple(problems))
            script_ref = self._artifacts.put_bytes(script.encode(), label=SCRIPT_FILE, media_type="text/x-python")
            (stage / SCRIPT_FILE).write_text(script)
            with self._slots if self._slots is not None else contextlib.nullcontext():
                outcome, exit_code, seconds = self._execute(stage)
            if exit_code is not None and exit_code < 0:
                cause = _SIGNALS.get(-exit_code, f"signal {-exit_code}")
                problems.append(f"the script was stopped by {cause}")
            elif outcome is AnalysisOutcome.TIMED_OUT:
                problems.append(f"the script did not finish within {self.limits.wall_seconds} s and was stopped")
            output = stage / OUTPUT_FILE
            output_ref = self._artifacts.put_file(output, media_type="text/plain") if output.is_file() else None
            products, product_problems = self._products(stage)
            results, checks, result_problems = (self._results(stage) if outcome is AnalysisOutcome.COMPLETED
                                                else ((), (), []))
            problems += product_problems + result_problems
            run = AnalysisRun(
                purpose=purpose, script=script_ref, inputs=tuple(staged), environment=environment_description(),
                outcome=outcome, exit_code=exit_code, duration_seconds=round(seconds, 3), output=output_ref,
                products=products, results=results, checks=checks, problems=tuple(problems))
            self._ledger.record(investigation_id, [EventDraft(actor=self._actor, payload=AnalysisRecorded(
                analysis_id=analysis_id, hypothesis_id=hypothesis_id, run=run))])
            return AnalysisReport(analysis_id, run, tuple(problems), self._shown(output))
        finally:
            _remove_tree(stage)

    # -- staging --------------------------------------------------------------------------

    def _stage(self, stage: Path, plan) -> tuple[list[AnalysisInput], list[str]]:
        staged, problems, total = [], [], 0
        for calculation_id, (calc, names) in plan.items():
            files: dict[str, ArtifactRef] = {}
            for name in names:
                ref, problem = self._input_file(calc, name)
                if problem:
                    problems.append(problem)
                    continue
                total += ref.size_bytes
                if total > self.limits.input_bytes:
                    return [], [f"the staged files exceed {self.limits.input_bytes} bytes; stage fewer or smaller "
                                "files"]
                target = stage / "inputs" / calculation_id / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self._artifacts.resolve(ref), target)
                target.chmod(0o444)
                files[name] = ref
            if files:
                staged.append(AnalysisInput(calculation_id=calculation_id, files=files))
        return staged, problems

    def _input_file(self, calc: CalculationRecord, name: str) -> tuple[ArtifactRef | None, str | None]:
        stored = {**calc.examination.files, **calc.run_check.outputs}
        if name in stored:
            return stored[name], None
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            return None, f"{name!r}: give a path relative to the run directory of {calc.calculation_id}"
        run_dir = (self._runs_root / calc.workdir).resolve()
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            return None, f"{name} is not a file in the run directory of {calc.calculation_id}"
        return self._artifacts.put_file(path), None

    # -- running --------------------------------------------------------------------------

    def _execute(self, stage: Path) -> tuple[AnalysisOutcome, int | None, float]:
        home, tmp = stage / ".home", stage / ".tmp"
        home.mkdir()
        tmp.mkdir()
        limits = {"cpu_seconds": self.limits.cpu_seconds, "memory_bytes": self.limits.memory_bytes,
                  "file_bytes": self.limits.file_bytes}
        env = {
            "PATH": "/usr/bin:/bin", "HOME": str(home), "TMPDIR": str(tmp), "MPLCONFIGDIR": str(home),
            "MPLBACKEND": "Agg", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            "S2R_ANALYSIS_ROOT": str(stage), "S2R_READ_ROOTS": os.pathsep.join(str(p) for p in self._read_roots),
            "S2R_LIMITS": json.dumps(limits),
        }
        started = time.monotonic()
        with open(stage / OUTPUT_FILE, "wb") as output:
            process = subprocess.Popen([sys.executable, "-I", "-B", str(GUARD)], cwd=stage, env=env,
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=self.limits.wall_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return AnalysisOutcome.TIMED_OUT, None, time.monotonic() - started
        outcome = AnalysisOutcome.COMPLETED if code == 0 else AnalysisOutcome.FAILED
        return outcome, code, time.monotonic() - started

    # -- collecting -----------------------------------------------------------------------

    def _products(self, stage: Path) -> tuple[dict[str, ArtifactRef], list[str]]:
        products, problems = {}, []
        skip = {stage / SCRIPT_FILE, stage / OUTPUT_FILE}
        for path in sorted(stage.rglob("*")):
            relative = path.relative_to(stage)
            if relative.parts[0] in ("inputs", *_PRIVATE) or path in skip or path.is_symlink() or not path.is_file():
                continue
            if len(products) >= MAX_PRODUCTS:
                problems.append(f"more than {MAX_PRODUCTS} files written; the rest were not stored")
                break
            products[relative.as_posix()] = self._artifacts.put_file(path)
        return products, problems

    @classmethod
    def _results(cls, stage: Path) -> tuple[tuple[AnalysisResult, ...], tuple[QualityCheck, ...], list[str]]:
        """The results and checks the script reported in results.json, and why any were not taken."""
        path = stage / RESULTS_FILE
        if not path.is_file():
            return (), (), [f"the script wrote no {RESULTS_FILE}, so no result can be recorded from this run"]
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return (), (), [f"{RESULTS_FILE} is not valid JSON: {exc}"]
        if not isinstance(data, dict) or not data or set(data) - {"results", "checks"}:
            return (), (), [f"{RESULTS_FILE} must be {RESULTS_FORMAT}"]
        problems: list[str] = []
        checks = cls._checks(data.get("checks", {}), "checks", problems)
        if checks is None:  # a malformed check could be a failed one; nothing is taken without it
            return (), (), problems + ["no result was taken because a check that applies to all of them is malformed"]
        results = []
        entries = data.get("results", {})
        if not isinstance(entries, dict):
            return (), checks, problems + [f"{RESULTS_FILE}: \"results\" must map result names to results"]
        for name, entry in entries.items():
            try:
                if not isinstance(entry, dict) or set(entry) - {"value", "unit", "description", "reference", "checks"}:
                    raise ValueError("expected {\"value\": number, \"unit\": text}, optionally \"description\", "
                                     "\"reference\" and \"checks\"")
                value = entry.get("value")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("value must be a finite number")
                own = cls._checks(entry.get("checks", {}), f"checks of {name!r}", problems)
                if own is None:
                    raise ValueError("one of its checks is malformed")
                results.append(AnalysisResult(
                    name=name, description=entry.get("description"), checks=own,
                    value=Quantity(value=float(value), unit=entry.get("unit"), reference=entry.get("reference"))))
            except ValidationError as exc:
                error = exc.errors()[0]
                where = ".".join(str(part) for part in error["loc"])
                problems.append(f"{RESULTS_FILE}: result {name!r} not taken: {where}: {error['msg']}")
            except (ValueError, TypeError) as exc:
                problems.append(f"{RESULTS_FILE}: result {name!r} not taken: {exc}")
        return tuple(results), checks, problems

    @staticmethod
    def _checks(entries, where: str, problems: list[str]) -> tuple[QualityCheck, ...] | None:
        """The checks in ``entries``, or None if any is malformed (its problem is added to ``problems``)."""
        if not isinstance(entries, dict):
            problems.append(f"{RESULTS_FILE}: {where} must map check names to {{\"passed\": true or false, "
                            "\"detail\": text}}")
            return None
        checks, malformed = [], False
        for name, entry in entries.items():
            if (not isinstance(name, str) or not _CHECK_NAME.fullmatch(name) or not isinstance(entry, dict)
                    or not isinstance(entry.get("passed"), bool) or set(entry) - {"passed", "detail"}
                    or not isinstance(entry.get("detail", ""), str)):
                problems.append(f"{RESULTS_FILE}: {where}: check {name!r} is malformed; a check is named with up to "
                                "64 letters, digits, '_', '.', ':' or '-', and is {\"passed\": true or false, "
                                "\"detail\": text}")
                malformed = True
                continue
            checks.append(QualityCheck(name=name, passed=entry["passed"], detail=entry.get("detail") or None))
        return None if malformed else tuple(checks)

    def _shown(self, output: Path) -> str:
        if not output.is_file():
            return ""
        text = output.read_text(errors="replace")
        limit = self.limits.shown_chars
        if len(text) <= limit:
            return text
        omitted = len(text) - limit
        return f"{text[:limit // 2]}\n…[{omitted} characters omitted]…\n{text[-limit // 2:]}"
