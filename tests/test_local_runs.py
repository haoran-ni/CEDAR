"""Calculations run locally, on the interactive node CEDAR runs on, inside the researcher's allocation.

The host is a TEST DOUBLE (an interactive node with a made-up allocation, or a login node); so are the Slurm
steps (``Steps``: a process pinned to the CPUs a step would get, started when the test says). The FHI-aims binary
is a SYNTHETIC shell script that prints a synthetic aims.out. The local runs really start, as detached
processes, and really end.
"""

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from factories import (
    AGENT,
    ENGINE,
    agent_for,
    authorize,
    examine,
    move,
    set_hypothesis_status,
    start_agent,
    submit,
    task,
)
from sites import aims_output
from test_orchestration import make_system
from test_swarm import INV, investigation
from worlds import World

from cedar.contracts import (
    CalculationState,
    EvidenceKind,
    EvidenceQuality,
    ExaminationVerdict,
    HypothesisStatus,
    Observable,
    OutputReading,
    PermissionAction,
    Quantity,
    SchedulerNoteRecorded,
    SchedulerState,
)
from cedar.contracts.evidence import FileCitation
from cedar.dashboard import agent_rows, job_status, work_summary
from cedar.dispatch import HOST, LOCK, Dispatcher, dispatcher_host, stop_dispatcher
from cedar.engine import JobManager, JobMonitor, ProtocolEngine, ResultValidator
from cedar.engine.workbench import Workbench
from cedar.errors import RuleViolation
from cedar.hpc.local import RECORDS, LocalBackend, SlurmSteps
from cedar.hpc.local_runner import main as run_locally
from cedar.hpc.node import CorePool, ExecutionHost
from cedar.hpc.slurm import JobRefused, SchedulerError, SchedulerObservation
from cedar.protocols import FhiAimsRun, FhiAimsRunSlurmOnly, ProtocolRegistry
from cedar.summaries import calculation_summary, short_id
from cedar.workspace import Workspace, services_for

ENERGY = -2345.678901234
TEMPLATE = """#!/bin/bash
#SBATCH --partition={{partition}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks={{ntasks}}
#SBATCH --time={{walltime}}

export OMP_NUM_THREADS=1
{{binary}} {{ntasks}} > {{output}}
"""


def interactive(cores=8, minutes=120) -> ExecutionHost:
    """An allocation of 8 cores by default: 4 for local DFT, 4 kept for the agents' commands."""
    return ExecutionHost(socket.gethostname(), allocation="777001", cores=cores,
                         ends_at=datetime.now(timezone.utc) + timedelta(minutes=minutes),
                         reason="TEST DOUBLE: an allocation that does not exist")


LOGIN = ExecutionHost("login1", cores=8, reason="not inside a Slurm allocation")
EIGHT_CPUS = pytest.mark.skipif(len(os.sched_getaffinity(0)) < 8, reason="pins test steps to CPUs 0-7")


class Steps:
    """TEST DOUBLE for Slurm steps: a step starts ``delay`` seconds after its launcher, pinned to ``cpus`` (where
    Slurm put it) or else to the CPUs CEDAR booked; meanwhile its launcher prints ``said``, as srun would. With
    ``fail`` the step never starts: its launcher prints that and exits 1."""

    def __init__(self) -> None:
        self.delay, self.cpus, self.said, self.fail = 0.0, None, None, None

    def unavailable(self) -> str | None:
        return None

    def launch(self, record) -> list[str]:
        if self.fail:
            return [sys.executable, "-c", f"print({self.fail!r}, flush=True); raise SystemExit(1)"]
        cpus = sorted(self.cpus or record["cpus"])
        return [sys.executable, "-c",
                "import os, sys, time\n"
                f"if {self.said!r}: print({self.said!r}, flush=True)\n"
                f"time.sleep({self.delay})\n"
                f"cpus = [c for c in {cpus!r} if c in os.sched_getaffinity(0)]\n"
                "if cpus: os.sched_setaffinity(0, cpus)\n"
                "os.execv(sys.argv[1], sys.argv[1:])\n"]

    def stop(self) -> list[str]:
        return []

    def environment(self, environ) -> dict[str, str]:
        return dict(environ)


class Local:
    """A World whose site's code is a SYNTHETIC script, with a local backend on a stand-in host."""

    def __init__(self, ledger, store, artifacts, site, tmp_path, host=None, cpus=tuple(range(8))):
        binary = Path(site.codes["fhi-aims"].binary)
        self.fixture = binary.with_name("aims.out.fixture")
        self.fixture.write_text(aims_output(energy=ENERGY))
        self.binary = binary
        self.code(sleep=0)
        site = site.model_copy(update={"job_template": TEMPLATE})
        self.world = World(ledger, store, artifacts, site)
        self.host = host or interactive()
        self.ended_allocations: set[str] = set()
        w = self.world
        w.registry = ProtocolRegistry([FhiAimsRun(site, host=lambda: self.host), FhiAimsRunSlurmOnly(site)])
        w.engine = ProtocolEngine(ledger, w.registry, artifacts, w.runs)
        self.pool = CorePool(list(cpus))
        self.steps = Steps()
        self.backend = self.backend_again()
        self.manager = JobManager(ledger, self.backend, w.runs, execution="local")
        self.monitor = JobMonitor(ledger, self.backend, execution="local")
        w.validator = ResultValidator(ledger, w.registry, artifacts, w.scheduler, w.runs, local_backend=self.backend)

    def backend_again(self, pool=None):
        """A local backend as a dispatcher (re)started on this host has it."""
        return LocalBackend(self.world.runs, host=lambda: self.host, pool=pool or self.pool, steps=self.steps,
                            allocation_state=lambda a: a not in self.ended_allocations)

    def code(self, *, sleep: float, fail: bool = False):
        """The SYNTHETIC binary: it waits, then prints the fixture (and its task count) or fails."""
        end = "exit 3" if fail else f"echo 'tasks' $1; cat {self.fixture}"
        self.binary.write_text(f"#!/bin/sh\nsleep {sleep}\n{end}\n")

    def request(self, **overrides):
        return self.world.request(version="0.2", **overrides)

    def local_task(self, k_grid=(4, 4, 1), **budget):
        task = self.world.task(version="0.2", numerical={"k_grid": list(k_grid)}).model_copy(
            update={"execution": "local"})
        if budget:
            task = task.model_copy(update={"budget": task.budget.model_copy(update=budget)})
        return task

    def request_local(self, k_grid=(4, 4, 1), **budget) -> str:
        w = self.world
        cid = w.ledger.request_calculation(w.inv, AGENT, w.hyp, self.local_task(k_grid, **budget),
                                           "test").calculation_id
        w.engine.examine(w.inv, cid)
        return cid

    def wait_until(self, cid, states, seconds=30.0):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.monitor.poll(self.world.inv)
            if self.world.calc(cid).state in states:
                return self.world.calc(cid)
            time.sleep(0.2)
        raise AssertionError(f"{cid} is still {self.world.calc(cid).state}")


@pytest.fixture
def local(ledger, store, artifacts, site, tmp_path):
    return Local(ledger, store, artifacts, site, tmp_path)


def findings(world, cid):
    return {f.code: f for f in world.calc(cid).examination.findings}


# -- examination -----------------------------------------------------------------------------------

def test_on_a_login_node_a_local_run_is_refused_at_once(local):
    local.host = LOGIN
    cid = local.request_local()
    calc = local.world.calc(cid)
    assert calc.state is CalculationState.REJECTED
    message = findings(local.world, cid)["local_on_login_node"].message
    assert "DFT cannot run locally on login nodes" in message and "login1" in message
    assert "not possible here" in local.world.registry.get(calc.task.protocol).describe()


def test_version_01_runs_through_slurm_only(local):
    w = local.world
    task = w.task(version="0.1").model_copy(update={"execution": "local"})
    cid = w.ledger.request_calculation(w.inv, AGENT, w.hyp, task, "test").calculation_id
    w.engine.examine(w.inv, cid)
    assert "no_local_runs" in findings(w, cid) and w.calc(cid).state is CalculationState.REJECTED
    assert findings(w, cid)["no_local_runs"].message.endswith("request fhiaims.run 0.4 for a local run")


def test_on_an_interactive_node_the_template_runs_without_its_slurm_lines(local):
    cid = local.request_local(max_walltime_hours=1)
    calc = local.world.calc(cid)
    assert calc.examination.verdict is ExaminationVerdict.APPROVED, calc.examination.findings
    script = local.world.artifacts.read(calc.examination.files["local.sh"]).decode()
    assert "#SBATCH" not in script and "#CEDAR --ntasks=4 --time=01:00:00" in script
    assert f"{local.binary} 4 > aims.out" in script and "job.sh" not in calc.examination.files
    assert calc.examination.execution_files == ("local.sh",) and calc.examination.requested_core_hours == 4


@pytest.mark.parametrize("budget, code", [({"cores": 8}, "too_many_tasks"),
                                          ({"max_walltime_hours": 3}, "past_the_allocation")])
def test_a_local_run_must_fit_the_allocation(local, budget, code):
    w = local.world
    task = local.local_task(**budget)
    script = (f"#!/bin/bash\n#CEDAR --ntasks={budget.get('cores', 4)} "
              f"--time={int(budget.get('max_walltime_hours', 1))}:00:00\n\nexport OMP_NUM_THREADS=1\n"
              f"{local.binary} {budget.get('cores', 4)} > aims.out\n")
    task = task.model_copy(update={"authored_files": {"local.sh": w.artifacts.put_bytes(script.encode())}})
    cid = w.ledger.request_calculation(w.inv, AGENT, w.hyp, task, "test").calculation_id
    w.engine.examine(w.inv, cid)
    assert code in findings(w, cid) and w.calc(cid).state is CalculationState.REJECTED


def test_a_local_runs_budget_findings_name_its_script(local):
    w = local.world
    script = (f"#!/bin/bash\n#CEDAR --ntasks=4 --time=01:00:00\n\nexport OMP_NUM_THREADS=1\n"
              f"{local.binary} 4 > aims.out\n")
    task = local.local_task(max_core_hours=1).model_copy(
        update={"authored_files": {"local.sh": w.artifacts.put_bytes(script.encode())}})
    cid = w.ledger.request_calculation(w.inv, AGENT, w.hyp, task, "test").calculation_id
    w.engine.examine(w.inv, cid)
    over = findings(w, cid)["over_budget"]
    assert over.file == "local.sh" and over.message == "4 core-hours requested; the task allows 1"


def test_a_local_run_that_no_longer_fits_once_approved_fails_at_once_with_the_reason(local):
    w = local.world
    cid = local.request_local(max_walltime_hours=1)
    authorize(w.ledger, w.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")
    local.host = interactive(minutes=30)  # the allocation ran on while the inputs were reviewed and approved
    with pytest.raises(JobRefused) as refused:
        local.manager.submit(w.inv, cid)
    assert refused.value.final  # its time left only shrinks: no point trying again
    calc = w.calc(cid)
    assert calc.state is CalculationState.SUBMISSION_FAILED and calc.submission_refusals == []
    assert calc.end_reason.startswith("60 minutes would run past the end of allocation 777001")
    assert not list((w.runs / RECORDS).glob("local-*.json"))
    assert job_status(w.state(), calc).endswith("refused by the scheduler")


def test_a_local_run_takes_no_slurm_script(local):
    w = local.world
    task = local.local_task()
    task = task.model_copy(update={"authored_files": {"job.sh": w.artifacts.put_bytes(b"#!/bin/bash\n")}})
    cid = w.ledger.request_calculation(w.inv, AGENT, w.hyp, task, "test").calculation_id
    w.engine.examine(w.inv, cid)
    assert findings(w, cid)["wrong_script"].message == "a local run takes local.sh, not job.sh"


# -- running ---------------------------------------------------------------------------------------

def test_an_approved_local_run_starts_ends_and_gives_validated_evidence(local):
    w = local.world
    cid = local.request_local(max_walltime_hours=1)
    with pytest.raises(RuleViolation, match="needs a user authorization"):
        local.manager.submit(w.inv, cid)
    authorize(w.ledger, w.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")
    job = local.manager.submit(w.inv, cid)
    assert job.startswith("local-") and w.calc(cid).state is CalculationState.SUBMITTED
    calc = local.wait_until(cid, {CalculationState.SCHEDULER_TERMINAL})
    assert calc.scheduler_state is SchedulerState.COMPLETED
    w.validator.check_run(w.inv, cid)
    check = w.calc(cid).run_check
    assert check.healthy and set(check.outputs) == {"aims.out", f"{job}.log"}
    assert check.method["execution"] == "local" and check.method["allocation"] == "777001"
    out = w.artifacts.read(check.outputs["aims.out"]).decode()
    assert out.startswith("tasks 4\n")
    lines = out.splitlines()
    line = next(i for i, text in enumerate(lines, start=1) if "Total energy of the DFT" in text)
    outcome = w.validator.record_reading(w.inv, cid, OutputReading(
        reader=AGENT, value=Quantity(value=ENERGY, unit="eV"), summary="local run",
        citations=(FileCitation(file="aims.out", line_start=line, line_end=line, quote=lines[line - 1].strip()),)),
        Observable(name="total energy", definition="final total energy", unit="eV"))
    ev = outcome.recorded.events[0].payload.evidence
    assert ev.kind is EvidenceKind.DIRECT_COMPUTATION and ev.quality is EvidenceQuality.VALIDATED
    assert ev.method["execution"] == "local"


def test_local_runs_wait_for_free_cores(ledger, store, artifacts, site, tmp_path):
    local = Local(ledger, store, artifacts, site, tmp_path, host=interactive(cores=5), cpus=(0, 1, 2, 3, 4))
    local.code(sleep=2)
    w = local.world
    first, second = local.request_local(max_core_hours=0.5), local.request_local((6, 6, 1), max_core_hours=0.5)
    for cid in (first, second):
        authorize(w.ledger, w.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")
        local.manager.submit(w.inv, cid)
    local.wait_until(first, {CalculationState.RUNNING})
    assert w.calc(second).state is CalculationState.QUEUED
    assert w.calc(second).scheduler_note == "waiting for 1 free core"
    assert w.calc(first).scheduler_note == "running on CPUs 0"
    local.wait_until(first, {CalculationState.SCHEDULER_TERMINAL})
    assert local.wait_until(second, {CalculationState.SCHEDULER_TERMINAL}).scheduler_state is SchedulerState.COMPLETED
    assert w.calc(first).scheduler_note is None  # it COMPLETED, which says it all


@EIGHT_CPUS
def test_a_run_is_running_only_once_slurm_started_its_step_and_the_panel_says_what_it_waits_for(local, monkeypatch):
    monkeypatch.setattr("cedar.hpc.local.START_SECONDS", 1.0)
    local.code(sleep=1)
    local.steps.delay, local.steps.said = 4.0, "srun: Job 777001 step creation temporarily disabled, retrying " \
                                               "(Requested nodes are busy)"
    w = local.world
    cid = local.request_local(max_walltime_hours=1)
    authorize(w.ledger, w.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")
    local.manager.submit(w.inv, cid)
    set_hypothesis_status(w.ledger, w.inv, w.hyp, HypothesisStatus.WAITING_FOR_JOB, awaiting=(cid,))
    local.wait_until(cid, {CalculationState.QUEUED})
    assert w.calc(cid).scheduler_note == "starting its Slurm step"
    deadline = time.monotonic() + 10
    while "Requested nodes" not in (w.calc(cid).scheduler_note or "") and time.monotonic() < deadline:
        local.monitor.poll(w.inv)
        time.sleep(0.2)
    calc = w.calc(cid)
    assert calc.state is CalculationState.QUEUED  # not RUNNING: Slurm has not started the step
    assert calc.scheduler_note == "waiting for Slurm to start its step (Requested nodes are busy)"
    state = w.state()
    assert job_status(state, calc).endswith("PENDING (waiting for Slurm to start its step (Requested nodes are busy))")
    row = next(r for r in agent_rows(state, w.store.read_stream(w.inv), [], datetime.now(timezone.utc))
               if r.hypothesis_id == w.hyp)
    assert (row.status, row.task) == ("waiting for jobs", f"calculations {short_id(cid)} (waiting for Slurm to start "
                                                          "its step (Requested nodes are busy))")
    assert "(local: 4 waiting)" in work_summary(state)
    calc = local.wait_until(cid, {CalculationState.RUNNING})
    assert calc.scheduler_note == "running on CPUs 0-3" and "(local: 4 cores running)" in work_summary(w.state())
    row = next(r for r in agent_rows(w.state(), w.store.read_stream(w.inv), [], datetime.now(timezone.utc))
               if r.hypothesis_id == w.hyp)
    assert row.task == f"calculations {short_id(cid)} (running on CPUs 0-3)"
    assert "running on CPUs 0-3" in calculation_summary(w.state(), calc)
    assert local.wait_until(cid, {CalculationState.SCHEDULER_TERMINAL}).scheduler_state is SchedulerState.COMPLETED
    notes = [e.payload.note for e in w.store.read_stream(w.inv) if isinstance(e.payload, SchedulerNoteRecorded)]
    assert notes == ["waiting for Slurm to start its step (Requested nodes are busy)"]


def test_a_step_slurm_would_not_start_fails_with_what_srun_said(local):
    local.steps.fail = ("srun: error: Unable to create step for job 777001: Invalid generic resource (gres) "
                        "specification")
    w = local.world
    cid = local.request_local(max_walltime_hours=1)
    authorize(w.ledger, w.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")
    local.manager.submit(w.inv, cid)
    calc = local.wait_until(cid, {CalculationState.SCHEDULER_TERMINAL})
    assert calc.scheduler_state is SchedulerState.FAILED
    assert calc.scheduler_note == ("its Slurm step did not start: error: Unable to create step for job 777001: "
                                   "Invalid generic resource (gres) specification")


@EIGHT_CPUS
def test_the_pool_holds_the_cpus_slurm_gave_the_step_not_the_ones_booked(local):
    local.code(sleep=3)
    local.steps.cpus = [4, 5, 6, 7]  # CEDAR booked 0-3; Slurm chose otherwise
    w = local.world
    cid = local.request_local(max_walltime_hours=1)
    authorize(w.ledger, w.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")
    local.manager.submit(w.inv, cid)
    assert local.wait_until(cid, {CalculationState.RUNNING}).scheduler_note == "running on CPUs 4-7"
    command = local.pool.acquire(4, timeout=0.1)  # the agents' commands get what the step does not use
    assert sorted(command) == [0, 1, 2, 3]
    local.pool.release(command)
    local.wait_until(cid, {CalculationState.SCHEDULER_TERMINAL})
    assert local.pool.take(8) is not None  # all given back


@EIGHT_CPUS
def test_a_restarted_dispatcher_keeps_exactly_the_cpus_its_runs_use(local):
    local.code(sleep=4)
    local.steps.cpus = [2, 3, 4, 5]
    w = local.world
    cid = local.request_local(max_walltime_hours=1)
    authorize(w.ledger, w.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")
    local.manager.submit(w.inv, cid)
    local.wait_until(cid, {CalculationState.RUNNING})
    pool = CorePool(list(range(8)))
    local.backend_again(pool)  # as a dispatcher started after the first one stopped
    assert sorted(pool.take(4)) == [0, 1, 6, 7]  # not 0-3, which would share 2 and 3 with the running step


def test_a_run_that_waited_until_its_limit_no_longer_fits_fails_without_starting(ledger, store, artifacts, site,
                                                                                    tmp_path):
    local = Local(ledger, store, artifacts, site, tmp_path, host=interactive(cores=5), cpus=(0, 1, 2, 3, 4))
    local.code(sleep=300)
    w = local.world
    first, second = local.request_local(max_walltime_hours=1), local.request_local((6, 6, 1), max_walltime_hours=1)
    for cid in (first, second):
        authorize(w.ledger, w.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")
        local.manager.submit(w.inv, cid)
    local.wait_until(first, {CalculationState.RUNNING})
    local.host = interactive(cores=5, minutes=50)  # time passed: the allocation ends in 50 minutes
    calc = local.wait_until(second, {CalculationState.SCHEDULER_TERMINAL})
    assert calc.scheduler_state is SchedulerState.FAILED
    assert calc.scheduler_note.startswith("it waited for free cores until its 60-minute limit no longer fit in "
                                          "allocation 777001") and calc.scheduler_note.endswith("it never started")
    assert not (w.runs / RECORDS / f"{calc.scheduler_job_id}.started").exists()
    authorize(w.ledger, w.inv, PermissionAction.CANCEL, first, intervention_id=f"cancel-{first}")
    local.manager.cancel(w.inv, first, "test")


def test_the_script_runs_without_the_steps_slurm_variables(tmp_path):
    record = tmp_path / "local-x.json"
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\nenv | grep '^SLURM_' > slurm.env\nexit 0\n")
    launch = [sys.executable, "-c", "import os, sys\nos.environ.update(SLURM_JOB_ID='777001', SLURM_STEP_ID='7', "
                                    "SLURM_CPU_BIND='quiet,mask_cpu:0xFF')\nos.execv(sys.argv[1], sys.argv[1:])\n"]
    record.write_text(json.dumps({"job_id": "local-x", "workdir": str(tmp_path), "script": str(script),
                                  "minutes": 5, "cpus": [0], "launch": launch, "stop": []}))
    run_locally(str(record))
    assert json.loads((tmp_path / "local-x.exit").read_text())["exit_code"] == 0
    assert (tmp_path / "slurm.env").read_text() == ""
    assert json.loads((tmp_path / "local-x.inside").read_text())["step"] == "777001.7"


def test_the_time_limit_counts_from_when_the_step_started(tmp_path):
    record = tmp_path / "local-x.json"
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\nsleep 300\n")
    launch = [sys.executable, "-c", "import os, sys, time\ntime.sleep(2)\nos.execv(sys.argv[1], sys.argv[1:])\n"]
    record.write_text(json.dumps({"job_id": "local-x", "workdir": str(tmp_path), "script": str(script),
                                  "minutes": 0.05, "cpus": [0], "launch": launch, "stop": []}))
    run_locally(str(record))
    started = json.loads((tmp_path / "local-x.started").read_text())["started_at"]
    inside = json.loads((tmp_path / "local-x.inside").read_text())["started_at"]
    ended = json.loads((tmp_path / "local-x.exit").read_text())
    assert ended["timed_out"] and inside - started >= 1.9 and ended["ended_at"] - inside >= 2.9


def test_each_local_run_is_a_slurm_step_of_its_own_size_on_this_node(monkeypatch):
    record = {"allocation": "674593", "node": "trace29", "ntasks": 60, "minutes": 240, "job_name": "cedar-ab12"}
    command = SlurmSteps().launch(record)
    assert command == ["srun", "--jobid=674593", "--nodes=1", "--nodelist=trace29", "--exact", "--gres=none",
                       "--mpi=none", "--ntasks=1", "--cpus-per-task=60", "--time=04:05:00", "--job-name=cedar-ab12"]
    shell = {"SLURM_JOB_ID": "674593", "SLURM_STEP_ID": "0", "SLURM_CPU_BIND": "quiet,mask_cpu:0xFFFF",
             "SLURM_CONF": "/etc/slurm/slurm.conf", "PATH": "/usr/bin"}
    assert SlurmSteps().environment(shell) == {"SLURM_CONF": "/etc/slurm/slurm.conf", "PATH": "/usr/bin"}
    monkeypatch.setattr("cedar.hpc.local.shutil.which", lambda name: None)
    assert "srun is not on this machine's PATH" in SlurmSteps().unavailable()
    backend = LocalBackend("/nowhere", host=interactive, steps=SlurmSteps())
    assert "srun is not on this machine's PATH" in backend.unavailable(interactive())


def test_a_cancelled_local_run_stops(local):
    local.code(sleep=300)
    w = local.world
    cid = local.request_local()
    authorize(w.ledger, w.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")
    job = local.manager.submit(w.inv, cid)
    local.wait_until(cid, {CalculationState.RUNNING})
    deadline = time.monotonic() + 10
    while not (w.runs / RECORDS / f"{job}.started").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    authorize(w.ledger, w.inv, PermissionAction.CANCEL, cid, intervention_id=f"cancel-{cid}")
    local.manager.cancel(w.inv, cid, "test")
    assert w.calc(cid).state is CalculationState.CANCELLED
    deadline = time.monotonic() + 10
    while local.backend.query([job])[job].state is not SchedulerState.CANCELLED and time.monotonic() < deadline:
        time.sleep(0.2)
    assert local.backend.query([job])[job].state is SchedulerState.CANCELLED


def test_the_runner_stops_a_run_at_its_time_limit(tmp_path):
    record = tmp_path / "local-x.json"
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\nsleep 300\n")
    record.write_text(json.dumps({"job_id": "local-x", "workdir": str(tmp_path), "script": str(script),
                                  "minutes": 0.02, "cpus": None}))
    started = time.monotonic()
    run_locally(str(record))
    ended = json.loads((tmp_path / "local-x.exit").read_text())
    assert ended["timed_out"] and ended["exit_code"] != 0 and time.monotonic() - started < 40


def test_a_restarted_dispatcher_finds_its_local_runs_without_starting_them_again(local):
    local.code(sleep=2)
    w = local.world
    cid = local.request_local()
    authorize(w.ledger, w.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")

    class Crashing:  # the dispatcher dies after the run started, before the job id was recorded
        def __init__(self, backend):
            self.backend = backend

        def __getattr__(self, name):
            return getattr(self.backend, name)

        def submit(self, *args):
            self.backend.submit(*args)
            raise KeyboardInterrupt

    crashing = JobManager(w.ledger, Crashing(local.backend), w.runs, execution="local")
    with pytest.raises(KeyboardInterrupt):
        crashing.submit(w.inv, cid)
    assert w.calc(cid).state is CalculationState.SUBMISSION_PENDING
    deadline = time.monotonic() + 10
    while not list((w.runs / RECORDS).glob("*.started")) and time.monotonic() < deadline:
        time.sleep(0.1)
    pool = CorePool(list(range(8)))
    again = local.backend_again(pool)
    assert pool.take(5) is None  # the running run keeps its 4 cores
    settled = JobManager(w.ledger, again, w.runs, execution="local").reconcile(w.inv)
    assert settled[cid].startswith("local-") and w.calc(cid).state is CalculationState.SUBMITTED
    assert len(list((w.runs / RECORDS).glob("local-*.json"))) == 1
    monitor = JobMonitor(w.ledger, again, execution="local")
    deadline = time.monotonic() + 30
    while w.calc(cid).state is not CalculationState.SCHEDULER_TERMINAL and time.monotonic() < deadline:
        monitor.poll(w.inv)
        time.sleep(0.2)
    assert w.calc(cid).scheduler_state is SchedulerState.COMPLETED


def test_a_run_whose_allocation_ended_failed(local):
    records = local.world.runs / RECORDS
    records.mkdir(parents=True)
    base = {"job_name": "cedar-x", "workdir": "/nowhere", "script": "/nowhere/local.sh", "host": "trace19",
            "allocation": "555", "ntasks": 2, "minutes": 60, "submitted_at": time.time(), "cpus": [0, 1],
            "launched_at": time.time()}
    for job in ("local-a", "local-b"):
        (records / f"{job}.json").write_text(json.dumps({**base, "job_id": job,
                                                         "allocation": "555" if job == "local-a" else "556"}))
    local.ended_allocations.add("555")
    seen = local.backend.query(["local-a", "local-b"])
    assert seen["local-a"].state is SchedulerState.FAILED and "allocation 555 ended" in seen["local-a"].raw_state
    assert seen["local-b"].state is SchedulerState.UNKNOWN


def test_nothing_starts_on_a_login_node(local):
    local.host = LOGIN
    with pytest.raises(SchedulerError, match="DFT cannot run locally on login nodes"):
        local.backend.submit(Path("/nowhere/local.sh"), Path("/nowhere"), "cedar-x")


# -- the dispatcher's host -------------------------------------------------------------------------------

def test_the_dispatcher_is_stopped_only_where_it_runs(tmp_path):
    workspace = Workspace.open(tmp_path / "data")
    (workspace.root / HOST).write_text("trace20")
    holder = subprocess.Popen([sys.executable, "-c", f"import fcntl, time\nf = open({str(workspace.root / LOCK)!r}, "
                               "'a')\nfcntl.lockf(f, fcntl.LOCK_EX)\nprint('locked', flush=True)\ntime.sleep(60)\n"],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        assert dispatcher_host(workspace) == "trace20"
        said = stop_dispatcher(workspace)
        assert said.startswith("The dispatcher runs on trace20, not on this machine") and holder.poll() is None
    finally:
        holder.kill()
        holder.wait()


def test_on_a_login_node_the_dispatcher_starts_no_local_run(tmp_path, site):
    system = make_system(tmp_path, site)
    workspace = system["workspace"]
    investigation(system, "hyp-a")
    start_agent(workspace.ledger, INV, "hyp-a")
    cid = workspace.ledger.request_calculation(INV, agent_for("hyp-a"), "hyp-a", task(execution="local"),
                                               "test").calculation_id
    examine(workspace.ledger, INV, cid)
    move(workspace.ledger, INV, cid, ENGINE, CalculationState.VALIDATED_INPUT)
    authorize(workspace.ledger, INV, PermissionAction.SUBMIT, cid, intervention_id="approve-local")
    dispatcher = Dispatcher(workspace, system["profiles"], system["clients"], backend=system["scheduler"],
                            workbench=Workbench(workspace.ledger, workspace.artifacts, host=lambda: LOGIN))
    notes = dispatcher.tick()
    assert any(f"{cid} waits: CEDAR is running on a login node (login1)" in note for note in notes), notes
    assert workspace.ledger.load(INV).calculations[cid].state is CalculationState.VALIDATED_INPUT
    assert system["scheduler"].submissions == []


class Says:
    """TEST DOUBLE for the local backend, as a query sees it: every run waits, and the backend says what for."""

    def __init__(self, note: str) -> None:
        self.note = note

    def query(self, job_ids):
        return {job_id: SchedulerObservation(job_id, SchedulerState.PENDING, self.note) for job_id in job_ids}


def test_the_dispatcher_logs_what_a_waiting_local_run_now_waits_for(tmp_path, site):
    system = make_system(tmp_path, site)
    workspace = system["workspace"]
    investigation(system, "hyp-a")
    start_agent(workspace.ledger, INV, "hyp-a")
    cid = workspace.ledger.request_calculation(INV, agent_for("hyp-a"), "hyp-a", task(execution="local"),
                                               "test").calculation_id
    submit(workspace.ledger, INV, cid, job_id="local-1")
    says = Says("waiting for 4 free cores")
    services = services_for(workspace, site, system["scheduler"], local_backend=says)
    dispatcher = system["dispatcher"]
    assert dispatcher._poll(INV, services) == [f"{cid} is QUEUED"]
    says.note = "waiting for Slurm to start its step"
    assert dispatcher._poll(INV, services) == [f"{cid}: waiting for Slurm to start its step"]  # still QUEUED
    assert workspace.ledger.load(INV).calculations[cid].scheduler_note == "waiting for Slurm to start its step"


def test_four_cores_always_stay_for_the_agents_commands(ledger, store, artifacts, site, tmp_path):
    from cedar.hpc.node import local_dft
    small = Local(ledger, store, artifacts, site, tmp_path, host=interactive(cores=4), cpus=(0, 1, 2, 3))
    cid = small.request_local(max_walltime_hours=1)
    assert "no_cores_for_dft" in findings(small.world, cid)
    assert "4 cores are always kept for the agents' commands, which leaves none for DFT" in local_dft(small.host)
    assert "a local run may use up to 4 cores, since 4 are kept" in local_dft(interactive(cores=8))
    pool = CorePool(list(range(8)))
    assert pool.take(5, keep=4) is None and len(pool.take(4, keep=4)) == 4
