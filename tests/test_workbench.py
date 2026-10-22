"""The workbench: agents' commands in their workspaces, in a sandbox, recorded; and where CEDAR runs.

Commands really run here, in bubblewrap where this machine has it (tests that need it are skipped otherwise),
on SYNTHETIC files under the test's temporary directory. No network, scheduler or DFT code is reached.
"""

import subprocess
from datetime import datetime, timedelta, timezone

import pytest
from factories import USER, agent_for, draft, start_agent
from test_orchestration import make_system
from test_swarm import INV, investigation
from worlds import World

from cedar.agents.toolkits import research_tools
from cedar.contracts import UserInterventionRecorded, WorkbenchOutcome, WorkbenchPolicyChanged
from cedar.engine.sandbox import SandboxView, bwrap_path, run_process
from cedar.engine.workbench import Workbench
from cedar.hpc.node import CorePool, ExecutionHost, detect_host, expand_nodelist, parse_time_left
from cedar.hpc.profile import LocalExecution
from cedar.notebook import render_notebook
from cedar.workspace import profile_for, research_context, services_for

needs_bwrap = pytest.mark.skipif(bwrap_path() is None, reason="this machine has no usable bubblewrap")
LOGIN = ExecutionHost("login1", cores=8, reason="not inside a Slurm allocation")


@pytest.fixture
def world(ledger, store, artifacts, site):
    return World(ledger, store, artifacts, site)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE KEY\n")
    (home / "projects").mkdir()
    (home / "projects" / "notes.txt").write_text("PBE prefers fcc\n")
    data = tmp_path / "data"
    workspace = data / "workspaces" / "inv-1" / "hyp-xc"
    workspace.mkdir(parents=True)
    other = data / "workspaces" / "inv-1" / "hyp-b"
    other.mkdir(parents=True)
    (other / "theirs.txt").write_text("hyp-b's work\n")
    runs = tmp_path / "runs"
    (runs / "own-calc").mkdir(parents=True)
    (runs / "own-calc" / "aims.out").write_text("own output\n")
    (runs / "their-calc").mkdir()
    (runs / "their-calc" / "aims.out").write_text("their output\n")
    code = tmp_path / "codes" / "aims.x"
    code.parent.mkdir()
    code.write_text("#!/bin/sh\necho DFT ran\n")
    code.chmod(0o755)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-never-shown-0000")
    return {"home": home, "data": data, "workspace": workspace, "runs": runs, "code": code}


def view_of(tree):
    return SandboxView(workspace=tree["workspace"],
                       hidden=(tree["home"] / ".ssh", tree["data"], tree["runs"], tree["code"]),
                       shown=(tree["runs"] / "own-calc",))


def bench(world, *, host=LOGIN, bwrap=bwrap_path, pool=None):
    return Workbench(world.ledger, world.artifacts, host=lambda: host, pool=pool or CorePool([0, 1, 2, 3]),
                     bwrap=bwrap)


def run(world, tree, command, *, workbench=None, **kwargs):
    workbench = workbench or bench(world)
    return workbench.run(world.inv, world.hyp, agent_for(world.hyp).id, view_of(tree), command=command,
                         purpose="test", local=kwargs.pop("local", LocalExecution()), **kwargs)


# -- the sandbox -----------------------------------------------------------------------------------

@needs_bwrap
def test_a_command_runs_in_the_workspace_and_is_recorded(world, tree):
    report = run(world, tree, "echo hello && python -c \"open('slab.in', 'w').write('atom 0 0 0 Cu')\"")
    assert report.run_id and report.run.outcome is WorkbenchOutcome.COMPLETED, report
    assert report.shown_output.strip() == "hello"
    assert (tree["workspace"] / "slab.in").read_text() == "atom 0 0 0 Cu"
    (record,) = world.state().workbench_runs.values()
    assert record.agent_id == agent_for(world.hyp).id and record.run.sandbox == "bwrap"
    assert set(record.run.changed) == {"slab.in"} and record.run.host == "login1" and record.run.cores == 1
    assert world.artifacts.read(record.run.output).decode().strip() == "hello"
    again = run(world, tree, "rm slab.in && mkdir -p sub && echo x > sub/y.txt")
    assert again.run.deleted == ("slab.in",) and set(again.run.changed) == {"sub/y.txt"}


@needs_bwrap
def test_secrets_records_and_other_work_are_hidden(world, tree):
    allowed = run(world, tree, f"cat {tree['home']}/projects/notes.txt {tree['runs']}/own-calc/aims.out")
    assert allowed.shown_output == "PBE prefers fcc\nown output\n"
    for path in (tree["home"] / ".ssh" / "id_ed25519", tree["runs"] / "their-calc" / "aims.out",
                 tree["data"] / "workspaces" / "inv-1" / "hyp-b" / "theirs.txt"):
        report = run(world, tree, f"cat {path}")
        assert report.run.outcome is WorkbenchOutcome.FAILED and "No such file" in report.shown_output, path
    listing = run(world, tree, f"ls -A {tree['data']}")
    assert listing.shown_output.split() == ["workspaces"]


@needs_bwrap
def test_nothing_outside_the_workspace_can_be_changed(world, tree):
    report = run(world, tree, f"touch {tree['home']}/projects/new.txt; echo x > /tmp/scratch && cat /tmp/scratch")
    assert "Read-only file system" in report.shown_output and report.shown_output.strip().endswith("x")
    assert not (tree["home"] / "projects" / "new.txt").exists()


@needs_bwrap
def test_there_is_no_network_no_scheduler_and_no_dft(world, tree):
    script = ("import socket\ns = socket.socket()\ns.settimeout(3)\ntry:\n    s.connect(('1.1.1.1', 53))\n"
              "    print('reachable')\nexcept OSError as exc:\n    print('blocked', exc)\n")
    (tree["workspace"] / "net.py").write_text(script)
    assert run(world, tree, "python net.py").shown_output.startswith("blocked")
    dft = run(world, tree, f"{tree['code']}")
    assert dft.run.outcome is WorkbenchOutcome.FAILED and "DFT ran" not in dft.shown_output


@needs_bwrap
def test_commands_get_no_credentials_and_are_pinned_to_their_cores(world, tree):
    report = run(world, tree, "env; python -c \"import os; print('cpus', len(os.sched_getaffinity(0)))\"")
    assert "sk-ant-test-key" not in report.shown_output and "ANTHROPIC_API_KEY" not in report.shown_output
    assert "cpus 1" in report.shown_output and "OMP_NUM_THREADS=1" in report.shown_output


def test_a_command_past_its_time_or_halted_is_stopped(tmp_path):
    started = datetime.now()
    result = run_process(["sleep", "30"], cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, output=tmp_path / "out",
                         time_limit=1)
    assert result.timed_out and result.exit_code is None and (datetime.now() - started).seconds < 10
    stopped = run_process(["sleep", "30"], cwd=tmp_path, env={"PATH": "/usr/bin:/bin"}, output=tmp_path / "out",
                          time_limit=60, should_stop=lambda: True)
    assert stopped.stopped and not stopped.timed_out


def test_without_bubblewrap_only_guarded_python_runs(world, tree):
    workbench = bench(world, bwrap=lambda: None)
    refused = run(world, tree, "ls", workbench=workbench)
    assert refused.run_id is None and "only Python scripts" in refused.problems[0]
    (tree["workspace"] / "check.py").write_text(
        "import sys\nprint(open(sys.argv[1]).read().strip())\nopen('made.txt', 'w').write('ok')\n"
        "try:\n    open(sys.argv[2]).read()\nexcept PermissionError as exc:\n    print('refused', exc)\n"
        "import subprocess\nsubprocess.run(['true'])\n")
    secret = tree["home"] / ".ssh" / "id_ed25519"
    report = run(world, tree, f"python check.py {tree['runs']}/own-calc/aims.out {secret}", workbench=workbench)
    assert report.run.sandbox == "python-guard" and report.run.outcome is WorkbenchOutcome.FAILED
    assert "own output" in report.shown_output and "refused" in report.shown_output
    assert "may not use subprocess.Popen" in report.shown_output
    assert set(report.run.changed) == {"made.txt"}


# -- limits and the researcher's setting -----------------------------------------------------------

def test_limits_follow_where_cedar_runs(world):
    local = LocalExecution()
    assert bench(world).limits(local) == (1, 30, 2)
    ends = datetime.now(timezone.utc) + timedelta(minutes=45)
    node = ExecutionHost("trace20", allocation="673421", cores=16, ends_at=ends)
    assert bench(world, host=node).limits(local)[:2] == (4, 43)  # the pool's 4 cores; the allocation's end
    assert bench(world, host=node, pool=CorePool(list(range(32)))).limits(local)[0] == 16


@needs_bwrap
def test_requests_beyond_the_limits_are_lowered_and_said_so(world, tree):
    report = run(world, tree, "true", minutes=600, cores=8)
    assert report.run.cores == 1 and report.run.time_limit_seconds == 30 * 60
    assert any("lowered from 600 to 30 minutes" in p for p in report.problems)
    assert any("1 core(s) instead of 8" in p for p in report.problems)


def test_the_researcher_can_turn_the_workbench_off(world, tree):
    world.ledger.record(world.inv, [
        draft(USER, UserInterventionRecorded(intervention_id="msg-1", text="/workbench off")),
        draft(USER, WorkbenchPolicyChanged(enabled=False, intervention_id="msg-1"))])
    report = run(world, tree, "true")
    assert report.run_id is None and "turned the workbench off" in report.problems[0]


def test_cores_are_shared_without_overlap():
    pool = CorePool([0, 1, 2])
    first = pool.acquire(2, timeout=0.1)
    assert first == (1, 2) and pool.acquire(2, timeout=0.1) is None
    assert pool.acquire(1, most_running=1, timeout=0.1) is None
    pool.release(first)
    assert sorted(pool.acquire(3, timeout=0.1)) == [0, 1, 2]


# -- where CEDAR runs ------------------------------------------------------------------------------

def test_slurm_node_lists_and_times_are_read():
    assert expand_nodelist("trace[01-03,07],gpu5") == ["trace01", "trace02", "trace03", "trace07", "gpu5"]
    assert expand_nodelist("trace20") == ["trace20"]
    assert parse_time_left("1-19:50:52") == timedelta(days=1, hours=19, minutes=50, seconds=52)
    assert parse_time_left("45:00") == timedelta(minutes=45)
    assert parse_time_left("UNLIMITED") is None


def fake_squeue(output, code=0):
    def runner(argv, **_):
        assert argv[:4] == ["squeue", "-h", "-j", "673421"]
        return subprocess.CompletedProcess(argv, code, stdout=output, stderr="")
    return runner


def test_an_interactive_node_is_a_running_allocations_own_node():
    env = {"SLURM_JOB_ID": "673421", "SLURM_JOB_NODELIST": "trace[19-20]", "SLURM_CPUS_ON_NODE": "2"}
    now = datetime(2026, 10, 21, tzinfo=timezone.utc)
    host = detect_host(env, runner=fake_squeue("RUNNING|2:00:00\n"), hostname=lambda: "trace20.cluster",
                       now=lambda: now)
    assert host.interactive and host.allocation == "673421" and host.cores <= 2
    assert host.ends_at == now + timedelta(hours=2) and "allocation 673421" in host.describe()


@pytest.mark.parametrize("env, hostname, squeue, reason", [
    ({}, "tracevm05", None, "not inside a Slurm allocation"),
    ({"SLURM_JOB_ID": "673421", "SLURM_JOB_NODELIST": "trace20"}, "tracevm05", None, "runs on trace20, not here"),
    ({"SLURM_JOB_ID": "673421", "SLURM_JOB_NODELIST": "trace20"}, "trace20", "PENDING|1:00:00", "not report"),
])
def test_anything_else_is_a_login_node(env, hostname, squeue, reason):
    host = detect_host(env, runner=fake_squeue(squeue or ""), hostname=lambda: hostname)
    assert not host.interactive and reason in host.reason and host.describe().startswith("login node")


# -- through the agent's tools and the researcher's chat ------------------------------------------------

@needs_bwrap
def test_an_agent_runs_commands_through_its_tools(tmp_path, site):
    system = make_system(tmp_path, site)
    chat = investigation(system, "hyp-a")
    workspace = system["workspace"]
    start_agent(workspace.ledger, INV, "hyp-a")
    state = workspace.ledger.load(INV)
    agent = state.active_agent("hyp-a")
    services = services_for(workspace, profile_for(state, system["profiles"]), system["scheduler"])
    workbench = Workbench(workspace.ledger, workspace.artifacts, host=lambda: LOGIN, pool=CorePool([0, 1]))
    context = research_context(workspace, INV, agent, services, deny=[tmp_path / "config"], workbench=workbench)
    tools = {t.name: t for t in research_tools(context)}
    view = context.bench.view()
    assert workspace.root.resolve() not in view.hidden and tmp_path.joinpath("config").resolve() in view.hidden
    assert site.codes["fhi-aims"].binary in {str(p) for p in view.hidden}

    out = tools["run_command"].invoke({"command": f"python -c \"import ase; print('ase', ase.__version__)\" && "
                                                  f"{site.codes['fhi-aims'].binary}; ls {workspace.root}",
                                       "purpose": "check the Python environment"})
    assert out.content.startswith("Workbench run wb_") and "ase 3." in out.content
    assert "synthetic test binary" not in out.content and "events.sqlite" in out.content  # CEDAR's records too
    assert "login node login1" in tools["workbench_runs"].invoke({}).content

    assert chat.handle("/workbench").startswith("The workbench is on for this investigation")
    assert chat.handle("/workbench off").startswith("No agent runs commands from now on")
    refused = tools["run_command"].invoke({"command": "true", "purpose": "x"})
    assert refused.is_error and "turned the workbench off" in refused.content
    assert "has turned the workbench off" in tools["workbench_runs"].invoke({}).content

    pages = render_notebook(workspace.store.read_stream(INV))
    (run_id,) = workspace.ledger.load(INV).workbench_runs
    assert f"- {run_id} by {agent.agent_id} for" in pages["hypotheses/hyp-a.md"]
    assert "## Workbench commands (context, not evidence)" in pages["hypotheses/hyp-a.md"]
    assert "Agents' workbench (sandboxed commands in their workspaces): **off**" in pages["index.md"]
    assert f"{run_id} for hyp-a:" in pages["timeline.md"] and "Agents' workbench turned off" in pages["timeline.md"]
