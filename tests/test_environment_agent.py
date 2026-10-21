"""Environment-setup agent and profile commands, on a SYNTHETIC site.

The ScriptedModel TEST DOUBLE plays the agent; scheduler commands are answered by a fake
runner; the researcher's answers are scripted.
"""

import json
import subprocess
from datetime import date

import pytest
from sites import JOB_TEMPLATE, aims_output, build_site

from cedar import cli
from cedar.agents import AgentRunner, TranscriptStore
from cedar.agents.prompts import ENVIRONMENT_SETUP
from cedar.agents.toolkits import FileAccess, file_tools
from cedar.agents.toolkits.environment import SetupSession, environment_tools
from cedar.hpc import ProfileStore
from cedar.hpc.queries import SchedulerQueries
from cedar.models import ModelSettings, TextPart, ToolCall
from cedar.models.scripted import ScriptedModel, reply

SETTINGS = ModelSettings(provider="scripted", model="scripted-model")
KEY = "sk-ant-test-0123456789abcdefghij"


def fake_runner(calls):
    def run(argv, **kwargs):
        calls.append((argv, kwargs["env"]))
        outputs = {"sinfo": "debug*|up|1:00:00|2|64|256000\ncompute|up|1-00:00:00|10|64|256000\n",
                   "scontrol": "PartitionName=compute MaxTime=1-00:00:00 DefMemPerCPU=4000\n"}
        if argv[0] == "sacctmgr":
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="sacctmgr: permission denied")
        return subprocess.CompletedProcess(argv, 0, stdout=outputs.get(argv[0], ""), stderr="")
    return run


class TestSchedulerQueries:
    def queries(self, calls, **kwargs):
        return SchedulerQueries(user="someone", runner=fake_runner(calls), environ={"PATH": "/bin",
                                "ANTHROPIC_API_KEY": KEY}, which=lambda c: f"/usr/bin/{c}" if c != "bsub" else None,
                                today=lambda: date(2026, 10, 21), **kwargs)

    def test_queries_are_fixed_commands_without_credentials(self):
        calls = []
        result = self.queries(calls).run("slurm_partition_detail", "compute")
        assert result.command == "scontrol show partition compute" and "MaxTime" in result.output
        assert calls == [(["scontrol", "show", "partition", "compute"], {"PATH": "/bin"})]

    def test_recent_jobs_use_a_fixed_window(self):
        calls = []
        self.queries(calls).run("slurm_recent_jobs")
        assert calls[0][0][:7] == ["sacct", "-u", "someone", "-n", "-P", "-X", "-S"] and "2026-09-21" in calls[0][0]

    @pytest.mark.parametrize("query, argument", [("rm", None), ("slurm_node_cpus", None),
                                                 ("slurm_node_cpus", "a; rm -rf ~"), ("slurm_partitions", "x")])
    def test_unknown_queries_and_bad_arguments_are_refused(self, query, argument):
        with pytest.raises(ValueError):
            self.queries([]).run(query, argument)

    def test_failures_and_detection_are_reported(self):
        queries = self.queries([])
        assert queries.run("slurm_associations").output.startswith("exit 1: sacctmgr: permission denied")
        detected = queries.run("detect_scheduler").output
        assert "sbatch=/usr/bin/sbatch" in detected and "bsub=not found" in detected


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A synthetic home with a working job script and an output written by the synthetic binary."""
    site = build_site(tmp_path / "site")
    code = site.codes["fhi-aims"]
    home = tmp_path / "home"
    old_run = home / "old_run"
    old_run.mkdir(parents=True)
    script = JOB_TEMPLATE.replace("{{partition}}", "compute").replace("{{nodes}}", "1").replace("{{ntasks}}", "64")
    script = script.replace("{{walltime}}", "24:00:00").replace("{{binary}}", code.binary).replace("{{output}}", "aims.out")
    (old_run / "submit.sh").write_text(script)
    (old_run / "aims.out").write_text(aims_output())
    config = tmp_path / "config"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CEDAR_CONFIG_DIR", str(config))
    return {"site": site, "home": home, "config": config, "old_run": old_run}


def proposal(home, **overrides):
    site = home["site"]
    code = site.codes["fhi-aims"]
    args = {
        "name": "synthetic", "scheduler": "slurm", "partitions": ["compute"], "default_partition": "compute",
        "cores_per_node": 64, "default_ntasks": 16, "max_nodes": 1, "default_walltime_minutes": 30,
        "max_walltime_minutes": 1440,
        "codes": {"fhi-aims": {"binary": code.binary, "version": "test-1.0", "data_paths": code.data_paths}},
        "job_template": site.job_template, "runs_root": str(home["home"] / "cedar_runs"),
        "source_notes": {str(home["old_run"] / "submit.sh"): "template, binary, environment script"},
    }
    return {**args, **overrides}


def setup_script(home):
    old_run = home["old_run"]
    return [
        reply(ToolCall(id="1", name="find_files", arguments={"root": str(home["home"]), "name_pattern": "*.sh"})),
        reply(ToolCall(id="2", name="read_file", arguments={"path": str(old_run / "submit.sh")}),
              ToolCall(id="3", name="search_file", arguments={"path": str(old_run / "aims.out"),
                                                              "pattern": "FHI-aims version"})),
        reply(ToolCall(id="4", name="inspect_path", arguments={"path": home["site"].codes["fhi-aims"].binary}),
              ToolCall(id="5", name="scheduler_query", arguments={"query": "slurm_partitions"})),
        reply(ToolCall(id="6", name="ask_user", arguments={"question": "Which partitions may CEDAR use?"})),
        reply(ToolCall(id="7", name="propose_profile", arguments=proposal(
            home, job_template="#!/bin/bash\nmpirun {{binary}}\n"))),
        reply(ToolCall(id="8", name="propose_profile", arguments=proposal(home))),
        reply(TextPart(text="Proposed profile 'synthetic' from your submit.sh; please review the walltime.")),
    ]


def test_setup_agent_proposes_a_profile_with_sources(home):
    store = ProfileStore(home["config"] / "profiles")
    session = SetupSession()
    access = FileAccess(roots=[home["home"]], deny=[home["config"]])
    tools = file_tools(access, on_read=session.file_read) + environment_tools(
        store, SchedulerQueries(runner=fake_runner([])), session, lambda q: "only compute", deny=[home["config"]])
    model = ScriptedModel(setup_script(home))
    result = AgentRunner(model, SETTINGS, TranscriptStore(home["config"] / "t")).run(
        "setup-1", system=ENVIRONMENT_SETUP, tools=tools, instruction="Propose a profile.")
    assert result.outcome == "finished"
    rejected = [p for m in model.calls[6].messages for p in m.parts if getattr(p, "call_id", "") == "7"][0]
    assert rejected.is_error and "exactly one template line" in rejected.content

    profile = store.load("synthetic")
    assert not store.is_approved("synthetic")
    kinds = {(s.kind, s.reference) for s in profile.sources}
    assert ("file", str(home["old_run"] / "submit.sh")) in kinds and ("file", str(home["old_run"] / "aims.out")) in kinds
    assert ("command_output", "sinfo -h -o %P|%a|%l|%D|%c|%m") in kinds
    answer = next(s for s in profile.sources if s.kind == "user_answer")
    assert (answer.reference, answer.note) == ("Which partitions may CEDAR use?", "only compute")
    script_source = next(s for s in profile.sources if s.reference.endswith("submit.sh"))
    assert script_source.note == "template, binary, environment script" and len(script_source.sha256) == 64


def test_inspect_path_cannot_see_the_credentials_directory(home):
    home["config"].mkdir()
    tools = {t.name: t for t in environment_tools(ProfileStore(home["config"] / "p"), SchedulerQueries(),
                                                   SetupSession(), lambda q: "", deny=[home["config"]])}
    out = tools["inspect_path"].invoke({"path": str(home["config"])})
    assert out.is_error and "not inspectable" in out.content
    assert "executable" in tools["inspect_path"].invoke({"path": home["site"].codes["fhi-aims"].binary}).content


def test_cli_setup_then_review_and_approve(home, monkeypatch, capsys):
    model = ScriptedModel(setup_script(home))
    monkeypatch.setattr("cedar.models.create_client", lambda settings, credentials: model)
    monkeypatch.setattr("cedar.hpc.queries.SchedulerQueries", lambda: SchedulerQueries(runner=fake_runner([])))
    monkeypatch.setattr(cli, "_ask_in_terminal", lambda question: "only compute")
    cli.main(["models", "init"])
    assert cli.main(["setup", "--name", "synthetic", "--root", str(home["site"].runs_root)]) == 0
    out = capsys.readouterr().out
    assert "- read_file" in out and "! propose_profile" in out and "cedar profile approve synthetic" in out
    log = json.loads((home["config"] / "setup_runs.jsonl").read_text().splitlines()[-1])
    assert log["profile"] == "synthetic" and log["outcome"] == "finished"

    cli.main(["profile", "list"])
    assert "synthetic: not approved" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli.main(["profile", "approve", "synthetic"]) == 1
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert cli.main(["profile", "approve", "synthetic"]) == 0
    cli.main(["profile", "show", "synthetic"])
    assert "Status: approved" in capsys.readouterr().out


def test_cli_refuses_to_approve_a_failing_profile(home, monkeypatch, capsys):
    store = ProfileStore(home["config"] / "profiles")
    broken = home["site"].model_copy(update={"name": "broken", "codes": {"fhi-aims": home["site"].codes[
        "fhi-aims"].model_copy(update={"binary": str(home["home"] / "missing.x")})}})
    store.save(broken)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("should not ask"))
    assert cli.main(["profile", "approve", "broken"]) == 1
    assert "fails its checks" in capsys.readouterr().out


def test_setup_can_resume_with_its_sources(home, monkeypatch, capsys):
    script = setup_script(home)
    first = ScriptedModel(script[:4] + [reply(TextPart(text="I need your answer before proposing."))])
    monkeypatch.setattr("cedar.hpc.queries.SchedulerQueries", lambda: SchedulerQueries(runner=fake_runner([])))
    monkeypatch.setattr(cli, "_ask_in_terminal", lambda question: "")
    monkeypatch.setattr("cedar.models.create_client", lambda settings, credentials: first)
    cli.main(["models", "init"])
    assert cli.main(["setup", "--name", "synthetic"]) == 1
    assert "synthetic" not in ProfileStore(home["config"] / "profiles").names()

    second = ScriptedModel([script[3], script[5], script[6]])
    monkeypatch.setattr(cli, "_ask_in_terminal", lambda question: "only compute")
    monkeypatch.setattr("cedar.models.create_client", lambda settings, credentials: second)
    assert cli.main(["setup", "--name", "synthetic", "--resume"]) == 0
    assert "Continue where you stopped" in second.calls[0].messages[-1].text
    assert len(second.calls[0].messages) > 5  # the first session's history came along
    profile = ProfileStore(home["config"] / "profiles").load("synthetic")
    references = {s.reference for s in profile.sources}
    assert str(home["old_run"] / "submit.sh") in references  # read during the first session
    assert any(s.kind == "user_answer" and s.note == "only compute" for s in profile.sources)
    assert "cached input read" in capsys.readouterr().out


def test_resume_without_an_earlier_session_fails(home, capsys):
    cli.main(["models", "init"])
    assert cli.main(["setup", "--name", "nothing", "--resume"]) == 1
