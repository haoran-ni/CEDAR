"""Chat, Chief Scientist, dispatcher, hypothesis agent and Evidence Judge together.

ScriptedModel and FakeScheduler TEST DOUBLES on a SYNTHETIC site; no model or job runs.
"""

import fcntl
import subprocess
import sys
from datetime import datetime, timezone

import pytest
from sites import aims_output, co_on_cu111, declared, geometry_text

from sim2real.agents.toolkits.chief import ChiefContext, chief_tools
from sim2real.chat import CHIEF, ChatSession
from sim2real.contracts import (
    Assessment,
    CalculationState,
    EvidenceQuality,
    HypothesisStatus,
    InvestigationStatus,
    SchedulerState,
)
from sim2real.dispatch import Dispatcher, current_activity, heartbeat_age
from sim2real.hpc import ProfileStore
from sim2real.hpc.fake import FakeScheduler
from sim2real.models import ModelSettings, TextPart, ToolCall
from sim2real.models.scripted import ScriptedModel, reply
from sim2real.state import replay
from sim2real.workspace import Workspace

ENERGY = -5000.123456789
INV, HYP = "co-cu111-site", "xc-frontier-orbitals"
NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)


class Clients:
    """Scripted models per agent role (TEST DOUBLE)."""

    def __init__(self):
        self.models: dict[str, ScriptedModel] = {}

    def add(self, role, *steps):
        self.models.setdefault(role, ScriptedModel([])).script.extend(steps)

    def settings(self, role):
        return ModelSettings(provider="scripted", model=f"scripted-{role}")

    def client(self, role):
        return self.models[role]

    def secrets(self):
        return []


def make_system(tmp_path, site):
    workspace = Workspace.open(tmp_path / "data")
    profiles = ProfileStore(tmp_path / "config" / "profiles")
    profiles.save(site)
    profiles.approve(site.name, "researcher", NOW)
    scheduler = FakeScheduler()
    clients = Clients()
    chat = ChatSession(workspace, profiles, clients, backend=scheduler, clock=lambda: NOW)
    dispatcher = Dispatcher(workspace, profiles, clients, backend=scheduler, deny=[tmp_path / "config"],
                            clock=lambda: NOW)
    return {"workspace": workspace, "profiles": profiles, "scheduler": scheduler, "clients": clients, "chat": chat,
            "dispatcher": dispatcher, "site": site}


@pytest.fixture
def system(tmp_path, site):
    return make_system(tmp_path, site)


def call(name, **arguments):
    return ToolCall(id=f"{name}-{len(arguments)}-{abs(hash(str(arguments))) % 10_000}", name=name, arguments=arguments)


def open_and_delegate(site, budget=1_000_000):
    return reply(
        call("open_investigation", investigation_id=INV, title="CO/Cu(111) adsorption site (synthetic test)",
             question="Why does the computed site preference differ from the observed one?",
             target_claim="The site discrepancy has identifiable causes.", objective="Test explanations cheaply",
             simulations=[{"label": "PBE slab", "code": "FHI-aims", "xc": "PBE"}],
             experiments=[{"label": "low-coverage measurement", "technique": "placeholder technique",
                           "observable": {"name": "preferred site", "definition": "occupied site at low coverage"}}],
             simulated_observable={"name": "site energy difference", "definition": "E(atop) - E(fcc)", "unit": "eV"},
             measured_observable={"name": "preferred site", "definition": "occupied site at low coverage"},
             compute_profile=site.name, model_token_budget=budget),
        call("set_investigation_status", status="ALIGNING_OBSERVABLES", reason="start"),
        call("record_alignment", status="REQUIRES_FORWARD_MODEL",
             rationale=["an energy difference predicts a site preference only under a thermodynamic model"]),
        call("set_investigation_status", status="PLANNING", reason="alignment recorded"),
        call("set_investigation_status", status="INVESTIGATING", reason="plan ready"),
        call("propose_hypothesis", hypothesis_id=HYP, title="XC error in CO frontier orbitals",
             statement="Semilocal XC misplaces CO frontier orbitals and favours hollow sites.",
             rationale="Site preference depends on CO-metal orbital mixing.",
             falsifiers=["Correcting frontier orbitals leaves the site ordering unchanged."]),
        call("set_hypothesis_status", hypothesis_id=HYP, status="SCREENED", reason="testable"),
        call("set_hypothesis_status", hypothesis_id=HYP, status="ACTIVE", reason="cheap first test"),
        call("delegate", hypothesis_id=HYP, instruction="Compute a PBE baseline total energy for CO atop."),
    )


def latest_calc(workspace):
    return next(reversed(workspace.ledger.load(INV).calculations))


def agent_requests(system):
    slab, bottom = co_on_cu111()
    workspace = system["workspace"]
    return [
        reply(call("set_hypothesis_status", status="PLANNING_TEST", reason="baseline first"),
              call("request_calculation", protocol_name="fhiaims.run", protocol_version="0.1",
                   observable_name="total energy", observable_definition="final total energy printed by FHI-aims",
                   observable_unit="eV", estimand="total energy of CO/Cu(111) atop",
                   system_label="CO/Cu(111) atop (synthetic)", structure_text=geometry_text(slab, fixed=bottom),
                   rationale="baseline", **declared())),
        lambda c: reply(call("register_prediction", prediction_id="pred-1", statement="converges",
                             observable_name="total energy", observable_definition="final energy",
                             expected_outcome="validated energy", falsified_if="fails checks",
                             calculation_ids=[latest_calc(workspace)])),
        lambda c: reply(TextPart(text="Requested the PBE baseline; waiting for it."),
                        call("wait_for_calculations", calculation_ids=[latest_calc(workspace)], reason="need baseline")),
    ]


HARTREE_SCRIPT = (
    "import json, re\nline = next(l for l in open('inputs/{calc}/aims.out') if 'Total energy of the DFT' in l)\n"
    "ev = float(re.findall(r'[-+]?\\d+\\.\\d+', line)[-1])\nprint('matched:', line.strip())\n"
    "json.dump({{'results': {{'total_energy_ha': {{'value': ev / 27.211386245988, 'unit': 'Ha'}}}},\n"
    "           'checks': {{'one_final_energy_line': {{'passed': True, 'detail': line.strip()}}}}}},\n"
    "          open('results.json', 'w'))\n")


def agent_reads(system, output):
    """The woken agent records a cited reading, computes a value with a script and records it."""
    line = next(i for i, l in enumerate(output.splitlines(), start=1) if "Total energy of the DFT" in l)
    workspace = system["workspace"]
    return [
        lambda c: reply(call("submit_reading", calculation_id=latest_calc(workspace), observable_name="total energy",
                             observable_definition="final total energy printed by FHI-aims", value=ENERGY,
                             unit="eV", summary="baseline total energy",
                             citations=[{"file": "aims.out", "line_start": line, "line_end": line,
                                         "quote": output.splitlines()[line - 1].strip()}])),
        lambda c: reply(call("run_analysis", purpose="total energy in Hartree",
                             script=HARTREE_SCRIPT.format(calc=latest_calc(workspace)),
                             inputs=[{"calculation_id": latest_calc(workspace)}])),
        lambda c: reply(call("record_result", analysis_id=next(iter(workspace.ledger.load(INV).analyses)),
                             result_name="total_energy_ha", observable_name="total energy in Hartree",
                             observable_definition="final total energy converted with CODATA 2018",
                             summary="printed eV value divided by 27.211386245988")),
        reply(call("set_hypothesis_status", status="ASSESSED", reason="baseline recorded")),
        reply(TextPart(text="Baseline validated; it alone does not test the hypothesis.")),
    ]


def test_full_loop(system):
    workspace, chat, dispatcher, clients = system["workspace"], system["chat"], system["dispatcher"], system["clients"]
    scheduler = system["scheduler"]

    # The researcher describes the problem; the Chief opens the investigation and delegates.
    clients.add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="Opened and delegated.")))
    assert chat.handle("PBE puts CO in the fcc hollow on Cu(111); experiments see atop.") == "Opened and delegated."
    state = workspace.ledger.load(INV)
    assert chat.investigation_id == INV and state.status is InvestigationStatus.INVESTIGATING
    assert state.interventions["msg-1"].text.startswith("PBE puts CO")
    assert state.interventions["msg-1"].actor_id == chat.user.id
    assert [r.role for r in state.agent_runs] == ["chief_scientist"] and len(state.pending_tasks) == 1

    # The dispatcher runs the hypothesis agent; nothing is submitted without authorization.
    clients.add("hypothesis_agent", *agent_requests(system))
    notes = dispatcher.tick()
    assert any("hypothesis_agent ran" in n and "waiting" in n for n in notes)
    assert not any("submit" in n for n in dispatcher.tick())  # not even attempted without authorization
    calc = latest_calc(workspace)
    assert workspace.ledger.load(INV).calculations[calc].state is CalculationState.VALIDATED_INPUT
    assert scheduler.submissions == [] and heartbeat_age(workspace, NOW) == 0
    notices = chat.notices()
    assert any("Requested the PBE baseline" in n for n in notices)
    assert any(f"/approve submit {calc}" in n for n in notices)

    # The researcher authorizes; the dispatcher submits; the job finishes.
    assert "authorized" in chat.handle(f"/approve submit {calc}")
    output = aims_output(energy=ENERGY)
    scheduler.on_submit = lambda job, workdir: (workdir / "aims.out").write_text(output)
    assert any(f"submitted {calc}" in n for n in dispatcher.tick())
    scheduler.set_state(next(iter(scheduler.jobs)), SchedulerState.COMPLETED)

    # The dispatcher notices, wakes the agent, and the agent reports a cited reading.
    clients.add("hypothesis_agent", *agent_reads(system, output))
    notes = dispatcher.tick()
    assert any("SCHEDULER_TERMINAL" in n for n in notes) and any("woke the agent" in n for n in notes)
    state = workspace.ledger.load(INV)
    evidence, computed = (state.evidence[e] for e in state.calculations[calc].evidence_ids)
    assert evidence.evidence.quality is EvidenceQuality.VALIDATED and evidence.evidence.citations
    assert computed.evidence.analysis_id is not None and computed.evidence.quality is EvidenceQuality.VALIDATED
    assert state.hypotheses[HYP].status is HypothesisStatus.ASSESSED

    # The Chief asks the Judge; the Judge records a partial assessment.
    clients.add("chief_scientist", reply(call("request_judgement", hypothesis_id=HYP,
                                              instruction="Assess the hypothesis on the evidence so far.")),
                reply(TextPart(text="Asked the Evidence Judge.")))
    chat.handle("Please get the judge's view.")
    clients.add("evidence_judge",
                reply(call("hypothesis_dossier", hypothesis_id=HYP)),
                reply(call("record_assessment", hypothesis_id=HYP, assessment="INCONCLUSIVE", completeness="PARTIAL",
                           evidence_ids=[evidence.evidence_id],
                           rationale="One baseline energy cannot test a claim about site ordering.")),
                reply(TextPart(text="INCONCLUSIVE: a functional comparison is needed.")))
    dispatcher.tick()
    state = workspace.ledger.load(INV)
    assert state.hypotheses[HYP].assessment is Assessment.INCONCLUSIVE
    assert state.pending_tasks == []
    assert {r.role for r in state.agent_runs} == {"chief_scientist", "hypothesis_agent", "evidence_judge"}
    assert state.interventions["msg-2"].authorizations[0].target_ids == (calc,)
    dossier_view = clients.models["evidence_judge"].calls[1].messages[-1].parts[0].content
    assert "Prediction pred-1" in dossier_view and "check citations_verified: passed" in dossier_view
    assert replay(workspace.store.read_stream(INV)) == state

    # The dispatcher keeps the research notebook in step with the record.
    notebook = workspace.notebook_dir(INV)
    index = (notebook / "index.md").read_text()
    assert f"| {calc} |" in index
    assert f"| COMPLETED | healthy | {evidence.evidence_id}: VALIDATED, total energy = {ENERGY!r} eV; " in index
    assert "NOT_TESTED → **INCONCLUSIVE** (partial)" in (notebook / "round-01.md").read_text()
    assert chat.handle("/report").startswith(f"Research notebook (already up to date): {notebook / 'index.md'}")


def test_commands_need_an_open_investigation_and_known_calculations(system):
    chat = system["chat"]
    assert "No investigation is open" in chat.handle("/status")
    system["clients"].add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="ok")))
    chat.handle("Investigate CO on Cu(111).")
    assert INV in chat.handle("/status")
    assert "No calculation calc_x" in chat.handle("/approve submit calc_x")
    assert "Unknown or incomplete command" in chat.handle("/approve")
    assert chat.handle("/pending") == "Nothing waits for your approval."


def test_on_hold_hypotheses_are_not_submitted(system):
    workspace, chat, dispatcher, clients = system["workspace"], system["chat"], system["dispatcher"], system["clients"]
    clients.add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="ok")))
    chat.handle("Investigate CO on Cu(111).")
    clients.add("hypothesis_agent", *agent_requests(system))
    dispatcher.tick()
    calc = latest_calc(workspace)
    chat.handle(f"/approve submit {calc}")
    clients.add("chief_scientist", reply(call("revise_plan", reason="researcher paused XC work",
                                              intervention_id="msg-3", priorities={HYP: "on_hold"})),
                reply(TextPart(text="XC work is on hold.")))
    chat.handle("Hold the XC work for now.")
    assert workspace.ledger.load(INV).plans[-1].intervention_id == "msg-3"
    assert not any("submitted" in n for n in dispatcher.tick())
    assert system["scheduler"].submissions == []

    clients.add("chief_scientist", reply(call("revise_plan", reason="resume", priorities={HYP: "normal"})),
                reply(TextPart(text="Resumed.")))
    chat.handle("Resume it.")
    assert any(f"submitted {calc}" in n for n in dispatcher.tick())


def test_spent_token_budget_leaves_queued_work_waiting(system):
    workspace, chat, dispatcher, clients = system["workspace"], system["chat"], system["dispatcher"], system["clients"]
    # Each scripted reply reports 15 tokens; the Chief's two replies spend the whole budget of 30.
    clients.add("chief_scientist", open_and_delegate(system["site"], budget=30), reply(TextPart(text="ok")))
    chat.handle("Investigate CO on Cu(111).")
    state = workspace.ledger.load(INV)
    assert state.model_tokens_remaining == 0 and len(state.pending_tasks) == 1
    notes = dispatcher.tick()
    assert any("budget is spent" in n for n in notes)
    assert len(workspace.ledger.load(INV).pending_tasks) == 1


def test_the_dispatcher_checks_every_finished_run(system):
    workspace, chat, dispatcher, clients = system["workspace"], system["chat"], system["dispatcher"], system["clients"]
    clients.add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="ok")))
    chat.handle("Investigate CO on Cu(111).")
    clients.add("hypothesis_agent", *agent_requests(system))
    dispatcher.tick()
    calc = latest_calc(workspace)
    chat.handle(f"/approve submit {calc}")
    output = aims_output(energy=ENERGY)
    system["scheduler"].on_submit = lambda job, workdir: (workdir / "aims.out").write_text(output)
    dispatcher.tick()
    system["scheduler"].set_state(next(iter(system["scheduler"].jobs)), SchedulerState.COMPLETED)
    clients.add("hypothesis_agent", reply(TextPart(text="I will look later.")))
    notes = dispatcher.tick()
    assert any(f"checked the run of {calc}: healthy" in n for n in notes)
    state = workspace.ledger.load(INV)
    assert state.calculations[calc].run_check.healthy and state.evidence == {}  # evidence comes only from agents
    woken = clients.models["hypothesis_agent"].calls[-1].messages[-1].text
    assert "run check: healthy" in woken and "protocol read total energy" in woken


def test_a_second_dispatcher_waits_for_the_first(system):
    lock = open(system["workspace"].root / "dispatcher.lock", "a")
    fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        code = ("from sim2real.workspace import Workspace; from sim2real.dispatch import Dispatcher;"
                "from sim2real.hpc import ProfileStore;"
                f"d = Dispatcher(Workspace.open({str(system['workspace'].root)!r}), ProfileStore('/nonexistent'), None);"
                "print(d.tick())")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        assert "another dispatcher" in out
    finally:
        fcntl.lockf(lock, fcntl.LOCK_UN)
        lock.close()


def test_only_the_researcher_authorizes(system):
    chat, clients = system["chat"], system["clients"]
    clients.add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="ok")))
    chat.handle("Investigate CO on Cu(111).")
    context = ChiefContext(ledger=system["workspace"].ledger, profiles=system["profiles"], registry_for=lambda n: None,
                           actor=CHIEF)
    names = {tool.name for tool in chief_tools(context)}
    assert not any(word in name for name in names for word in ("approve", "authorize", "submit", "cancel"))
    interventions = [e for e in system["workspace"].store.read_stream(INV) if e.type == "user_intervention_recorded"]
    assert interventions and all(e.actor == chat.user for e in interventions)


def test_early_messages_get_ids_at_opening_and_delegation_waits_for_investigating(system):
    chat, clients, workspace = system["chat"], system["clients"], system["workspace"]
    opening = open_and_delegate(system["site"]).message.tool_calls[0]
    clients.add("chief_scientist", reply(opening),
                reply(call("delegate", hypothesis_id=HYP, instruction="too early")),
                reply(TextPart(text="Opened.")))
    chat.handle("CO on Cu(111): PBE prefers fcc, experiment atop.")
    model = clients.models["chief_scientist"]
    open_result = model.calls[1].messages[-1].parts[0].content
    assert "recorded as msg-1" in open_result
    delegate_result = model.calls[2].messages[-1].parts[0]
    assert delegate_result.is_error and "move it to INVESTIGATING" in delegate_result.content
    state = workspace.ledger.load(INV)
    assert state.interventions["msg-1"].text.startswith("CO on Cu(111)") and state.pending_tasks == []


def test_cli_start_and_dispatch(system, monkeypatch, capsys, tmp_path):
    from sim2real import cli

    clients = system["clients"]
    monkeypatch.setenv("SIM2REAL_DATA_DIR", str(system["workspace"].root))
    monkeypatch.setattr(cli, "_profiles", lambda: system["profiles"])
    monkeypatch.setattr(cli, "_model_clients", lambda: clients)
    clients.add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="Opened and delegated.")))
    lines = iter(["Investigate CO on Cu(111).", "/status", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    assert cli.main(["start"]) == 0
    out = capsys.readouterr().out
    assert "no dispatcher has run recently" in out and "Opened and delegated." in out
    assert f"Investigation {INV}" in out and f"Resume with: sim2real start {INV}" in out

    cli.main(["investigations"])
    assert f"{INV}: INVESTIGATING; 1 hypotheses" in capsys.readouterr().out
    clients.add("hypothesis_agent", *agent_requests(system))
    monkeypatch.setattr(cli, "_make_dispatcher", lambda: system["dispatcher"])
    assert cli.main(["dispatch", "--once"]) == 0
    assert "hypothesis_agent ran" in capsys.readouterr().out

    lines = iter(["/pending", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    assert cli.main(["start", INV]) == 0
    out = capsys.readouterr().out
    assert "Requested the PBE baseline" in out and "/approve submit calc_" in out
    assert cli.main(["start", "no-such-investigation"]) == 1

    assert cli.main(["report"]) == 0
    assert f"{INV}: {system['workspace'].notebook_dir(INV) / 'index.md'}" in capsys.readouterr().out
    assert cli.main(["report", "no-such-investigation"]) == 1


def test_dispatcher_describes_the_agent_run_in_progress(system):
    workspace, chat, dispatcher, clients = system["workspace"], system["chat"], system["dispatcher"], system["clients"]
    clients.add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="ok")))
    chat.handle("Investigate CO on Cu(111).")
    seen = []

    def look(_request):
        seen.append(current_activity(workspace))
        return reply(TextPart(text="Looked around."))

    clients.add("hypothesis_agent", reply(call("investigation_overview")), look)
    dispatcher.tick()
    (activity,) = seen
    assert (activity.investigation_id, activity.hypothesis_id, activity.role) == (INV, HYP, "hypothesis_agent")
    assert activity.last_step == "investigation_overview" and activity.started_at == NOW
    assert current_activity(workspace) is None and heartbeat_age(workspace, NOW) == 0


def test_calculations_can_be_named_by_a_unique_start_of_their_id(system):
    workspace, chat, dispatcher, clients = system["workspace"], system["chat"], system["dispatcher"], system["clients"]
    clients.add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="ok")))
    chat.handle("Investigate CO on Cu(111).")
    clients.add("hypothesis_agent", *agent_requests(system))
    dispatcher.tick()
    calc = latest_calc(workspace)
    assert chat.handle(f"/approve submit {calc[5:8]}") == f"No calculation {calc[5:8]} in this investigation."
    assert "authorized" in chat.handle(f"/approve submit {calc[:12]}")
    assert workspace.ledger.load(INV).interventions["msg-2"].authorizations[0].target_ids == (calc,)


def test_reports_that_arrive_during_a_chief_turn_are_still_shown(system):
    chat, dispatcher, clients = system["chat"], system["dispatcher"], system["clients"]
    clients.add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="ok")))
    chat.handle("Investigate CO on Cu(111).")
    clients.add("hypothesis_agent", *agent_requests(system))

    def answer_while_the_dispatcher_works(_request):
        dispatcher.tick()  # records the agent's report while the Chief is answering
        return reply(TextPart(text="Noted."))

    clients.add("chief_scientist", answer_while_the_dispatcher_works)
    assert chat.handle("Any news?") == "Noted."
    notices = chat.notices()
    assert any("Requested the PBE baseline" in n for n in notices)
    assert sum("[waiting for you]" in n for n in notices) == 1
    assert chat.notices() == []  # each report and each request for approval is shown once


def test_start_opens_the_full_screen_interface_only_where_it_can_work(system, monkeypatch, capsys):
    from argparse import Namespace

    from sim2real import cli
    from sim2real import tui

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert cli._full_screen(Namespace(plain=False))
    assert not cli._full_screen(Namespace(plain=True))
    monkeypatch.setenv("TERM", "dumb")
    assert not cli._full_screen(Namespace(plain=False))
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert not cli._full_screen(Namespace(plain=False))

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setenv("SIM2REAL_DATA_DIR", str(system["workspace"].root))
    monkeypatch.setattr(cli, "_profiles", lambda: system["profiles"])
    monkeypatch.setattr(cli, "_model_clients", lambda: system["clients"])
    opened = []
    monkeypatch.setattr(tui, "run_tui", lambda session: opened.append(session))
    assert cli.main(["start"]) == 0 and len(opened) == 1 and opened[0].investigation_id is None
