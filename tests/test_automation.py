"""Full automation: the approver takes the researcher's routine decisions and the Evidence Judge assesses new evidence
before every review, within limits the researcher set; human-in-the-loop is unchanged.

ScriptedModel and FakeScheduler TEST DOUBLES on a SYNTHETIC site; no model or job runs.
"""

import pytest
from factories import CHIEF, USER, add_hypothesis, agent_for, draft, start_investigation
from pydantic import ValidationError
from sites import aims_output
from test_orchestration import (
    HYP,
    INV,
    ROLE,
    agent_reads,
    agent_requests,
    call,
    latest_calc,
    make_system,
    open_and_delegate,
    open_and_start,
)

from sim2real.contracts import (
    ActorKind,
    AgentRunRecorded,
    AgentsStarted,
    ApproverDecision,
    InvestigationMode,
    InvestigationStatus,
    ModeChanged,
    PermissionAction,
    SchedulerState,
)
from sim2real.contracts.base import Actor
from sim2real.dashboard import mode_line
from sim2real.dispatch import SYSTEM
from sim2real.errors import RuleViolation
from sim2real.models import TextPart
from sim2real.models.scripted import reply
from sim2real.notebook import render_notebook
from sim2real.summaries import approval_requests, investigation_summary

APPROVER = Actor(kind=ActorKind.APPROVER, id="approver")


@pytest.fixture
def system(tmp_path, site):
    return make_system(tmp_path, site)


def says(text):
    return reply(TextPart(text=text))


def test_in_full_automation_the_approver_and_the_judge_carry_the_investigation(tmp_path, site):
    system = make_system(tmp_path, site, reviews=True)
    ledger, clients, dispatcher, chat, scheduler = (system["workspace"].ledger, system["clients"],
                                                    system["dispatcher"], system["chat"], system["scheduler"])
    clients.add("chief_scientist", open_and_delegate(site), says("Opened; one agent proposed."))
    chat.handle("Investigate CO on Cu(111).")
    answer = chat.handle("/mode full-automation 1000 2,000,000")
    assert answer.startswith(f"{INV} is in full automation (msg-2). While you are away, an approver decides")
    proposal = ledger.load(INV).pending_proposal
    assert proposal.feedback_ids == []  # a command is not a reply that holds the proposal
    assert approval_requests(ledger.load(INV)) == []  # it is the approver's to decide, not the researcher's

    # The approver starts the proposed agent at once; no review window.
    clients.add("approver", reply(call("approve", target_id=proposal.proposal_id, reason="A cheap baseline first.")),
                says("Approved the baseline agent."))
    notes = dispatcher.tick()
    assert f"{INV}: queued the approver's decisions" in notes
    agent = ledger.load(INV).agents[f"agent-{HYP}"]
    assert agent.started_by.kind is ActorKind.APPROVER
    shown = clients.models["approver"].calls[0].messages[-1].text
    assert f"Agents proposed by the Chief Scientist ({proposal.proposal_id}). Why: one cheap baseline first" in shown
    assert f"  - {HYP} (Semilocal XC misplaces CO frontier orbitals" in shown and f"{ROLE}; first task: " in shown
    texts = [n.text for n in chat.notices()]
    assert (f"[approver, full automation] approved the agents of {proposal.proposal_id}: A cheap baseline "
            "first.") in texts

    # The agent requests a calculation; the approver approves its submission within the core-hours.
    clients.add("hypothesis_agent", *agent_requests(system))
    dispatcher.tick()
    calc = latest_calc(system["workspace"])
    hours = ledger.load(INV).calculations[calc].examination.requested_core_hours
    assert hours and approval_requests(ledger.load(INV)) == []
    clients.add("approver", reply(call("approve", target_id=calc, reason="It is the chosen baseline.")),
                says("Approved the baseline run."))
    dispatcher.tick()
    state = ledger.load(INV)
    assert state.automation_core_hours_left == pytest.approx(1000 - hours)
    assert mode_line(state) == ("Mode: full-automation", True)

    output = aims_output(energy=-5000.123456789)
    scheduler.on_submit = lambda job, workdir: (workdir / "aims.out").write_text(output)
    clients.add("chief_scientist", says("The baseline runs; nothing else to do yet."))
    assert any(f"submitted {calc}" in n for n in dispatcher.tick())
    scheduler.set_state(next(iter(scheduler.jobs)), SchedulerState.COMPLETED)

    # The agent records evidence; the Judge assesses it before the Chief Scientist reviews.
    clients.add("hypothesis_agent", *agent_reads(system, output))
    dispatcher.tick()
    reading = ledger.load(INV).calculations[calc].evidence_ids[0]
    clients.add("evidence_judge",
                reply(call("record_assessment", hypothesis_id=HYP, assessment="INCONCLUSIVE", completeness="PARTIAL",
                           evidence_ids=[reading], rationale="One baseline energy cannot rank adsorption sites.")),
                says("INCONCLUSIVE: a comparison is needed."))
    notes = dispatcher.tick()
    assert f"{INV}: queued the Evidence Judge on {HYP}" in notes
    assert not any("queued the Chief Scientist's review" in n for n in notes)  # it waits for the Judge
    assert ledger.load(INV).hypotheses[HYP].assessment.value == "INCONCLUSIVE"
    clients.add("chief_scientist", says("The Judge finds the baseline inconclusive; a comparison comes next."))
    assert f"{INV}: queued the Chief Scientist's review of the agents' reports" in dispatcher.tick()
    first, second = (c.messages[-1].text for c in clients.models["chief_scientist"].calls[-2:])
    assert f"- approved submitting {calc}: It is the chosen baseline." in first
    assert "The investigation is in full automation (msg-2): the researcher is away." in second
    assert f"- {HYP}: INCONCLUSIVE (partial), citing {reading}" in second
    assert "approved submitting" not in second  # only what is new since the first review


def test_the_researcher_switches_modes_and_sees_which_one_is_on(system, monkeypatch):
    import sim2real.chat

    monkeypatch.setattr(sim2real.chat, "stop_dispatcher", lambda ws, timeout, force: "Stopped.")
    ledger, chat = system["workspace"].ledger, system["chat"]
    open_and_start(system)  # msg-1 and msg-2
    usage = "Usage: /mode, /mode human-in-the-loop, or /mode full-automation <core-hours> <tokens>"
    assert chat.handle("/mode") == f"{INV} is in human-in-the-loop.\n{usage}"
    assert chat.handle("/mode full-automation 500") == usage
    assert chat.handle("/mode full-automation 0 1000").startswith("The mode could not be changed:")
    chat.handle("/mode Full-Automation 500 2000000")
    state = ledger.load(INV)
    assert state.mode is InvestigationMode.FULL_AUTOMATION and state.automation.intervention_id == "msg-3"
    left = "500 of 500 core-hours and 2,000,000 of 2,000,000 tokens left"
    assert f"  mode: full automation: {left}" in investigation_summary(state)
    assert mode_line(state) == ("Mode: full-automation", True)
    assert chat.handle("/mode").startswith(f"{INV} is in full automation: {left}.")
    assert chat.handle("/mode human-in-the-loop") == (f"{INV} is in human-in-the-loop (msg-4): agent proposals and "
                                                      "submissions wait for you again.")

    chat.handle("/mode full-automation 500 2000000")
    assert "  - switched full automation off" in chat.handle("/halt")  # the researcher's brake ends it too
    state = ledger.load(INV)
    assert state.mode is InvestigationMode.HUMAN_IN_THE_LOOP and state.status is InvestigationStatus.PAUSED
    timeline = render_notebook(system["workspace"].store.read_stream(INV))["timeline.md"]
    assert ("Full automation, within 500 core-hours and 2,000,000 tokens: “switched by the researcher (msg-3)”"
            in timeline)
    assert "Back to human-in-the-loop: “halted by the researcher (msg-6)" in timeline


def test_the_approver_decides_only_in_full_automation_and_within_its_core_hours(system):
    from sim2real.automation import approver_due
    from sim2real.state.projection import authorization_for

    ledger, chat, dispatcher, scheduler = (system["workspace"].ledger, system["chat"], system["dispatcher"],
                                           system["scheduler"])
    open_and_start(system)
    system["clients"].add("hypothesis_agent", *agent_requests(system))
    dispatcher.tick()
    calc = latest_calc(system["workspace"])
    hours = ledger.load(INV).calculations[calc].examination.requested_core_hours

    def decide(decision_id="d-1", approved=True):
        ledger.record(INV, [draft(APPROVER, ApproverDecision(decision_id=decision_id, action=PermissionAction.SUBMIT,
                                                             target_id=calc, approved=approved, reason="r"))])

    with pytest.raises(RuleViolation, match="the approver decides only while the investigation is in full automation"):
        decide()
    with pytest.raises(ValidationError, match="the approver does not decide approve_inputs"):
        ApproverDecision(decision_id="d", action=PermissionAction.APPROVE_INPUTS, target_id=calc, approved=True,
                         reason="r")
    with pytest.raises(RuleViolation, match="only the researcher switches full automation on"):
        ledger.record(INV, [draft(SYSTEM, ModeChanged(mode=InvestigationMode.FULL_AUTOMATION, reason="r",
                                                      max_core_hours=1, max_tokens=1))])

    # A submission over the core-hours left is not offered to the approver: it waits for the researcher.
    chat.handle(f"/mode full-automation {hours / 2:g} 1000000")
    with pytest.raises(RuleViolation, match=f"requests {hours:g} core-hours; {hours / 2:g} of full automation's"):
        decide()
    assert not approver_due(ledger.load(INV))
    assert [command for command, _ in approval_requests(ledger.load(INV))] == [f"/approve submit {calc}"]

    # Within them it is offered, once; the approval authorizes the submission while full automation lasts.
    chat.handle(f"/mode full-automation {hours * 2:g} 1000000")
    assert approver_due(ledger.load(INV)) and approval_requests(ledger.load(INV)) == []
    decide()
    assert authorization_for(ledger.load(INV), PermissionAction.SUBMIT, calc) == "d-1"
    with pytest.raises(RuleViolation, match=f"has already decided on {calc} in this stretch"):
        decide("d-2")
    chat.handle("/mode human-in-the-loop")
    assert authorization_for(ledger.load(INV), PermissionAction.SUBMIT, calc) is None
    assert [command for command, _ in approval_requests(ledger.load(INV))] == [f"/approve submit {calc}"]

    # A submission the approval let start is recorded even if full automation ends while sbatch runs.
    chat.handle(f"/mode full-automation {hours * 2:g} 1000000")
    decide("d-3")
    scheduler.on_submit = lambda job, workdir: chat.handle("/mode human-in-the-loop")
    assert any(f"submitted {calc}" in n for n in dispatcher.tick())
    calculation = ledger.load(INV).calculations[calc]
    assert calculation.scheduler_job_id is not None and calculation.state.value != "SUBMISSION_PENDING"
    assert ledger.load(INV).mode is InvestigationMode.HUMAN_IN_THE_LOOP


def test_full_automation_ends_and_pauses_once_its_tokens_are_used(system):
    ledger, chat, dispatcher = system["workspace"].ledger, system["chat"], system["dispatcher"]
    open_and_start(system)
    chat.handle("/mode full-automation 100 50")
    chat.notices()
    ledger.record(INV, [draft(agent_for(HYP), AgentRunRecorded(
        session_id="s", role="hypothesis_agent", provider="p", requested_model="m", turns=1, input_tokens=60,
        output_tokens=0, cache_read_tokens=0, cache_write_tokens=0, outcome="finished"))])
    notes = dispatcher.tick()
    assert (f"{INV}: full automation's allowance of 50 tokens is used up; the investigation waits for the "
            "researcher: back to human-in-the-loop, paused") in notes
    state = ledger.load(INV)
    assert state.mode is InvestigationMode.HUMAN_IN_THE_LOOP and state.status is InvestigationStatus.PAUSED
    assert state.modes[-1].actor.kind is ActorKind.SYSTEM
    assert [n.text for n in chat.notices() if n.text.startswith("[full automation ended]")] == [
        "[full automation ended] full automation's allowance of 50 tokens is used up; the investigation waits for "
        "the researcher. It is paused: ask the Chief Scientist to resume it, or switch full automation on again "
        "with /mode."]


def test_a_declined_proposal_is_withdrawn_and_the_chief_reviews_why(tmp_path, site):
    system = make_system(tmp_path, site, reviews=True)
    ledger, clients, dispatcher, chat = (system["workspace"].ledger, system["clients"], system["dispatcher"],
                                         system["chat"])
    clients.add("chief_scientist", open_and_delegate(site), says("Opened."))
    chat.handle("Investigate CO on Cu(111).")
    chat.handle("/mode full-automation 100 1000000")
    proposal = ledger.load(INV).pending_proposal
    with pytest.raises(RuleViolation, match="in full automation the approver, not the review window, starts"):
        ledger.record(INV, [draft(SYSTEM, AgentsStarted(proposal_id=proposal.proposal_id,
                                                        agent_ids={HYP: f"agent-{HYP}"}))])

    system["clock"].advance(minutes=11)  # past the review window: in full automation it starts nothing
    clients.add("approver", reply(call("decline", target_id=proposal.proposal_id,
                                       reason="The brief names no protocol.")), says("Declined the proposal."))
    notes = dispatcher.tick()
    assert not any("review window" in n for n in notes)
    state = ledger.load(INV)
    assert state.proposals[proposal.proposal_id].status.value == "withdrawn" and not state.agents
    clients.add("chief_scientist", says("I will name the protocol and propose again."))
    assert f"{INV}: queued the Chief Scientist's review of the agents' reports" in dispatcher.tick()
    material = clients.models["chief_scientist"].calls[-1].messages[-1].text
    assert f"- declined starting the agents of {proposal.proposal_id}: The brief names no protocol." in material
    round_page = next(v for k, v in render_notebook(system["workspace"].store.read_stream(INV)).items()
                      if k.startswith("round-01"))
    assert f"- The approver declined starting the agents of {proposal.proposal_id} (decision_" in round_page
