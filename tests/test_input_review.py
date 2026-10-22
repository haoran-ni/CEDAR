"""The researcher's say over examined inputs: approving calculations to run, those needing a look included, one by one
or in bulk; rejecting inputs with a reason that reaches the agent; and seeing where the files are.

Everything here is SYNTHETIC: the models are scripted (TEST DOUBLES), the scheduler is the FakeScheduler TEST
DOUBLE, and every directory is under the test's temporary directory.
"""

from pathlib import Path

import pytest
from factories import AGENT, add_hypothesis, agent_for, draft
from test_orchestration import make_system
from test_teamwork import record_run
from worlds import World

from cedar.chat import ChatSession
from cedar.contracts import CalculationState, ExaminationVerdict, HypothesisStatus, PermissionAction
from cedar.contracts.events import HypothesisStatusChanged
from cedar.engine.jobs import MAX_SUBMISSION_ATTEMPTS
from cedar.summaries import gist, short_id
from cedar.workspace import services_for


@pytest.fixture
def system(tmp_path, site):
    return make_system(tmp_path, site)


@pytest.fixture
def world(system):
    workspace = system["workspace"]
    return World(workspace.ledger, workspace.store, workspace.artifacts, system["site"],
                 compute_profile=system["site"].name)


@pytest.fixture
def chat(system, world):
    return ChatSession(system["workspace"], system["profiles"], system["clients"], investigation_id=world.inv,
                       backend=system["scheduler"], clock=system["clock"])


@pytest.fixture
def baseline(world):
    """A calculation whose inputs passed examination: it waits for the researcher to authorize its submission."""
    calc = world.request()
    world.engine.examine(world.inv, calc)
    assert world.calc(calc).state is CalculationState.VALIDATED_INPUT
    return calc


def needs_review(world, baseline, keyword, hypothesis="hyp-xc"):
    """A calculation whose control.in carries a keyword the examiner does not know, so it waits for review."""
    control = world.artifacts.read(world.calc(baseline).examination.files["control.in"])
    unusual = world.artifacts.put_bytes(control.replace(b"\nspin", f"\n{keyword} 1\nspin".encode(), 1),
                                        label="control.in")
    calc = world.ledger.request_calculation(world.inv, agent_for(hypothesis), hypothesis,
                                            world.task(authored={"control.in": unusual}), "unusual").calculation_id
    world.engine.examine(world.inv, calc)
    assert world.calc(calc).examination.verdict is ExaminationVerdict.NEEDS_REVIEW
    return calc


def test_pending_names_calculations_briefly_and_says_where_their_inputs_are(system, world, chat, baseline):
    calc = needs_review(world, baseline, "some_new_keyword")
    pending = chat.handle("/pending")
    runs = Path(system["site"].runs_root)
    assert (f"/approve submit {short_id(calc)}  (A1 agent-hyp-xc on hyp-xc; CO/Cu(111) atop (synthetic): total energy "
            "of CO/Cu(111) atop; through Slurm") in pending
    assert ("core-hours; needs review: some_new_keyword is not known to fhiaims.run 0.1; its effect cannot be verified "
            "(control.in:") in pending
    assert f"in the run directory {runs / world.inv / calc})" in pending
    assert f"/approve submit {short_id(baseline)}  (" in pending and "/approve inputs" not in pending
    assert f"in the run directory {runs / world.inv / baseline})" in pending
    assert "Refuse any of these calculations with /reject inputs <calc> <reason>" in pending
    assert (runs / world.inv / calc / "control.in").read_text().count("some_new_keyword 1") == 1


def test_short_ids_grow_until_they_tell_calculations_apart():
    first, second = "calc_1eea8563dd" + "0" * 17, "calc_1eea8563de" + "0" * 17
    assert short_id(first) == "1eea8563"
    assert short_id(first, [first, second]) == "1eea8563dd" and short_id(second, [first, second]) == "1eea8563de"


def test_the_conversation_shows_the_gist_of_an_agents_message():
    calc = "calc_4870f57d" + "b" * 24
    # The first sentence, or two if the first is short, without Markdown emphasis, ids as the chat takes them.
    assert gist(f"**Done.** Requested `{calc}` at 8x8x1. Then more.\n\nDetails.", [calc]) == \
        "Done. Requested 4870f57d at 8x8x1."
    assert gist("- The 4x4x1 k-grid is not converged for the Xe height, e.g. it moves by 0.05 Å at 8x8x1. It "
                "lowers Xe.") == "The 4x4x1 k-grid is not converged for the Xe height, e.g. it moves by 0.05 Å at 8x8x1."
    long = "The site does not explain the gap: " + "the height stays within 0.01 Å at every site " * 6
    assert len(gist(long)) <= 200 and gist(long).endswith(" within 0.01…")  # at a word


def test_one_approval_lets_a_calculation_run_and_in_bulk_takes_only_inputs_the_researcher_was_shown(
        world, chat, baseline):
    first, second = needs_review(world, baseline, "keyword_one"), needs_review(world, baseline, "keyword_two")
    assert sum("[waiting for you] /approve submit" in n.text for n in chat.notices()) == 3  # the baseline too
    late = needs_review(world, baseline, "keyword_three")  # arrives after the researcher looked

    answer = chat.handle("/approve submit all")
    assert answer.startswith("Approved 3 calculations to run (msg-1). The dispatcher submits them on its next tick:")
    assert baseline in answer and first in answer and second in answer and "keyword_one is not known" in answer
    assert answer.endswith(f"Not approved, because you have not been shown them yet: {late}. /pending shows them.")
    state = world.state()
    assert [state.calculations[c].state for c in (first, second, late)] == [
        CalculationState.VALIDATED_INPUT, CalculationState.VALIDATED_INPUT, CalculationState.SPECIFIED]
    inputs, submission = state.interventions["msg-1"].authorizations
    assert inputs.action is PermissionAction.APPROVE_INPUTS and inputs.target_ids == (first, second)
    assert submission.action is PermissionAction.SUBMIT and submission.target_ids == (baseline, first, second)

    assert chat.handle("/approve submit all") == (f"Not approved, because you have not been shown them yet: {late}. "
                                                  "/pending shows them.")
    assert f"/approve submit {short_id(late)}" in chat.handle("/pending")
    assert chat.handle("/approve submit all").startswith("Approved 1 calculation to run (msg-2).")
    assert chat.handle("/approve submit all") == "No calculation waits for your approval to run."
    assert chat.handle("/approve inputs all").startswith("A calculation's inputs and its submission are approved "
                                                         "together")


def test_approving_one_agents_inputs_leaves_the_others(world, chat, baseline):
    other = add_hypothesis(world.ledger, world.inv, "hyp-b")
    mine, theirs = needs_review(world, baseline, "keyword_one"), needs_review(world, baseline, "keyword_two", other)
    chat.handle("/pending")
    answer = chat.handle("/approve submit A2")
    assert answer.startswith("Approved 1 calculation of A2 agent-hyp-b to run") and theirs in answer
    assert world.calc(theirs).state is CalculationState.VALIDATED_INPUT
    assert world.calc(mine).state is CalculationState.SPECIFIED
    assert chat.handle("/approve submit A2") == "No calculation of A2 agent-hyp-b waits for your approval to run."


def test_a_rejection_reaches_the_agent_with_the_researchers_reason(system, world, chat, baseline):
    calc = needs_review(world, baseline, "some_new_keyword")
    record_run(world, "agent-hyp-xc", world.state().version)  # the agent ended its task before the rejection
    answer = chat.handle(f"/reject inputs {short_id(calc)} don't add keywords the examiner can't check")
    assert answer == (f"Rejected the inputs of {calc} (msg-1). The dispatcher passes your reason to "
                      "A1 agent-hyp-xc (test specialist).")
    rejected = world.calc(calc)
    assert rejected.state is CalculationState.REJECTED and rejected.rejected_by_researcher
    assert (rejected.end_reason, rejected.ended_by) == ("don't add keywords the examiner can't check", "msg-1")
    (authorization,) = world.state().interventions["msg-1"].authorizations
    assert authorization.action is PermissionAction.REJECT_INPUTS and authorization.target_ids == (calc,)
    assert f"/approve submit {short_id(calc)}" not in chat.handle("/pending")

    assert system["dispatcher"]._wake(world.inv) == [f"woke the agent for {world.hyp}"]
    (task,) = world.state().pending_tasks
    assert task.instruction.startswith("Your calculations have ended:")
    assert "rejected by the researcher (msg-1): don't add keywords the examiner can't check" in task.instruction
    assert f"The inputs of {calc} were rejected; who rejected them and why is shown above." in task.instruction
    assert "Look at the outputs" not in task.instruction
    assert system["dispatcher"]._wake(world.inv) == []  # once is enough: the task is queued


def test_an_agent_waiting_for_other_calculations_hears_of_a_rejection_at_once(system, world, chat, baseline):
    calc = needs_review(world, baseline, "some_new_keyword")
    record_run(world, "agent-hyp-xc", world.state().version)
    world.ledger.record(world.inv, [draft(AGENT, HypothesisStatusChanged(
        hypothesis_id=world.hyp, status=HypothesisStatus.WAITING_FOR_JOB, reason="waiting",
        awaiting_calculation_ids=(baseline, calc)))])
    assert system["dispatcher"]._wake(world.inv) == []  # nothing has ended

    chat.handle(f'/reject inputs {short_id(calc)} "use the light defaults unmodified"')
    assert world.calc(calc).end_reason == "use the light defaults unmodified"
    assert system["dispatcher"]._wake(world.inv) == [f"woke the agent for {world.hyp}"]
    hypothesis = world.state().hypotheses[world.hyp]
    assert hypothesis.status is HypothesisStatus.ANALYZING
    (task,) = world.state().pending_tasks
    assert calc in task.instruction and baseline not in task.instruction  # the baseline has not ended


def test_a_calculation_waiting_for_submission_can_be_rejected_until_it_is_submitted(world, chat, baseline):
    assert chat.handle(f"/reject inputs {short_id(baseline)} use an 8x8x1 k-grid").startswith(
        f"Rejected the inputs of {baseline} (msg-1).")
    assert world.calc(baseline).state is CalculationState.REJECTED
    assert chat.handle(f"/reject inputs {short_id(baseline)} again") == (
        f"The inputs of {baseline} are already rejected.")

    submitted = world.request(numerical={"k_grid": [8, 8, 1]})
    world.engine.examine(world.inv, submitted)
    world.approve_submission(submitted)
    world.manager.submit(world.inv, submitted)
    assert chat.handle(f"/reject inputs {short_id(submitted)} too late") == (
        f"{submitted} is already SUBMITTED; /cancel {short_id(submitted)} <reason> stops it.")
    assert chat.handle(f"/reject inputs {short_id(submitted)}").startswith("Usage: /reject inputs <calc> <reason>")
    assert chat.handle("/reject inputs calc_nothing at all") == "No calculation calc_nothing in this investigation."
    assert [i for i in world.state().interventions if i.startswith("msg-")] == ["msg-1"]  # refusals record nothing


def test_an_agent_hears_at_once_that_the_scheduler_refused_its_calculation(system, world, chat, baseline):
    other = needs_review(world, baseline, "some_new_keyword")
    record_run(world, "agent-hyp-xc", world.state().version)
    world.ledger.record(world.inv, [draft(AGENT, HypothesisStatusChanged(
        hypothesis_id=world.hyp, status=HypothesisStatus.WAITING_FOR_JOB, reason="waiting",
        awaiting_calculation_ids=(baseline, other)))])
    assert chat.handle(f"/approve submit {short_id(baseline)}").startswith("Approved 1 calculation")
    refusal = "sbatch: error: Batch job submission failed: Invalid partition name specified"
    system["scheduler"].refusal = refusal
    services = services_for(system["workspace"], system["site"], system["scheduler"])
    dispatcher = system["dispatcher"]
    for _ in range(MAX_SUBMISSION_ATTEMPTS - 1):
        assert dispatcher._submit(world.inv, services) == [
            f"the scheduler refused {baseline}: {refusal}; trying again on the next pass"]
        assert dispatcher._wake(world.inv) == []  # it may still be taken
    assert dispatcher._submit(world.inv, services) == [
        f"the scheduler refused {baseline}: {refusal}; its agent is told"]
    assert world.calc(baseline).state is CalculationState.SUBMISSION_FAILED
    assert dispatcher._submit(world.inv, services) == []  # nothing is tried again

    assert dispatcher._wake(world.inv) == [f"woke the agent for {world.hyp}"]  # without waiting for the other one
    hypothesis = world.state().hypotheses[world.hyp]
    assert hypothesis.status is HypothesisStatus.ANALYZING
    (woken,) = [e.payload for e in system["workspace"].store.read_stream(world.inv)
                if isinstance(e.payload, HypothesisStatusChanged) and e.payload.status is HypothesisStatus.ANALYZING]
    assert woken.reason == f"the scheduler refused {baseline}"
    (task,) = world.state().pending_tasks
    assert (f"the scheduler refused to run it: {refusal} (refused {MAX_SUBMISSION_ATTEMPTS} times)"
            in task.instruction)
    assert f"The scheduler refused to run {baseline}; what it said is shown above." in task.instruction
    assert other not in task.instruction and "Look at the outputs" not in task.instruction
    assert (f"The scheduler refused to run {baseline}: {refusal} (refused {MAX_SUBMISSION_ATTEMPTS} times). Its "
            "agent is asked to request it again with the cause fixed.") == chat.handle(
        f"/approve submit {short_id(baseline)}")
