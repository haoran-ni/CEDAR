"""The input reviewer: an agent judges the scientific content of inputs under fhiaims.run 0.3, while deterministic
checks keep to what runs on the machine and to the task's own declarations.

Everything here is SYNTHETIC: the models are scripted (TEST DOUBLES), the scheduler is the FakeScheduler TEST
DOUBLE, and every directory is under the test's temporary directory.
"""

import pytest
from factories import AGENT, ENGINE, MANAGER, add_hypothesis, agent_for, authorize, draft, move
from test_orchestration import make_system
from test_teamwork import record_run
from worlds import World

from cedar.agents.toolkits.judge import JudgeContext, judge_tools
from cedar.chat import ChatSession
from cedar.contracts import (
    Actor,
    ActorKind,
    AgentTaskQueued,
    CalculationState,
    ExaminationVerdict,
    InputReviewRecorded,
    PermissionAction,
    ReviewDecision,
)
from cedar.dispatch import needs_dispatcher
from cedar.errors import RuleViolation
from cedar.state.projection import MAX_REVIEW_ATTEMPTS, MAX_REVIEW_REJECTIONS
from cedar.summaries import awaiting_review, short_id

REVIEWER = Actor(kind=ActorKind.INPUT_REVIEWER, id="input-reviewer")
SYSTEM = Actor(kind=ActorKind.SYSTEM, id="dispatcher")


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


def examined(world, hypothesis="hyp-xc", **task):
    """A calculation requested under fhiaims.run 0.3 and examined."""
    cid = world.ledger.request_calculation(world.inv, agent_for(hypothesis), hypothesis,
                                           world.task(version="0.3", **task), "test").calculation_id
    world.engine.examine(world.inv, cid)
    return cid


def generated(world, name="control.in") -> bytes:
    """A file the protocol generates for the baseline task, from a calculation cancelled at once so that tests can
    request the same declarations with files of their own."""
    task = world.task(version="0.3").model_copy(update={"estimand": "only for its generated files"})
    cid = world.ledger.request_calculation(world.inv, AGENT, world.hyp, task, "test").calculation_id
    world.engine.examine(world.inv, cid)
    if world.calc(cid).state is CalculationState.SPECIFIED:  # else rejected as identical to one already requested
        move(world.ledger, world.inv, cid, MANAGER, CalculationState.CANCELLED, reason="only for its generated files")
    return world.artifacts.read(world.calc(cid).examination.files[name])


def with_control(world, text: bytes, **task) -> str:
    return examined(world, authored={"control.in": world.artifacts.put_bytes(text, label="control.in")}, **task)


# -- the examination under 0.3 -----------------------------------------------------------------


def test_keywords_no_list_knows_go_to_the_reviewer_as_facts_not_to_the_researcher(world):
    control = generated(world).replace(b"\nspin", b"\nd3\noverride_kgrid_checks .true.\nspin", 1)
    cid = with_control(world, control)
    calc = world.calc(cid)
    exam = calc.examination
    assert exam.verdict is ExaminationVerdict.APPROVED and exam.agent_review
    assert calc.state is CalculationState.SPECIFIED  # waiting for the input reviewer, not for the researcher
    assert exam.settings["d3"] == "" and exam.settings["override_kgrid_checks"] == ".true."
    undeclared = [f for f in exam.findings if f.code == "undeclared_setting"]
    line = control.decode().splitlines().index("d3") + 1
    assert [(f.severity.value, f.line) for f in undeclared] == [("info", line), ("info", line + 1)]
    assert awaiting_review(world.state()) == []


def test_any_keyword_can_be_declared_and_must_then_be_set_as_declared(world):
    cid = examined(world, controls={"d3": True})
    assert world.calc(cid).examination.settings["d3"] == ""  # generated into control.in
    contradicting = generated(world).replace(b"xc pbe", b"xc pbe0", 1)
    calc = world.calc(with_control(world, contradicting, numerical={"k_grid": [3, 3, 1]}))
    assert calc.state is CalculationState.REJECTED
    assert "contradicts_task" in {f.code for f in calc.examination.findings}


def test_facts_about_species_and_structure_are_recorded_for_the_reviewer(world):
    calc = world.calc(with_control(world, generated(world) + b"    cut_pot 4.0 2.0 1.0\n"))  # in the last block
    codes = {f.code: f for f in calc.examination.findings}
    assert codes["species_differs"].severity.value == "info"
    assert "atoms (" in codes["structure"].message and "shortest interatomic distance" in codes["structure"].message
    assert "Å of empty space along the third lattice vector" in codes["structure"].message
    assert calc.examination.verdict is ExaminationVerdict.APPROVED


def test_the_job_script_and_forbidden_keywords_are_still_checked_deterministically(world):
    script = generated(world, "job.sh") + b"echo unexpected\n"
    calc = world.calc(examined(world, authored={"job.sh": world.artifacts.put_bytes(script, label="job.sh")},
                               numerical={"k_grid": [5, 5, 1]}))
    assert calc.examination.verdict is ExaminationVerdict.NEEDS_REVIEW
    assert "unapproved_command" in {f.code for f in calc.examination.findings}
    forbidden = with_control(world, generated(world) + b"postprocess_anyway .true.\n", numerical={"k_grid": [6, 6, 1]})
    assert world.calc(forbidden).state is CalculationState.REJECTED


def test_new_requests_cannot_use_the_superseded_versions(world):
    assert world.registry.current_refs()[0].version == "0.3"
    assert [r.version for r in world.registry.current_refs()] == ["0.3"]


# -- the reviewer's verdicts ------------------------------------------------------------------------


def test_an_approval_lets_the_inputs_wait_for_the_researchers_authorization(world):
    cid = examined(world)
    outcome = world.review(cid)
    assert not outcome.is_error and "approved; the inputs wait for the researcher" in outcome.content
    calc = world.calc(cid)
    assert calc.state is CalculationState.VALIDATED_INPUT
    assert calc.input_review.decision is ReviewDecision.APPROVED
    assert calc.input_review.settings["xc"] == "physical_model"


def test_a_warning_sends_the_inputs_to_the_researcher_with_what_to_look_at(world, chat):
    cid = examined(world)
    world.review(cid, "input_agent_warned", summary="Unusual mixing.",
                 findings=[{"severity": "warning", "text": "Check the smearing width for this metal.",
                            "file": "control.in", "line": 7, "keyword": "occupation_type"}])
    assert world.calc(cid).state is CalculationState.SPECIFIED and awaiting_review(world.state()) == [cid]
    pending = chat.handle("/pending")
    assert (f"/approve inputs {short_id(cid)}  (A1 agent-hyp-xc on hyp-xc; input reviewer: input_agent_warned: "
            "Unusual mixing. warning: Check the smearing width for this metal. (control.in:7)") in pending
    chat.handle("/approve inputs all")
    assert world.calc(cid).state is CalculationState.VALIDATED_INPUT


def test_a_rejection_goes_back_to_the_agent_with_the_reviewers_reasons(system, world):
    cid = examined(world)
    record_run(world, "agent-hyp-xc", world.state().version)
    outcome = world.review(cid, "input_agent_rejected", summary="The k-grid cannot converge this cell.",
                           findings=[{"severity": "problem", "text": "Use at least 8x8x1 k-points.",
                                      "file": "control.in", "line": 5, "keyword": "k_grid"}])
    assert "rejected; the agent is woken" in outcome.content
    calc = world.calc(cid)
    assert calc.rejected_by_reviewer and calc.ended_by == calc.input_review.review_id
    assert calc.end_reason == ("The k-grid cannot converge this cell. Problems: Use at least 8x8x1 k-points. "
                               "(control.in:5)")
    assert system["dispatcher"]._wake(world.inv) == [f"woke the agent for {world.hyp}"]
    (task,) = world.state().pending_tasks
    assert f"rejected by the input reviewer ({calc.ended_by}): The k-grid cannot converge" in task.instruction
    assert f"The inputs of {cid} were rejected" in task.instruction


@pytest.mark.parametrize("decision, findings, problem", [
    ("input_agent_approved", [{"severity": "problem", "text": "x"}], "is input_agent_rejected, not"),
    ("input_agent_rejected", [{"severity": "note", "text": "x"}], "is input_agent_approved, not"),
    ("input_agent_warned", [], "is input_agent_approved, not"),
])
def test_the_decision_must_follow_from_the_findings(world, decision, findings, problem):
    cid = examined(world)
    outcome = world.review(cid, decision, findings=findings)
    assert outcome.is_error and problem in outcome.content
    assert world.calc(cid).input_review is None


def test_every_keyword_must_be_classified(world):
    cid = examined(world)
    outcome = world.review(cid, settings={"xc": "physical_model"})
    assert outcome.is_error and "classify every keyword control.in sets; missing: charge" in outcome.content


def test_quotes_from_local_documentation_are_checked(world, tmp_path):
    manual = tmp_path / "manual.tex"
    manual.write_text("intro\n\\keydefinition{d3}{control.in}\nSwitches on Grimme's D3 dispersion correction.\n")

    def cite(quote, lines=None, source=None):
        return [{"severity": "note", "text": "d3 adds a dispersion correction.", "keyword": "d3",
                 "sources": [{"source": source or str(manual), "quote": quote, "lines": lines}]}]

    cid = examined(world)
    bad = world.review(cid, findings=cite("Switches on Becke-Johnson damping."))
    assert bad.is_error and "the quote is not in" in bad.content
    wrong_lines = world.review(cid, findings=cite("Switches on Grimme's D3", lines="1-2"))
    assert wrong_lines.is_error and "at lines 1-2" in wrong_lines.content
    assert not world.review(cid, findings=cite("Switches on   Grimme's D3\ndispersion", lines="3")).is_error
    (source,) = world.calc(cid).input_review.findings[0].sources
    assert source.verified and source.source == str(manual)

    other = examined(world, numerical={"k_grid": [5, 5, 1]})
    assert not world.review(other, findings=cite("D3", source="https://fhi-aims.org/manual")).is_error
    assert world.calc(other).input_review.findings[0].sources[0].verified is False

    secret = examined(world, numerical={"k_grid": [6, 6, 1]})
    refused = world.review(secret, findings=cite("x"), deny=[tmp_path])
    assert refused.is_error and "not readable by agents" in refused.content


def test_after_three_rejected_revisions_the_researcher_decides(world, chat):
    problem = [{"severity": "problem", "text": "Wrong spin state."}]
    for k in range(MAX_REVIEW_REJECTIONS):
        cid = examined(world, numerical={"k_grid": [3 + k, 3 + k, 1]})
        assert "rejected" in world.review(cid, "input_agent_rejected", findings=problem).content
    last = examined(world, numerical={"k_grid": [9, 9, 1]})
    assert f"you sent back ({MAX_REVIEW_REJECTIONS} of {MAX_REVIEW_REJECTIONS}; you cannot send this one back" \
        in world.reviewer_material(last)
    outcome = world.review(last, "input_agent_rejected", findings=problem)
    assert "the researcher decides, since you sent back the most revisions" in outcome.content
    assert world.calc(last).state is CalculationState.SPECIFIED and awaiting_review(world.state()) == [last]
    assert f"(it sent back {MAX_REVIEW_REJECTIONS} revisions of this request already, so you decide)" in \
        chat.handle("/pending")
    with pytest.raises(RuleViolation, match="the researcher decides on this one"):
        move(world.ledger, world.inv, last, ENGINE, CalculationState.REJECTED, reason="again",
             authorization_id=world.calc(last).input_review.review_id)


def test_an_approved_review_does_not_clear_job_script_findings(world):
    script = generated(world, "job.sh") + b"echo unexpected\n"
    cid = examined(world, authored={"job.sh": world.artifacts.put_bytes(script, label="job.sh")})
    assert "the researcher decides, since the job script needs their review too" in world.review(cid).content
    assert world.calc(cid).state is CalculationState.SPECIFIED and awaiting_review(world.state()) == [cid]


def test_the_researcher_decides_when_the_reviewer_reaches_no_decision(world):
    cid = examined(world)
    for k in range(MAX_REVIEW_ATTEMPTS):
        world.ledger.record(world.inv, [draft(SYSTEM, AgentTaskQueued(
            task_id=f"t{k}", role="input_reviewer", hypothesis_id=world.hyp, calculation_ids=(cid,),
            instruction="Review."))])
        record_run_for(world, f"t{k}")
    assert awaiting_review(world.state()) == [cid]


def record_run_for(world, task_id):
    from cedar.contracts import AgentRunRecorded
    world.ledger.record(world.inv, [draft(REVIEWER, AgentRunRecorded(
        session_id="s", role="input_reviewer", task_id=task_id, provider="p", requested_model="m", turns=1,
        input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0, outcome="finished",
        report="Could not decide."))])


# -- what the record allows -------------------------------------------------------------------------


def test_reviewed_inputs_run_only_after_an_approval(world):
    cid = examined(world)
    with pytest.raises(RuleViolation, match="approve_inputs"):
        move(world.ledger, world.inv, cid, ENGINE, CalculationState.VALIDATED_INPUT)
    with pytest.raises(RuleViolation, match="hypothesis_agent may not record input_review_recorded"):
        world.ledger.record(world.inv, [draft(AGENT, InputReviewRecorded(
            review_id="r", calculation_id=cid, decision=ReviewDecision.APPROVED, summary="fine"))])
    authorize(world.ledger, world.inv, PermissionAction.APPROVE_INPUTS, cid, intervention_id="msg-9")
    world.engine.approve(world.inv, cid)  # the researcher may approve before the reviewer has
    assert world.calc(cid).state is CalculationState.VALIDATED_INPUT


def test_inputs_are_reviewed_once(world):
    cid = examined(world)
    world.review(cid, "input_agent_warned", findings=[{"severity": "warning", "text": "look"}])
    with pytest.raises(RuleViolation, match="were reviewed already"):
        world.ledger.record(world.inv, [draft(REVIEWER, InputReviewRecorded(
            review_id="r2", calculation_id=cid, decision=ReviewDecision.APPROVED, summary="fine"))])


# -- the dispatcher ---------------------------------------------------------------------------------


def test_the_dispatcher_queues_one_review_per_hypothesis_once(system, world):
    other = add_hypothesis(world.ledger, world.inv, "hyp-b")
    mine = [examined(world), examined(world, numerical={"k_grid": [5, 5, 1]})]
    theirs = examined(world, other, numerical={"k_grid": [6, 6, 1]})
    assert needs_dispatcher(world.state())
    dispatcher = system["dispatcher"]
    notes = dispatcher._queue_input_reviews(world.inv)
    assert notes == [f"queued the input reviewer for {', '.join(mine)}", f"queued the input reviewer for {theirs}"]
    assert dispatcher._queue_input_reviews(world.inv) == []
    tasks = [t for t in world.state().pending_tasks if t.role == "input_reviewer"]
    assert [(t.hypothesis_id, t.calculation_ids) for t in tasks] == [(world.hyp, tuple(mine)), (other, (theirs,))]
    assert dispatcher._key(world.inv, tasks[0]) != dispatcher._key(world.inv, tasks[1])  # they run side by side


def test_the_reviewer_is_shown_the_installation_the_files_and_the_other_calculations(world, tmp_path):
    first = examined(world)
    second = with_control(world, generated(world).replace(b"\nspin", b"\nd3\nspin", 1))
    material = world.reviewer_material(second)
    assert f"Installed FHI-aims {world.site.codes['fhi-aims'].version}" in material
    line = world.artifacts.read(world.calc(second).examination.files["control.in"]).decode().splitlines().index("d3")
    assert "--- control.in ---" in material and f"{line + 1:>5}  d3\n" in material
    assert (f"  {first} (the same request), SPECIFIED: CO/Cu(111) atop (synthetic): total energy of CO/Cu(111) "
            "atop; d3: '(absent)' vs ''") in material
    assert "[info] undeclared_setting: d3 is set but not declared by the task (control.in:" in material
    assert "Keywords control.in sets (classify every one of them): " in material


# -- the Judge compares what the inputs actually set ------------------------------------------------


def test_an_undeclared_difference_makes_a_comparison_uncontrolled(world):
    plain = examined(world)
    with_d3 = with_control(world, generated(world).replace(b"\nspin", b"\nd3\nspin", 1))
    judge = {t.name: t for t in judge_tools(JudgeContext(ledger=world.ledger, investigation_id=world.inv,
                                                          actor=Actor(kind=ActorKind.EVIDENCE_JUDGE, id="j")))}
    text = judge["compare_calculations"].invoke({"calculation_ids": [plain, with_d3], "varied": []}).content
    assert text.startswith("Controlled comparison: NO") and "    d3: ['(absent)', '']" in text

    world.review(plain)
    world.review(with_d3, settings={**{k: "physical_model" for k in world.calc(with_d3).examination.settings},
                                    "d3": "output"})
    text = judge["compare_calculations"].invoke({"calculation_ids": [plain, with_d3], "varied": []}).content
    assert "    d3: ['(absent)', ''] (output)" in text and "as the input reviewer classified them" in text
    assert judge["compare_calculations"].invoke({"calculation_ids": [plain, with_d3], "varied": ["d3"]}).content \
        .startswith("Controlled comparison: yes")
    dossier = judge["hypothesis_dossier"].invoke({"hypothesis_id": world.hyp}).content
    assert "  control.in actually sets: " in dossier and "input reviewer (review_" in dossier



# -- after the run ----------------------------------------------------------------------------------


def test_a_relaxation_the_task_does_not_declare_is_still_checked_for_convergence(world):
    from sites import aims_output

    from cedar.contracts import EvidenceQuality, SchedulerState

    cid = with_control(world, generated(world).replace(b"\nspin", b"\nrelax_geometry bfgs 1e-2\nspin", 1))
    world.review(cid)
    world.approve_submission(cid)
    job = world.manager.submit(world.inv, cid)
    (world.runs / world.calc(cid).workdir / "aims.out").write_text(aims_output(relax=True, geometry_converged=False))
    world.scheduler.set_state(job, SchedulerState.COMPLETED)
    world.monitor.poll(world.inv)
    world.validator.check_run(world.inv, cid)
    check = world.calc(cid).run_check
    assert {c.name: c.passed for c in check.checks}["geometry_converged"] is False
    assert check.failure_quality is EvidenceQuality.INCONCLUSIVE
