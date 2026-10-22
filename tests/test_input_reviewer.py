"""The input reviewer: an agent checks that FHI-aims will run the inputs as the calculation the task describes, and
that nothing in them is harmful. Under fhiaims.run 0.4 deterministic checks keep to what runs on the machine; under
0.3, kept for the calculations recorded under it, they also held the files to the task's declarations.

Everything here is SYNTHETIC: the models are scripted (TEST DOUBLES), the scheduler is the FakeScheduler TEST
DOUBLE, and every directory is under the test's temporary directory.
"""

import pytest
from factories import AGENT, ENGINE, MANAGER, add_hypothesis, agent_for, authorize, draft, move
from sites import control_text
from test_orchestration import make_system
from test_teamwork import record_run
from worlds import World

from cedar.agents.toolkits.judge import JudgeContext, judge_tools
from cedar.chat import ChatSession
from cedar.contracts import (
    Actor,
    ActorKind,
    AgentTaskQueued,
    Prediction,
    PredictionRegistered,
    CalculationState,
    ExaminationVerdict,
    InputReviewRecorded,
    PermissionAction,
    ResourceBudget,
    ReviewDecision,
)
from cedar.dispatch import needs_dispatcher
from cedar.errors import RuleViolation
from cedar.protocols import TOTAL_ENERGY
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


def examined(world, hypothesis="hyp-xc", version="0.4", geometry: bytes | None = None, budget=None, **task):
    """A calculation requested under fhiaims.run ``version`` and examined; under 0.4 with a control.in that sets
    what the task declares, unless the task gives one. ``geometry`` replaces the structure, ``budget`` the budget."""
    authored = dict(task.pop("authored", None) or {})
    structure = world.structure if geometry is None else world.artifacts.put_bytes(geometry, label="geometry.in")
    if version == "0.4" and "control.in" not in authored:
        groups = {k: task[k] for k in ("interventions", "controls", "numerical") if k in task}
        text = control_text(world.site, world.artifacts.read(structure).decode(), **groups)
        authored["control.in"] = world.artifacts.put_bytes(text.encode(), label="control.in")
    spec = world.task(version=version, authored=authored, **task).model_copy(
        update={"structure": structure} | ({"budget": budget} if budget is not None else {}))
    cid = world.ledger.request_calculation(world.inv, agent_for(hypothesis), hypothesis, spec, "test").calculation_id
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


# -- the examination under 0.4 ----------------------------------------------------------------------


def test_nothing_about_the_dft_settings_is_enforced(world):
    """Declarations in the agent's own terms, settings they do not mention and keywords no list knows: none of it
    is held against the files, which go to the input reviewer."""
    geometry = world.artifacts.read(world.structure).decode()
    control = control_text(world.site, geometry, interventions={"xc": "pbe0"}).encode().replace(
        b"\nspin", b"\nKS_method parallel\ncompute_forces .true.\nsome_new_keyword 1\nspin", 1)
    cid = with_control(world, control, interventions={"adsorption_site": "top (x, y fixed)", "xc": "pbe"},
                       numerical={"KS_method": "parallel", "compute_forces": True, "n_layers": 4})
    calc = world.calc(cid)
    exam = calc.examination
    assert exam.verdict is ExaminationVerdict.APPROVED and exam.agent_review and not exam.declarations_checked
    assert calc.state is CalculationState.SPECIFIED and awaiting_review(world.state()) == []  # for the reviewer
    assert {f.severity.value for f in exam.findings} <= {"info", "warning"}
    assert exam.settings["xc"] == "pbe0" and exam.settings["ks_method"] == "parallel"
    assert exam.settings["some_new_keyword"] == "1"


def test_the_agent_writes_control_in_and_only_the_job_script_is_generated(world):
    cid = world.ledger.request_calculation(world.inv, AGENT, world.hyp, world.task(version="0.4"), "test").calculation_id
    world.engine.examine(world.inv, cid)
    calc = world.calc(cid)
    assert calc.state is CalculationState.REJECTED and calc.rejected_by is None
    assert calc.end_reason == "control.in is missing; write it, this protocol generates only the job script"
    written = world.calc(examined(world))
    assert sorted(written.examination.files) == ["control.in", "geometry.in", "job.sh"]
    assert [f.file for f in written.examination.findings if f.code == "generated"] == ["job.sh"]


def test_the_job_script_is_still_checked_deterministically(world):
    script = generated(world, "job.sh") + b"echo unexpected\n"
    calc = world.calc(examined(world, authored={"job.sh": world.artifacts.put_bytes(script, label="job.sh")}))
    assert calc.examination.verdict is ExaminationVerdict.NEEDS_REVIEW
    assert "unapproved_command" in {f.code for f in calc.examination.findings}
    over = world.calc(examined(world, numerical={"k_grid": [5, 5, 1]}, budget=ResourceBudget(max_core_hours=0.01),
                               authored={"job.sh": world.artifacts.put_bytes(generated(world, "job.sh"),
                                                                             label="job.sh")}))
    assert over.state is CalculationState.REJECTED and "core-hours requested; the task allows" in over.end_reason


def test_the_structure_and_species_blocks_are_facts_for_the_reviewer(world):
    calc = world.calc(examined(world))
    facts = {f.code: f for f in calc.examination.findings}
    assert facts["species_level"].message == "every species block equals the light species defaults"
    assert "shortest interatomic distance" in facts["structure"].message
    changed = world.calc(with_control(world, generated(world) + b"    cut_pot 4.0 2.0 1.0\n",
                                      numerical={"k_grid": [5, 5, 1]}))
    facts = {f.code: f for f in changed.examination.findings}
    assert facts["species_not_a_default"].severity.value == "info" and "sha256:" in facts["species_level"].message
    odd = world.calc(examined(world, numerical={"k_grid": [6, 6, 1]},
                              geometry=world.artifacts.read(world.structure) + b"atom 1.0 two 3.0 Cu\n"))
    unread = [f for f in odd.examination.findings if f.code == "geometry_unread"]
    assert unread and unread[0].severity.value == "info"  # FHI-aims is the judge of its own input
    assert odd.examination.verdict is ExaminationVerdict.APPROVED


# -- the examination under 0.3, kept for the calculations recorded under it ---------------------


def test_keywords_no_list_knows_go_to_the_reviewer_as_facts_not_to_the_researcher(world):
    control = generated(world).replace(b"\nspin", b"\nd3\noverride_kgrid_checks .true.\nspin", 1)
    cid = with_control(world, control, version="0.3")
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
    cid = examined(world, version="0.3", controls={"d3": True})
    assert world.calc(cid).examination.settings["d3"] == ""  # generated into control.in
    contradicting = generated(world).replace(b"xc pbe", b"xc pbe0", 1)
    calc = world.calc(with_control(world, contradicting, version="0.3", numerical={"k_grid": [3, 3, 1]}))
    assert calc.state is CalculationState.REJECTED
    assert "contradicts_task" in {f.code for f in calc.examination.findings}


def test_facts_about_species_and_structure_are_recorded_for_the_reviewer(world):
    calc = world.calc(with_control(world, generated(world) + b"    cut_pot 4.0 2.0 1.0\n",  # in the last block
                                   version="0.3"))
    codes = {f.code: f for f in calc.examination.findings}
    assert codes["species_differs"].severity.value == "info"
    assert "atoms (" in codes["structure"].message and "shortest interatomic distance" in codes["structure"].message
    assert "Å of empty space along the third lattice vector" in codes["structure"].message
    assert calc.examination.verdict is ExaminationVerdict.APPROVED


def test_the_job_script_and_forbidden_keywords_are_still_checked_deterministically(world):
    script = generated(world, "job.sh") + b"echo unexpected\n"
    calc = world.calc(examined(world, version="0.3", numerical={"k_grid": [5, 5, 1]},
                               authored={"job.sh": world.artifacts.put_bytes(script, label="job.sh")}))
    assert calc.examination.verdict is ExaminationVerdict.NEEDS_REVIEW
    assert "unapproved_command" in {f.code for f in calc.examination.findings}
    forbidden = with_control(world, generated(world) + b"postprocess_anyway .true.\n", version="0.3",
                             numerical={"k_grid": [6, 6, 1]})
    assert world.calc(forbidden).state is CalculationState.REJECTED


def test_new_requests_cannot_use_the_superseded_versions(world):
    assert [r.version for r in world.registry.current_refs()] == ["0.4"]


# -- the reviewer's verdicts ------------------------------------------------------------------------


def test_an_approval_lets_the_inputs_wait_for_the_researchers_authorization(world):
    cid = examined(world)
    outcome = world.review(cid)
    assert not outcome.is_error and "approved; the researcher is asked to approve the run" in outcome.content
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
    assert f"/approve submit {short_id(cid)}  (A1 agent-hyp-xc on hyp-xc; " in pending
    assert ("input reviewer: input_agent_warned: Unusual mixing. warning: Check the smearing width for this metal. "
            "(control.in:7)") in pending
    chat.handle("/approve submit all")
    assert world.calc(cid).state is CalculationState.VALIDATED_INPUT
    assert chat.handle("/pending") == "Nothing waits for your approval."  # the one approval let it run


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
    assert outcome.is_error and "classify every setting the inputs make; missing: charge" in outcome.content


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


def test_the_reviewer_is_shown_the_installation_and_the_files_not_the_science_around_them(world, tmp_path):
    first = examined(world)
    world.ledger.record(world.inv, [draft(AGENT, PredictionRegistered(
        prediction_id="pred-1", hypothesis_id=world.hyp, prediction=Prediction(
            statement="atop is preferred", observable=TOTAL_ENERGY, expected_outcome="lower than fcc",
            falsified_if="higher than fcc", calculation_ids=(first,))))])
    second = with_control(world, generated(world).replace(b"\nspin", b"\nd3\nspin", 1),
                          controls={"dispersion": "D3"})
    material = world.reviewer_material(second)
    assert f"Installed FHI-aims {world.site.codes['fhi-aims'].version}" in material
    line = world.artifacts.read(world.calc(second).examination.files["control.in"]).decode().splitlines().index("d3")
    assert "--- control.in ---" in material and f"{line + 1:>5}  d3\n" in material
    assert "  declared (in the agent's own terms, not checked against the files): " in material
    assert "'dispersion': 'D3'" in material
    assert "Settings the inputs make (classify every one of them): " in material
    # What it checks needs no hypothesis, predictions or other calculations to compare with.
    assert first not in material and "atop is preferred" not in world.reviewer_material(first)
    assert "for hypothesis" not in material
    older = examined(world, version="0.3", numerical={"k_grid": [7, 7, 1]})
    assert "declared (checked against control.in by the examination)" in world.reviewer_material(older)


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
    assert "  its inputs actually set: " in dossier and "input reviewer (review_" in dossier



def test_comparisons_rest_on_what_the_files_set_when_declarations_are_not_checked(world):
    geometry = world.artifacts.read(world.structure).decode()
    coarse = examined(world)
    fine = with_control(world, control_text(world.site, geometry, numerical={"k_grid": [8, 8, 1]}).encode())
    assert world.calc(fine).task.numerical_settings["k_grid"] == [4, 4, 1]  # declared as coarse's, set otherwise
    judge = {t.name: t for t in judge_tools(JudgeContext(ledger=world.ledger, investigation_id=world.inv,
                                                          actor=Actor(kind=ActorKind.EVIDENCE_JUDGE, id="j")))}

    def compare():
        return judge["compare_calculations"].invoke({"calculation_ids": [coarse, fine], "varied": []}).content

    assert compare().startswith("Controlled comparison: NO") and "    k_grid: ['4 4 1', '8 8 1']" in compare()
    for cid in (coarse, fine):
        world.review(cid, settings={**{k: "physical_model" for k in world.calc(cid).examination.settings},
                                    "k_grid": "numerical"})
    text = compare()
    assert text.startswith("Controlled comparison: yes")
    assert "Accuracy settings set differently in the inputs, as the input reviewer classified them:" in text
    assert "    k_grid: ['4 4 1', '8 8 1'] (numerical)" in text and "converged with respect to them" in text


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
    assert any("the settings the task declares are its own description" in text for text in check.limitations)
