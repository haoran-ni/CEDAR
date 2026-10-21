"""The research notebook: Markdown pages generated from the event log. Synthetic data throughout."""

import re
from pathlib import Path

import pytest
from factories import (
    AGENT,
    CHIEF,
    ENGINE,
    JUDGE,
    SITE_ENERGY,
    USER,
    VALIDATOR,
    add_hypothesis,
    analysis_evidence,
    draft,
    examine,
    investigation_spec,
    move,
    plan,
    record_analysis,
    record_evidence,
    run_to_parsing,
    start_investigation,
    task,
)

from sim2real.agents.toolkits.chief import ChiefContext, chief_tools
from sim2real.contracts import (
    AgentRunRecorded,
    Assessment,
    AssessmentRecorded,
    CalculationState,
    Completeness,
    EvidenceQuality,
    EvidenceRecorded,
    InvestigationStatus,
    InvestigationStatusChanged,
    PlanRevised,
    Prediction,
    PredictionRegistered,
    Priority,
    QualityCheck,
    RoundStarted,
    SchedulerState,
    UserInterventionRecorded,
)
from sim2real.notebook import ROUND_SECTIONS, render_notebook, update_notebook
from sim2real.workspace import Workspace

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "ROUND_REPORT.md"
REPORT_WITH_HEADING = "Baseline done.\n## What is next?\nRun the hybrid functional."


def build(ledger) -> dict:
    """Round 0: setup and a message. Round 1: a validated run, a failed run, a judgement and a plan change."""
    inv = "inv-1"
    ledger.create_investigation(USER, investigation_spec(), plan(), investigation_id=inv)
    for status in (InvestigationStatus.ALIGNING_OBSERVABLES, InvestigationStatus.PLANNING):
        ledger.record(inv, [draft(CHIEF, InvestigationStatusChanged(status=status, reason=f"to {status.value}"))])
    ledger.record(inv, [draft(USER, UserInterventionRecorded(
        intervention_id="msg-1", text="Check the XC explanation first.\nKeep costs low."))])
    ledger.record(inv, [draft(CHIEF, RoundStarted(round=1, reason="first tests")),
                        draft(CHIEF, InvestigationStatusChanged(status=InvestigationStatus.INVESTIGATING,
                                                                reason="plan ready"))])
    hyp = add_hypothesis(ledger, inv, "hyp-xc")
    good = ledger.request_calculation(inv, AGENT, hyp, task(), "baseline at PBE").calculation_id
    ledger.record(inv, [draft(AGENT, PredictionRegistered(prediction_id="pred-1", hypothesis_id=hyp, prediction=Prediction(
        statement="The baseline converges.", observable=SITE_ENERGY, expected_outcome="a validated value",
        falsified_if="the run fails its checks", calculation_ids=(good,))))])
    run_to_parsing(ledger, inv, good, job_id="1001")
    record_evidence(ledger, inv, good, "ev-1")
    record_analysis(ledger, inv, hyp, "an-1", good)
    ledger.record(inv, [draft(VALIDATOR, EvidenceRecorded(evidence_id="ev-gap",
                                                          evidence=analysis_evidence("an-1", good)))])
    bad = ledger.request_calculation(inv, AGENT, hyp, task(xc="PBE0"), "hybrid comparison").calculation_id
    run_to_parsing(ledger, inv, bad, job_id="1002", scheduler_state=SchedulerState.FAILED)
    record_evidence(ledger, inv, bad, "ev-2", quality=EvidenceQuality.INVALID,
                    checks=(QualityCheck(name="scf_converged", passed=False, detail="no converged SCF cycle"),))
    ledger.record(inv, [draft(AGENT, AgentRunRecorded(
        session_id="inv-1.hyp-xc", role="hypothesis_agent", provider="scripted", requested_model="m",
        served_models=("m",), turns=4, input_tokens=10, output_tokens=5, cache_read_tokens=100, cache_write_tokens=0,
        outcome="finished", report=REPORT_WITH_HEADING))])
    ledger.record(inv, [draft(JUDGE, AssessmentRecorded(
        hypothesis_id=hyp, assessment=Assessment.INCONCLUSIVE, completeness=Completeness.PARTIAL,
        evidence_ids=("ev-1",), rationale="One baseline value cannot test a site ordering."))])
    ledger.record(inv, [draft(CHIEF, PlanRevised(version=2, plan=plan(**{hyp: Priority.ON_HOLD}),
                                                 reason="the researcher asked to pause", intervention_id="msg-1"))])
    return {"inv": inv, "hyp": hyp, "good": good, "bad": bad}


@pytest.fixture
def workspace(tmp_path):
    return Workspace.open(tmp_path / "data")


@pytest.fixture
def pages(workspace):
    ids = build(workspace.ledger)
    events = workspace.store.read_stream(ids["inv"])
    return ids, render_notebook(events), events


def headings(text: str) -> list[str]:
    return re.findall(r"^## (.+)$", text, flags=re.MULTILINE)


def section(page: str, title: str) -> str:
    return page.split(f"## {title}\n\n", 1)[1].split("\n## ", 1)[0]


def test_round_pages_use_the_template_headings(pages):
    _, rendered, _ = pages
    assert headings(TEMPLATE.read_text()) == list(ROUND_SECTIONS)
    rounds = sorted(name for name in rendered if name.startswith("round-"))
    assert rounds == ["round-00.md", "round-01.md"]
    # An agent's report containing a heading is quoted, so it cannot add a section.
    assert all(headings(rendered[name]) == list(ROUND_SECTIONS) for name in rounds)
    assert "  > ## What is next?" in rendered["round-01.md"]


def test_each_event_appears_in_the_round_it_belongs_to(pages):
    ids, rendered, _ = pages
    setup, first = rendered["round-00.md"], rendered["round-01.md"]
    assert "> Check the XC explanation first.\n  > Keep costs low." in section(setup, "Researcher's messages")
    assert "[Round 1](round-01.md) started next" in section(setup, "What is next?")
    assert "pred-1" not in setup and ids["good"] not in setup
    assert first.splitlines()[4] == "Started by Chief Scientist: “first tests”"
    assert "pred-1" in section(first, "What did we predict?")
    messages = section(first, "Researcher's messages")
    assert "msg-1" not in messages and f"(authorizes submit of {ids['good']}):" in messages


def test_round_page_answers_from_the_record(pages):
    ids, rendered, _ = pages
    page = rendered["round-01.md"]
    tested = section(page, "What did we test?")
    assert f"**{ids['good']}** for [hyp-xc](hypotheses/hyp-xc.md)" in tested
    assert "Why, by agent-1: “baseline at PBE”" in tested and "Varied: xc = PBE." in tested
    assert f"**{ids['good']}** submitted to the scheduler as job 1001, authorized by approve-{ids['good']}" in tested

    observed = section(page, "What did we observe?")
    assert "**ev-1**" in observed and "**VALIDATED**; site energy difference = 0.1 eV" in observed
    assert "[aims.out line 10](../../artifacts/objects/aa/" + "a" * 64 + "#L10)" in observed
    assert f"**{ids['bad']}**: the job ended FAILED" in observed
    assert "Check failed: scf_converged: no converged SCF cycle" in observed
    assert f"- **{ids['good']}**: run checked, healthy" in observed
    assert f"- **{ids['bad']}**: run checked, failed scf_converged" in observed
    assert ("- **an-1**, a script written for [hyp-xc](hypotheses/hyp-xc.md): “synthetic analysis” **completed**"
            in observed)
    assert "  - Result gap = 7.0 eV" in observed and "[Script](../../artifacts/objects/bb/" in observed
    assert f"- **ev-gap** from analysis an-1 of {ids['good']} for [hyp-xc](hypotheses/hyp-xc.md)" in observed

    assessed = section(page, "What changed our assessment?")
    assert "NOT_TESTED → **INCONCLUSIVE** (partial), by Evidence Judge, citing ev-1" in assessed
    why = section(page, "Why?")
    assert "Plan v2 by Chief Scientist, prompted by msg-1: “the researcher asked to pause”" in why
    assert "hyp-xc normal → on_hold" in why

    unknown = section(page, "What remains unknown?")
    assert "UNASSESSED, so no numerical discrepancy is defined" in unknown
    assert "is INCONCLUSIVE (partial)" in unknown and "ev-2 is INVALID; failed checks: scf_converged." in unknown
    assert "Priorities: hyp-xc on_hold." in section(page, "What is next?")
    reports = section(page, "Agent reports")
    assert reports.startswith("In the agents' own words. A report is not evidence")
    assert "Model tokens processed this round: 115." in reports


def test_index_and_hypothesis_pages(pages):
    ids, rendered, events = pages
    index = rendered["index.md"]
    rows = [line for line in index.splitlines() if line.startswith(f"| {ids['bad']}")]
    assert rows and rows[0].endswith(
        "| 1002 | FAILED | failed scf_converged | ev-2: INVALID, site energy difference = 0.1 eV |")
    assert "- [Round 1](round-01.md): events #5–" in index
    assert "Discrepancy: not defined; observable alignment is UNASSESSED" in index
    hypothesis = rendered["hypotheses/hyp-xc.md"]
    assert "status PLANNING_TEST · assessment INCONCLUSIVE · priority on_hold" in hypothesis
    assert "[aims.out line 10](../../../artifacts/objects/aa/" in hypothesis
    assert "Round 1, event #" in section(hypothesis, "Assessments")
    assert section(hypothesis, "Analyses").startswith("- **an-1**")
    assert "**ev-gap** from analysis an-1" in section(hypothesis, "Evidence")
    assert f"Generated from events #1–#{len(events)};" in index
    timeline = [line for line in rendered["timeline.md"].splitlines() if line.startswith("- #")]
    assert [line.split(" · ")[0] for line in timeline] == [f"- #{e.stream_version}" for e in events]


def test_what_is_next_lists_unfinished_work_and_approvals(workspace):
    inv = start_investigation(workspace.ledger)
    hyp = add_hypothesis(workspace.ledger, inv)
    calc = workspace.ledger.request_calculation(inv, AGENT, hyp, task(), "baseline").calculation_id
    examine(workspace.ledger, inv, calc)
    move(workspace.ledger, inv, calc, ENGINE, CalculationState.VALIDATED_INPUT)
    upcoming = section(render_notebook(workspace.store.read_stream(inv))["round-00.md"], "What is next?")
    assert f"- Waiting for the researcher: `/approve submit {calc}` (CO/Cu(111) p(2x2): " in upcoming
    assert f"- {calc} not finished: VALIDATED_INPUT" in upcoming
    assert "Nothing is queued" not in upcoming


def test_updates_write_only_changed_pages(workspace):
    ids = build(workspace.ledger)
    first = update_notebook(workspace, ids["inv"])
    assert set(first.changed) == {"index.md", "round-00.md", "round-01.md", "timeline.md", "hypotheses/hyp-xc.md"}
    assert first.index.read_text() == render_notebook(workspace.store.read_stream(ids["inv"]))["index.md"]
    assert update_notebook(workspace, ids["inv"]).changed == ()

    workspace.ledger.record(ids["inv"], [draft(USER, UserInterventionRecorded(intervention_id="msg-2",
                                                                              text="What next?"))])
    assert set(update_notebook(workspace, ids["inv"]).changed) == {"index.md", "round-01.md", "timeline.md"}


def test_entering_investigating_starts_a_round(workspace):
    inv = "inv-1"
    workspace.ledger.create_investigation(USER, investigation_spec(), plan(), investigation_id=inv)
    context = ChiefContext(ledger=workspace.ledger, profiles=None, registry_for=lambda n: None, actor=CHIEF,
                           investigation_id=inv)
    status = next(t for t in chief_tools(context) if t.name == "set_investigation_status")
    for target in ("ALIGNING_OBSERVABLES", "PLANNING", "INVESTIGATING", "REVIEWING", "INVESTIGATING"):
        outcome = status.invoke({"status": target, "reason": f"to {target}"})
        assert not outcome.is_error, outcome.content
    assert workspace.ledger.load(inv).round == 2
    # Resuming from a pause continues the same round.
    workspace.ledger.record(inv, [draft(CHIEF, InvestigationStatusChanged(status=InvestigationStatus.PAUSED,
                                                                          reason="pause"))])
    assert "round" not in status.invoke({"status": "INVESTIGATING", "reason": "resume"}).content
    assert workspace.ledger.load(inv).round == 2
    assert set(render_notebook(workspace.store.read_stream(inv))) >= {"round-00.md", "round-01.md", "round-02.md"}
