"""The Chief Scientist's recorded next steps, and the view of what every hypothesis predicted for each run.

ScriptedModel and FakeScheduler TEST DOUBLES on a SYNTHETIC site; no model or job runs.
"""

import pytest
from factories import CHIEF, USER, acting_for, add_hypothesis, draft, start_investigation, task
from pydantic import ValidationError
from test_orchestration import make_system
from test_swarm import INV, investigation, propose

from cedar.agents.toolkits.chief import ChiefContext, chief_tools
from cedar.contracts import (
    AgentsProposed,
    AgentSpec,
    CandidateAction,
    InvestigationStatus,
    InvestigationStatusChanged,
    NextActionsRecommended,
    PredictionRegistered,
)
from cedar.contracts.hypothesis import Prediction
from cedar.errors import RuleViolation
from cedar.notebook import render_notebook, round_file
from cedar.summaries import calculation_summary, investigation_summary


@pytest.fixture
def system(tmp_path, site):
    return make_system(tmp_path, site)


def candidate(candidate_id="a1", kind="calculation", hypothesis_ids=("hyp-a",), expected=(), **fields):
    return {"candidate_id": candidate_id, "kind": kind, "action": f"do {candidate_id}",
            "hypothesis_ids": list(hypothesis_ids),
            "expected_outcomes": [{"hypothesis_id": h, "outcome": o} for h, o in expected],
            "rationale": f"why {candidate_id}", "cost": "small", "feasibility": "possible now",
            "priority": "medium", "priority_reason": f"because {candidate_id}", **fields}


def test_the_chief_records_the_next_steps_it_weighed_and_carries_out_its_choice(system):
    ledger = system["workspace"].ledger
    chat = investigation(system, "hyp-a", "hyp-b")
    propose(ledger, "p-1", "hyp-a")
    chat.handle("/approve agents")
    chat.notices()
    tools = {t.name: t for t in chief_tools(ChiefContext(
        ledger=ledger, profiles=system["profiles"], registry_for=lambda name: None, actor=CHIEF, investigation_id=INV,
        clock=system["clock"]))}

    def use(name, **arguments):
        return tools[name].invoke(arguments)

    outcome = use("recommend_next_actions", reason="a1 tells A from B at the lowest cost.", chosen=["a1"], candidates=[
        candidate("a1", hypothesis_ids=("hyp-a", "hyp-b"), priority="high",
                  expected=(("hyp-a", "atop lower"), ("hyp-b", "hollow lower")),
                  action="Relax both sites with a hybrid functional.", cost="2 relaxations, about 400 core-hours"),
        candidate("a2", kind="measurement", hypothesis_ids=("hyp-b",), action="Suggest low-temperature IRAS."),
        candidate("a3", kind="close", hypothesis_ids=(), priority="low")])
    assert not outcome.is_error and "3 candidates; taking a1" in outcome.content
    (rec_id,) = ledger.load(INV).recommendations
    assert f"recommendation_id {rec_id}" in outcome.content

    assert not use("delegate", hypothesis_id="hyp-a", instruction="Carry out a1.", recommendation_id=rec_id).is_error
    (task_record,) = [t for t in ledger.load(INV).pending_tasks if t.instruction == "Carry out a1."]
    assert task_record.recommendation_id == rec_id
    outcome = use("delegate", hypothesis_id="hyp-a", instruction="x", recommendation_id="rec-unknown")
    assert outcome.is_error and "unknown recommendation rec-unknown" in outcome.content

    # The researcher sees it once, in full; the Chief's state view gives one line per candidate.
    (shown,) = [n.text for n in chat.notices() if "Next steps recommended" in n.text]
    assert shown.splitlines()[:6] == [
        f"[Next steps recommended by the Chief Scientist ({rec_id}, round 0). Taking: a1. Why: a1 tells A from B at "
        "the lowest cost.",
        "  a1 (chosen; calculation, high priority) on hyp-a, hyp-b: Relax both sites with a hybrid functional.",
        "    if hyp-a holds: atop lower",
        "    if hyp-b holds: hollow lower",
        "    informative because: why a1",
        "    cost: 2 relaxations, about 400 core-hours; feasibility: possible now"]
    assert "  a2 (measurement, medium priority) on hyp-b: Suggest low-temperature IRAS." in shown
    assert shown.endswith("  a3 (close, low priority): do a3\n    informative because: why a3\n    cost: small; "
                          "feasibility: possible now\n    priority because: because a3]")
    assert not [n for n in chat.notices() if "Next steps recommended" in n.text]
    state_view = use("investigation_state").content
    assert "Latest recommendation:\n  Next steps recommended by the Chief Scientist" in state_view
    assert "  a2 (measurement, medium priority) on hyp-b: Suggest low-temperature IRAS." in state_view
    assert "informative because" not in state_view

    pages = render_notebook(system["workspace"].store.read_stream(INV))
    round_page = pages[round_file(ledger.load(INV).round)]
    assert f"- Next steps recommended by Chief Scientist ({rec_id}). Taking: a1. Why: “a1 tells A" in round_page
    assert "  - **a2** (measurement, medium priority) on " in round_page
    assert "holds: “hollow lower”" in round_page
    assert f"- Queued for agent-hyp-a on " in round_page and f"carrying out {rec_id}: “Carry out a1.”" in round_page
    assert f"{rec_id}: 3 next steps compared; taking a1" in pages["timeline.md"]


def test_a_recommendation_must_be_consistent_and_about_known_hypotheses(ledger):
    inv = start_investigation(ledger)
    add_hypothesis(ledger, inv, "hyp-a", agent=False)

    def recommendation(rec_id="rec-1", chosen=(), *candidates):
        return NextActionsRecommended(recommendation_id=rec_id, reason="r", chosen=chosen,
                                      candidates=candidates or (CandidateAction(**candidate()),))

    with pytest.raises(ValidationError, match="chosen names distinct candidates"):
        recommendation("rec-1", ("a9",))
    with pytest.raises(ValidationError, match="candidate ids must be unique"):
        recommendation("rec-1", (), CandidateAction(**candidate()), CandidateAction(**candidate()))
    with pytest.raises(ValidationError, match="a calculation candidate names the hypotheses it bears on"):
        CandidateAction(**candidate(hypothesis_ids=()))
    with pytest.raises(ValidationError, match="expected outcomes are for hypotheses the candidate bears on"):
        CandidateAction(**candidate(expected=(("hyp-b", "x"),)))

    def record(actor, payload):
        ledger.record(inv, [draft(actor, payload)])

    with pytest.raises(RuleViolation, match="user may not record next_actions_recommended"):
        record(USER, recommendation())
    with pytest.raises(RuleViolation, match="unknown hypotheses hyp-b; propose them before"):
        record(CHIEF, recommendation("rec-1", (), CandidateAction(**candidate(hypothesis_ids=("hyp-a", "hyp-b")))))
    record(CHIEF, recommendation())
    with pytest.raises(RuleViolation, match="recommendation rec-1 already exists"):
        record(CHIEF, recommendation())
    with pytest.raises(RuleViolation, match="unknown recommendation rec-9"):
        record(CHIEF, AgentsProposed(proposal_id="p-1", rationale="r", recommendation_id="rec-9", agents=(
            AgentSpec(hypothesis_id="hyp-a", role="r", brief="b"),)))
    record(CHIEF, InvestigationStatusChanged(status=InvestigationStatus.CANCELLED, reason="test"))
    with pytest.raises(RuleViolation, match="investigation is CANCELLED"):
        record(CHIEF, recommendation("rec-2"))


def test_the_chief_sees_what_every_hypothesis_predicted_for_each_calculation(ledger):
    inv = start_investigation(ledger)
    a, b = add_hypothesis(ledger, inv, "hyp-a"), add_hypothesis(ledger, inv, "hyp-b")
    calc = ledger.request_calculation(inv, acting_for(ledger, inv, a), a, task(), "a's test").calculation_id
    ledger.request_calculation(inv, acting_for(ledger, inv, b), b, task(), "the same run tests b")
    assert ledger.load(inv).calculations[calc].hypothesis_ids == [a, b]  # one shared run

    def predict(hypothesis_id, prediction_id, expected, falsified_if, calculation_ids=(calc,)):
        ledger.record(inv, [draft(acting_for(ledger, inv, hypothesis_id), PredictionRegistered(
            prediction_id=prediction_id, hypothesis_id=hypothesis_id, prediction=Prediction(
                statement=f"{hypothesis_id} expects this", observable=task().observable, expected_outcome=expected,
                falsified_if=falsified_if, calculation_ids=calculation_ids)))])

    predict(a, "pred-a", "E(atop) - E(fcc) < 0", "E(atop) - E(fcc) > 0")
    predict(b, "pred-b", "E(atop) - E(fcc) > 0", "E(atop) - E(fcc) < 0")
    predict(b, "pred-b2", "the gap closes at higher coverage", "it does not", calculation_ids=())
    state = ledger.load(inv)
    summary = investigation_summary(state)
    assert ("    predicted by hyp-a (pred-a): E(atop) - E(fcc) < 0; falsified if E(atop) - E(fcc) > 0\n"
            "    predicted by hyp-b (pred-b): E(atop) - E(fcc) > 0; falsified if E(atop) - E(fcc) < 0") in summary
    assert ("  prediction pred-b2 (about no particular calculation): hyp-b expects this Expected: the gap closes at "
            "higher coverage") in summary
    # An agent's own view of the run shows no other hypothesis's predictions.
    assert "predicted by" not in calculation_summary(state, state.calculations[calc], viewer=a)
