import pytest

from cedar.contracts.status import (
    CALCULATION_MACHINE,
    HYPOTHESIS_MACHINE,
    INVESTIGATION_MACHINE,
    CalculationState as C,
    HypothesisStatus as H,
    InvestigationStatus as I,
)
from cedar.errors import InvalidTransition


def walk(machine, start, *targets):
    status, resume_to = start, None
    for target in targets:
        status, resume_to = machine.advance(status, target, resume_to)
    return status, resume_to


def test_investigation_follows_the_documented_lifecycle():
    status, _ = walk(INVESTIGATION_MACHINE, I.CREATED, I.ALIGNING_OBSERVABLES, I.PLANNING,
                     I.INVESTIGATING, I.REVIEWING, I.INVESTIGATING, I.REVIEWING, I.INCONCLUSIVE)
    assert status is I.INCONCLUSIVE


def test_observable_alignment_cannot_be_skipped():
    with pytest.raises(InvalidTransition):
        INVESTIGATION_MACHINE.check(I.CREATED, I.INVESTIGATING)


def test_pause_resumes_only_to_the_interrupted_state():
    status, resume_to = walk(INVESTIGATION_MACHINE, I.INVESTIGATING, I.PAUSED)
    assert (status, resume_to) == (I.PAUSED, I.INVESTIGATING)
    with pytest.raises(InvalidTransition, match="resume"):
        INVESTIGATION_MACHINE.check(status, I.REVIEWING, resume_to)
    assert INVESTIGATION_MACHINE.advance(status, I.INVESTIGATING, resume_to) == (I.INVESTIGATING, None)
    assert INVESTIGATION_MACHINE.advance(status, I.CANCELLED, resume_to) == (I.CANCELLED, None)


@pytest.mark.parametrize("terminal", [I.COMPLETED, I.INCONCLUSIVE, I.CANCELLED])
def test_terminal_investigations_stay_closed(terminal):
    with pytest.raises(InvalidTransition, match="terminal"):
        INVESTIGATION_MACHINE.check(terminal, I.INVESTIGATING)


def test_same_state_is_not_a_transition():
    with pytest.raises(InvalidTransition, match="already"):
        INVESTIGATION_MACHINE.check(I.PLANNING, I.PLANNING)


def test_calculation_cannot_skip_submission():
    with pytest.raises(InvalidTransition):
        CALCULATION_MACHINE.check(C.SPECIFIED, C.RUNNING)
    with pytest.raises(InvalidTransition):
        CALCULATION_MACHINE.check(C.SCHEDULER_TERMINAL, C.QUALITY_ASSESSED)


def test_calculation_can_be_cancelled_until_assessed():
    for state in (C.SPECIFIED, C.SUBMITTED, C.RUNNING, C.PARSING):
        CALCULATION_MACHINE.check(state, C.CANCELLED)
    with pytest.raises(InvalidTransition, match="terminal"):
        CALCULATION_MACHINE.check(C.QUALITY_ASSESSED, C.CANCELLED)


def test_hypothesis_can_be_revived_but_not_after_being_superseded():
    assert walk(HYPOTHESIS_MACHINE, H.ACTIVE, H.FAILED, H.ACTIVE)[0] is H.ACTIVE
    assert walk(HYPOTHESIS_MACHINE, H.PROPOSED, H.DISCARDED, H.SCREENED)[0] is H.SCREENED
    with pytest.raises(InvalidTransition, match="terminal"):
        HYPOTHESIS_MACHINE.check(H.SUPERSEDED, H.ACTIVE)


def test_hypothesis_cycle_returns_to_planning_after_assessment():
    status, _ = walk(HYPOTHESIS_MACHINE, H.PLANNING_TEST, H.WAITING_FOR_JOB, H.ANALYZING,
                     H.ASSESSED, H.PLANNING_TEST)
    assert status is H.PLANNING_TEST
