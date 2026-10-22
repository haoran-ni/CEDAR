"""Status namespaces and their allowed transitions.

Investigation, hypothesis lifecycle, calculation, scheduler, evidence quality and
hypothesis assessment are deliberately separate enums: a finished Slurm job is not a
validated result, and an assessed agent is not a supported hypothesis.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from cedar.errors import InvalidTransition


class InvestigationStatus(StrEnum):
    CREATED = "CREATED"
    ALIGNING_OBSERVABLES = "ALIGNING_OBSERVABLES"
    PLANNING = "PLANNING"
    INVESTIGATING = "INVESTIGATING"
    REVIEWING = "REVIEWING"
    COMPLETED = "COMPLETED"
    INCONCLUSIVE = "INCONCLUSIVE"
    PAUSED = "PAUSED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class HypothesisStatus(StrEnum):
    PROPOSED = "PROPOSED"
    SCREENED = "SCREENED"
    ACTIVE = "ACTIVE"
    PLANNING_TEST = "PLANNING_TEST"
    WAITING_FOR_JOB = "WAITING_FOR_JOB"
    ANALYZING = "ANALYZING"
    ASSESSED = "ASSESSED"
    NEEDS_DATA = "NEEDS_DATA"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    DISCARDED = "DISCARDED"
    SUPERSEDED = "SUPERSEDED"


class CalculationState(StrEnum):
    SPECIFIED = "SPECIFIED"
    VALIDATED_INPUT = "VALIDATED_INPUT"
    REJECTED = "REJECTED"
    SUBMISSION_PENDING = "SUBMISSION_PENDING"
    SUBMISSION_FAILED = "SUBMISSION_FAILED"
    """The scheduler refused to take the job; its agent requests the calculation again with the cause fixed."""
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SCHEDULER_TERMINAL = "SCHEDULER_TERMINAL"
    PARSING = "PARSING"
    QUALITY_ASSESSED = "QUALITY_ASSESSED"
    CANCELLED = "CANCELLED"


class SchedulerState(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    NODE_FAIL = "NODE_FAIL"
    PREEMPTED = "PREEMPTED"
    BOOT_FAIL = "BOOT_FAIL"
    DEADLINE = "DEADLINE"
    UNKNOWN = "UNKNOWN"


SCHEDULER_TERMINAL_STATES = frozenset(SchedulerState) - {
    SchedulerState.PENDING,
    SchedulerState.RUNNING,
    SchedulerState.UNKNOWN,
}


class EvidenceQuality(StrEnum):
    VALIDATED = "VALIDATED"
    PROVISIONAL = "PROVISIONAL"
    INCONCLUSIVE = "INCONCLUSIVE"
    INVALID = "INVALID"


class Assessment(StrEnum):
    SUPPORTED = "SUPPORTED"
    CHALLENGED = "CHALLENGED"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_TESTED = "NOT_TESTED"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class InvestigationMode(StrEnum):
    """Who takes the researcher's routine decisions: the researcher, or, while they are away, the approver."""

    HUMAN_IN_THE_LOOP = "human-in-the-loop"
    FULL_AUTOMATION = "full-automation"


class Completeness(StrEnum):
    PARTIAL = "PARTIAL"
    FINAL = "FINAL"


@dataclass(frozen=True)
class StateMachine[S: StrEnum]:
    """Allowed transitions for one status namespace.

    ``flow`` holds the normal forward edges. A ``suspend`` state can be entered from any
    non-terminal state and left only by resuming to the state it interrupted (or by
    aborting). An ``abort`` state can be entered from any non-terminal state.
    """

    name: str
    flow: Mapping[S, frozenset[S]]
    suspend: frozenset[S] = field(default_factory=frozenset)
    abort: frozenset[S] = field(default_factory=frozenset)
    terminal: frozenset[S] = field(default_factory=frozenset)

    def check(self, current: S, target: S, resume_to: S | None = None) -> None:
        if current in self.terminal:
            raise InvalidTransition(f"{self.name}: {current} is terminal")
        if target == current:
            raise InvalidTransition(f"{self.name}: already {current}")
        if target in self.abort:
            return
        if current in self.suspend:
            if target == resume_to:
                return
            raise InvalidTransition(f"{self.name}: {current} can only resume to {resume_to} or abort")
        if target in self.suspend or target in self.flow.get(current, frozenset()):
            return
        raise InvalidTransition(f"{self.name}: {current} -> {target} is not allowed")

    def advance(self, current: S, target: S, resume_to: S | None = None) -> tuple[S, S | None]:
        """Check a transition and return the new ``(status, resume_to)`` pair."""
        self.check(current, target, resume_to)
        if target in self.suspend:
            return target, current
        return target, None


def _graph[S: StrEnum](edges: Mapping[S, Iterable[S]]) -> dict[S, frozenset[S]]:
    return {source: frozenset(targets) for source, targets in edges.items()}


_I = InvestigationStatus
INVESTIGATION_MACHINE = StateMachine(
    name="investigation",
    flow=_graph({
        _I.CREATED: {_I.ALIGNING_OBSERVABLES},
        _I.ALIGNING_OBSERVABLES: {_I.PLANNING},
        _I.PLANNING: {_I.INVESTIGATING},
        _I.INVESTIGATING: {_I.REVIEWING},
        _I.REVIEWING: {_I.INVESTIGATING, _I.COMPLETED, _I.INCONCLUSIVE},
    }),
    suspend=frozenset({_I.PAUSED, _I.BLOCKED}),
    abort=frozenset({_I.CANCELLED}),
    terminal=frozenset({_I.COMPLETED, _I.INCONCLUSIVE, _I.CANCELLED}),
)

_H = HypothesisStatus
HYPOTHESIS_MACHINE = StateMachine(
    name="hypothesis",
    flow=_graph({
        _H.PROPOSED: {_H.SCREENED, _H.DISCARDED},
        _H.SCREENED: {_H.ACTIVE, _H.DISCARDED},
        _H.ACTIVE: {_H.PLANNING_TEST, _H.NEEDS_DATA},
        _H.PLANNING_TEST: {_H.WAITING_FOR_JOB, _H.ANALYZING, _H.NEEDS_DATA},
        _H.WAITING_FOR_JOB: {_H.ANALYZING, _H.PLANNING_TEST},
        _H.ANALYZING: {_H.ASSESSED, _H.PLANNING_TEST},
        _H.ASSESSED: {_H.PLANNING_TEST},
        _H.NEEDS_DATA: {_H.ACTIVE, _H.PLANNING_TEST},
        _H.FAILED: {_H.ACTIVE},
        _H.DISCARDED: {_H.SCREENED},
    }),
    suspend=frozenset({_H.PAUSED}),
    abort=frozenset({_H.FAILED, _H.SUPERSEDED}),
    terminal=frozenset({_H.SUPERSEDED}),
)

_C = CalculationState
CALCULATION_MACHINE = StateMachine(
    name="calculation",
    flow=_graph({
        _C.SPECIFIED: {_C.VALIDATED_INPUT, _C.REJECTED},
        _C.VALIDATED_INPUT: {_C.SUBMISSION_PENDING, _C.REJECTED},
        _C.SUBMISSION_PENDING: {_C.SUBMITTED, _C.SUBMISSION_FAILED},
        _C.SUBMITTED: {_C.QUEUED, _C.RUNNING, _C.SCHEDULER_TERMINAL},
        _C.QUEUED: {_C.RUNNING, _C.SCHEDULER_TERMINAL},
        _C.RUNNING: {_C.QUEUED, _C.SCHEDULER_TERMINAL},
        _C.SCHEDULER_TERMINAL: {_C.PARSING},
        _C.PARSING: {_C.QUALITY_ASSESSED},
    }),
    abort=frozenset({_C.CANCELLED}),
    terminal=frozenset({_C.QUALITY_ASSESSED, _C.REJECTED, _C.SUBMISSION_FAILED, _C.CANCELLED}),
)
