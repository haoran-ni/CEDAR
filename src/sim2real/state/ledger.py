"""The single write path into the research record.

Every write reloads the investigation from the log, runs the new events through
``apply`` (which enforces the research rules), and appends them only if no other writer
got there first. Events carrying an idempotency key are recorded at most once, so a
replayed job-completion or a retried request does not produce a second event.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from sim2real.contracts.base import Actor, new_id, utc_now
from sim2real.contracts.events import (
    CalculationLinked,
    CalculationRequested,
    EventDraft,
    InvestigationCreated,
    RecordedEvent,
    content_hash,
)
from sim2real.contracts.investigation import InvestigationSpec
from sim2real.contracts.plan import ResearchPlan
from sim2real.contracts.task import ScientificTaskSpec
from sim2real.state.projection import InvestigationState, apply, replay, reusable_calculation
from sim2real.store.event_store import ConcurrencyConflict, IdempotencyConflict, SQLiteEventStore


@dataclass(frozen=True)
class RecordResult:
    events: tuple[RecordedEvent, ...]
    state: InvestigationState
    deduplicated: bool = False
    """True if nothing new was recorded because the events already existed."""


@dataclass(frozen=True)
class CalculationRequestOutcome:
    calculation_id: str
    reused: bool
    result: RecordResult


class ResearchLedger:
    def __init__(
        self,
        store: SQLiteEventStore,
        *,
        clock: Callable[[], datetime] = utc_now,
        max_attempts: int = 5,
    ) -> None:
        self._store = store
        self._clock = clock
        self._max_attempts = max_attempts

    def load(self, investigation_id: str) -> InvestigationState:
        state = replay(self._store.read_stream(investigation_id))
        if state is None:
            raise KeyError(f"unknown investigation {investigation_id}")
        return state

    def create_investigation(
        self,
        actor: Actor,
        spec: InvestigationSpec,
        plan: ResearchPlan,
        *,
        investigation_id: str | None = None,
    ) -> RecordResult:
        """Pass a fixed ``investigation_id`` to make creation safe to retry."""
        investigation_id = investigation_id or new_id("inv")
        draft = EventDraft(
            actor=actor,
            payload=InvestigationCreated(spec=spec, plan=plan),
            idempotency_key="investigation_created",
        )
        return self.record(investigation_id, [draft])

    def record(self, investigation_id: str, drafts: Sequence[EventDraft]) -> RecordResult:
        """Validate and atomically append ``drafts``; all or none are recorded."""
        drafts = tuple(drafts)
        return self.record_with(investigation_id, lambda _state: drafts)

    def record_with(
        self,
        investigation_id: str,
        build: Callable[[InvestigationState | None], Sequence[EventDraft]],
    ) -> RecordResult:
        """Like ``record``, but builds the drafts from the freshly loaded state.

        ``build`` is called again on every retry, so decisions that depend on state
        (such as reusing an existing calculation) are made against the state the events
        are validated and appended on.
        """
        for _ in range(self._max_attempts):
            state = replay(self._store.read_stream(investigation_id))
            drafts = tuple(build(state))
            if not drafts:
                if state is None:
                    raise ValueError("nothing to record for a new investigation")
                return RecordResult((), state, deduplicated=True)
            existing = self._already_recorded(investigation_id, drafts)
            if existing is not None:
                return RecordResult(existing, self.load(investigation_id), deduplicated=True)

            expected = state.version if state else 0
            events = []
            for draft in drafts:
                event = self._stamp(investigation_id, state, draft)
                state = apply(state, event)
                events.append(event)
            try:
                self._store.append(events, expected_version=expected)
            except ConcurrencyConflict:
                continue
            return RecordResult(tuple(events), state)
        raise ConcurrencyConflict(
            f"{investigation_id}: gave up after {self._max_attempts} conflicting attempts"
        )

    def request_calculation(
        self,
        investigation_id: str,
        actor: Actor,
        hypothesis_id: str,
        task: ScientificTaskSpec,
        rationale: str,
        *,
        plan_version: int | None = None,
    ) -> CalculationRequestOutcome:
        """Request a calculation, or link the hypothesis to an identical existing one."""
        task_hash = task.identity_hash()
        new_calculation_id = new_id("calc")

        def build(state: InvestigationState | None) -> list[EventDraft]:
            if state is None:
                raise KeyError(f"unknown investigation {investigation_id}")
            existing = reusable_calculation(state, task_hash)
            if existing is None:
                payload = CalculationRequested(
                    calculation_id=new_calculation_id,
                    hypothesis_id=hypothesis_id,
                    task=task,
                    task_hash=task_hash,
                    rationale=rationale,
                )
            elif hypothesis_id in state.calculations[existing].hypothesis_ids:
                return []
            else:
                payload = CalculationLinked(
                    calculation_id=existing, hypothesis_id=hypothesis_id, rationale=rationale
                )
            return [EventDraft(actor=actor, payload=payload, plan_version=plan_version)]

        result = self.record_with(investigation_id, build)
        calculation_id = reusable_calculation(result.state, task_hash)
        if calculation_id is None:
            raise RuntimeError(f"calculation for task {task_hash} missing after it was recorded")
        return CalculationRequestOutcome(calculation_id, calculation_id != new_calculation_id, result)

    def _stamp(self, investigation_id: str, state: InvestigationState | None, draft: EventDraft) -> RecordedEvent:
        return RecordedEvent(
            event_id=new_id("evt"),
            investigation_id=investigation_id,
            stream_version=(state.version if state else 0) + 1,
            occurred_at=self._clock(),
            actor=draft.actor,
            round=state.round if state else 0,
            plan_version=draft.plan_version or (state.plan_version if state else 1),
            idempotency_key=draft.idempotency_key,
            causation_id=draft.causation_id,
            payload=draft.payload,
        )

    def _already_recorded(
        self, investigation_id: str, drafts: Sequence[EventDraft]
    ) -> tuple[RecordedEvent, ...] | None:
        keys = [d.idempotency_key for d in drafts if d.idempotency_key is not None]
        if len(set(keys)) != len(keys):
            raise ValueError("idempotency keys within one batch must be distinct")
        found = self._store.find_by_idempotency_keys(investigation_id, keys)
        if not found:
            return None
        if len(found) != len(drafts):
            raise IdempotencyConflict("batch only partly matches previously recorded events")
        for draft in drafts:
            recorded = found[draft.idempotency_key]
            if content_hash(draft.actor, draft.payload) != content_hash(recorded.actor, recorded.payload):
                raise IdempotencyConflict(
                    f"idempotency key {draft.idempotency_key!r} was already used for different content"
                )
        return tuple(found[d.idempotency_key] for d in drafts)
