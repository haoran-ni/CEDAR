"""The dispatcher: moves investigations forward with or without anyone in the chat.

Each pass, for every open investigation:
1. submit calculations the researcher has authorized (never those of on-hold hypotheses); one the scheduler keeps
   refusing fails, and its agent is woken at once with what the scheduler said;
2. poll the scheduler and record job state changes;
3. check the health of every run that ended and store its outputs;
4. start proposed agents whose review window has passed without a reply from the researcher;
5. wake hypothesis agents whose awaited calculations have all ended, or any of whose calculations has ended
   since their last run started;
6. queue a review by the Chief Scientist once every agent given work has reported;
7. start queued agent work, within the budgets, in dollars.

Evidence comes only from agents' cited readings and recorded analyses, after the run check.
Steps 1-3 need the investigation's approved compute profile. Without one, agents still run,
with every tool but those that request calculations or read runs.

Agents run in background threads, so submission and polling go on while they work: several
agents at once (``max_agents``), but never two runs of the same agent, and never two runs
of the Evidence Judge or of the Chief Scientist's review in one investigation. Hypothesis
agents run only while their investigation is INVESTIGATING; the Evidence Judge and the
Chief Scientist's reviews also while it is REVIEWING. A run in progress stops at its next
model call once that no longer holds or its agent is retired, as when the researcher halts
the investigation. Nothing is submitted while an investigation is paused or blocked. Agents running together share the
investigation's budget: before every model call each checks what has been recorded and what every run in progress has
spent so far, so together they overspend it by at most one call each. A run whose model has no known prices does not
start while a budget applies.

The dispatcher then regenerates the research notebook of every investigation whose log
grew, closed ones included.

Only one dispatcher works on a data directory at a time: it holds a lock file for as long
as it runs. The researcher may run it themselves (``cedar dispatch``), or the chat
starts one in the background whenever an investigation has work for it and none is
running (``start_background_dispatcher``); such a dispatcher stops by itself once it has
had nothing to do for a while. It leaves a heartbeat at every pass and after every agent step, so the chat can
warn when no dispatcher is running, and describes each agent run in progress in a file
under ``activity/``, so the live panel can show them.
"""

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cedar.agents import AgentRunner, RunLimits, record_agent_run
from cedar.agents.prompts import APPROVER, CHIEF_SCIENTIST, EVIDENCE_JUDGE, HYPOTHESIS_AGENT, INPUT_REVIEWER
from cedar.agents.toolkits import file_tools, research_tools
from cedar.agents.toolkits.files import agent_access
from cedar.agents.toolkits.approver import ApproverContext, approver_tools, decisions_text
from cedar.agents.toolkits.chief import ChiefContext, chief_tools
from cedar.agents.toolkits.judge import JudgeContext, judge_tools
from cedar.agents.toolkits.reviewer import ReviewerContext, reviewer_tools
from cedar.agents.toolkits.materials import material_tools, recorded_materials
from cedar.contracts.base import Actor, ActorKind, new_id
from cedar.automation import approver_due, judgements_due
from cedar.briefing import agent_briefing, review_briefing, role_briefing, session_name
from cedar.contracts.events import (
    AgentTaskQueued,
    EventDraft,
    HypothesisStatusChanged,
    InvestigationStatusChanged,
    ModeChanged,
)
from cedar.contracts.plan import PermissionAction, Priority
from cedar.contracts.status import (
    INVESTIGATION_MACHINE,
    CalculationState,
    HypothesisStatus,
    InvestigationMode,
    InvestigationStatus,
)
from cedar.engine.jobs import InputsChanged
from cedar.engine.workbench import Workbench
from cedar.errors import RuleViolation
from cedar.hpc.local import LocalBackend
from cedar.hpc.node import local_dft
from cedar.hpc.profile import ProfileNotApproved, ProfileStore
from cedar.hpc.slurm import JobRefused, SchedulerBackend, SchedulerError
from cedar.models.prices import dollars, prices_for
from cedar.notebook import update_notebook
from cedar.state.projection import (AgentTaskRecord, CalculationRecord, InvestigationState, authorization_for,
                                    awaits_reviewer, review_deadline)
from cedar.summaries import calculation_summary, task_owner
from cedar.swarm import (
    CHIEF,
    MAX_CONTINUATIONS,
    REVIEW_INSTRUCTION,
    WAKE_TASK,
    account_of_run,
    agent_prompt,
    continuation_text,
    ended_unseen,
    review_due,
    review_material,
    start_drafts,
    task_chain,
)
from cedar.web import web_access
from cedar.workspace import (ModelClients, Services, Workspace, investigation_registrar, profile_for, registry_for,
                             research_context, services_for)

SYSTEM = Actor(kind=ActorKind.SYSTEM, id="dispatcher")
JUDGE = Actor(kind=ActorKind.EVIDENCE_JUDGE, id="evidence-judge")
APPROVER_ACTOR = Actor(kind=ActorKind.APPROVER, id="approver")
REVIEWER_ACTOR = Actor(kind=ActorKind.INPUT_REVIEWER, id="input-reviewer")
_ENDED = frozenset({CalculationState.SCHEDULER_TERMINAL, CalculationState.PARSING, CalculationState.QUALITY_ASSESSED,
                    CalculationState.REJECTED, CalculationState.SUBMISSION_FAILED, CalculationState.CANCELLED})
_ORDER = {Priority.HIGH: 0, Priority.NORMAL: 1, Priority.LOW: 2, Priority.ON_HOLD: 3}
HEARTBEAT = "dispatcher.heartbeat"
ACTIVITY = "activity"
LOCK = "dispatcher.lock"
PID = "dispatcher.pid"
HOST = "dispatcher.host"
"""The machine the dispatcher holding the lock runs on; local calculations start there."""
LOG = "dispatcher.log"
IDLE_MINUTES = 30
"""A dispatcher the chat started stops after this long with nothing to do."""
_WORKING = frozenset({InvestigationStatus.INVESTIGATING, InvestigationStatus.REVIEWING})
"""Investigation states in which work goes on: submissions, the Evidence Judge and reviews."""
_LIVE = frozenset({CalculationState.SUBMISSION_PENDING, CalculationState.SUBMITTED, CalculationState.QUEUED,
                   CalculationState.RUNNING, CalculationState.SCHEDULER_TERMINAL, CalculationState.PARSING})
_RESTING = frozenset({HypothesisStatus.PAUSED, HypothesisStatus.DISCARDED, HypothesisStatus.SUPERSEDED,
                      HypothesisStatus.FAILED})
"""Hypotheses whose agents are not woken for results."""
_LOOK_AT_OUTPUTS = (
    "Look at the outputs (run_outputs, file tools). Record the values that matter: a value printed in an output with "
    "submit_reading, citing its lines; anything that must be computed with run_analysis and then record_result.")
_ANSWER_REJECTION = (
    "The inputs of {calcs} were rejected; who rejected them and why is shown above. Address the reasons: change what "
    "they concern and request the calculation again, or, if you cannot or disagree, say why in your report and ask "
    "the researcher. Never request the same inputs again unchanged.")
_ANSWER_REFUSAL = (
    "The scheduler refused to run {calcs}; what it said is shown above. Its inputs had been examined and approved, so "
    "the cause is usually in how the job asks to run (the job script's partition, time limit, task count or account) "
    "or a limit of the cluster; it can also be a passing problem of the scheduler. Request the calculation again with "
    "the cause fixed: the same DFT inputs may be given again, and the request is reviewed and approved again. If the "
    "fix needs the compute profile or the cluster changed, say so in your report and ask the researcher.")


def wake_instruction(state: InvestigationState, ended: list[CalculationRecord]) -> str:
    """The task that wakes an agent for calculations of its that have ended: what each came to, and what to do."""
    rejected = [c.calculation_id for c in ended if c.refused]
    refused = [c.calculation_id for c in ended if c.submission_failed]
    lines = [WAKE_TASK, *(calculation_summary(state, c, findings=False) for c in ended)]
    if len(rejected) + len(refused) < len(ended):
        lines.append(_LOOK_AT_OUTPUTS)
    if rejected:
        lines.append(_ANSWER_REJECTION.format(calcs=", ".join(rejected)))
    if refused:
        lines.append(_ANSWER_REFUSAL.format(calcs=", ".join(refused)))
    lines.append("Say what this means for your hypothesis, then set its status (ASSESSED, or PLANNING_TEST if another "
                 "test is needed).")
    return "\n".join(lines)


def needs_dispatcher(state: InvestigationState) -> bool:
    """Whether the investigation has work only a dispatcher carries out: a proposal waiting out its review
    window, agent work that can run, a review that is due, in full automation a decision or an assessment to
    queue (or its allowance to end), inputs waiting for the input reviewer, or a calculation authorized, at the
    scheduler or waiting for its run check."""
    if state.status in INVESTIGATION_MACHINE.terminal:
        return False
    proposal = state.pending_proposal
    if proposal is not None and not proposal.feedback_ids:
        return True
    if any(Dispatcher._runnable(state, t) for t in state.pending_tasks) or review_due(state):
        return True
    if state.automation is not None and (approver_due(state) or judgements_due(state)
                                         or state.automation_cost_left <= 0):
        return True
    for calc in state.calculations.values():
        if calc.state in _LIVE or (state.status in _WORKING and awaits_reviewer(state, calc)):
            return True
        if (calc.state is CalculationState.VALIDATED_INPUT and state.status in _WORKING
                and not all(state.plan.priority(h) is Priority.ON_HOLD for h in calc.hypothesis_ids)
                and (not state.spec.constraints.submit_requires_approval
                     or authorization_for(state, PermissionAction.SUBMIT, calc.calculation_id) is not None)):
            return True
    return False


@dataclass(frozen=True)
class Activity:
    """One agent run the dispatcher is carrying out right now."""

    investigation_id: str
    task_id: str
    role: str
    hypothesis_id: str | None
    agent_id: str | None
    started_at: datetime
    last_step: str | None = None
    """The tool the agent called most recently."""
    last_step_at: datetime | None = None
    calls: int = 0
    """Model calls so far."""
    tokens: int = 0
    """Tokens processed so far, cached input included."""
    cost_usd: float | None = None
    """Dollars spent so far, when the model's prices are known."""


def _write_atomically(path: Path, text: str) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.partial")
    partial.write_text(text)
    os.replace(partial, path)


@dataclass
class _Run:
    investigation_id: str
    task: AgentTaskRecord
    spent: float = 0.0
    """Dollars it has spent so far."""
    note: str = ""
    done: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class Dispatcher:
    def __init__(self, workspace: Workspace, profiles: ProfileStore, clients: ModelClients, *,
                 backend: SchedulerBackend | None = None, deny: list[Path] = (),
                 limits: RunLimits = RunLimits(), max_agents: int = 4,
                 analysis_slots: int = 2, chief_reviews: bool = True,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 workbench: Workbench | None = None) -> None:
        """``max_agents`` agent runs at once; ``analysis_slots`` analysis scripts at once on this node; the
        ``workbench`` runs agents' commands, sharing this machine's cores between them."""
        self.workspace = workspace
        self.profiles = profiles
        self.clients = clients
        self.backend = backend
        self.deny = list(deny)
        self.limits = limits
        self.max_agents = max_agents
        self.chief_reviews = chief_reviews
        self.clock = clock
        self._analysis_slots = threading.BoundedSemaphore(analysis_slots)
        self.workbench = workbench or Workbench(workspace.ledger, workspace.artifacts)
        self._local_backends: dict[str, LocalBackend] = {}
        self._runs: dict[str, _Run] = {}
        self._held = None
        self._notebook_versions: dict[str, int] = {}

    # -- running --------------------------------------------------------------------------------

    def tick(self) -> list[str]:
        """One pass over every investigation, then wait for the agent runs it started."""
        with self._exclusive() as mine:
            if not mine:
                return ["another dispatcher is working on this data directory; skipped"]
            notes = self._pass()
            notes += self.wait()
            notes += self._refresh_notebooks()
            self._beat()
            return notes

    def run_forever(self, interval: float, log: Callable[[str], None] = print,
                    stop: threading.Event | None = None, idle_seconds: float | None = None) -> None:
        """Pass every ``interval`` seconds while agents work in the background; holds the lock throughout.

        It ends, after the agent runs in progress, when ``stop`` is set, on SIGTERM (``cedar dispatch
        --stop``), or once it has had nothing to do for ``idle_seconds``. On Ctrl-C it also waits for the runs,
        so none is cut off; a second Ctrl-C stops at once.
        """
        stop = stop or threading.Event()
        with self._exclusive() as mine:
            if not mine:
                log("another dispatcher is working on this data directory; stopping")
                return
            pid_path = self.workspace.root / PID
            _write_atomically(pid_path, str(os.getpid()))
            terminate = None
            if threading.current_thread() is threading.main_thread():
                terminate = signal.signal(signal.SIGTERM, lambda *_: stop.set())
            idle_since = None
            try:
                while not stop.is_set():
                    for note in self._pass():
                        log(f"{self.clock():%Y-%m-%d %H:%M:%S} {note}")
                    idle_since = None if self._has_work() else (idle_since or time.monotonic())
                    if idle_seconds is not None and idle_since is not None \
                            and time.monotonic() - idle_since >= idle_seconds:
                        log(f"{self.clock():%Y-%m-%d %H:%M:%S} nothing to do for {idle_seconds / 60:g} min; "
                            "stopping (the chat starts a dispatcher again when there is work)")
                        break
                    stop.wait(interval)
                for note in self.wait():
                    log(note)
            except KeyboardInterrupt:
                if self._runs:
                    log(f"Stopping: waiting for {len(self._runs)} agent run{'s' if len(self._runs) > 1 else ''} to "
                        "finish (Ctrl-C again stops at once; an interrupted run resumes next time).")
                    for note in self.wait():
                        log(note)
                raise
            finally:
                if terminate is not None:
                    signal.signal(signal.SIGTERM, terminate)
                pid_path.unlink(missing_ok=True)

    def _has_work(self) -> bool:
        if self._runs:
            return True
        return any(needs_dispatcher(self.workspace.ledger.load(i)) for i in self.workspace.store.investigation_ids())

    def wait(self) -> list[str]:
        """Wait for every agent run in progress to end; their notes."""
        for run in list(self._runs.values()):
            while not run.done.wait(0.5):
                pass
        return self._collect()

    @contextmanager
    def _exclusive(self) -> Iterator[bool]:
        if self._held is not None:
            yield True
            return
        lock = open(self.workspace.root / LOCK, "a")
        try:
            fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()
            yield False
            return
        self._held = lock
        _write_atomically(self.workspace.root / HOST, socket.gethostname())
        self._clear_activity()  # descriptions left by a dispatcher that stopped mid-run
        try:
            yield True
        finally:
            self._held = None
            fcntl.lockf(lock, fcntl.LOCK_UN)
            lock.close()

    def _pass(self) -> list[str]:
        self._beat()
        notes = self._collect()
        for investigation_id in self.workspace.store.investigation_ids():
            notes += self._advance(investigation_id)
        notes += self._refresh_notebooks()
        self._beat()
        return notes

    def _beat(self) -> None:
        _write_atomically(self.workspace.root / HEARTBEAT, self.clock().isoformat())

    # -- one investigation ---------------------------------------------------------------------

    def _advance(self, investigation_id: str) -> list[str]:
        state = self.workspace.ledger.load(investigation_id)
        if state.status in INVESTIGATION_MACHINE.terminal:
            return []
        notes = []
        try:
            profile = profile_for(state, self.profiles)
            services = services_for(self.workspace, profile, self.backend, deny=self.deny,
                                    analysis_slots=self._analysis_slots, local_backend=self._local_backend(profile))
        except (LookupError, ProfileNotApproved, FileNotFoundError) as exc:
            services = None  # agents can still plan and report; calculations wait for a usable profile
            if state.calculations:
                notes.append(f"calculations wait for a usable compute profile ({exc})")
        if services is not None:
            notes += self._submit(investigation_id, services)
            notes += self._poll(investigation_id, services)
            notes += self._check_runs(investigation_id, services)
        notes += self._end_spent_automation(investigation_id)
        notes += self._start_proposed(investigation_id)
        notes += self._wake(investigation_id)
        notes += self._queue_input_reviews(investigation_id)
        notes += self._queue_judgements(investigation_id)
        notes += self._queue_approver(investigation_id)
        notes += self._queue_review(investigation_id)
        notes += self._launch(investigation_id, services)
        return [f"{investigation_id}: {n}" for n in notes]

    def _local_backend(self, profile) -> LocalBackend:
        """One per runs directory, so it keeps track of the local runs it started; it shares this machine's cores
        with the workbench."""
        if profile.runs_root not in self._local_backends:
            self._local_backends[profile.runs_root] = LocalBackend(profile.runs_root, host=self.workbench.host,
                                                                   pool=self.workbench.pool)
        return self._local_backends[profile.runs_root]

    def _submit(self, investigation_id: str, services: Services) -> list[str]:
        state = self.workspace.ledger.load(investigation_id)
        notes = []
        for calc in state.calculations.values():
            if calc.state is CalculationState.VALIDATED_INPUT:
                if state.status not in _WORKING:
                    continue
                if all(state.plan.priority(h) is Priority.ON_HOLD for h in calc.hypothesis_ids):
                    continue
                if state.spec.constraints.submit_requires_approval and authorization_for(
                        state, PermissionAction.SUBMIT, calc.calculation_id) is None:
                    continue
            elif calc.state is not CalculationState.SUBMISSION_PENDING:
                continue
            if calc.task.execution == "local":
                problem = services.local_backend.unavailable(self.workbench.host())
                if problem:
                    notes.append(f"{calc.calculation_id} waits: {problem}")
                    continue
            try:
                job = services.manager_for(calc).submit(investigation_id, calc.calculation_id)
                notes.append(f"submitted {calc.calculation_id} as job {job}")
            except JobRefused as exc:
                now = self.workspace.ledger.load(investigation_id).calculations[calc.calculation_id]
                notes.append(f"the scheduler refused {calc.calculation_id}: {exc}; "
                             + ("its agent is told" if now.submission_failed else "trying again on the next pass"))
            except (RuleViolation, InputsChanged, SchedulerError, OSError) as exc:
                notes.append(f"could not submit {calc.calculation_id}: {exc}")
        return notes

    def _poll(self, investigation_id: str, services: Services) -> list[str]:
        notes = []
        for monitor in (services.monitor, services.local_monitor):
            try:
                result = monitor.poll(investigation_id)
            except SchedulerError as exc:
                notes.append(f"{monitor.execution} query failed: {exc}")
                continue
            notes += [f"{e.payload.calculation_id} is {e.payload.state.value}"
                      + (f" ({e.payload.scheduler_state.value})" if e.payload.scheduler_state else "")
                      for e in result.recorded]
        return notes

    def _check_runs(self, investigation_id: str, services: Services) -> list[str]:
        notes = []
        for calc in self.workspace.ledger.load(investigation_id).calculations.values():
            if calc.state not in (CalculationState.SCHEDULER_TERMINAL, CalculationState.PARSING):
                continue
            try:
                result = services.validator.check_run(investigation_id, calc.calculation_id)
            except (RuleViolation, OSError) as exc:
                notes.append(f"could not check the run of {calc.calculation_id}: {exc}")
                continue
            check = result.state.calculations[calc.calculation_id].run_check
            failed = [c.name for c in check.checks if not c.passed]
            notes.append(f"checked the run of {calc.calculation_id}: "
                         + (f"failed {', '.join(failed)}" if failed else "healthy"))
        return notes

    def _start_proposed(self, investigation_id: str) -> list[str]:
        """Start a proposal whose review window passed without a word from the researcher."""
        state = self.workspace.ledger.load(investigation_id)
        proposal = state.pending_proposal
        if (proposal is None or proposal.feedback_ids or state.status is not InvestigationStatus.INVESTIGATING
                or state.automation is not None or self.clock() < review_deadline(state, proposal)):
            return []  # in full automation the approver decides instead

        def build(current: InvestigationState) -> list[EventDraft]:
            waiting = current.pending_proposal
            if waiting is None or waiting.proposal_id != proposal.proposal_id or waiting.feedback_ids:
                return []
            return start_drafts(current, waiting, SYSTEM)

        try:
            result = self.workspace.ledger.record_with(investigation_id, build)
        except RuleViolation as exc:
            return [f"could not start the agents of {proposal.proposal_id}: {exc}"]
        if result.deduplicated:
            return []
        started = result.events[0].payload.agent_ids.values()
        return [f"the review window of {proposal.proposal_id} passed without a reply; started {', '.join(started)}"]

    def _wake(self, investigation_id: str) -> list[str]:
        """Wake an agent once the calculations it waits for have all ended, or, when it is not waiting, once any of
        its calculations has ended that it has not been told of: since its last run started."""
        state = self.workspace.ledger.load(investigation_id)
        notes = []
        for hyp in state.hypotheses.values():
            agent = state.active_agent(hyp.hypothesis_id)
            if agent is None:
                continue
            drafts = []
            runs = [r for r in state.agent_runs if r.actor_id == agent.agent_id]
            last = runs[-1] if runs else None
            since = 0 if last is None else last.run.started_version if last.run.started_version is not None \
                else last.recorded_at
            if hyp.status is HypothesisStatus.WAITING_FOR_JOB:
                awaited = [state.calculations[c] for c in hyp.awaiting_calculation_ids]
                # A rejection by the researcher or the input reviewer, or the scheduler's refusal, is answered at
                # once, without waiting for the other calculations.
                rejected = [state.calculations[c] for c in ended_unseen(state, hyp.hypothesis_id, since)
                            if state.calculations[c].refused or state.calculations[c].submission_failed]
                if all(c.state in _ENDED for c in awaited):
                    ended = awaited + [c for c in rejected if c not in awaited]
                    why = "the calculations it was waiting for have ended"
                elif rejected and state.plan.priority(hyp.hypothesis_id) is not Priority.ON_HOLD:
                    ended = rejected
                    inputs = [c.calculation_id for c in rejected if c.refused]
                    jobs = [c.calculation_id for c in rejected if c.submission_failed]
                    why = "; ".join(([f"the inputs of {', '.join(inputs)} were rejected"] if inputs else [])
                                    + ([f"the scheduler refused {', '.join(jobs)}"] if jobs else []))
                else:
                    continue
                drafts.append(EventDraft(actor=SYSTEM, payload=HypothesisStatusChanged(
                    hypothesis_id=hyp.hypothesis_id, status=HypothesisStatus.ANALYZING, reason=why)))
            else:
                if (not runs or hyp.status in _RESTING or state.plan.priority(hyp.hypothesis_id) is Priority.ON_HOLD
                        or any(t.agent_id == agent.agent_id for t in state.pending_tasks)):
                    continue
                ended = [state.calculations[c] for c in ended_unseen(state, hyp.hypothesis_id, since)]
                if not ended:
                    continue
            drafts.append(EventDraft(actor=SYSTEM, payload=AgentTaskQueued(
                task_id=new_id("task"), role="hypothesis_agent", hypothesis_id=hyp.hypothesis_id,
                agent_id=agent.agent_id, instruction=wake_instruction(state, ended))))
            self.workspace.ledger.record(investigation_id, drafts)
            notes.append(f"woke the agent for {hyp.hypothesis_id}")
        return notes

    def _end_spent_automation(self, investigation_id: str) -> list[str]:
        """Once full automation's spending allowance is used up, switch back to human-in-the-loop and pause."""
        state = self.workspace.ledger.load(investigation_id)
        if state.automation is None or state.automation_cost_left > 0:
            return []
        reason = (f"full automation's allowance of {dollars(state.automation.max_usd)} is spent; the investigation "
                  "waits for the researcher")

        def build(current: InvestigationState) -> list[EventDraft]:
            if current.automation is None or current.automation_cost_left > 0:
                return []
            drafts = [EventDraft(actor=SYSTEM, payload=ModeChanged(mode=InvestigationMode.HUMAN_IN_THE_LOOP,
                                                                   reason=reason))]
            if current.status not in INVESTIGATION_MACHINE.suspend | INVESTIGATION_MACHINE.terminal:
                drafts.append(EventDraft(actor=SYSTEM, payload=InvestigationStatusChanged(
                    status=InvestigationStatus.PAUSED, reason=reason)))
            return drafts

        result = self.workspace.ledger.record_with(investigation_id, build)
        return [] if result.deduplicated or not result.events else [f"{reason}: back to human-in-the-loop, paused"]

    def _queue_input_reviews(self, investigation_id: str) -> list[str]:
        """The input reviewer reviews inputs that wait for it, one task per hypothesis for the calculations its
        agent requested, so it sees them together."""

        def build(state: InvestigationState) -> list[EventDraft]:
            if state.status not in _WORKING:
                return []
            taken = {c for t in state.pending_tasks if t.role == "input_reviewer" for c in t.calculation_ids}
            waiting: dict[str, list[str]] = {}
            for calc in state.calculations.values():
                if calc.calculation_id not in taken and awaits_reviewer(state, calc):
                    waiting.setdefault(calc.hypothesis_ids[0], []).append(calc.calculation_id)
            return [EventDraft(actor=SYSTEM, payload=AgentTaskQueued(
                task_id=new_id("task"), role="input_reviewer", hypothesis_id=hypothesis_id, calculation_ids=tuple(ids),
                instruction=f"Review the inputs of {', '.join(ids)}, requested for {hypothesis_id}, before they can "
                            "run.")) for hypothesis_id, ids in waiting.items()]

        result = self.workspace.ledger.record_with(investigation_id, build)
        return [f"queued the input reviewer for {', '.join(e.payload.calculation_ids)}" for e in result.events]

    def _queue_judgements(self, investigation_id: str) -> list[str]:
        """In full automation, the Evidence Judge assesses every hypothesis that has new evidence, before the Chief
        Scientist's review."""
        due = judgements_due(self.workspace.ledger.load(investigation_id))
        if not due:
            return []

        def build(state: InvestigationState) -> list[EventDraft]:
            return [EventDraft(actor=SYSTEM, payload=AgentTaskQueued(
                task_id=new_id("task"), role="evidence_judge", hypothesis_id=hypothesis_id,
                instruction=f"Assess {hypothesis_id}: evidence {', '.join(evidence)} was recorded since it was last "
                            "assessed. The investigation is in full automation, so your assessment is the check of "
                            "these results against the hypothesis's predictions before the Chief Scientist decides "
                            "what follows.")) for hypothesis_id, evidence in judgements_due(state)]

        result = self.workspace.ledger.record_with(investigation_id, build)
        return [f"queued the Evidence Judge on {e.payload.hypothesis_id}" for e in result.events]

    def _queue_approver(self, investigation_id: str) -> list[str]:
        """In full automation, wake the approver for decisions it has not been offered yet."""
        if not approver_due(self.workspace.ledger.load(investigation_id)):
            return []
        result = self.workspace.ledger.record_with(investigation_id, lambda state: [EventDraft(
            actor=SYSTEM, payload=AgentTaskQueued(
                task_id=new_id("task"), role="approver",
                instruction="Decide, on the researcher's behalf, what waits for you."))] if approver_due(state) else [])
        return [] if result.deduplicated or not result.events else ["queued the approver's decisions"]

    def _queue_review(self, investigation_id: str) -> list[str]:
        """Fan-in: once every agent given work has reported, the Chief Scientist reviews what came in."""
        if not self.chief_reviews or not review_due(self.workspace.ledger.load(investigation_id)):
            return []
        result = self.workspace.ledger.record_with(investigation_id, lambda state: [EventDraft(
            actor=SYSTEM, payload=AgentTaskQueued(task_id=new_id("task"), role="chief_scientist",
                                                  instruction=REVIEW_INSTRUCTION))] if review_due(state) else [])
        return [] if result.deduplicated else ["queued the Chief Scientist's review of the agents' reports"]

    # -- agent runs ----------------------------------------------------------------------------

    @staticmethod
    def _key(investigation_id: str, task: AgentTaskRecord) -> str:
        """Runs with the same key never overlap: one per agent, one Judge and one review per investigation, and one
        review of inputs per hypothesis."""
        if task.role == "input_reviewer":
            return f"{investigation_id}/{task.role}/{task.hypothesis_id}"
        return f"{investigation_id}/{task.agent_id or task.role}"

    def _interruption(self, investigation_id: str, task: AgentTaskRecord) -> str | None:
        """Why a run in progress must stop now: its agent was retired, or its investigation left the states its
        role works in."""
        state = self.workspace.ledger.load(investigation_id)
        if task.agent_id is not None and not state.agents[task.agent_id].active:
            return f"{task.agent_id} was retired"
        if task.role == "approver" and state.automation is None:
            return "the investigation is no longer in full automation"
        allowed = _WORKING if task.role != "hypothesis_agent" else (InvestigationStatus.INVESTIGATING,)
        if state.status not in allowed:
            return f"the investigation is {state.status.value}"
        return None

    @staticmethod
    def _runnable(state: InvestigationState, task: AgentTaskRecord) -> bool:
        if task.hypothesis_id is not None and state.plan.priority(task.hypothesis_id) is Priority.ON_HOLD:
            return False
        if task.role == "hypothesis_agent":
            return state.status is InvestigationStatus.INVESTIGATING
        if task.role == "approver" and state.automation is None:
            return False
        return state.status in (InvestigationStatus.INVESTIGATING, InvestigationStatus.REVIEWING)

    def _shared_left(self, investigation_id: str, run: "_Run | None" = None, own: float = 0.0) -> float | None:
        """What the investigation's budget and full automation's allowance have left once the runs in progress have
        spent what they have (``run``, spending ``own``, among them); None if neither applies."""
        state = self.workspace.ledger.load(investigation_id)
        live = sum(r.spent for r in list(self._runs.values()) if r.investigation_id == investigation_id and r is not run)
        caps = [left - live - own for left in (state.budget_left, state.automation_cost_left) if left is not None]
        return min(caps) if caps else None

    def _limits(self, state: InvestigationState, task: AgentTaskRecord) -> tuple[RunLimits | None, str | None]:
        """The run's limits: what its agent has left to spend (the budget it shares with the other runs is checked
        before every model call); or why it must wait."""
        shared = self._shared_left(state.investigation_id)
        if shared is not None and shared <= 0:
            if any(left is not None and left <= 0 for left in (state.budget_left, state.automation_cost_left)):
                return None, ("the budget is spent; queued agent work waits until the researcher raises it with "
                              "/budget")
            return None, None  # the runs in progress are spending what is left
        agent_left = None
        if task.agent_id is not None:
            spec = state.agents[task.agent_id].spec
            if spec.budget_usd is not None:
                agent_left = spec.budget_usd - state.agent_cost(task.agent_id)
                if agent_left <= 0:
                    return None, (f"{task.agent_id} has spent its budget of {dollars(spec.budget_usd)}; its queued "
                                  "work waits")
        caps = [c for c in (self.limits.max_cost_usd, agent_left) if c is not None]
        settings = self.clients.settings(task.role)
        if (caps or shared is not None) and prices_for(settings) is None:
            return None, (f"the prices of {settings.provider} {settings.model}, the {task.role.replace('_', ' ')}'s "
                          "model, are not known, so its spending cannot be kept within the budget; set them in "
                          "models.toml")
        return RunLimits(max_turns=self.limits.max_turns, max_cost_usd=min(caps) if caps else None), None

    def _launch(self, investigation_id: str, services: Services | None) -> list[str]:
        state = self.workspace.ledger.load(investigation_id)

        def order(task: AgentTaskRecord) -> tuple:
            priority = state.plan.priority(task.hypothesis_id) if task.hypothesis_id else Priority.HIGH
            return task.role != "approver", task.role != "chief_scientist", _ORDER[priority], task.queued_at

        notes = []
        for task in sorted(state.pending_tasks, key=order):
            key = self._key(investigation_id, task)
            if key in self._runs or not self._runnable(state, task):
                continue
            if len(self._runs) >= self.max_agents:
                break
            limits, problem = self._limits(state, task)
            if limits is None:
                if problem and problem not in notes:
                    notes.append(problem)
                continue
            run = _Run(investigation_id, task)
            run.thread = threading.Thread(target=self._execute, args=(run, services, limits), daemon=True,
                                          name=f"cedar {key}")
            self._runs[key] = run
            run.thread.start()
            notes.append(f"started {task_owner(task)} on {task.task_id}")
        return notes

    def _execute(self, run: _Run, services: Services | None, limits: RunLimits) -> None:
        task = run.task
        about = task.hypothesis_id or ("the decisions" if task.role == "approver" else "the review")
        try:
            outcome = self.run_task(run.investigation_id, task, services, limits)
            run.note = f"{run.investigation_id}: {task.role} ran {task.task_id} for {about}: {outcome}"
        except Exception as exc:  # report it; the task stays queued and is tried again on a later pass
            run.note = (f"{run.investigation_id}: {task.role} could not carry out {task.task_id} for {about}: "
                        f"{type(exc).__name__}: {exc}")
        finally:
            run.done.set()

    def _collect(self) -> list[str]:
        notes = []
        for key, run in list(self._runs.items()):
            if run.done.is_set():
                notes.append(run.note)
                del self._runs[key]
        return notes

    def run_task(self, investigation_id: str, task: AgentTaskRecord, services: Services | None,
                 limits: RunLimits) -> str:
        """Carry out one task; ``services`` is None while the investigation has no usable compute profile."""
        role = task.role
        settings = self.clients.settings(role)
        runner = AgentRunner(self.clients.client(role), settings, self.workspace.transcripts,
                             secrets=self.clients.secrets())
        instruction = f"[task {task.task_id}, queued by {task.queued_by}] {task.instruction}"
        state = self.workspace.ledger.load(investigation_id)
        transcripts, notebook = self.workspace.transcripts.root, self.workspace.notebook_dir(investigation_id)
        if role == "hypothesis_agent":
            agent = state.agents[task.agent_id]
            context = research_context(self.workspace, investigation_id, agent, services, deny=self.deny,
                                       analysis_slots=self._analysis_slots, workbench=self.workbench,
                                       should_stop=lambda: self._interruption(investigation_id, task) is not None)
            actor = context.actor
            tools = research_tools(context) + file_tools(context.files)
            system, session = HYPOTHESIS_AGENT + agent_prompt(agent), session_name(investigation_id, agent.agent_id,
                                                                                  task.task_id)
            briefing = agent_briefing(state, agent, workspace=context.workspace, runs_root=context.runs_root,
                                      transcripts=transcripts, notebook=notebook)
        elif role == "evidence_judge":
            actor = JUDGE
            try:
                runs_root = profile_for(state, self.profiles).runs_root
            except (LookupError, ProfileNotApproved, FileNotFoundError):
                runs_root = None
            tools = judge_tools(JudgeContext(ledger=self.workspace.ledger, investigation_id=investigation_id,
                                             actor=actor, artifacts=self.workspace.artifacts, runs_root=runs_root))
            system, session = EVIDENCE_JUDGE, session_name(investigation_id, "judge", task.task_id)
            briefing = role_briefing(state, transcripts=transcripts, notebook=notebook, actor_id=actor.id, role=role)
        elif role == "input_reviewer":
            actor = REVIEWER_ACTOR
            try:
                profile = profile_for(state, self.profiles)
            except (LookupError, ProfileNotApproved, FileNotFoundError):
                profile = None
            tools = reviewer_tools(ReviewerContext(ledger=self.workspace.ledger, investigation_id=investigation_id,
                                                   actor=actor, artifacts=self.workspace.artifacts, task=task,
                                                   files=agent_access(self.deny), profile=profile))
            system, session = INPUT_REVIEWER, session_name(investigation_id, "input-reviewer", task.task_id)
            briefing = role_briefing(state, transcripts=transcripts, notebook=notebook, actor_id=actor.id, role=role)
        elif role == "approver":
            actor = APPROVER_ACTOR
            tools = approver_tools(ApproverContext(ledger=self.workspace.ledger, investigation_id=investigation_id,
                                                   actor=actor))
            system, session = APPROVER, session_name(investigation_id, "approver", task.task_id)
            briefing = role_briefing(state, transcripts=transcripts, notebook=notebook, actor_id=actor.id, role=role)
            instruction += "\n\n" + decisions_text(state)
        else:
            actor = CHIEF
            tools = chief_tools(ChiefContext(
                ledger=self.workspace.ledger, profiles=self.profiles, actor=CHIEF, investigation_id=investigation_id,
                registry_for=lambda name: registry_for(self.profiles.load_approved(name)), clock=self.clock,
                dispatcher=lambda: "running; it is carrying out this review",
                registrar=lambda inv: investigation_registrar(self.workspace, self.profiles, inv, deny=self.deny),
                where=lambda: local_dft(self.workbench.host()),
                running=lambda: [f"{state.agents[r.task.agent_id].tag} ({r.task.agent_id})"
                                 for r in list(self._runs.values()) if r.investigation_id == investigation_id
                                 and r.task.role == "hypothesis_agent" and r.task.agent_id in state.agents]))
            system, session = CHIEF_SCIENTIST, session_name(investigation_id, "chief-review", task.task_id)
            briefing = review_briefing(state, transcripts=transcripts, notebook=notebook, actor_id=actor.id)
            instruction += "\n\n" + review_material(state, task)
        instruction = f"{briefing}\n\n{instruction}"
        ledger = self.workspace.ledger
        if role != "hypothesis_agent":  # every agent reads any file but credentials
            tools += file_tools(agent_access(self.deny))
        tools += material_tools(lambda: recorded_materials(ledger.load(investigation_id)), self.workspace.artifacts)
        activity = {"investigation_id": investigation_id, "task_id": task.task_id, "role": role,
                    "hypothesis_id": task.hypothesis_id, "agent_id": task.agent_id,
                    "started_at": self.clock().isoformat()}
        directory = self.workspace.root / ACTIVITY
        directory.mkdir(exist_ok=True)
        activity_path = directory / f"{task.task_id}.json"

        def step(call, _outcome) -> None:
            activity.update(last_step=call.name, last_step_at=self.clock().isoformat())
            _write_atomically(activity_path, json.dumps(activity))
            self._beat()

        run = self._runs.get(self._key(investigation_id, task))

        def turn(so_far) -> None:
            activity.update(calls=so_far.turns, tokens=so_far.usage.total, cost_usd=so_far.cost_usd)
            _write_atomically(activity_path, json.dumps(activity))
            if run is not None:
                run.spent = so_far.cost_usd or 0.0

        _write_atomically(activity_path, json.dumps(activity))
        try:
            result = runner.run(session, system=system, tools=tools, instruction=instruction, limits=limits,
                                observer=step, should_stop=lambda: self._interruption(investigation_id, task),
                                web=web_access(state.retrieval, role), on_turn=turn,
                                spend_left=lambda own: self._shared_left(investigation_id, run, own))
        finally:
            activity_path.unlink(missing_ok=True)
        structured = context.report if role == "hypothesis_agent" else None
        account = None
        if structured is None and not result.final_text.strip():  # no report: say from the record what it did
            account = account_of_run(self.workspace.ledger.load(investigation_id), task, actor.id, state.version,
                                     outcome=result.outcome, error=result.error, turns=result.turns,
                                     last_step=activity.get("last_step"))
        record_agent_run(self.workspace.ledger, investigation_id, actor, session_id=session, role=role,
                         settings=settings, result=result, task_id=task.task_id, started_version=state.version,
                         account=account, structured=structured)
        continued = self._continue(investigation_id, task, result)
        return result.outcome + (f"; {continued}" if continued else "")

    def _continue(self, investigation_id: str, task: AgentTaskRecord, result) -> str | None:
        """Carry on with a task whose run was cut off by a temporary model-service error, or because the
        investigation left the states its role works in; the new task runs once it can. Not after a halt or a
        retirement, nor more than ``MAX_CONTINUATIONS`` times."""
        if task.role == "approver":
            return None
        state = self.workspace.ledger.load(investigation_id)
        if state.status in INVESTIGATION_MACHINE.terminal:
            return None
        if task.role == "input_reviewer" and not any(awaits_reviewer(state, state.calculations[c])
                                                     for c in task.calculation_ids):
            return None
        if task.agent_id is not None and not state.agents[task.agent_id].active:
            return None
        if result.outcome == "model_error" and result.retryable:
            why = f"a temporary model-service error: {result.error}"
        elif result.outcome == "stopped" and state.status not in (InvestigationStatus.INVESTIGATING,
                                                                  InvestigationStatus.REVIEWING):
            why = result.error or "the investigation stopped"
        elif result.outcome == "stopped" and task.role == "hypothesis_agent" \
                and state.status is not InvestigationStatus.INVESTIGATING:
            why = result.error or "the investigation stopped"
        else:
            return None
        record = state.tasks[task.task_id]
        if len(task_chain(state, record)) > MAX_CONTINUATIONS:
            return None
        task_id = new_id("task")
        self.workspace.ledger.record(investigation_id, [EventDraft(actor=SYSTEM, payload=AgentTaskQueued(
            task_id=task_id, role=task.role, hypothesis_id=task.hypothesis_id, agent_id=task.agent_id,
            instruction=continuation_text(state, record, why), recommendation_id=task.recommendation_id,
            continues=task.task_id, calculation_ids=task.calculation_ids))])
        return f"to be continued as {task_id}"

    def _clear_activity(self) -> None:
        directory = self.workspace.root / ACTIVITY
        if directory.is_dir():
            for path in directory.iterdir():
                path.unlink(missing_ok=True)

    # -- the notebook --------------------------------------------------------------------------

    def _refresh_notebooks(self) -> list[str]:
        notes = []
        for investigation_id in self.workspace.store.investigation_ids():
            version = self.workspace.store.current_version(investigation_id)
            if self._notebook_versions.get(investigation_id) == version:
                continue
            try:
                update_notebook(self.workspace, investigation_id)
            except Exception as exc:  # the notebook is a view of the log; never let it stop submissions and polling
                notes.append(f"{investigation_id}: could not update the research notebook: {exc!r}")
                continue
            self._notebook_versions[investigation_id] = version
        return notes


def dispatcher_running(workspace: Workspace) -> bool:
    """Whether a dispatcher holds this data directory's lock right now."""
    with open(workspace.root / LOCK, "a") as lock:
        try:
            fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.lockf(lock, fcntl.LOCK_UN)
        return False


def start_background_dispatcher(workspace: Workspace, idle_minutes: float = IDLE_MINUTES) -> int:
    """Start ``cedar dispatch`` for this data directory as a separate, detached process, so it keeps working
    after the chat closes; it stops after ``idle_minutes`` with nothing to do. Its output goes to the log file
    in the data directory. Returns its process id."""
    environment = {**os.environ, "CEDAR_DATA_DIR": str(workspace.root)}
    with open(workspace.root / LOG, "a") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "cedar", "dispatch", "--until-idle", f"{idle_minutes:g}"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=workspace.root, env=environment,
            start_new_session=True, close_fds=True)
    return process.pid


def stop_dispatcher(workspace: Workspace, timeout: float = 600, force: bool = False) -> str:
    """Ask the dispatcher of this data directory to stop (SIGTERM) and wait until it has, for at most
    ``timeout`` seconds; it first ends the agent runs in progress. With ``force`` it is then killed, losing the
    token counts of runs still going."""
    if not dispatcher_running(workspace):
        return "No dispatcher is running on this data directory."
    host = dispatcher_host(workspace)
    if host is not None and host != socket.gethostname():
        return (f"The dispatcher runs on {host}, not on this machine ({socket.gethostname()}); stop it there, with "
                "`cedar dispatch --stop` on that machine.")
    try:
        pid = int((workspace.root / PID).read_text().strip())
    except (FileNotFoundError, ValueError):
        return "A dispatcher holds the lock but left no process id; stop it where it runs (Ctrl-C)."
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return f"The dispatcher's recorded process {pid} no longer exists, yet the lock is held; stop it where it runs."
    deadline = time.monotonic() + timeout
    while dispatcher_running(workspace):
        if time.monotonic() > deadline:
            if not force:
                return f"Asked the dispatcher (process {pid}) to stop; it is still finishing agent runs."
            os.kill(pid, signal.SIGKILL)
            for _ in range(100):
                if not dispatcher_running(workspace):
                    break
                time.sleep(0.1)
            (workspace.root / PID).unlink(missing_ok=True)
            for path in (workspace.root / ACTIVITY).glob("*.json"):
                path.unlink(missing_ok=True)
            return (f"Stopped the dispatcher (process {pid}) by force: its agent runs did not end within "
                    f"{timeout:g} s, so their last token counts were not recorded.")
        time.sleep(0.2)
    return f"Stopped the dispatcher (process {pid})."


def dispatcher_host(workspace: Workspace) -> str | None:
    """The machine the running dispatcher, or the last one, worked on; None if no dispatcher has."""
    try:
        return (workspace.root / HOST).read_text().strip() or None
    except OSError:
        return None


def heartbeat_age(workspace: Workspace, now: datetime | None = None) -> float | None:
    """Seconds since the dispatcher last showed it was alive, or None if none has run here."""
    path = workspace.root / HEARTBEAT
    if not path.exists():
        return None
    last = datetime.fromisoformat(path.read_text().strip())
    return ((now or datetime.now(timezone.utc)) - last).total_seconds()


def current_activity(workspace: Workspace) -> list[Activity]:
    """The agent runs in progress, as the dispatcher last described them, oldest first."""
    directory = workspace.root / ACTIVITY
    found = []
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
        try:
            data = json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            continue  # the run ended while we looked
        found.append(Activity(
            investigation_id=data["investigation_id"], task_id=data["task_id"], role=data["role"],
            hypothesis_id=data.get("hypothesis_id"), agent_id=data.get("agent_id"),
            started_at=datetime.fromisoformat(data["started_at"]), last_step=data.get("last_step"),
            last_step_at=datetime.fromisoformat(data["last_step_at"]) if data.get("last_step_at") else None,
            calls=data.get("calls", 0), tokens=data.get("tokens", 0), cost_usd=data.get("cost_usd")))
    return sorted(found, key=lambda a: a.started_at)
