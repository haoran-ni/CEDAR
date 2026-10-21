"""The dispatcher: moves investigations forward with or without anyone in the chat.

Each pass, for every open investigation:
1. submit calculations the researcher has authorized (never those of on-hold hypotheses);
2. poll the scheduler and record job state changes;
3. check the health of every run that ended and store its outputs;
4. start proposed agents whose review window has passed without a reply from the researcher;
5. wake hypothesis agents whose awaited calculations have all ended;
6. queue a review by the Chief Scientist once every agent given work has reported;
7. start queued agent work, within the token budgets.

Evidence comes only from agents' cited readings and recorded analyses, after the run check.
Steps 1-3 need the investigation's approved compute profile. Without one, agents still run,
with every tool but those that request calculations or read runs.

Agents run in background threads, so submission and polling go on while they work: several
agents at once (``max_agents``), but never two runs of the same agent, and never two runs
of the Evidence Judge or of the Chief Scientist's review in one investigation. Hypothesis
agents run only while their investigation is INVESTIGATING; the Evidence Judge and the
Chief Scientist's reviews also while it is REVIEWING. A run in progress stops at its next
model call once that no longer holds or its agent is retired, as when the researcher halts
the investigation. Nothing is submitted while an investigation is paused or blocked. Each agent run sets aside the tokens
it may use, so agents running together cannot overspend a budget.

The dispatcher then regenerates the research notebook of every investigation whose log
grew, closed ones included.

Only one dispatcher works on a data directory at a time: it holds a lock file for as long
as it runs. The researcher may run it themselves (``sim2real dispatch``), or the chat
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
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sim2real.agents import AgentRunner, RunLimits, record_agent_run
from sim2real.agents.prompts import APPROVER, CHIEF_SCIENTIST, EVIDENCE_JUDGE, HYPOTHESIS_AGENT
from sim2real.agents.toolkits import ResearchContext, file_tools, research_tools
from sim2real.agents.toolkits.approver import ApproverContext, approver_tools, decisions_text
from sim2real.agents.toolkits.chief import ChiefContext, chief_tools
from sim2real.agents.toolkits.judge import JudgeContext, judge_tools
from sim2real.agents.toolkits.materials import material_tools, recorded_materials
from sim2real.agents.toolkits.research import agent_file_access
from sim2real.contracts.base import Actor, ActorKind, new_id
from sim2real.automation import approver_due, judgements_due
from sim2real.contracts.events import (
    AgentTaskQueued,
    EventDraft,
    HypothesisStatusChanged,
    InvestigationStatusChanged,
    ModeChanged,
)
from sim2real.contracts.plan import PermissionAction, Priority
from sim2real.contracts.status import (
    INVESTIGATION_MACHINE,
    CalculationState,
    HypothesisStatus,
    InvestigationMode,
    InvestigationStatus,
)
from sim2real.engine.jobs import InputsChanged
from sim2real.errors import RuleViolation
from sim2real.hpc.profile import ProfileNotApproved, ProfileStore
from sim2real.hpc.slurm import SchedulerBackend, SchedulerError
from sim2real.notebook import update_notebook
from sim2real.protocols.base import ProtocolRegistry
from sim2real.state.projection import AgentTaskRecord, InvestigationState, authorization_for, review_deadline
from sim2real.summaries import calculation_summary, task_owner
from sim2real.swarm import (
    CHIEF,
    REVIEW_INSTRUCTION,
    WAKE_TASK,
    agent_prompt,
    review_due,
    review_material,
    start_drafts,
)
from sim2real.web import web_access
from sim2real.workspace import ModelClients, Services, Workspace, profile_for, registry_for, services_for

SYSTEM = Actor(kind=ActorKind.SYSTEM, id="dispatcher")
JUDGE = Actor(kind=ActorKind.EVIDENCE_JUDGE, id="evidence-judge")
APPROVER_ACTOR = Actor(kind=ActorKind.APPROVER, id="approver")
_ENDED = frozenset({CalculationState.SCHEDULER_TERMINAL, CalculationState.PARSING, CalculationState.QUALITY_ASSESSED,
                    CalculationState.REJECTED, CalculationState.CANCELLED})
_ORDER = {Priority.HIGH: 0, Priority.NORMAL: 1, Priority.LOW: 2, Priority.ON_HOLD: 3}
HEARTBEAT = "dispatcher.heartbeat"
ACTIVITY = "activity"
LOCK = "dispatcher.lock"
PID = "dispatcher.pid"
LOG = "dispatcher.log"
IDLE_MINUTES = 30
"""A dispatcher the chat started stops after this long with nothing to do."""
_WORKING = frozenset({InvestigationStatus.INVESTIGATING, InvestigationStatus.REVIEWING})
"""Investigation states in which work goes on: submissions, the Evidence Judge and reviews."""
_LIVE = frozenset({CalculationState.SUBMISSION_PENDING, CalculationState.SUBMITTED, CalculationState.QUEUED,
                   CalculationState.RUNNING, CalculationState.SCHEDULER_TERMINAL, CalculationState.PARSING})


def needs_dispatcher(state: InvestigationState) -> bool:
    """Whether the investigation has work only a dispatcher carries out: a proposal waiting out its review
    window, agent work that can run, a review that is due, in full automation a decision or an assessment to
    queue (or its allowance to end), or a calculation authorized, at the scheduler or waiting for its run check."""
    if state.status in INVESTIGATION_MACHINE.terminal:
        return False
    proposal = state.pending_proposal
    if proposal is not None and not proposal.feedback_ids:
        return True
    if any(Dispatcher._runnable(state, t) for t in state.pending_tasks) or review_due(state):
        return True
    if state.automation is not None and (approver_due(state) or judgements_due(state)
                                         or state.automation_tokens_left <= 0):
        return True
    for calc in state.calculations.values():
        if calc.state in _LIVE:
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


def _write_atomically(path: Path, text: str) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.partial")
    partial.write_text(text)
    os.replace(partial, path)


@dataclass
class _Run:
    investigation_id: str
    task: AgentTaskRecord
    tokens: int | None
    """Tokens set aside for it until it ends."""
    note: str = ""
    done: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class Dispatcher:
    def __init__(self, workspace: Workspace, profiles: ProfileStore, clients: ModelClients, *,
                 backend: SchedulerBackend | None = None, deny: list[Path] = (),
                 limits: RunLimits = RunLimits(max_turns=40), max_agents: int = 4,
                 analysis_slots: int = 2, chief_reviews: bool = True,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        """``max_agents`` agent runs at once; ``analysis_slots`` analysis scripts at once on this node."""
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

        It ends, after the agent runs in progress, when ``stop`` is set, on SIGTERM (``sim2real dispatch
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
            services = services_for(self.workspace, profile_for(state, self.profiles), self.backend, deny=self.deny,
                                    analysis_slots=self._analysis_slots)
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
        notes += self._queue_judgements(investigation_id)
        notes += self._queue_approver(investigation_id)
        notes += self._queue_review(investigation_id)
        notes += self._launch(investigation_id, services)
        return [f"{investigation_id}: {n}" for n in notes]

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
            try:
                job = services.manager.submit(investigation_id, calc.calculation_id)
                notes.append(f"submitted {calc.calculation_id} as job {job}")
            except (RuleViolation, InputsChanged, SchedulerError, OSError) as exc:
                notes.append(f"could not submit {calc.calculation_id}: {exc}")
        return notes

    def _poll(self, investigation_id: str, services: Services) -> list[str]:
        try:
            result = services.monitor.poll(investigation_id)
        except SchedulerError as exc:
            return [f"scheduler query failed: {exc}"]
        return [f"{e.payload.calculation_id} is {e.payload.state.value}"
                + (f" ({e.payload.scheduler_state.value})" if e.payload.scheduler_state else "")
                for e in result.recorded]

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
        state = self.workspace.ledger.load(investigation_id)
        notes = []
        for hyp in state.hypotheses.values():
            if hyp.status is not HypothesisStatus.WAITING_FOR_JOB:
                continue
            agent = state.active_agent(hyp.hypothesis_id)
            awaited = [state.calculations[c] for c in hyp.awaiting_calculation_ids]
            if agent is None or not all(c.state in _ENDED for c in awaited):
                continue
            summaries = "\n".join(calculation_summary(state, c, findings=False, viewer=hyp.hypothesis_id)
                                  for c in awaited)
            instruction = (WAKE_TASK + "\n" + summaries + "\nLook at "
                           "the outputs (run_outputs, file tools). Record the values that matter: a value printed in "
                           "an output with submit_reading, citing its lines; anything that must be computed with "
                           "run_analysis and then record_result. Say what the results mean for your hypothesis, then "
                           "set its status (ASSESSED, or PLANNING_TEST if another test is needed).")
            self.workspace.ledger.record(investigation_id, [
                EventDraft(actor=SYSTEM, payload=HypothesisStatusChanged(
                    hypothesis_id=hyp.hypothesis_id, status=HypothesisStatus.ANALYZING,
                    reason="the calculations it was waiting for have ended")),
                EventDraft(actor=SYSTEM, payload=AgentTaskQueued(
                    task_id=new_id("task"), role="hypothesis_agent", hypothesis_id=hyp.hypothesis_id,
                    agent_id=agent.agent_id, instruction=instruction)),
            ])
            notes.append(f"woke the agent for {hyp.hypothesis_id}")
        return notes

    def _end_spent_automation(self, investigation_id: str) -> list[str]:
        """Once full automation's token allowance is used up, switch back to human-in-the-loop and pause."""
        state = self.workspace.ledger.load(investigation_id)
        if state.automation is None or state.automation_tokens_left > 0:
            return []
        reason = (f"full automation's allowance of {state.automation.max_tokens:,} tokens is used up; the "
                  "investigation waits for the researcher")

        def build(current: InvestigationState) -> list[EventDraft]:
            if current.automation is None or current.automation_tokens_left > 0:
                return []
            drafts = [EventDraft(actor=SYSTEM, payload=ModeChanged(mode=InvestigationMode.HUMAN_IN_THE_LOOP,
                                                                   reason=reason))]
            if current.status not in INVESTIGATION_MACHINE.suspend | INVESTIGATION_MACHINE.terminal:
                drafts.append(EventDraft(actor=SYSTEM, payload=InvestigationStatusChanged(
                    status=InvestigationStatus.PAUSED, reason=reason)))
            return drafts

        result = self.workspace.ledger.record_with(investigation_id, build)
        return [] if result.deduplicated or not result.events else [f"{reason}: back to human-in-the-loop, paused"]

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
        """Runs with the same key never overlap: one per agent, one Judge and one review per investigation."""
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

    def _limits(self, state: InvestigationState, task: AgentTaskRecord) -> tuple[RunLimits | None, str | None]:
        """The run's limits, within what the investigation and the agent have left; or why it must wait."""
        investigation_left = state.model_tokens_remaining
        automation_left = state.automation_tokens_left
        if automation_left is not None:  # full automation's allowance caps every run, like a budget
            investigation_left = automation_left if investigation_left is None else min(investigation_left,
                                                                                        automation_left)
        if investigation_left is not None:
            if investigation_left <= 0:
                return None, ("the model token budget is spent; queued agent work waits until the researcher "
                              "raises it with /budget")
            investigation_left -= sum(r.tokens or 0 for r in self._runs.values()
                                      if r.investigation_id == state.investigation_id)
            if investigation_left <= 0:
                return None, None  # set aside for the agents running now
        agent_left, max_turns = None, self.limits.max_turns
        if task.agent_id is not None:
            spec = state.agents[task.agent_id].spec
            max_turns = spec.max_turns or max_turns
            if spec.token_budget is not None:
                agent_left = spec.token_budget - state.agent_tokens_used(task.agent_id)
                if agent_left <= 0:
                    return None, (f"{task.agent_id} has used its budget of {spec.token_budget:,} tokens; its queued "
                                  "work waits")
        caps = [c for c in (self.limits.max_tokens, investigation_left, agent_left) if c is not None]
        return RunLimits(max_turns=max_turns, max_tokens=min(caps) if caps else None), None

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
            run = _Run(investigation_id, task, limits.max_tokens)
            run.thread = threading.Thread(target=self._execute, args=(run, services, limits), daemon=True,
                                          name=f"sim2real {key}")
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
        if role == "hypothesis_agent":
            agent = state.agents[task.agent_id]
            actor = Actor(kind=ActorKind.HYPOTHESIS_AGENT, id=agent.agent_id)
            if services is not None:
                files = services.agent_files(self.workspace, investigation_id, agent.hypothesis_id)
                context = ResearchContext(ledger=self.workspace.ledger, investigation_id=investigation_id,
                                          hypothesis_id=agent.hypothesis_id, actor=actor, registry=services.registry,
                                          artifacts=self.workspace.artifacts, engine=services.engine,
                                          validator=services.validator, files=files, analysis=services.analysis)
            else:
                files = agent_file_access(self.workspace.ledger, investigation_id, agent.hypothesis_id, None,
                                          deny=self.deny)
                context = ResearchContext(ledger=self.workspace.ledger, investigation_id=investigation_id,
                                          hypothesis_id=agent.hypothesis_id, actor=actor, registry=ProtocolRegistry([]),
                                          artifacts=self.workspace.artifacts, engine=None, validator=None,
                                          files=files, analysis=None)
            tools = research_tools(context) + file_tools(files)
            system, session = HYPOTHESIS_AGENT + agent_prompt(agent), f"{investigation_id}.{agent.agent_id}"
        elif role == "evidence_judge":
            actor = JUDGE
            tools = judge_tools(JudgeContext(ledger=self.workspace.ledger, investigation_id=investigation_id,
                                             actor=actor, artifacts=self.workspace.artifacts))
            system, session = EVIDENCE_JUDGE, f"{investigation_id}.judge"
        elif role == "approver":
            actor = APPROVER_ACTOR
            tools = approver_tools(ApproverContext(ledger=self.workspace.ledger, investigation_id=investigation_id,
                                                   actor=actor))
            system, session = APPROVER, f"{investigation_id}.approver"
            instruction += "\n\n" + decisions_text(state)
        else:
            actor = CHIEF
            tools = chief_tools(ChiefContext(
                ledger=self.workspace.ledger, profiles=self.profiles, actor=CHIEF, investigation_id=investigation_id,
                registry_for=lambda name: registry_for(self.profiles.load_approved(name)), clock=self.clock,
                dispatcher=lambda: "running; it is carrying out this review"))
            system, session = CHIEF_SCIENTIST, f"{investigation_id}.chief-review"
            instruction += "\n\n" + review_material(state, task)
        ledger = self.workspace.ledger
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

        _write_atomically(activity_path, json.dumps(activity))
        try:
            result = runner.run(session[:128], system=system, tools=tools, instruction=instruction, limits=limits,
                                observer=step, should_stop=lambda: self._interruption(investigation_id, task),
                                web=web_access(state.retrieval, role))
        finally:
            activity_path.unlink(missing_ok=True)
        record_agent_run(self.workspace.ledger, investigation_id, actor, session_id=session[:128], role=role,
                         settings=settings, result=result, task_id=task.task_id)
        return result.outcome

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
    """Start ``sim2real dispatch`` for this data directory as a separate, detached process, so it keeps working
    after the chat closes; it stops after ``idle_minutes`` with nothing to do. Its output goes to the log file
    in the data directory. Returns its process id."""
    environment = {**os.environ, "SIM2REAL_DATA_DIR": str(workspace.root)}
    with open(workspace.root / LOG, "a") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "sim2real", "dispatch", "--until-idle", f"{idle_minutes:g}"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=workspace.root, env=environment,
            start_new_session=True, close_fds=True)
    return process.pid


def stop_dispatcher(workspace: Workspace, timeout: float = 600, force: bool = False) -> str:
    """Ask the dispatcher of this data directory to stop (SIGTERM) and wait until it has, for at most
    ``timeout`` seconds; it first ends the agent runs in progress. With ``force`` it is then killed, losing the
    token counts of runs still going."""
    if not dispatcher_running(workspace):
        return "No dispatcher is running on this data directory."
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
            last_step_at=datetime.fromisoformat(data["last_step_at"]) if data.get("last_step_at") else None))
    return sorted(found, key=lambda a: a.started_at)
