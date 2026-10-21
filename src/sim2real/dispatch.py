"""The dispatcher: moves investigations forward with or without anyone in the chat.

Each tick, for every open investigation:
1. submit calculations the researcher has authorized (never those of on-hold hypotheses);
2. poll the scheduler and record job state changes;
3. wake hypothesis agents whose awaited calculations have all ended;
4. run queued agent tasks, within the investigation's token budget;
5. record evidence, from the protocol's own reading, for finished runs no agent reported on.

It then regenerates the research notebook of every investigation whose log grew, closed
ones included.

Only one dispatcher works on a data directory at a time (a lock file). It leaves a
heartbeat at the start and end of each tick and after every agent step, so the chat can
warn when no dispatcher is running, and while an agent works it describes that run in an
activity file, so the live panel can show it.
"""

import fcntl
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sim2real.agents import AgentRunner, RunLimits, record_agent_run
from sim2real.agents.prompts import EVIDENCE_JUDGE, HYPOTHESIS_AGENT
from sim2real.agents.toolkits import ResearchContext, file_tools, research_tools
from sim2real.agents.toolkits.judge import JudgeContext, judge_tools
from sim2real.contracts.base import Actor, ActorKind, new_id
from sim2real.contracts.events import AgentTaskQueued, EventDraft, HypothesisStatusChanged
from sim2real.contracts.plan import PermissionAction, Priority
from sim2real.contracts.status import INVESTIGATION_MACHINE, CalculationState, HypothesisStatus
from sim2real.engine.jobs import InputsChanged
from sim2real.errors import RuleViolation
from sim2real.hpc.profile import ProfileNotApproved, ProfileStore
from sim2real.hpc.slurm import SchedulerBackend, SchedulerError
from sim2real.notebook import update_notebook
from sim2real.state.projection import AgentTaskRecord, authorization_for
from sim2real.summaries import calculation_summary
from sim2real.workspace import ModelClients, Services, Workspace, profile_for, services_for

SYSTEM = Actor(kind=ActorKind.SYSTEM, id="dispatcher")
JUDGE = Actor(kind=ActorKind.EVIDENCE_JUDGE, id="evidence-judge")
_ENDED = frozenset({CalculationState.SCHEDULER_TERMINAL, CalculationState.PARSING, CalculationState.QUALITY_ASSESSED,
                    CalculationState.REJECTED, CalculationState.CANCELLED})
HEARTBEAT = "dispatcher.heartbeat"
ACTIVITY = "dispatcher.activity"


@dataclass(frozen=True)
class Activity:
    """The agent run the dispatcher is carrying out right now."""

    investigation_id: str
    task_id: str
    role: str
    hypothesis_id: str
    started_at: datetime
    last_step: str | None = None
    """The tool the agent called most recently."""
    last_step_at: datetime | None = None


def _write_atomically(path: Path, text: str) -> None:
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(text)
    os.replace(partial, path)


def hypothesis_actor(hypothesis_id: str) -> Actor:
    return Actor(kind=ActorKind.HYPOTHESIS_AGENT, id=f"agent-{hypothesis_id}"[:128])


class Dispatcher:
    def __init__(self, workspace: Workspace, profiles: ProfileStore, clients: ModelClients, *,
                 backend: SchedulerBackend | None = None, deny: list[Path] = (),
                 limits: RunLimits = RunLimits(max_turns=40, max_tokens=1_000_000),
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.workspace = workspace
        self.profiles = profiles
        self.clients = clients
        self.backend = backend
        self.deny = list(deny)
        self.limits = limits
        self.clock = clock
        self._notebook_versions: dict[str, int] = {}

    def tick(self) -> list[str]:
        lock_path = self.workspace.root / "dispatcher.lock"
        with open(lock_path, "a") as lock:
            try:
                fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return ["another dispatcher is working on this data directory; skipped"]
            try:
                self._beat()
                notes = []
                for investigation_id in self.workspace.store.investigation_ids():
                    notes += self._advance(investigation_id)
                    notes += self._refresh_notebook(investigation_id)
                self._beat()
                return notes
            finally:
                fcntl.lockf(lock, fcntl.LOCK_UN)

    def _beat(self) -> None:
        _write_atomically(self.workspace.root / HEARTBEAT, self.clock().isoformat())

    def run_forever(self, interval: float, log: Callable[[str], None] = print) -> None:
        while True:
            for note in self.tick():
                log(f"{self.clock():%Y-%m-%d %H:%M:%S} {note}")
            time.sleep(interval)

    # -- one investigation ---------------------------------------------------------------

    def _advance(self, investigation_id: str) -> list[str]:
        state = self.workspace.ledger.load(investigation_id)
        if state.status in INVESTIGATION_MACHINE.terminal:
            return []
        try:
            services = services_for(self.workspace, profile_for(state, self.profiles), self.backend, deny=self.deny)
        except (LookupError, ProfileNotApproved, FileNotFoundError) as exc:
            return [f"{investigation_id}: waiting for a usable compute profile ({exc})"] if state.calculations \
                or state.pending_tasks else []
        notes = []
        notes += self._submit(investigation_id, services)
        notes += self._poll(investigation_id, services)
        notes += self._wake(investigation_id)
        notes += self._run_tasks(investigation_id, services)
        notes += self._assess_unreported(investigation_id, services)
        return [f"{investigation_id}: {n}" for n in notes]

    def _submit(self, investigation_id: str, services: Services) -> list[str]:
        state = self.workspace.ledger.load(investigation_id)
        notes = []
        for calc in state.calculations.values():
            if calc.state is CalculationState.VALIDATED_INPUT:
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

    def _wake(self, investigation_id: str) -> list[str]:
        state = self.workspace.ledger.load(investigation_id)
        notes = []
        for hyp in state.hypotheses.values():
            if hyp.status is not HypothesisStatus.WAITING_FOR_JOB:
                continue
            awaited = [state.calculations[c] for c in hyp.awaiting_calculation_ids]
            if not all(c.state in _ENDED for c in awaited):
                continue
            summaries = "\n".join(calculation_summary(state, c, findings=False) for c in awaited)
            instruction = ("Your calculations have ended:\n" + summaries + "\nRead their outputs with the file tools, "
                           "submit cited readings for the values that matter, and say what they mean for your "
                           "hypothesis. Then set its status (ASSESSED, or PLANNING_TEST if another test is needed).")
            self.workspace.ledger.record(investigation_id, [
                EventDraft(actor=SYSTEM, payload=HypothesisStatusChanged(
                    hypothesis_id=hyp.hypothesis_id, status=HypothesisStatus.ANALYZING,
                    reason="the calculations it was waiting for have ended")),
                EventDraft(actor=SYSTEM, payload=AgentTaskQueued(
                    task_id=new_id("task"), role="hypothesis_agent", hypothesis_id=hyp.hypothesis_id,
                    instruction=instruction)),
            ])
            notes.append(f"woke the agent for {hyp.hypothesis_id}")
        return notes

    def _run_tasks(self, investigation_id: str, services: Services) -> list[str]:
        notes = []
        for task in sorted(self.workspace.ledger.load(investigation_id).pending_tasks, key=lambda t: t.queued_at):
            state = self.workspace.ledger.load(investigation_id)
            if state.plan.priority(task.hypothesis_id) is Priority.ON_HOLD:
                continue
            remaining = state.model_tokens_remaining
            if remaining is not None and remaining <= 0:
                notes.append("the model token budget is spent; queued agent work waits for a larger budget")
                break
            limit = self.limits.max_tokens if remaining is None else min(self.limits.max_tokens or remaining, remaining)
            limits = RunLimits(max_turns=self.limits.max_turns, max_tokens=limit)
            outcome = self.run_task(investigation_id, task, services, limits)
            notes.append(f"{task.role} ran {task.task_id} for {task.hypothesis_id}: {outcome}")
        return notes

    def _refresh_notebook(self, investigation_id: str) -> list[str]:
        version = self.workspace.store.current_version(investigation_id)
        if self._notebook_versions.get(investigation_id) == version:
            return []
        try:
            update_notebook(self.workspace, investigation_id)
        except Exception as exc:  # the notebook is a view of the log; never let it stop submissions and polling
            return [f"{investigation_id}: could not update the research notebook: {exc!r}"]
        self._notebook_versions[investigation_id] = version
        return []

    def run_task(self, investigation_id: str, task: AgentTaskRecord, services: Services, limits: RunLimits) -> str:
        role = task.role
        settings = self.clients.settings(role)
        runner = AgentRunner(self.clients.client(role), settings, self.workspace.transcripts,
                             secrets=self.clients.secrets())
        instruction = f"[task {task.task_id}, queued by {task.queued_by}] {task.instruction}"
        if role == "hypothesis_agent":
            actor = hypothesis_actor(task.hypothesis_id)
            context = ResearchContext(ledger=self.workspace.ledger, investigation_id=investigation_id,
                                      hypothesis_id=task.hypothesis_id, actor=actor, registry=services.registry,
                                      artifacts=self.workspace.artifacts, engine=services.engine,
                                      validator=services.validator, files=services.files)
            tools = research_tools(context) + file_tools(services.files)
            system, session = HYPOTHESIS_AGENT, f"{investigation_id}.{task.hypothesis_id}"
        else:
            actor = JUDGE
            tools = judge_tools(JudgeContext(ledger=self.workspace.ledger, investigation_id=investigation_id,
                                             actor=actor))
            system, session = EVIDENCE_JUDGE, f"{investigation_id}.judge"
        activity = {"investigation_id": investigation_id, "task_id": task.task_id, "role": role,
                    "hypothesis_id": task.hypothesis_id, "started_at": self.clock().isoformat()}
        activity_path = self.workspace.root / ACTIVITY

        def step(call, _outcome) -> None:
            activity.update(last_step=call.name, last_step_at=self.clock().isoformat())
            _write_atomically(activity_path, json.dumps(activity))
            self._beat()

        _write_atomically(activity_path, json.dumps(activity))
        try:
            result = runner.run(session[:128], system=system, tools=tools, instruction=instruction, limits=limits,
                                observer=step)
        finally:
            activity_path.unlink(missing_ok=True)
        record_agent_run(self.workspace.ledger, investigation_id, actor, session_id=session[:128], role=role,
                         settings=settings, result=result, task_id=task.task_id)
        return result.outcome

    def _assess_unreported(self, investigation_id: str, services: Services) -> list[str]:
        state = self.workspace.ledger.load(investigation_id)
        busy = {t.hypothesis_id for t in state.pending_tasks} | {
            h.hypothesis_id for h in state.hypotheses.values() if h.status is HypothesisStatus.WAITING_FOR_JOB}
        notes = []
        for calc in state.calculations.values():
            if calc.state not in (CalculationState.SCHEDULER_TERMINAL, CalculationState.PARSING):
                continue
            if busy.intersection(calc.hypothesis_ids):
                continue
            outcome = services.validator.assess(investigation_id, calc.calculation_id)
            if outcome.recorded is not None:
                record = outcome.recorded.state.evidence[outcome.recorded.state.calculations[calc.calculation_id]
                                                         .evidence_id]
                notes.append(f"recorded {record.evidence.quality.value} evidence for {calc.calculation_id} from the "
                             "protocol's own reading")
        return notes


def heartbeat_age(workspace: Workspace, now: datetime | None = None) -> float | None:
    """Seconds since the dispatcher last showed it was alive, or None if none has run here."""
    path = workspace.root / HEARTBEAT
    if not path.exists():
        return None
    last = datetime.fromisoformat(path.read_text().strip())
    return ((now or datetime.now(timezone.utc)) - last).total_seconds()


def current_activity(workspace: Workspace) -> Activity | None:
    """The agent run in progress, as the dispatcher last described it, or None."""
    try:
        data = json.loads((workspace.root / ACTIVITY).read_text())
        return Activity(
            investigation_id=data["investigation_id"], task_id=data["task_id"], role=data["role"],
            hypothesis_id=data["hypothesis_id"], started_at=datetime.fromisoformat(data["started_at"]),
            last_step=data.get("last_step"),
            last_step_at=datetime.fromisoformat(data["last_step_at"]) if data.get("last_step_at") else None)
    except FileNotFoundError:
        return None
