"""The conversation with the Chief Scientist, independent of any terminal interface.

Every researcher message is recorded verbatim as a user intervention once an investigation
is open (messages sent before it opens are recorded right after). Slash commands are
handled here, not by a model: authorizations to submit, approve inputs or cancel can only
come from the researcher's own command.
"""

import getpass
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sim2real.agents import AgentRunner, RunLimits, RunResult, record_agent_run
from sim2real.agents.prompts import CHIEF_SCIENTIST
from sim2real.agents.toolkits.chief import ChiefContext, chief_tools
from sim2real.contracts.base import Actor, ActorKind
from sim2real.contracts.events import EventDraft, UserInterventionRecorded
from sim2real.contracts.plan import Authorization, PermissionAction
from sim2real.errors import RuleViolation
from sim2real.hpc.profile import ProfileStore
from sim2real.hpc.slurm import SchedulerBackend
from sim2real.notebook import update_notebook
from sim2real.summaries import investigation_summary, pending_approvals
from sim2real.workspace import ModelClients, Workspace, profile_for, registry_for, services_for

CHIEF = Actor(kind=ActorKind.CHIEF_SCIENTIST, id="chief-scientist")
HELP = """Commands:
  /status                      the investigation's state
  /pending                     what waits for your approval
  /approve submit <calc>       allow a calculation with approved inputs to be submitted
  /approve inputs <calc>       allow inputs the examiner could not verify
  /cancel <calc> <reason>      cancel a calculation, stopping its job if it has one
  /report                      update the research notebook and show where it is
  /help                        this list
  /quit                        leave (queued work and jobs continue with the dispatcher)
Anything else goes to the Chief Scientist."""


@dataclass
class ChatSession:
    workspace: Workspace
    profiles: ProfileStore
    clients: ModelClients
    investigation_id: str | None = None
    user: Actor = field(default_factory=lambda: Actor(kind=ActorKind.USER, id=getpass.getuser()))
    backend: SchedulerBackend | None = None
    limits: RunLimits = field(default_factory=lambda: RunLimits(max_turns=30, max_tokens=600_000))
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    _unrecorded_messages: list[str] = field(default_factory=list)
    _unrecorded_runs: list[RunResult] = field(default_factory=list)
    _seen_version: int = 0
    chief_session: str = ""

    def __post_init__(self) -> None:
        if self.investigation_id is not None:
            state = self.workspace.ledger.load(self.investigation_id)
            # Show what happened since the conversation last moved: the researcher's last message or
            # the Chief's last reply.
            self._seen_version = max(
                [i.recorded_at for i in state.interventions.values()]
                + [r.recorded_at for r in state.agent_runs if r.role == "chief_scientist"], default=0)
            chief_runs = [r for r in state.agent_runs if r.role == "chief_scientist"]
            self.chief_session = chief_runs[-1].session_id if chief_runs else f"{self.investigation_id}.chief"
        else:
            self.chief_session = f"chief-{self.clock():%Y%m%d%H%M%S}"

    # -- input ------------------------------------------------------------------------------

    def handle(self, line: str) -> str:
        line = line.strip()
        if not line:
            return ""
        if line.startswith("/"):
            return self._command(line)
        return self._ask_chief(line)

    def _command(self, line: str) -> str:
        try:
            words = shlex.split(line)
        except ValueError as exc:
            return f"Cannot read that command: {exc}"
        name, args = words[0], words[1:]
        if name == "/help":
            return HELP
        if self.investigation_id is None:
            return "No investigation is open yet. Describe the discrepancy and the Chief Scientist will open one."
        state = self.workspace.ledger.load(self.investigation_id)
        if name == "/status":
            return investigation_summary(state)
        if name == "/pending":
            return "\n".join(pending_approvals(state)) or "Nothing waits for your approval."
        if name == "/report":
            update = update_notebook(self.workspace, self.investigation_id)
            return (f"Research notebook ({update.summary}): {update.index}\n"
                    f"This round: {update.directory / f'round-{state.round:02d}.md'}")
        if name == "/approve" and len(args) == 2 and args[0] in ("submit", "inputs"):
            action = PermissionAction.SUBMIT if args[0] == "submit" else PermissionAction.APPROVE_INPUTS
            return self._authorize(action, args[1], line)
        if name == "/cancel" and len(args) >= 2:
            return self._cancel(args[0], " ".join(args[1:]), line)
        return f"Unknown or incomplete command: {line}\n{HELP}"

    def _authorize(self, action: PermissionAction, calculation_id: str, line: str) -> str:
        state = self.workspace.ledger.load(self.investigation_id)
        if calculation_id not in state.calculations:
            return f"No calculation {calculation_id} in this investigation."
        intervention = self._record_message(line, (Authorization(action=action, target_ids=(calculation_id,)),))
        if action is PermissionAction.APPROVE_INPUTS:
            services = services_for(self.workspace, profile_for(state, self.profiles), self.backend)
            try:
                services.engine.approve(self.investigation_id, calculation_id)
            except RuleViolation as exc:
                return f"Recorded your approval ({intervention}), but the inputs could not be validated: {exc}"
            return f"Inputs of {calculation_id} approved ({intervention}). They still need /approve submit to run."
        return f"Submission of {calculation_id} authorized ({intervention}). The dispatcher submits it on its next tick."

    def _cancel(self, calculation_id: str, reason: str, line: str) -> str:
        state = self.workspace.ledger.load(self.investigation_id)
        if calculation_id not in state.calculations:
            return f"No calculation {calculation_id} in this investigation."
        intervention = self._record_message(line, (Authorization(action=PermissionAction.CANCEL,
                                                                 target_ids=(calculation_id,)),))
        services = services_for(self.workspace, profile_for(state, self.profiles), self.backend)
        try:
            services.manager.cancel(self.investigation_id, calculation_id, reason)
        except RuleViolation as exc:
            return f"Recorded your cancellation ({intervention}), but it could not be applied: {exc}"
        return f"Cancelled {calculation_id} ({intervention})."

    # -- the Chief Scientist ------------------------------------------------------------------

    def _ask_chief(self, text: str) -> str:
        if self.investigation_id is not None:
            message_id = self._record_message(text)
            instruction = f"[researcher, recorded as {message_id}] {text}"
        else:
            self._unrecorded_messages.append(text)
            instruction = f"[researcher] {text}"
        context = ChiefContext(ledger=self.workspace.ledger, profiles=self.profiles, actor=CHIEF,
                               registry_for=lambda name: registry_for(self.profiles.load_approved(name)),
                               investigation_id=self.investigation_id, on_open=self._opened)
        settings = self.clients.settings("chief_scientist")
        runner = AgentRunner(self.clients.client("chief_scientist"), settings, self.workspace.transcripts,
                             secrets=self.clients.secrets())
        result = runner.run(self.chief_session, system=CHIEF_SCIENTIST, tools=chief_tools(context),
                            instruction=instruction, limits=self.limits)
        if self.investigation_id is None:
            self._unrecorded_runs.append(result)
        else:
            for earlier in [*self._unrecorded_runs, result]:
                record_agent_run(self.workspace.ledger, self.investigation_id, CHIEF, session_id=self.chief_session,
                                 role="chief_scientist", settings=settings, result=earlier)
            self._unrecorded_runs.clear()
            self._seen_version = self.workspace.ledger.load(self.investigation_id).version
        reply = result.final_text or ""
        if result.outcome != "finished":
            reply += f"\n[the Chief Scientist stopped: {result.outcome}{': ' + result.error if result.error else ''}]"
        return reply.strip()

    def _opened(self, investigation_id: str) -> list[str]:
        """Record the messages sent before the investigation existed, in order, as soon as it does."""
        self.investigation_id = investigation_id
        recorded = [self._record_message(text) for text in self._unrecorded_messages]
        self._unrecorded_messages.clear()
        return recorded

    def _record_message(self, text: str, authorizations: tuple[Authorization, ...] = ()) -> str:
        state = self.workspace.ledger.load(self.investigation_id)
        number = 1 + sum(1 for i in state.interventions if i.startswith("msg-"))
        message_id = f"msg-{number}"
        self.workspace.ledger.record(self.investigation_id, [EventDraft(actor=self.user, payload=UserInterventionRecorded(
            intervention_id=message_id, text=text, authorizations=authorizations))])
        return message_id

    # -- what happened meanwhile ----------------------------------------------------------------

    def notices(self) -> list[str]:
        """Agent reports and requests for approval that arrived since the researcher last looked."""
        if self.investigation_id is None:
            return []
        state = self.workspace.ledger.load(self.investigation_id)
        fresh = [r for r in state.agent_runs if r.recorded_at > self._seen_version and r.role != "chief_scientist"]
        lines = []
        for run in fresh:
            report = (run.run.report or "(no report)").strip().splitlines()[0][:300]
            lines.append(f"[{run.role} {run.actor_id}: {run.run.outcome}] {report}")
        if fresh or state.version > self._seen_version:
            lines += [f"[waiting for you] {a}" for a in pending_approvals(state)]
        self._seen_version = state.version
        return lines
