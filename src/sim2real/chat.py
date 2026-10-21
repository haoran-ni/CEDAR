"""The conversation with the Chief Scientist, independent of any terminal interface.

Every researcher message is recorded verbatim as a user intervention once an investigation
is open (messages sent before it opens are recorded right after). Slash commands are
handled here, not by a model: authorizations to submit, approve inputs or cancel can only
come from the researcher's own command.

Commands may run while the Chief Scientist is still working on an earlier message (the
terminal interface does this); every record they make is built from the state at the
moment it is written.
"""

import getpass
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sim2real.agents import AgentRunner, RunLimits, RunResult, record_agent_run
from sim2real.agents.tools import ToolOutcome
from sim2real.agents.prompts import CHIEF_SCIENTIST
from sim2real.agents.toolkits.chief import ChiefContext, chief_tools
from sim2real.contracts.base import Actor, ActorKind
from sim2real.contracts.events import EventDraft, UserInterventionRecorded
from sim2real.contracts.plan import Authorization, PermissionAction
from sim2real.errors import RuleViolation
from sim2real.hpc.profile import ProfileStore
from sim2real.hpc.slurm import SchedulerBackend
from sim2real.models.types import ToolCall
from sim2real.notebook import round_file, update_notebook
from sim2real.state.projection import InvestigationState
from sim2real.summaries import approval_requests, investigation_summary, pending_approvals
from sim2real.workspace import ModelClients, Workspace, profile_for, registry_for, services_for

CHIEF = Actor(kind=ActorKind.CHIEF_SCIENTIST, id="chief-scientist")
HELP = """Commands:
  /status                      the investigation's state
  /pending                     what waits for your approval
  /approve submit <calc>       allow a calculation with approved inputs to be submitted
  /approve inputs <calc>       allow inputs the examiner could not verify
  /cancel <calc> <reason>      cancel a calculation, stopping its job if it has one
                               (<calc> may be the first characters of its id, if unique)
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
    _seen_runs: int = 0
    _shown_approvals: set[str] = field(default_factory=set)
    chief_session: str = ""

    def __post_init__(self) -> None:
        if self.investigation_id is not None:
            state = self.workspace.ledger.load(self.investigation_id)
            # Show the agent reports since the conversation last moved: the researcher's last message
            # or the Chief's last reply.
            self._seen_runs = max(
                [i.recorded_at for i in state.interventions.values()]
                + [r.recorded_at for r in state.agent_runs if r.role == "chief_scientist"], default=0)
            chief_runs = [r for r in state.agent_runs if r.role == "chief_scientist"]
            self.chief_session = chief_runs[-1].session_id if chief_runs else f"{self.investigation_id}.chief"
        else:
            self.chief_session = f"chief-{self.clock():%Y%m%d%H%M%S}"

    # -- input ------------------------------------------------------------------------------

    def handle(self, line: str, observer: Callable[[ToolCall, ToolOutcome], None] | None = None) -> str:
        """Answer one line: a command, or a message to the Chief Scientist, whose tool steps
        ``observer`` sees as they happen."""
        line = line.strip()
        if not line:
            return ""
        if line.startswith("/"):
            return self._command(line)
        return self._ask_chief(line, observer)

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
                    f"This round: {update.directory / round_file(state.round)}")
        if name == "/approve" and len(args) == 2 and args[0] in ("submit", "inputs"):
            action = PermissionAction.SUBMIT if args[0] == "submit" else PermissionAction.APPROVE_INPUTS
            return self._authorize(action, args[1], line)
        if name == "/cancel" and len(args) >= 2:
            return self._cancel(args[0], " ".join(args[1:]), line)
        return f"Unknown or incomplete command: {line}\n{HELP}"

    @staticmethod
    def _find_calculation(state: InvestigationState, given: str) -> tuple[str | None, str]:
        """The calculation ``given`` names, by full id or by a unique start of at least 4 characters."""
        if given in state.calculations:
            return given, ""
        stem = given.removeprefix("calc_")
        matches = [c for c in state.calculations if len(stem) >= 4 and c.removeprefix("calc_").startswith(stem)]
        if len(matches) == 1:
            return matches[0], ""
        if matches:
            return None, f"{given} matches several calculations ({', '.join(matches)}); give more of the id."
        return None, f"No calculation {given} in this investigation."

    def _authorize(self, action: PermissionAction, given: str, line: str) -> str:
        state = self.workspace.ledger.load(self.investigation_id)
        calculation_id, problem = self._find_calculation(state, given)
        if calculation_id is None:
            return problem
        intervention = self._record_message(line, (Authorization(action=action, target_ids=(calculation_id,)),))
        if action is PermissionAction.APPROVE_INPUTS:
            services = services_for(self.workspace, profile_for(state, self.profiles), self.backend)
            try:
                services.engine.approve(self.investigation_id, calculation_id)
            except RuleViolation as exc:
                return f"Recorded your approval ({intervention}), but the inputs could not be validated: {exc}"
            return f"Inputs of {calculation_id} approved ({intervention}). They still need /approve submit to run."
        return f"Submission of {calculation_id} authorized ({intervention}). The dispatcher submits it on its next tick."

    def _cancel(self, given: str, reason: str, line: str) -> str:
        state = self.workspace.ledger.load(self.investigation_id)
        calculation_id, problem = self._find_calculation(state, given)
        if calculation_id is None:
            return problem
        intervention = self._record_message(line, (Authorization(action=PermissionAction.CANCEL,
                                                                 target_ids=(calculation_id,)),))
        services = services_for(self.workspace, profile_for(state, self.profiles), self.backend)
        try:
            services.manager.cancel(self.investigation_id, calculation_id, reason)
        except RuleViolation as exc:
            return f"Recorded your cancellation ({intervention}), but it could not be applied: {exc}"
        return f"Cancelled {calculation_id} ({intervention})."

    # -- the Chief Scientist ------------------------------------------------------------------

    def _ask_chief(self, text: str, observer: Callable[[ToolCall, ToolOutcome], None] | None = None) -> str:
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
                            instruction=instruction, limits=self.limits, observer=observer)
        if self.investigation_id is None:
            self._unrecorded_runs.append(result)
        else:
            for earlier in [*self._unrecorded_runs, result]:
                record_agent_run(self.workspace.ledger, self.investigation_id, CHIEF, session_id=self.chief_session,
                                 role="chief_scientist", settings=settings, result=earlier)
            self._unrecorded_runs.clear()
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
        message_id = []

        def build(state: InvestigationState) -> list[EventDraft]:
            # Numbered from the state the event is appended to, so concurrent writers never collide.
            message_id[:] = [f"msg-{1 + sum(1 for i in state.interventions if i.startswith('msg-'))}"]
            return [EventDraft(actor=self.user, payload=UserInterventionRecorded(
                intervention_id=message_id[0], text=text, authorizations=authorizations))]

        self.workspace.ledger.record_with(self.investigation_id, build)
        return message_id[0]

    # -- what happened meanwhile ----------------------------------------------------------------

    def notices(self) -> list[str]:
        """Agent reports that arrived since the researcher last looked, and each request for
        approval the first time it is seen."""
        if self.investigation_id is None:
            return []
        state = self.workspace.ledger.load(self.investigation_id)
        fresh = [r for r in state.agent_runs if r.recorded_at > self._seen_runs and r.role != "chief_scientist"]
        self._seen_runs = state.version
        lines = []
        for run in fresh:
            report = (run.run.report or "(no report)").strip().splitlines()[0][:300]
            lines.append(f"[{run.role} {run.actor_id}: {run.run.outcome}] {report}")
        for command, note in approval_requests(state):
            if command not in self._shown_approvals:
                self._shown_approvals.add(command)
                lines.append(f"[waiting for you] {command}  ({note})")
        return lines
