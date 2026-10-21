"""The conversation with the Chief Scientist, independent of any terminal interface.

Every researcher message is recorded verbatim as a user intervention once an investigation
is open (messages sent before it opens are recorded right after). A message sent while
proposed agents wait for review holds them until the Chief Scientist proposes again. Slash
commands are handled here, not by a model: authorizations to submit, approve inputs,
cancel, start agents early or close the investigation can only come from the researcher's
own command.

The Chief Scientist also reviews the agents' reports in the background, in a conversation
of its own; what it concluded is shown here, and passed to it at the start of its next
turn in this conversation.

With ``auto_dispatch`` the session starts a dispatcher in the background whenever the
investigation has work for one and none is running, so approved agents start without the
researcher starting anything.

Commands may run while the Chief Scientist is still working on an earlier message (the
terminal interface does this); every record they make is built from the state at the
moment it is written.
"""

import getpass
import shlex
import time
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
from sim2real.contracts.status import INVESTIGATION_MACHINE, CalculationState
from sim2real.errors import RuleViolation
from sim2real.hpc.profile import ProfileStore
from sim2real.hpc.profile import ProfileNotApproved
from sim2real.hpc.slurm import SchedulerBackend, SchedulerError
from sim2real.models.types import ToolCall
from sim2real.dashboard import dispatcher_status
from sim2real.dispatch import (
    IDLE_MINUTES,
    LOG,
    current_activity,
    dispatcher_running,
    heartbeat_age,
    needs_dispatcher,
    start_background_dispatcher,
    stop_dispatcher,
)
from sim2real.notebook import round_file, update_notebook
from sim2real.state.projection import AgentRunRecord, InvestigationState
from sim2real.summaries import approval_requests, investigation_summary, pending_approvals, recommendation_text
from sim2real.swarm import CHIEF, agent_label, agents_summary, close_drafts, halt_drafts, start_drafts
from sim2real.workspace import ModelClients, Workspace, profile_for, registry_for, services_for

@dataclass(frozen=True)
class Command:
    usage: str
    description: str

    @property
    def stem(self) -> str:
        """What is typed before any argument: the usage up to its first <placeholder>."""
        return self.usage.split("<", 1)[0]

    @property
    def takes_arguments(self) -> bool:
        return "<" in self.usage


COMMANDS = (
    Command("/status", "the investigation's state"),
    Command("/agents", "the hypothesis agents, and any the Chief Scientist proposes"),
    Command("/approve agents", "start the proposed agents now instead of after the review window"),
    Command("/pending", "what waits for your approval"),
    Command("/approve submit <calc>", "allow a calculation with approved inputs to be submitted"),
    Command("/approve inputs <calc>", "allow inputs the examiner could not verify"),
    Command("/approve close", "close the investigation for good, as the Chief Scientist proposes"),
    Command("/cancel <calc> <reason>", "cancel a calculation, stopping its job if it has one"),
    Command("/report", "update the research notebook and show where it is"),
    Command("/halt <reason>", "stop this investigation now: cancel its jobs, retire its agents, pause it (reason "
                              "optional)"),
    Command("/halt all <reason>", "the same for every investigation, and stop the background dispatcher"),
    Command("/help", "this list"),
    Command("/quit", "leave (queued work and jobs continue with the dispatcher)"),
    Command("/exit", "leave, like /quit"),
)
"""Every command of the conversation, wherever it runs; interfaces may add their own."""


def command_lines(commands: tuple[Command, ...]) -> str:
    return "\n".join(f"  {c.usage:<28} {c.description}" for c in commands)


RESTART_SECONDS = 60
"""The least time between two dispatchers this chat starts."""
HALT_WAIT_SECONDS = 60
"""How long /halt lets agent runs end before it kills the dispatcher."""
_AT_SCHEDULER = frozenset({CalculationState.SUBMISSION_PENDING, CalculationState.SUBMITTED, CalculationState.QUEUED,
                           CalculationState.RUNNING})

HELP = f"""Commands:
{command_lines(COMMANDS)}
A <calc> may be given by the first characters of its id, if they are unique. Anything else goes to
the Chief Scientist. A message sent while agents are proposed holds them until the Chief Scientist
proposes again."""


@dataclass(frozen=True)
class Notice:
    """Something that happened while the researcher was not looking."""

    text: str
    review: bool = False
    """The Chief Scientist's background review, shown as its message."""


def _is_review(run: AgentRunRecord) -> bool:
    return run.role == "chief_scientist" and run.run.task_id is not None


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
    _shown_starts: set[str] = field(default_factory=set)
    chief_session: str = ""
    auto_dispatch: bool = False
    """Start a dispatcher in the background when there is work for one and none is running."""
    start_dispatcher: Callable[[Workspace], int] | None = None
    """How to start one; by default ``start_background_dispatcher``."""
    _last_start: float | None = None

    def __post_init__(self) -> None:
        if self.investigation_id is not None:
            state = self.workspace.ledger.load(self.investigation_id)
            # Show the agent reports since the conversation last moved: the researcher's last message
            # or the Chief's last reply.
            self._seen_runs = max(
                [i.recorded_at for i in state.interventions.values()]
                + [r.recorded_at for r in state.agent_runs if r.role == "chief_scientist" and not _is_review(r)],
                default=0)
            chats = [r for r in state.agent_runs if r.role == "chief_scientist" and not _is_review(r)]
            self.chief_session = chats[-1].session_id if chats else f"{self.investigation_id}.chief"
            self._shown_starts = set(state.agents)
        else:
            self.chief_session = f"chief-{self.clock():%Y%m%d%H%M%S}"

    # -- input ------------------------------------------------------------------------------

    def handle(self, line: str, observer: Callable[[ToolCall, ToolOutcome], None] | None = None) -> str:
        """Answer one line: a command, or a message to the Chief Scientist, whose tool steps
        ``observer`` sees as they happen."""
        line = line.strip()
        if not line:
            return ""
        reply = self._command(line) if line.startswith("/") else self._ask_chief(line, observer)
        started = self.ensure_dispatcher()
        return f"{reply}\n\n{started}" if started else reply

    def ensure_dispatcher(self) -> str | None:
        """With ``auto_dispatch``, start a background dispatcher if the investigation has work for one and none
        is running; say so, or return None."""
        if not self.auto_dispatch or self.investigation_id is None:
            return None
        if not needs_dispatcher(self.workspace.ledger.load(self.investigation_id)) \
                or dispatcher_running(self.workspace):
            return None
        if self._last_start is not None and time.monotonic() - self._last_start < RESTART_SECONDS:
            return None  # one that keeps failing is not restarted over and over; its log says why
        self._last_start = time.monotonic()
        pid = (self.start_dispatcher or start_background_dispatcher)(self.workspace)
        return (f"[dispatcher] Started in the background (process {pid}) to carry out the queued work. It keeps "
                f"going after you leave and stops by itself after {IDLE_MINUTES} minutes with nothing to do; its log "
                f"is {self.workspace.root / LOG}. `sim2real dispatch --stop` stops it.")

    def _command(self, line: str) -> str:
        try:
            words = shlex.split(line)
        except ValueError as exc:
            return f"Cannot read that command: {exc}"
        name, args = words[0], words[1:]
        if name == "/help":
            return HELP
        if name == "/halt" and args[:1] == ["all"]:
            open_ones = [i for i in self.workspace.store.investigation_ids()
                         if self.workspace.ledger.load(i).status not in INVESTIGATION_MACHINE.terminal]
            return self._halt(open_ones, " ".join(args[1:]), line, everything=True)
        if self.investigation_id is None:
            return "No investigation is open yet. Describe the discrepancy and the Chief Scientist will open one."
        state = self.workspace.ledger.load(self.investigation_id)
        if name == "/status":
            return investigation_summary(state)
        if name == "/agents":
            return agents_summary(state, self.clock())
        if name == "/approve" and args == ["agents"]:
            return self._start_agents(line)
        if name == "/approve" and args == ["close"]:
            return self._close(line)
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
        if name == "/halt":
            return self._halt([self.investigation_id], " ".join(args), line)
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

    def _start_agents(self, line: str) -> str:
        found = []

        def build(state: InvestigationState) -> list[EventDraft]:
            proposal = state.pending_proposal
            if proposal is None:
                return []
            message = self._message_draft(state, line, (Authorization(action=PermissionAction.START_AGENTS,
                                                                      target_ids=(proposal.proposal_id,)),))
            found[:] = [message.payload.intervention_id]
            return [message, *start_drafts(state, proposal, self.user, message.payload.intervention_id)]

        try:
            result = self.workspace.ledger.record_with(self.investigation_id, build)
        except RuleViolation as exc:
            return f"The proposed agents could not start: {exc}"
        if not found:
            return "No agents are waiting to start."
        (message_id,) = found
        started = result.events[1].payload.agent_ids
        return (f"Started {', '.join(f'{a} on {h}' for h, a in started.items())} ({message_id}). The dispatcher runs "
                "their first tasks.")

    def _close(self, line: str) -> str:
        found = []

        def build(state: InvestigationState) -> list[EventDraft]:
            closing = state.pending_closing
            if closing is None:
                return []
            message = self._message_draft(state, line, (Authorization(action=PermissionAction.CLOSE_INVESTIGATION,
                                                                      target_ids=(closing.closing_id,)),))
            found[:] = [message.payload.intervention_id, closing]
            return [message, *close_drafts(state, closing, self.user, message.payload.intervention_id)]

        try:
            result = self.workspace.ledger.record_with(self.investigation_id, build)
        except RuleViolation as exc:
            return f"The investigation could not be closed: {exc}"
        if not found:
            return "Nothing to close: the Chief Scientist has not proposed closing the investigation."
        message_id, closing = found
        done = [event.payload for event in result.events[1:]]
        lines = [f"{self.investigation_id} is now {closing.status.value}, closed for good ({message_id}). The Chief "
                 f"Scientist proposed it ({closing.closing_id}): {closing.reason}"]
        retired = [p.agent_id for p in done if p.type == "agent_retired"]
        if retired:
            lines.append(f"  - retired {', '.join(retired)}")
        lines += [f"  - withdrew the proposal {p.proposal_id}" for p in done if p.type == "proposal_withdrawn"]
        return "\n".join(lines)

    def _halt(self, investigation_ids: list[str], reason: str, line: str, everything: bool = False) -> str:
        """Stop the investigations at once: cancel their jobs, withdraw their proposals, retire their agents,
        pause them; then stop the background dispatcher unless another investigation still needs it."""
        lines = [self._halt_one(i, reason, line) for i in investigation_ids] or ["No open investigation to halt."]
        others = [i for i in self.workspace.store.investigation_ids()
                  if i not in investigation_ids and needs_dispatcher(self.workspace.ledger.load(i))]
        if others and not everything:
            lines.append(f"The background dispatcher keeps running for {', '.join(others)}.")
        else:
            lines.append(stop_dispatcher(self.workspace, timeout=HALT_WAIT_SECONDS, force=True))
        return "\n".join(lines)

    def _halt_one(self, investigation_id: str, reason: str, line: str) -> str:
        if self.workspace.ledger.load(investigation_id).status in INVESTIGATION_MACHINE.terminal:
            return f"{investigation_id}: already closed; nothing to halt."
        found = []

        def build(state: InvestigationState) -> list[EventDraft]:
            live = tuple(c.calculation_id for c in state.calculations.values() if c.state in _AT_SCHEDULER)
            message = self._message_draft(state, line, (Authorization(action=PermissionAction.CANCEL,
                                                                      target_ids=live),) if live else ())
            why = f"halted by the researcher ({message.payload.intervention_id})" + (f": {reason}" if reason else "")
            found[:] = [message.payload.intervention_id, why, live]
            return [message, *halt_drafts(state, self.user, why)]

        result = self.workspace.ledger.record_with(investigation_id, build)
        message_id, why, live = found
        done = [event.payload for event in result.events[1:]]
        lines = [f"{investigation_id}: halted ({message_id})."]
        cancelled, failed = self._cancel_jobs(investigation_id, live, why)
        if live:
            lines.append(f"  - cancelled {len(cancelled)} job{'s' if len(cancelled) != 1 else ''}"
                         + (f": {', '.join(cancelled)}" if cancelled else ""))
        lines += [f"  - could not cancel {problem}" for problem in failed]
        retired = [p.agent_id for p in done if p.type == "agent_retired"]
        if retired:
            lines.append(f"  - retired {', '.join(retired)}; a run in progress stops at its next model call")
        lines += [f"  - withdrew the proposal {p.proposal_id}" for p in done if p.type == "proposal_withdrawn"]
        paused = any(p.type == "investigation_status_changed" for p in done)
        lines.append("  - paused: nothing new starts until you ask the Chief Scientist to resume it" if paused else
                     f"  - left {self.workspace.ledger.load(investigation_id).status.value}, so nothing new starts")
        return "\n".join(lines)

    def _cancel_jobs(self, investigation_id: str, live: tuple[str, ...], reason: str) -> tuple[list[str], list[str]]:
        if not live:
            return [], []
        state = self.workspace.ledger.load(investigation_id)
        try:
            manager = services_for(self.workspace, profile_for(state, self.profiles), self.backend).manager
        except (LookupError, ProfileNotApproved, FileNotFoundError) as exc:
            return [], [f"{', '.join(live)}: no usable compute profile ({exc})"]
        cancelled, failed = [], []
        for calculation_id in live:
            job = state.calculations[calculation_id].scheduler_job_id
            try:
                manager.cancel(investigation_id, calculation_id, reason)
                cancelled.append(f"{calculation_id}" + (f" (job {job})" if job else ""))
            except (RuleViolation, SchedulerError, OSError) as exc:
                failed.append(f"{calculation_id}: {exc}")
        return cancelled, failed

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
            instruction = self._reviews_since_last_turn() + f"[researcher, recorded as {message_id}] {text}"
        else:
            self._unrecorded_messages.append(text)
            instruction = f"[researcher] {text}"
        context = ChiefContext(ledger=self.workspace.ledger, profiles=self.profiles, actor=CHIEF,
                               registry_for=lambda name: registry_for(self.profiles.load_approved(name)),
                               investigation_id=self.investigation_id, on_open=self._opened, clock=self.clock,
                               dispatcher=self._dispatcher_status)
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

    def _dispatcher_status(self) -> str:
        now = self.clock()
        if self.auto_dispatch and not dispatcher_running(self.workspace):
            return ("none is running now; this chat starts one in the background by itself right after your turn "
                    "whenever there is queued work, so the researcher need not start anything")
        text, _ = dispatcher_status(heartbeat_age(self.workspace, now), current_activity(self.workspace), now,
                                    automatic=self.auto_dispatch)
        return text

    def _reviews_since_last_turn(self) -> str:
        """The Chief Scientist's background reviews this conversation has not seen (the last three), so it answers
        knowing them."""
        state = self.workspace.ledger.load(self.investigation_id)
        last = max((r.recorded_at for r in state.agent_runs if r.session_id == self.chief_session), default=0)
        reviews = [r for r in state.agent_runs if _is_review(r) and r.recorded_at > last and r.run.report][-3:]
        return "".join(f"[record: your background review of the agents' reports, run {r.run.outcome}]\n"
                       f"{r.run.report.strip()}\n\n" for r in reviews)

    def _opened(self, investigation_id: str) -> list[str]:
        """Record the messages sent before the investigation existed, in order, as soon as it does."""
        self.investigation_id = investigation_id
        recorded = [self._record_message(text) for text in self._unrecorded_messages]
        self._unrecorded_messages.clear()
        return recorded

    def _message_draft(self, state: InvestigationState, text: str,
                       authorizations: tuple[Authorization, ...] = ()) -> EventDraft:
        # Numbered from the state the event is appended to, so concurrent writers never collide.
        message_id = f"msg-{1 + sum(1 for i in state.interventions if i.startswith('msg-'))}"
        return EventDraft(actor=self.user, payload=UserInterventionRecorded(
            intervention_id=message_id, text=text, authorizations=authorizations))

    def _record_message(self, text: str, authorizations: tuple[Authorization, ...] = ()) -> str:
        message_id = []

        def build(state: InvestigationState) -> list[EventDraft]:
            draft = self._message_draft(state, text, authorizations)
            message_id[:] = [draft.payload.intervention_id]
            return [draft]

        self.workspace.ledger.record_with(self.investigation_id, build)
        return message_id[0]

    # -- what happened meanwhile ----------------------------------------------------------------

    def notices(self) -> list[Notice]:
        """What arrived since the researcher last looked: agent reports, the Chief Scientist's reviews and the next
        steps it recommended, agents that started, and each request for approval the first time it is seen."""
        if self.investigation_id is None:
            return []
        state = self.workspace.ledger.load(self.investigation_id)
        fresh = [r for r in state.agent_runs if r.recorded_at > self._seen_runs
                 and (r.role != "chief_scientist" or _is_review(r))]
        recommended = [r for r in state.recommendations.values() if r.recorded_at > self._seen_runs]
        self._seen_runs = state.version
        notices = []
        for agent in state.agents.values():  # the researcher's own /approve agents already said who started
            if agent.agent_id not in self._shown_starts and agent.authorization_id is None:
                notices.append(Notice(f"[agent started after the review window, with no reply] {agent_label(agent)} "
                                      f"on {agent.hypothesis_id}"))
            self._shown_starts.add(agent.agent_id)
        for run in fresh:
            if _is_review(run):
                notices.append(Notice((run.run.report or f"(the review ended {run.run.outcome} without a "
                                                         "message)").strip(), review=True))
                continue
            agent = state.agents.get(run.actor_id)
            who = f"{agent_label(agent)}" if agent else f"{run.role} {run.actor_id}"
            report = (run.run.report or "(no report)").strip().splitlines()[0][:300]
            notices.append(Notice(f"[{who}: {run.run.outcome}] {report}"))
        notices += [Notice(f"[{recommendation_text(r)}]") for r in recommended]
        for command, note in approval_requests(state):
            if f"{command} {note}" not in self._shown_approvals:
                self._shown_approvals.add(f"{command} {note}")
                notices.append(Notice(f"[waiting for you] {command}  ({note})"))
        return notices
