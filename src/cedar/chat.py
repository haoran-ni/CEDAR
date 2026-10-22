"""The conversation with the Chief Scientist, independent of any terminal interface.

Every researcher message is recorded verbatim as a user intervention once an investigation
is open (messages sent before it opens are recorded right after). A message sent while
proposed agents wait for review holds them until the Chief Scientist proposes again. Slash
commands are handled here, not by a model: authorizations to run a calculation, cancel,
start agents early or close the investigation can only come from the researcher's own
command. A calculation is approved to run once, after the input reviewer has finished
with it: that approval covers its inputs and its submission, and is the last step before
it runs. A session reopened on an investigation rebuilds the conversation so far from the
record, with the Chief Scientist's answers and the notices shown meanwhile.

The Chief Scientist also reviews the agents' reports in the background, in a conversation
of its own; what it concluded is shown here, and passed to it at the start of its next
turn in this conversation. Once this conversation has grown past ``fresh_after_tokens``, it
continues in a new one that opens with a briefing built from the record, the latest exchanges
included.

With ``auto_dispatch`` the session starts a dispatcher in the background whenever the
investigation has work for one and none is running, so approved agents start without the
researcher starting anything.

Commands may run while the Chief Scientist is still working on an earlier message (the
terminal interface does this); every record they make is built from the state at the
moment it is written.

``/web`` sets whether the agents may use the web, ``/attach`` gives them a file to view, and
``/budget`` sets what the investigation's agents may spend on models in total, in dollars
($15 unless the researcher sets another amount), which also limits the Chief Scientist's
answers here. The size of this conversation is a separate, token limit: past
``fresh_after_tokens`` it continues in a new one. All three work before the investigation exists: what they set is
recorded, with the researcher's command, the moment it opens, and the Chief Scientist works
under it meanwhile.
"""

import getpass
import mimetypes
import re
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from cedar.agents import AgentRunner, RunLimits, RunResult, record_agent_run
from cedar.agents.tools import ToolOutcome
from cedar.agents.prompts import CHIEF_SCIENTIST
from cedar.agents.transcripts import estimated_tokens
from cedar.briefing import chat_briefing, session_name, sessions_of
from cedar.agents.media import viewable
from cedar.agents.toolkits.chief import ChiefContext, chief_tools
from cedar.agents.toolkits.files import agent_access, file_tools
from cedar.agents.toolkits.materials import Material, material_line, material_tools, recorded_materials
from cedar.contracts.base import Actor, ActorKind
from pydantic import ValidationError

from cedar.automation import mode_summary
from cedar.contracts.base import ArtifactRef
from cedar.contracts.events import (
    EventDraft,
    MaterialAttached,
    ModeChanged,
    RetrievalPolicyChanged,
    BudgetChanged,
    UserInterventionRecorded,
    WorkbenchPolicyChanged,
)
from cedar.contracts.investigation import DEFAULT_BUDGET_USD, RetrievalPolicy
from cedar.contracts.plan import Authorization, PermissionAction
from cedar.contracts.examination import ExaminationVerdict
from cedar.contracts.status import INVESTIGATION_MACHINE, CalculationState, InvestigationMode
from cedar.errors import RuleViolation
from cedar.hpc.profile import ProfileStore
from cedar.hpc.profile import ProfileNotApproved
from cedar.hpc.node import ExecutionHost, HostWatch, local_dft
from cedar.hpc.slurm import SchedulerBackend, SchedulerError
from cedar.models.prices import dollars
from cedar.models.types import ToolCall
from cedar.dashboard import dispatcher_status
from cedar.dispatch import (
    IDLE_MINUTES,
    LOG,
    current_activity,
    dispatcher_host,
    dispatcher_running,
    heartbeat_age,
    needs_dispatcher,
    start_background_dispatcher,
    stop_dispatcher,
)
from cedar.notebook import round_file, update_notebook
from cedar.paths import config_dir
from cedar.state.projection import (
    AgentRunRecord,
    CalculationRecord,
    InvestigationState,
    RecommendationRecord,
    RetrievalRecord,
    awaits_reviewer,
    find_agent,
)
from cedar.summaries import (
    agent_requests,
    approval_requests,
    awaiting_researcher,
    awaiting_review,
    gist,
    needs_inputs_approval,
    short_id,
    submission_note,
    calculation_summary,
    investigation_summary,
    pending_approvals,
)
from cedar.swarm import CHIEF, agent_label, agents_summary, close_drafts, halt_drafts, start_drafts
from cedar.web import web_access, web_summary
from cedar.workspace import ModelClients, Workspace, investigation_registrar, profile_for, registry_for, services_for

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
    Command("/approve submit <calc>", "let a calculation run, once the input reviewer has finished with it"),
    Command("/approve submit all", "let every calculation waiting for you run (those needing a look: once shown)"),
    Command("/approve submit <agent>", "let the calculations of one agent waiting for you run, e.g. A2"),
    Command("/reject inputs <calc> <reason>", "refuse a calculation's inputs before it runs; the reason goes to the "
                                              "agent"),
    Command("/approve close", "close the investigation for good, as the Chief Scientist proposes"),
    Command("/mode", "show the mode: human-in-the-loop, or full automation and what it has left"),
    Command("/mode human-in-the-loop", "take your decisions back: agent proposals and submissions wait for you"),
    Command("/mode full-automation <core-hours> <dollars>", "while you are away, an approver decides on agents and "
            "submissions and the Evidence Judge assesses new evidence, within these limits"),
    Command("/budget", "show what the investigation's agents have spent on models, and the budget"),
    Command("/budget <dollars>", "limit what all its agents may spend on models in total, e.g. 40 "
            f"(${DEFAULT_BUDGET_USD:g} by default)"),
    Command("/budget off", "no limit on spending"),
    Command("/web", "show whether the agents may use the web"),
    Command("/web on", "let the Chief Scientist and the hypothesis agents search and read the web"),
    Command("/web only <domains>", "the same, limited to these domains, e.g. arxiv.org doi.org"),
    Command("/web off", "no web for any agent (the default; keeps blind tests blind)"),
    Command("/workbench", "show whether the agents may run commands in their workspaces, and where they run"),
    Command("/workbench on", "let them (the default): sandboxed, no network, no DFT, writing only in their workspaces"),
    Command("/workbench off", "no commands for any agent of this investigation"),
    Command("/attach <path> <note>", "give the agents an image, PDF or text file to view; the note is optional"),
    Command("/materials", "the files you attached"),
    Command("/register <dir> <note>", "register a finished FHI-aims run computed outside CEDAR, shared with every "
                                      "agent, so evidence can come from it; --output <file> if it is not aims.out"),
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
MAX_ATTACH_BYTES = 100_000_000
HALT_WAIT_SECONDS = 60
"""How long /halt lets agent runs end before it kills the dispatcher."""
_AT_SCHEDULER = frozenset({CalculationState.SUBMISSION_PENDING, CalculationState.SUBMITTED, CalculationState.QUEUED,
                           CalculationState.RUNNING})

HELP = f"""Commands:
{command_lines(COMMANDS)}
A <calc> may be given by the first characters of its id, if they are unique. Anything else goes to
the Chief Scientist. A message sent while agents are proposed holds them until the Chief Scientist
proposes again."""


def _not_ready(state: InvestigationState, calc: CalculationRecord) -> str:
    """Why a calculation cannot be approved to run now."""
    cid = calc.calculation_id
    if calc.registered:
        return f"{cid} is a registered run, computed outside CEDAR; it does not run."
    if calc.state is CalculationState.SPECIFIED and calc.examination is None:
        return f"The inputs of {cid} have not been examined yet; you are asked once the input reviewer has finished."
    if awaits_reviewer(state, calc):
        return (f"The input reviewer is still checking the inputs of {cid}; you are asked to approve it once it has "
                "finished, and your approval is the last step before it runs.")
    if calc.state is CalculationState.REJECTED:
        return f"The inputs of {cid} were rejected: {calc.end_reason}"
    if calc.state is CalculationState.VALIDATED_INPUT:
        return f"{cid} is approved already; the dispatcher submits it on its next tick."
    if calc.state is CalculationState.SPECIFIED:
        return f"{cid} waits for nothing from you."
    if calc.state in _AT_SCHEDULER:
        return f"{cid} is already {calc.state.value}."
    return f"{cid} has ended ({calc.state.value})."


def _not_rejectable(calc: CalculationRecord) -> str | None:
    """Why a calculation's inputs cannot be rejected now, or None if they wait for the researcher."""
    cid = calc.calculation_id
    if calc.registered:
        return f"{cid} is a registered run, computed outside CEDAR; it has no inputs to reject."
    if calc.state is CalculationState.VALIDATED_INPUT or (
            calc.state is CalculationState.SPECIFIED and calc.examination is not None
            and calc.examination.verdict is not ExaminationVerdict.REJECTED):
        return None  # waiting for review (the researcher's or the input reviewer's) or for submission
    if calc.state is CalculationState.SPECIFIED:
        return f"The inputs of {cid} have not been examined yet."
    if calc.state is CalculationState.REJECTED:
        return f"The inputs of {cid} are already rejected."
    if calc.state in _AT_SCHEDULER:
        return f"{cid} is already {calc.state.value}; /cancel {short_id(cid)} <reason> stops it."
    return f"{cid} has ended ({calc.state.value}); there is nothing to reject."


@dataclass(frozen=True)
class Notice:
    """Something that happened while the researcher was not looking, in a few words: the full reports are in the
    notebook and the transcripts."""

    body: str
    source: str = ""
    """Who it comes from: an agent by its tag and id (e.g. "A3 agent-h-kgrid"), "input reviewer", "Evidence Judge",
    "approver" or "Chief Scientist"; empty for CEDAR's own notices."""
    label: str = ""
    """What it is: how the agent's task ended, "asks you", "waiting for you"."""
    attention: bool = False
    """It needs the researcher: a question for them, something waiting for their approval, or an agent that could
    not go on."""
    command: str = ""
    """The command that grants what waits for the researcher; the body then says what it concerns."""
    review: bool = False
    """The Chief Scientist's background review, shown as its message."""

    @property
    def text(self) -> str:
        """The notice as one plain line."""
        head = ": ".join(part for part in (self.source, self.label) if part)
        body = f"{self.command}  ({self.body})" if self.command else self.body
        return f"[{head}] {body}" if head else body


@dataclass(frozen=True)
class Said:
    """A turn of the conversation as recorded: the researcher's message or command, or the Chief Scientist's answer."""

    text: str
    researcher: bool


_ROLE_NAMES = {"input_reviewer": "input reviewer", "evidence_judge": "Evidence Judge", "approver": "approver",
               "chief_scientist": "Chief Scientist"}
_FAILED_RUNS = {"turn_limit", "budget_limit", "truncated", "refused", "model_error", "stopped"}
"""Run outcomes that stopped an agent before it finished its task."""
_NEEDS_RESEARCHER = {"needs_researcher_data", "could_not_proceed"}


def _report_notice(state: InvestigationState, run: AgentRunRecord) -> Notice:
    """An agent's finished task in one line: who, how it ended, and the first sentence of what it said."""
    agent = state.agents.get(run.actor_id)
    who = f"{agent.tag} {agent.agent_id}" if agent else _ROLE_NAMES.get(run.role, f"{run.role} {run.actor_id}")
    report = run.run.structured
    ended = report.outcome.value if report is not None else run.run.outcome
    said = report.summary if report is not None else run.run.report or "(no report)"
    return Notice(gist(said, state.calculations), who, ended.replace("_", " "),
                  attention=ended in _NEEDS_RESEARCHER or ended in _FAILED_RUNS)


def _next_steps_notice(state: InvestigationState, record: RecommendationRecord) -> Notice:
    """The Chief Scientist's choice of next steps: what it takes and the gist of why, and each chosen step in a line;
    the other candidates and the reasons are on the round's notebook page."""
    rec = record.recommendation
    among = state.calculations
    lines = [f"Taking {', '.join(rec.chosen) or 'none for now'} ({rec.recommendation_id}, round {record.round}): "
             f"{gist(rec.reason, among)}"]
    chosen = [c for c in rec.candidates if c.candidate_id in rec.chosen]
    lines += [f"  {c.candidate_id} ({c.kind.value}, {c.priority.value} priority"
              + (f", on {', '.join(c.hypothesis_ids)}" if c.hypothesis_ids else "") + f"): {gist(c.action, among, 160)}"
              for c in chosen]
    others = len(rec.candidates) - len(chosen)
    lines.append("  In full" + (f", with the {others} other candidate{'s' if others > 1 else ''}," if others else "")
                 + " on the round's notebook page (/round).")
    return Notice("\n".join(lines), "Chief Scientist", "next steps")


def _is_review(run: AgentRunRecord) -> bool:
    return run.role == "chief_scientist" and run.run.task_id is not None


def _recorded_notices(state: InvestigationState, after: int, until: int) -> list[tuple[int, Notice]]:
    """The notices for what was recorded after event ``after`` up to event ``until``, each with the event it came
    from: agent reports and what they ask the researcher, the Chief Scientist's reviews and next steps, the approver's
    decisions, and full automation ending."""
    def new(recorded_at: int) -> bool:
        return after < recorded_at <= until

    fresh = [r for r in state.agent_runs if new(r.recorded_at) and (r.role != "chief_scientist" or _is_review(r))]
    recommended = [r for r in state.recommendations.values() if new(r.recorded_at)]
    decided = [a for a in state.approvals.values() if new(a.recorded_at)]
    ended = [m for m in state.modes if new(m.recorded_at) and m.actor.kind is not ActorKind.USER]
    notices = []
    for run in fresh:
        if _is_review(run):
            notices.append((run.recorded_at, Notice((run.run.report or f"(the review ended {run.run.outcome} without "
                                                                       "a message)").strip(), review=True)))
            continue
        if run.role == "approver" and run.run.outcome not in _FAILED_RUNS:
            continue  # each of its decisions is shown below, with its reason
        notice = _report_notice(state, run)
        notices.append((run.recorded_at, notice))
        asked = run.run.structured.requests if run.run.structured is not None else ()
        notices += [(run.recorded_at, Notice(r.text, notice.source, "asks you", attention=True)) for r in asked
                    if r.to == "researcher"]
    notices += [(r.recorded_at, _next_steps_notice(state, r)) for r in recommended]
    for a in decided:
        what = (f"the agents of {a.target_id}" if a.action is PermissionAction.START_AGENTS else
                f"submitting {a.target_id}" + (f" ({a.core_hours:g} core-hours)" if a.core_hours else ""))
        notices.append((a.recorded_at, Notice(f"{'approved' if a.approved else 'declined'} {what}: {a.reason}",
                                              "approver", "full automation")))
    notices += [(m.recorded_at, Notice(f"{m.reason}. It is paused: ask the Chief Scientist to resume it, or switch "
                                       "full automation on again with /mode.", label="full automation ended",
                                       attention=True)) for m in ended]
    return notices


def _answer(run: AgentRunRecord) -> Said:
    """The Chief Scientist's answer to the researcher, as recorded (the record keeps the first REPORT_LIMIT
    characters)."""
    stopped = f"\n\n[the Chief Scientist stopped: {run.run.outcome}]" if run.run.outcome != "finished" else ""
    return Said(((run.run.report or "") + stopped).strip() or "(no reply)", researcher=False)


def _in_turns(entries: list[tuple[int, Said | Notice]]) -> list[Said | Notice]:
    """The entries, each with the event it came from, in the order they were shown. Messages sent before the
    investigation opened are recorded as it opens, and the answers to them together at the end of that turn: each of
    those answers is put right after its message. A message whose answer was never recorded (the window was closed
    while the Chief Scientist worked) stays without one."""
    entries = sorted(entries, key=lambda e: e[0])
    keys = [(at, 0) for at, _ in entries]
    asked: list[int] = []  # the messages since the last answers, by the event that recorded them
    answers: list[int] = []  # answers recorded one after another, by their place in ``entries``

    def pair() -> None:
        # The last answer is to the last message and stays where it was recorded; each before it goes after its own.
        for place, at in list(zip(reversed(answers), reversed(asked)))[1:]:
            keys[place] = (at, 1)
        asked.clear()
        answers.clear()

    for place, (at, entry) in enumerate(entries):
        if isinstance(entry, Said) and not entry.researcher:
            answers.append(place)
        elif isinstance(entry, Said) and not entry.text.startswith("/"):
            if answers:
                pair()
            asked.append(at)
    pair()
    return [entry for _, (_, entry) in sorted(zip(keys, entries), key=lambda keyed: keyed[0])]


@dataclass
class ChatSession:
    workspace: Workspace
    profiles: ProfileStore
    clients: ModelClients
    investigation_id: str | None = None
    user: Actor = field(default_factory=lambda: Actor(kind=ActorKind.USER, id=getpass.getuser()))
    backend: SchedulerBackend | None = None
    limits: RunLimits = field(default_factory=RunLimits)
    """For each answer of the Chief Scientist: no limit on model calls; what the budget has left caps it."""
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    _unrecorded: list[tuple] = field(default_factory=list)
    """What the researcher said and set before the investigation existed, in order: ("message", text),
    ("web", line, policy, domains), ("attach", line, Material) and ("budget", line, max_usd)."""
    _announced_materials: set[str] = field(default_factory=set)
    _unrecorded_runs: list[RunResult] = field(default_factory=list)
    _seen_runs: int = 0
    _history_until: int = 0
    """The last event ``history`` shows; ``notices`` shows what came after."""
    _shown_approvals: set[str] = field(default_factory=set)
    _shown_reviews: set[str] = field(default_factory=set)
    """Calculations whose inputs needing a look the researcher has been shown; /approve submit all approves only these
    of them."""
    _shown_starts: set[str] = field(default_factory=set)
    chief_session: str = ""
    auto_dispatch: bool = False
    """Start a dispatcher in the background when there is work for one and none is running."""
    start_dispatcher: Callable[[Workspace], int] | None = None
    """How to start one; by default ``start_background_dispatcher``."""
    host: Callable[[], ExecutionHost] = field(default_factory=HostWatch)
    """Where the chat runs: a login node, or an interactive node inside a Slurm allocation."""
    fresh_after_tokens: int = 800_000
    """Once the conversation's input has grown this large, it continues in a new conversation, briefed."""
    _last_start: float | None = None
    _context_tokens: int | None = None
    _warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.investigation_id is not None:
            state = self.workspace.ledger.load(self.investigation_id)
            # Show the agent reports since the conversation last moved: the researcher's last message
            # or the Chief's last reply.
            self._seen_runs = max(
                [i.recorded_at for i in state.interventions.values()]
                + [r.recorded_at for r in state.agent_runs if r.role == "chief_scientist" and not _is_review(r)],
                default=0)
            self._history_until = self._seen_runs
            chats = [r for r in state.agent_runs if r.role == "chief_scientist" and not _is_review(r)]
            self.chief_session = chats[-1].session_id if chats else f"{self.investigation_id}.chief"
            self._shown_starts = set(state.agents)
            self._announced_materials = set(state.materials)
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
                f"is {self.workspace.root / LOG}. `cedar dispatch --stop` stops it.")

    def _command(self, line: str) -> str:
        if line.split(maxsplit=1)[0] == "/reject":
            return self._reject(line)  # read from the line as typed: a reason is prose, apostrophes and all
        try:
            words = shlex.split(line)
        except ValueError as exc:
            return f"Cannot read that command: {exc}"
        name, args = words[0], words[1:]
        if name == "/help":
            return HELP
        if name == "/budget":
            return self._budget(args, line)
        if name == "/web":
            return self._web(args, line)
        if name == "/attach":
            return self._attach(args, line)
        if name == "/materials":
            return "\n".join(material_line(m) for m in self._materials()) or "You have attached no materials."
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
        if name == "/mode":
            return self._mode(args, line)
        if name == "/pending":
            requests = agent_requests(state)
            pending = pending_approvals(state, self._runs_root(state))
            self._shown_reviews.update(awaiting_review(state))
            text = "\n".join(pending) or "Nothing waits for your approval."
            if any(p.startswith("/approve submit") for p in pending):
                text += ("\nRefuse any of these calculations with /reject inputs <calc> <reason>; the reason goes to "
                         "its agent.")
            return text + ("\n\nRequests from agents:\n" + "\n".join(f"  {r}" for r in requests) if requests else "")
        if name == "/report":
            update = update_notebook(self.workspace, self.investigation_id)
            return (f"Research notebook ({update.summary}): {update.index}\n"
                    f"This round: {update.directory / round_file(state.round)}")
        if name == "/approve" and len(args) == 2 and args[0] == "submit":
            return self._authorize(args[1], line)
        if name == "/approve" and args[:1] in (["inputs"], ["input"]):
            return ("A calculation's inputs and its submission are approved together, once the input reviewer has "
                    "finished with it: /approve submit <calc>, all, or an agent such as A2. /pending shows what "
                    "waits for you.")
        if name == "/cancel" and len(args) >= 2:
            return self._cancel(args[0], " ".join(args[1:]), line)
        if name == "/register":
            return self._register(args, line)
        if name == "/workbench":
            return self._workbench(args, line)
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

    def _authorize(self, given: str, line: str) -> str:
        """Let calculations run whose input review has finished: approve their inputs where they wait for the
        researcher's look, and authorize their submission."""
        state = self.workspace.ledger.load(self.investigation_id)
        if given.lower() == "all" or find_agent(state, given) is not None:
            return self._authorize_many(state, given, line)
        calculation_id, problem = self._find_calculation(state, given)
        if calculation_id is None:
            return problem
        if calculation_id not in awaiting_researcher(state):
            return _not_ready(state, state.calculations[calculation_id])
        return self._approve_to_run(state, [state.calculations[calculation_id]], line, "")

    def _authorize_many(self, state: InvestigationState, given: str, line: str) -> str:
        """Let every calculation waiting for the researcher run, or those of one agent's hypothesis; inputs waiting
        for their look only if they have been shown them."""
        agent = None if given.lower() == "all" else find_agent(state, given)
        whose = f" of {agent.tag} {agent.agent_id}" if agent else ""

        def theirs(calc: CalculationRecord) -> bool:
            return agent is None or agent.hypothesis_id in calc.hypothesis_ids

        waiting = [c for c in (state.calculations[c] for c in awaiting_researcher(state)) if theirs(c)]
        # Inputs the researcher has not seen are never approved in bulk: they may have arrived after /pending.
        review = set(awaiting_review(state))
        unseen = [c.calculation_id for c in waiting if c.calculation_id in review
                  and c.calculation_id not in self._shown_reviews]
        waiting = [c for c in waiting if c.calculation_id not in unseen]
        left_out = (f"Not approved, because you have not been shown them yet: {', '.join(unseen)}. /pending shows "
                    "them.") if unseen else ""
        if not waiting:
            if left_out:
                return left_out
            reviewing = [c for c in state.calculations.values() if awaits_reviewer(state, c) and theirs(c)]
            later = (f" {len(reviewing)} {'is' if len(reviewing) == 1 else 'are'} still with the input reviewer; you "
                     "are asked once it has finished." if reviewing else "")
            return f"No calculation{whose} waits for your approval to run.{later}"
        return self._approve_to_run(state, waiting, line, whose, left_out)

    def _approve_to_run(self, state: InvestigationState, calcs: list[CalculationRecord], line: str, whose: str,
                        left_out: str = "") -> str:
        """Record the researcher's approval of ``calcs`` to run, in one message: of their inputs where those wait for
        the researcher, and of their submission; then validate those inputs."""
        review = [c.calculation_id for c in calcs if needs_inputs_approval(state, c)]
        ids = tuple(c.calculation_id for c in calcs)
        authorizations = ((Authorization(action=PermissionAction.APPROVE_INPUTS, target_ids=tuple(review)),)
                          if review else ())
        if state.spec.constraints.submit_requires_approval:
            authorizations += (Authorization(action=PermissionAction.SUBMIT, target_ids=ids),)
        intervention = self._record_message(line, authorizations)
        root = self._runs_root(state)
        notes = {c.calculation_id: submission_note(state, c, root) for c in calcs}
        failed: dict[str, str] = {}
        if review:
            services = services_for(self.workspace, profile_for(state, self.profiles), self.backend)
            for calculation_id in review:
                try:
                    services.engine.approve(self.investigation_id, calculation_id)
                except RuleViolation as exc:
                    failed[calculation_id] = str(exc)
        running = [c for c in ids if c not in failed]
        plural = "s" * (len(running) != 1)
        lines = [f"Approved {len(running)} calculation{plural}{whose} to run ({intervention}). The dispatcher "
                 f"submits {'them' if plural else 'it'} on its next tick:"] if running else []
        lines += [f"  - {c}: {notes[c]}" for c in running]
        if failed:
            lines.append(f"Recorded your approval ({intervention}), but these inputs could not be validated:")
            lines += [f"  - {c}: {problem}" for c, problem in failed.items()]
        return "\n".join(lines + ([left_out] if left_out else []))

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

    def _mode(self, args: list[str], line: str) -> str:
        usage = "Usage: /mode, /mode human-in-the-loop, or /mode full-automation <core-hours> <dollars>"
        if not args:
            state = self.workspace.ledger.load(self.investigation_id)
            return f"{self.investigation_id} is in {mode_summary(state)}.\n{usage}"
        wanted, limits = args[0].lower(), {}
        if wanted == "human-in-the-loop" and len(args) == 1:
            mode = InvestigationMode.HUMAN_IN_THE_LOOP
        elif wanted == "full-automation" and len(args) == 3:
            mode = InvestigationMode.FULL_AUTOMATION
            try:
                limits = {"max_core_hours": float(args[1].replace(",", "")), "max_usd": self._amount(args[2])}
            except ValueError:
                return usage
            if limits["max_usd"] is None:
                return usage
        else:
            return usage

        def build(state: InvestigationState) -> list[EventDraft]:
            message = self._message_draft(state, line, ())
            message_id = message.payload.intervention_id
            return [message, EventDraft(actor=self.user, payload=ModeChanged(
                mode=mode, reason=f"switched by the researcher ({message_id})", intervention_id=message_id,
                **limits))]

        try:
            result = self.workspace.ledger.record_with(self.investigation_id, build)
        except (RuleViolation, ValidationError) as exc:
            return f"The mode could not be changed: {str(exc).splitlines()[0]}\n{usage}"
        message_id = result.events[0].payload.intervention_id
        if mode is InvestigationMode.HUMAN_IN_THE_LOOP:
            return (f"{self.investigation_id} is in human-in-the-loop ({message_id}): agent proposals and submissions "
                    "wait for you again.")
        return (f"{self.investigation_id} is in full automation ({message_id}). While you are away, an approver "
                "decides on agent proposals and submissions, and the Evidence Judge assesses new evidence before each "
                f"review of the Chief Scientist, within {limits['max_core_hours']:g} core-hours and "
                f"{dollars(limits['max_usd'])} of model use from now. When that is spent, the investigation switches "
                "back to human-in-the-loop and pauses. Calculations whose inputs need your look, /reject inputs, "
                "/approve close, /cancel and /halt stay yours; "
                "/mode human-in-the-loop takes everything back.")

    # -- the budget -----------------------------------------------------------------------------

    def _spending(self) -> tuple[float, float | None]:
        """Dollars spent and the budget, None for none; before the investigation exists, the Chief Scientist's
        answers so far and the researcher's latest /budget, else the default."""
        if self.investigation_id is not None:
            state = self.workspace.ledger.load(self.investigation_id)
            return state.model_cost, state.budget_usd
        budgets = [entry[2] for entry in self._unrecorded if entry[0] == "budget"]
        return (sum(r.cost_usd or 0 for r in self._unrecorded_runs),
                budgets[-1] if budgets else DEFAULT_BUDGET_USD)

    def _budget_left(self) -> float | None:
        spent, budget = self._spending()
        return None if budget is None else max(0.0, budget - spent)

    def _shared_left(self, own: float) -> float | None:
        """What the budget has left while the Chief Scientist answers, given what it has spent on this answer and
        what the dispatcher's agent runs in progress have spent; None for no budget."""
        left = self.workspace.ledger.load(self.investigation_id).budget_left
        if left is None:
            return None
        live = sum(a.cost_usd or 0 for a in current_activity(self.workspace)
                   if a.investigation_id == self.investigation_id)
        return left - live - own

    @staticmethod
    def _amount(text: str) -> float | None:
        """Dollars as the researcher types them: 15, $15, 7.50, 1,000."""
        match = re.fullmatch(r"\$?(\d+(?:\.\d+)?)", text.replace(",", "").replace("_", ""))
        if match is None:
            return None
        amount = float(match.group(1))
        return amount if amount > 0 else None

    def _budget(self, args: list[str], line: str) -> str:
        usage = "Usage: /budget, /budget <dollars> (e.g. 15 or $40), or /budget off"
        spent, budget = self._spending()
        if not args:
            if budget is None:
                return f"No budget: agents may spend any amount on models. {dollars(spent)} spent so far.\n{usage}"
            return (f"Budget: {dollars(spent)} of {dollars(budget)} spent on models, "
                    f"{dollars(max(0.0, budget - spent))} left.\n{usage}")
        if len(args) != 1:
            return usage
        if args[0].lower() == "off":
            max_usd = None
        else:
            max_usd = self._amount(args[0])
            if max_usd is None:
                return usage
        if self.investigation_id is None:
            self._unrecorded.append(("budget", line, max_usd))
            recorded = "it is recorded when the investigation opens"
        else:
            try:
                recorded = self._record_budget(line, max_usd)
            except RuleViolation as exc:
                return f"The budget could not be changed: {exc}"
        if max_usd is None:
            return f"No budget from now on ({recorded}). {dollars(spent)} spent so far."
        if spent >= max_usd:
            return (f"Budget {dollars(max_usd)} ({recorded}): {dollars(spent)} is already spent, so no agent works and "
                    "the Chief Scientist does not answer until you raise it.")
        return (f"Budget {dollars(max_usd)} ({recorded}): {dollars(spent)} spent, {dollars(max_usd - spent)} left for "
                "all agents together, the Chief Scientist's answers included, at the price of each kind of token.")

    def _record_budget(self, line: str, max_usd: float | None) -> str:
        def build(state: InvestigationState) -> list[EventDraft]:
            message = self._message_draft(state, line, ())
            return [message, EventDraft(actor=self.user, payload=BudgetChanged(
                max_usd=max_usd, intervention_id=message.payload.intervention_id))]

        return self.workspace.ledger.record_with(self.investigation_id, build).events[0].payload.intervention_id

    # -- the agents' web access and materials --------------------------------------------------

    def web_setting(self) -> RetrievalRecord:
        """The setting the agents work under now; before the investigation exists, the researcher's latest /web."""
        if self.investigation_id is not None:
            return self.workspace.ledger.load(self.investigation_id).retrieval
        pending = [entry for entry in self._unrecorded if entry[0] == "web"]
        if not pending:
            return RetrievalRecord(RetrievalPolicy.DISABLED, (), None, 0)
        _, _, policy, domains = pending[-1]
        return RetrievalRecord(policy, domains, None, 0)

    def _web(self, args: list[str], line: str) -> str:
        usage = "Usage: /web, /web on, /web only <domain> [<domain> ...], or /web off"
        if not args:
            return f"Web access for the agents: {web_summary(self.web_setting())}.\n{usage}"
        wanted = args[0].lower()
        if wanted in ("on", "off") and len(args) == 1:
            policy, domains = (RetrievalPolicy.ENABLED if wanted == "on" else RetrievalPolicy.DISABLED), ()
        elif wanted == "only" and len(args) > 1:
            policy = RetrievalPolicy.CURATED_ONLY
            names = [d for arg in args[1:] for d in arg.replace(",", " ").split()]
            domains = tuple(d.removeprefix("https://").removeprefix("http://").rstrip("/") for d in names)
        else:
            return usage
        try:
            check = RetrievalPolicyChanged(policy=policy, allowed_domains=domains, intervention_id="msg-0")
        except ValidationError:
            return f"Give plain domain names such as arxiv.org, not paths or addresses.\n{usage}"
        if self.investigation_id is None:
            self._unrecorded.append(("web", line, policy, check.allowed_domains))
            return (f"Web access will be {web_summary(self.web_setting())} for this investigation; it is recorded "
                    "when the investigation opens, and the Chief Scientist works under it meanwhile.")
        try:
            message_id = self._record_web(line, policy, check.allowed_domains)
        except RuleViolation as exc:
            return f"Web access could not be changed: {exc}"
        if policy is RetrievalPolicy.DISABLED:
            return (f"No agent may use the web from now on ({message_id}). Pages read earlier stay in the agents' "
                    "conversations.")
        return (f"The Chief Scientist and the hypothesis agents may use the web from their next runs: "
                f"{web_summary(self.web_setting())} ({message_id}). What they search for reaches the search "
                "provider, and what they read there is untrusted and is not evidence. The Evidence Judge and the "
                "approver keep to the research record. /web off takes it back.")

    def _record_web(self, line: str, policy: RetrievalPolicy, domains: tuple[str, ...]) -> str:
        def build(state: InvestigationState) -> list[EventDraft]:
            message = self._message_draft(state, line, ())
            return [message, EventDraft(actor=self.user, payload=RetrievalPolicyChanged(
                policy=policy, allowed_domains=domains, intervention_id=message.payload.intervention_id))]

        return self.workspace.ledger.record_with(self.investigation_id, build).events[0].payload.intervention_id

    def _materials(self) -> list[Material]:
        if self.investigation_id is not None:
            return recorded_materials(self.workspace.ledger.load(self.investigation_id))
        return [entry[2] for entry in self._unrecorded if entry[0] == "attach"]

    def _attach(self, args: list[str], line: str) -> str:
        usage = "Usage: /attach <path> <note>, e.g. /attach ~/papers/stm.pdf \"Fig. 3 shows the tilted phase\""
        if not args:
            return usage
        path = Path(args[0]).expanduser().resolve()
        if path == config_dir().resolve() or path.is_relative_to(config_dir().resolve()):
            return "Files in CEDAR's configuration directory, which holds your API keys, cannot be attached."
        if not path.is_file():
            return f"No file {path}.\n{usage}"
        if path.stat().st_size > MAX_ATTACH_BYTES:
            return f"{path} is larger than {MAX_ATTACH_BYTES:,} bytes."
        data = path.read_bytes()
        if not viewable(data) and b"\x00" in data[:8192]:
            return f"{path.name} is neither an image, a PDF nor a text file, so no agent could view it."
        artifact = self.workspace.artifacts.put_bytes(data, label=path.name,
                                                      media_type=mimetypes.guess_type(path.name)[0])
        note = " ".join(args[1:]).strip() or None
        if self.investigation_id is None:
            material = Material(f"mat-{len(self._materials()) + 1}", path.name, artifact, note)
            self._unrecorded.append(("attach", line, material))
            return (f"Attached {material_line(material)}. It is recorded when the investigation opens; the Chief "
                    "Scientist can view it meanwhile.")
        try:
            material_id, message_id = self._record_material(line, path.name, artifact, note)
        except RuleViolation as exc:
            return f"The file could not be attached: {exc}"
        material = next(m for m in self._materials() if m.material_id == material_id)
        return (f"Attached {material_line(material)} ({message_id}). Every agent of the investigation can view it; "
                "the Chief Scientist hears of it with your next message.")

    def _record_material(self, line: str, name: str, artifact: ArtifactRef, note: str | None) -> tuple[str, str]:
        def build(state: InvestigationState) -> list[EventDraft]:
            message = self._message_draft(state, line, ())
            return [message, EventDraft(actor=self.user, payload=MaterialAttached(
                material_id=f"mat-{len(state.materials) + 1}", name=name, artifact=artifact, note=note,
                intervention_id=message.payload.intervention_id))]

        events = self.workspace.ledger.record_with(self.investigation_id, build).events
        return events[1].payload.material_id, events[0].payload.intervention_id

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
            return [message, *halt_drafts(state, self.user, why, message.payload.intervention_id)]

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
        if any(p.type == "mode_changed" for p in done):
            lines.append("  - switched full automation off")
        paused = any(p.type == "investigation_status_changed" for p in done)
        lines.append("  - paused: nothing new starts until you ask the Chief Scientist to resume it" if paused else
                     f"  - left {self.workspace.ledger.load(investigation_id).status.value}, so nothing new starts")
        return "\n".join(lines)

    def _cancel_jobs(self, investigation_id: str, live: tuple[str, ...], reason: str) -> tuple[list[str], list[str]]:
        if not live:
            return [], []
        state = self.workspace.ledger.load(investigation_id)
        try:
            services = services_for(self.workspace, profile_for(state, self.profiles), self.backend)
        except (LookupError, ProfileNotApproved, FileNotFoundError) as exc:
            return [], [f"{', '.join(live)}: no usable compute profile ({exc})"]
        cancelled, failed = [], []
        for calculation_id in live:
            job = state.calculations[calculation_id].scheduler_job_id
            try:
                services.manager_for(state.calculations[calculation_id]).cancel(investigation_id, calculation_id,
                                                                                reason)
                cancelled.append(f"{calculation_id}" + (f" (job {job})" if job else ""))
            except (RuleViolation, SchedulerError, OSError) as exc:
                failed.append(f"{calculation_id}: {exc}")
        return cancelled, failed

    def _workbench(self, args: list[str], line: str) -> str:
        state = self.workspace.ledger.load(self.investigation_id)
        if not args:
            setting = "on" if state.workbench_enabled else "off"
            return (f"The workbench is {setting} for this investigation: agents "
                    + ("may run sandboxed commands in their workspaces (no network, no DFT, writing nowhere else)."
                       if state.workbench_enabled else "run no commands.")
                    + f" CEDAR runs on the {self.host().describe()}.\nUsage: /workbench on, or /workbench off")
        if args not in (["on"], ["off"]):
            return "Usage: /workbench, /workbench on, or /workbench off"
        enabled = args == ["on"]

        def build(current: InvestigationState) -> list[EventDraft]:
            message = self._message_draft(current, line, ())
            return [message, EventDraft(actor=self.user, payload=WorkbenchPolicyChanged(
                enabled=enabled, intervention_id=message.payload.intervention_id))]

        try:
            events = self.workspace.ledger.record_with(self.investigation_id, build).events
        except RuleViolation as exc:
            return f"The workbench could not be changed: {exc}"
        message_id = events[0].payload.intervention_id
        if enabled:
            return f"Agents may run commands in their workspaces again from their next command ({message_id})."
        return (f"No agent runs commands from now on ({message_id}); a command already running finishes. "
                "/workbench on allows them again.")

    def _register(self, args: list[str], line: str) -> str:
        usage = ("Usage: /register <dir> <note> [--output <file>], e.g. /register ~/runs/co-fcc \"my PBE fcc run, "
                 "the one that disagrees\"")
        output = "aims.out"
        if "--output" in args:
            at = args.index("--output")
            if at + 1 >= len(args):
                return usage
            output, args = args[at + 1], args[:at] + args[at + 2:]
        if not args:
            return usage
        directory = Path(args[0]).expanduser().resolve()
        if directory == config_dir().resolve() or directory.is_relative_to(config_dir().resolve()):
            return "Nothing in CEDAR's configuration directory, which holds your API keys, can be registered."
        missing = [name for name in ("control.in", "geometry.in", output) if not (directory / name).is_file()]
        if missing:
            return f"{directory} has no {', '.join(missing)}.\n{usage}"
        note = " ".join(args[1:]).strip() or f"a run the researcher registered from {directory}"
        message_id = self._record_message(line)
        registrar = investigation_registrar(self.workspace, self.profiles, self.investigation_id,
                                            deny=[config_dir()])
        outcome = registrar.register(self.investigation_id, directory, requested_by=self.user, description=note,
                                     output_file=output, intervention_id=message_id)
        if outcome.calculation_id is None:
            return f"Recorded your message ({message_id}), but nothing was registered: {'; '.join(outcome.problems)}"
        state = self.workspace.ledger.load(self.investigation_id)
        summary = calculation_summary(state, state.calculations[outcome.calculation_id])
        if outcome.reused:
            return f"That run is already registered ({message_id}):\n{summary}"
        return (f"Registered ({message_id}); every agent can read and analyse it, and evidence from it is marked as "
                f"computed outside CEDAR:\n{summary}")

    def _cancel(self, given: str, reason: str, line: str) -> str:
        state = self.workspace.ledger.load(self.investigation_id)
        calculation_id, problem = self._find_calculation(state, given)
        if calculation_id is None:
            return problem
        intervention = self._record_message(line, (Authorization(action=PermissionAction.CANCEL,
                                                                 target_ids=(calculation_id,)),))
        services = services_for(self.workspace, profile_for(state, self.profiles), self.backend)
        try:
            services.manager_for(state.calculations[calculation_id]).cancel(self.investigation_id, calculation_id,
                                                                            reason)
        except RuleViolation as exc:
            return f"Recorded your cancellation ({intervention}), but it could not be applied: {exc}"
        return f"Cancelled {calculation_id} ({intervention})."

    def _reject(self, line: str) -> str:
        """Refuse inputs that wait for the researcher, to be reviewed or submitted; the reason goes to the agents."""
        match = re.fullmatch(r"/reject\s+inputs?\s+(\S+)\s+(.*\S)\s*", line, re.DOTALL)
        if match is None:
            return ("Usage: /reject inputs <calc> <reason>, e.g. /reject inputs 1eea8563 use the light species "
                    "defaults unmodified. The reason goes to the agent that asked for the calculation.")
        if self.investigation_id is None:
            return "No investigation is open yet. Describe the discrepancy and the Chief Scientist will open one."
        given, reason = match.groups()
        if len(reason) > 1 and reason[0] == reason[-1] and reason[0] in "\"'":
            reason = reason[1:-1].strip() or reason
        state = self.workspace.ledger.load(self.investigation_id)
        calculation_id, problem = self._find_calculation(state, given)
        if calculation_id is None:
            return problem
        refusal = _not_rejectable(state.calculations[calculation_id])
        if refusal is not None:
            return refusal
        intervention = self._record_message(line, (Authorization(action=PermissionAction.REJECT_INPUTS,
                                                                 target_ids=(calculation_id,)),))
        services = services_for(self.workspace, profile_for(state, self.profiles), self.backend)
        try:
            services.engine.reject(self.investigation_id, calculation_id, reason)
        except RuleViolation as exc:
            return f"Recorded your rejection ({intervention}), but it could not be applied: {exc}"
        state = self.workspace.ledger.load(self.investigation_id)
        agents = [a for a in (state.active_agent(h) for h in state.calculations[calculation_id].hypothesis_ids) if a]
        told = (f"The dispatcher passes your reason to {', '.join(agent_label(a) for a in agents)}." if agents else
                "No active agent works on it; your reason stays in the record.")
        return f"Rejected the inputs of {calculation_id} ({intervention}). {told}"

    def _runs_root(self, state: InvestigationState) -> Path | None:
        """Where the investigation's run directories are, so the paths shown can be opened; None without an approved
        compute profile."""
        try:
            return Path(profile_for(state, self.profiles).runs_root)
        except (LookupError, ProfileNotApproved, FileNotFoundError):
            return None

    # -- the Chief Scientist ------------------------------------------------------------------

    def _ask_chief(self, text: str, observer: Callable[[ToolCall, ToolOutcome], None] | None = None) -> str:
        left = self._budget_left()
        if left == 0:
            return ("Not sent: the investigation's budget is spent. Raise it with /budget <dollars>, or remove it with "
                    "/budget off.")
        limits = self.limits
        if left is not None:
            limits = replace(limits, max_cost_usd=left if limits.max_cost_usd is None else min(left, limits.max_cost_usd))
        attached = self._newly_attached()
        if self.investigation_id is not None:
            fresh = self._fresh_start()
            message_id = self._record_message(text)
            instruction = (fresh + self._reviews_since_last_turn() + attached
                           + f"[researcher, recorded as {message_id}] {text}")
        else:
            self._unrecorded.append(("message", text))
            instruction = attached + f"[researcher] {text}"
        context = ChiefContext(ledger=self.workspace.ledger, profiles=self.profiles, actor=CHIEF,
                               registry_for=lambda name: registry_for(self.profiles.load_approved(name)),
                               investigation_id=self.investigation_id, on_open=self._opened, clock=self.clock,
                               dispatcher=self._dispatcher_status,
                               registrar=lambda inv: investigation_registrar(self.workspace, self.profiles, inv,
                                                                             deny=[config_dir()]),
                               where=self._where, running=self._agents_at_work, warn=self._warnings.append)
        settings = self.clients.settings("chief_scientist")
        runner = AgentRunner(self.clients.client("chief_scientist"), settings, self.workspace.transcripts,
                             secrets=self.clients.secrets())
        tools = (chief_tools(context) + file_tools(agent_access([config_dir()]))
                 + material_tools(self._materials, self.workspace.artifacts))
        result = runner.run(self.chief_session, system=CHIEF_SCIENTIST, tools=tools, instruction=instruction,
                            limits=limits, observer=observer,
                            web=web_access(self.web_setting(), "chief_scientist"),
                            spend_left=self._shared_left if self.investigation_id is not None else None)
        self._context_tokens = result.context_tokens
        if self.investigation_id is None:
            self._unrecorded_runs.append(result)
        else:
            for earlier in [*self._unrecorded_runs, result]:
                record_agent_run(self.workspace.ledger, self.investigation_id, CHIEF, session_id=self.chief_session,
                                 role="chief_scientist", settings=settings, result=earlier)
            self._unrecorded_runs.clear()
        reply = "".join(f"[warning] {w}\n\n" for w in self._warnings) + (result.final_text or "")
        self._warnings.clear()
        if result.outcome != "finished":
            reply += f"\n[the Chief Scientist stopped: {result.outcome}{': ' + result.error if result.error else ''}]"
        return reply.strip()

    def _agents_at_work(self) -> list[str]:
        """The hypothesis agents of this investigation that the dispatcher is running now."""
        if self.investigation_id is None:
            return []
        state = self.workspace.ledger.load(self.investigation_id)
        return [f"{state.agents[a.agent_id].tag} ({a.agent_id})" for a in current_activity(self.workspace)
                if a.investigation_id == self.investigation_id and a.role == "hypothesis_agent"
                and a.agent_id in state.agents]

    def _fresh_start(self) -> str:
        """If the conversation has grown past ``fresh_after_tokens``, continue in a new one and return the briefing
        that opens it; otherwise nothing."""
        size = self._context_tokens
        if size is None:
            size = estimated_tokens(self.workspace.transcripts.load(self.chief_session))
        if size < self.fresh_after_tokens:
            return ""
        state = self.workspace.ledger.load(self.investigation_id)
        earlier = list(dict.fromkeys([*sessions_of(state, CHIEF.id, reviews=False), self.chief_session]))
        self.chief_session = session_name(self.investigation_id, "chief", str(len(earlier) + 1))
        self._context_tokens = None
        return chat_briefing(state, transcripts=self.workspace.transcripts.root,
                             notebook=self.workspace.notebook_dir(self.investigation_id), sessions=earlier) + "\n\n"

    def _where(self) -> str:
        """Whether calculations can run locally: they start where the dispatcher runs, which the chat starts here."""
        here = self.host()
        elsewhere = dispatcher_host(self.workspace)
        if dispatcher_running(self.workspace) and elsewhere is not None and elsewhere != here.hostname:
            return (f"local calculations start on {elsewhere}, where the dispatcher runs, not on the "
                    f"{here.describe()} where this chat runs; ask the researcher to run CEDAR on one node")
        return local_dft(here)

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

    def _newly_attached(self) -> str:
        """The materials the Chief Scientist has not been told of, as notes for its next instruction."""
        fresh = [m for m in self._materials() if m.material_id not in self._announced_materials]
        self._announced_materials.update(m.material_id for m in fresh)
        return "".join(f"[record: the researcher attached {material_line(m)}]\n" for m in fresh)

    def _opened(self, investigation_id: str) -> list[str]:
        """Record what the researcher said and set before the investigation existed, in order, as soon as it does;
        returns the ids of their messages."""
        self.investigation_id = investigation_id
        recorded = []
        for kind, *entry in self._unrecorded:
            if kind == "message":
                recorded.append(self._record_message(entry[0]))
            elif kind == "web":
                self._record_web(*entry)
            elif kind == "budget":
                self._record_budget(*entry)
            else:
                line, material = entry
                self._record_material(line, material.name, material.artifact, material.note)
        self._unrecorded.clear()
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

    def history(self) -> list[Said | Notice]:
        """The conversation before this session, rebuilt from the record in the order it was shown: the researcher's
        messages and the commands that changed the record, the Chief Scientist's answers, and the notices that came
        meanwhile. What other commands answered is not recorded; requests for approval are shown by ``notices``
        while they wait."""
        if self.investigation_id is None:
            return []
        state = self.workspace.ledger.load(self.investigation_id)
        until = self._history_until
        entries: list[tuple[int, Said | Notice]] = [
            (i.recorded_at, Said(i.text, researcher=True)) for i in state.interventions.values()
            if i.recorded_at <= until]
        entries += [(r.recorded_at, _answer(r)) for r in state.agent_runs
                    if r.role == "chief_scientist" and not _is_review(r) and r.recorded_at <= until]
        return _in_turns(entries + _recorded_notices(state, 0, until))

    def notices(self) -> list[Notice]:
        """What arrived since the researcher last looked: agent reports, the Chief Scientist's reviews and the next
        steps it recommended, agents that started, and each request for approval the first time it is seen."""
        if self.investigation_id is None:
            return []
        state = self.workspace.ledger.load(self.investigation_id)
        notices = []
        for agent in state.agents.values():  # the researcher's own /approve agents already said who started
            if agent.agent_id not in self._shown_starts and agent.authorization_id is None:
                notices.append(Notice(f"on {agent.hypothesis_id}, after the review window passed with no reply",
                                      f"{agent.tag} {agent.agent_id}", "started"))
            self._shown_starts.add(agent.agent_id)
        notices += [notice for _, notice in _recorded_notices(state, self._seen_runs, state.version)]
        self._seen_runs = state.version
        for command, note in approval_requests(state, brief=True):
            if f"{command} {note}" not in self._shown_approvals:
                self._shown_approvals.add(f"{command} {note}")
                notices.append(Notice(note, label="waiting for you", attention=True, command=command))
        self._shown_reviews.update(awaiting_review(state))
        return notices
