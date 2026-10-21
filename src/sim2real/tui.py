"""The full-screen terminal interface: a live panel above the Chief Scientist conversation.

The interface is only a client. It reads the record and the dispatcher's status files,
and talks to the Chief Scientist through ``ChatSession``. Closing it interrupts nothing,
and reopening it shows the same investigation.

Top to bottom: the panel, always shown: the logo, set across its top border, with the
tagline and the models in use beside it; the status lines; which sub-agents are working;
and a table with one row per hypothesis, keeping lifecycle, agent role and activity,
scheduler state, evidence quality and assessment in separate columns. Then the scrolling
conversation and the input line, above which a menu of matching commands opens when a
line starts with "/". The Chief Scientist answers in a
background thread, so the panel keeps updating and commands keep working while it thinks.
Messages sent meanwhile wait their turn. The Chief Scientist's background reviews of the
agents' reports appear in the conversation as its messages.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Collapsible, DataTable, Footer, Input, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from sim2real.chat import COMMANDS, HELP, ChatSession, Command, Notice, command_lines
from sim2real.dashboard import (
    SwarmRow,
    dispatcher_status,
    model_line,
    status_line,
    subagent_status,
    swarm_rows,
    work_summary,
)
from sim2real.dispatch import current_activity, heartbeat_age
from sim2real.notebook import round_file, update_notebook
from sim2real.state.projection import InvestigationState, replay

TUI_COMMANDS = (
    Command("/notebook", "open the research notebook's index"),
    Command("/round", "open this round's notebook page"),
    Command("/timeline", "open the timeline of every event"),
    Command("/panel", "hide or show the panel's table of hypotheses"),
)
MENU_COMMANDS = COMMANDS + TUI_COMMANDS

TUI_HELP = f"""In this window:
{command_lines(TUI_COMMANDS)}
A line starting with / lists the commands that match it: ↑/↓ choose, Tab or Enter completes, Esc closes
the list; Enter on a complete command runs it. On narrow terminals the panel's table starts hidden.
Tab moves to the panel; Enter on a hypothesis opens its page. Esc closes a page, Backspace goes back."""


def suggestions(text: str) -> list[Command]:
    """The commands whose typed part begins with ``text``; none unless ``text`` starts with "/"."""
    if not text.startswith("/"):
        return []
    typed = text.lower()
    return [c for c in MENU_COMMANDS if c.stem.lower().startswith(typed)]


class Composer(Input):
    """The input line. While the command menu is open, ↑/↓ move through it, Tab completes and Esc closes it;
    otherwise these keys do what they always do."""

    BINDINGS = [Binding("up", "menu_move(-1)", show=False), Binding("down", "menu_move(1)", show=False),
                Binding("tab", "menu_complete", show=False), Binding("escape", "menu_close", show=False)]

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action.startswith("menu_"):
            return self.app.menu_open
        return super().check_action(action, parameters)

    def action_menu_move(self, step: int) -> None:
        self.app.move_menu(step)

    def action_menu_complete(self) -> None:
        self.app.complete_command()

    def action_menu_close(self) -> None:
        self.app.close_menu()

NARROW_COLUMNS = 110
"""Below this terminal width the panel's table starts hidden; the status lines still summarize it."""

TAGLINE = "An Autonomous Scientific Agent Swarm for Reconciling Simulation and Experiment"
_LOGO_ANSI = (
    "\x1b[0;96;46m▓░\x1b[0;36m▀\x1b[0;96;46m▓\x1b[0;37m \x1b[0;96;46m▓▒\x1b[0;36m▀▀\x1b[0;37m \x1b[0;96;46m▓▒\x1b[0;36m▀▄"
    "\x1b[0;37m \x1b[0;96;46m▓▒\x1b[0;36m▀\x1b[0;96;46m▓\x1b[0;37m \x1b[0;96;46m▓█\x1b[0;36m▀\x1b[0;96;46m▓\x1b[0m",
    "\x1b[0;96;46m▒▒\x1b[0;37m   \x1b[0;96;46m▒▓\x1b[0;36m▀▀\x1b[0;37m \x1b[0;96;46m▒▓\x1b[0;97m \x1b[0;96;46m░"
    "\x1b[0;37m \x1b[0;96;46m▒░\x1b[0;36m▄\x1b[0;96;46m▒\x1b[0;37m \x1b[0;96;46m▒▓\x1b[0;36m▄▀\x1b[0m",
    "\x1b[0;96;46m░▓\x1b[0;36m▄\x1b[0;96;46m░\x1b[0;37m \x1b[0;96;46m░ \x1b[0;36m▄▄\x1b[0;37m \x1b[0;96;46m░ \x1b[0;36m▄▀"
    "\x1b[0;37m \x1b[0;96;46m░ \x1b[0;97m \x1b[0;96;46m░\x1b[0;37m \x1b[0;96;46m░▒\x1b[0;97m \x1b[0;96;46m▒\x1b[0m",
)
"""The logo, three rows of shaded blocks in two cyans, as ANSI text."""
LOGO = tuple(Text.from_ansi(row) for row in _LOGO_ANSI)
LOGO_WIDTH = max(row.cell_len for row in LOGO)


class PanelTop(Widget):
    """The logo's upper two rows, with the panel's top border running through the second, a space either side."""

    DEFAULT_CSS = "PanelTop { height: 2; }"

    def render(self) -> Text:
        rest = max(0, self.size.width - LOGO_WIDTH - 5)
        return Text("\n").join([Text("   ") + LOGO[0], Text("╭─ ") + LOGO[1] + Text(" " + "─" * rest + "╮")])

FILE_WINDOW_LINES = 20_000
"""Longer files are shown as a window of this many lines around the cited line."""

_PAGES = {"/notebook": "index.md", "/timeline": "timeline.md"}


class FileScreen(Screen):
    """A stored file, read-only, opened at a cited line."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Close")]

    def __init__(self, path: Path, line: int | None = None) -> None:
        super().__init__()
        self.path, self.line = path, line

    def compose(self) -> ComposeResult:
        lines = self.path.read_text(errors="replace").splitlines()
        start = 1
        if len(lines) > FILE_WINDOW_LINES:
            start = max(1, (self.line or 1) - FILE_WINDOW_LINES // 2)
            lines = lines[start - 1: start - 1 + FILE_WINDOW_LINES]
        self.start = start
        shown = f" (lines {start}–{start + len(lines) - 1})" if start > 1 or len(lines) == FILE_WINDOW_LINES else ""
        yield Static(Text(f"{self.path}{shown}"), classes="title")
        yield TextArea("\n".join(lines), read_only=True, show_line_numbers=True, line_number_start=start,
                       soft_wrap=False)
        yield Footer()

    def on_mount(self) -> None:
        area = self.query_one(TextArea)
        area.focus()
        if self.line:
            area.move_cursor((self.line - self.start, 0), center=True)


class NotebookScreen(Screen):
    """Pages of the research notebook, with links between them and to the stored files they cite."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Close"), Binding("backspace", "back", "Back")]

    def __init__(self, directory: Path, page: str, artifacts: Path) -> None:
        super().__init__()
        self.directory, self.artifacts = directory.resolve(), artifacts.resolve()
        self.history = [self.directory / page]

    def compose(self) -> ComposeResult:
        yield Static(classes="title")
        with VerticalScroll(id="page"):
            yield Markdown(open_links=False)
        yield Footer()

    async def on_mount(self) -> None:
        await self._show(self.history[-1])

    async def _show(self, path: Path) -> None:
        self.query_one(".title", Static).update(Text(str(path)))
        await self.query_one(Markdown).update(path.read_text())
        self.query_one("#page").scroll_home(animate=False)
        self.query_one(Markdown).focus()

    async def on_markdown_link_clicked(self, message: Markdown.LinkClicked) -> None:
        message.stop()
        location, _, anchor = message.href.partition("#")
        target = (self.history[-1].parent / location).resolve()
        if target.suffix == ".md" and target.is_relative_to(self.directory) and target.is_file():
            self.history.append(target)
            await self._show(target)
        elif target.is_relative_to(self.artifacts) and target.is_file():
            line = int(anchor[1:]) if anchor[:1] == "L" and anchor[1:].isdigit() else None
            self.app.push_screen(FileScreen(target, line))
        else:
            self.notify(f"Not a notebook page or stored file: {message.href}", severity="warning", markup=False)

    async def action_back(self) -> None:
        if len(self.history) > 1:
            self.history.pop()
            await self._show(self.history[-1])


@dataclass
class _Snapshot:
    status: str
    summary: str
    alarm: bool
    subagents: str
    models: str
    rows: list[SwarmRow] = field(default_factory=list)
    notices: list[Notice] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    round: int = 0


class Sim2RealApp(App):
    TITLE = "Sim2Real Scientist"
    CSS = """
    #panel { height: auto; }
    PanelTop { color: $accent; }
    #panel-body { height: auto; border: round $accent; border-top: none; padding: 0 1; }
    #masthead { height: auto; margin-bottom: 1; }
    #logo-bottom { width: 27; }
    #about { width: 1fr; height: auto; }
    #tagline { text-style: bold; }
    #models { color: $text-muted; }
    #status { height: auto; }
    #status.alarm { color: $warning; text-style: bold; }
    #subagents { height: auto; color: $success; }
    #swarm { height: auto; max-height: 40vh; margin-top: 1; }
    #chat { height: 1fr; padding: 0 1; }
    #working { height: auto; padding: 0 1; color: $text-muted; display: none; }
    #commands { height: auto; max-height: 12; margin: 0 1; display: none; }
    #commands.open { display: block; }
    #working.busy { display: block; }
    .user { margin-top: 1; color: $accent; text-style: bold; }
    .speaker { margin-top: 1; color: $success; text-style: bold; }
    .notice { color: $text-muted; }
    .command { padding-left: 2; }
    .error { color: $error; }
    .title { height: auto; padding: 0 1; background: $boost; }
    Collapsible { margin: 0; padding: 0; border: none; }
    """
    ENABLE_COMMAND_PALETTE = False
    """Commands in the conversation are typed, starting with "/"; there are no shortcut keys."""
    COLUMNS = ("Hypothesis", "Lifecycle", "Agent", "Jobs (scheduler)", "Evidence (validator)", "Assessment (judge)")

    def __init__(self, session: ChatSession, *, refresh_seconds: float = 2.0,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc), started: str | None = None) -> None:
        super().__init__()
        self.session = session
        self._started = started
        self.workspace = session.workspace
        self.refresh_seconds = refresh_seconds
        self.clock = clock
        self.chief_busy = False
        self.waiting: deque[str] = deque()
        self._refreshing = False
        self._version: int | None = None
        self._events: list = []
        self._state: InvestigationState | None = None
        self._rows: list[SwarmRow] | None = None
        self._seen_results: set[tuple[str, ...]] | None = None
        self._round = 0
        self._working_text = ""
        self._panel_choice: bool | None = None
        """Whether the researcher chose to show the panel; None follows the terminal's width."""
        self._menu: list[Command] = []
        self._menu_closed = False
        """The researcher closed the menu with Esc; it stays closed until the line changes."""

    # -- layout -------------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="panel"):
            yield PanelTop()
            with Vertical(id="panel-body"):
                with Horizontal(id="masthead"):
                    yield Static(Text(" ") + LOGO[2], id="logo-bottom")
                    with Vertical(id="about"):
                        yield Static(TAGLINE, id="tagline")
                        yield Static(id="models")
                yield Static(id="status")
                yield Static("No working sub-agents", id="subagents")
                yield DataTable(id="swarm", cursor_type="row", zebra_stripes=True)
        yield VerticalScroll(id="chat")
        yield Static(id="working")
        menu = OptionList(id="commands")
        menu.can_focus = False
        yield menu
        yield Composer(placeholder="Message the Chief Scientist, or a /command (/help)", id="composer")

    def on_mount(self) -> None:
        self.query_one(DataTable).add_columns(*self.COLUMNS)
        if self.session.investigation_id:
            self.say(f"Investigation {self.session.investigation_id}. Type /help for commands.", "notice")
        else:
            self.say("New conversation with the Chief Scientist. Describe the simulation-experiment discrepancy you "
                     "want to investigate. Type /help for commands.", "notice")
        if self._started:
            self.say(self._started, "notice")
        self.query_one(Input).focus()
        self._fit_panel()
        self.refresh_state()
        self.set_interval(self.refresh_seconds, self.refresh_state)

    def on_resize(self) -> None:
        self._fit_panel()

    def _fit_panel(self) -> None:
        show = self._panel_choice if self._panel_choice is not None else self.size.width >= NARROW_COLUMNS
        self.query_one(DataTable).display = show and bool(self._rows)

    def say(self, text: str, kind: str) -> None:
        """Add a line to the conversation, shown exactly as written."""
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(Static(Text(text), classes=kind))
        chat.scroll_end(animate=False)

    # -- input --------------------------------------------------------------------------------

    # -- the command menu ---------------------------------------------------------------------

    @property
    def menu_open(self) -> bool:
        return self.query_one("#commands", OptionList).has_class("open")

    def on_input_changed(self, event: Input.Changed) -> None:
        self._menu_closed = False
        self._show_menu(event.value)

    def _show_menu(self, text: str) -> None:
        menu = self.query_one("#commands", OptionList)
        matches = [] if self._menu_closed else suggestions(text)
        if matches != self._menu:
            self._menu = matches
            width = max((len(c.usage) for c in matches), default=0)
            menu.clear_options()
            menu.add_options(Option(Text.assemble((c.usage.ljust(width + 2), "bold"), (c.description, "dim")))
                             for c in matches)
            if matches:
                menu.highlighted = 0
        menu.set_class(bool(matches), "open")

    def move_menu(self, step: int) -> None:
        menu = self.query_one("#commands", OptionList)
        menu.highlighted = ((menu.highlighted or 0) + step) % len(self._menu)

    def _chosen(self) -> Command | None:
        index = self.query_one("#commands", OptionList).highlighted
        return self._menu[index] if self.menu_open and index is not None else None

    def complete_command(self, command: Command | None = None) -> None:
        command = command or self._chosen()
        if command is not None:
            composer = self.query_one(Composer)
            composer.value = command.stem
            composer.cursor_position = len(command.stem)
            composer.focus()

    def close_menu(self) -> None:
        self._menu_closed = True
        self._show_menu(self.query_one(Composer).value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.complete_command(self._menu[event.option_index])

    # -- input --------------------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        chosen = self._chosen()
        if chosen is not None and event.value.strip() != chosen.stem.strip():
            self.complete_command(chosen)  # Enter on a partial command completes it; Enter again runs it
            return
        line = event.value.strip()
        event.input.clear()
        if not line:
            return
        if line in ("/quit", "/exit"):
            self.exit()
        elif line == "/help":
            self.say(f"{HELP}\n\n{TUI_HELP}", "command")
        elif line in _PAGES:
            self.action_page(_PAGES[line])
        elif line == "/round":
            self.action_round_page()
        elif line == "/panel":
            self.action_toggle_panel()
        elif line.startswith("/"):
            self.say(f"› {line}", "user")
            self.run_worker(partial(self._command, line), thread=True, group="commands", exit_on_error=False)
        else:
            self.say(f"you › {line}", "user")
            if self.chief_busy:
                self.waiting.append(line)
                self.say("(waiting until the Chief Scientist has answered the previous message)", "notice")
                self._working(self._working_text)
            else:
                self._ask(line)

    def _command(self, line: str) -> None:
        try:
            reply = self.session.handle(line)
        except Exception as exc:  # show the failure; the record is unchanged by a failed command
            self.call_from_thread(self.say, f"{type(exc).__name__}: {exc}", "error")
            return
        self.call_from_thread(self.say, reply, "command")
        if line.split()[0] == "/report" and self.session.investigation_id:
            self.call_from_thread(self.action_round_page)
        self.call_from_thread(self.refresh_state)

    # -- the Chief Scientist ----------------------------------------------------------------------

    def _ask(self, line: str) -> None:
        self.chief_busy = True
        steps = Static(Text(""), classes="notice")
        box = Collapsible(steps, title="Chief Scientist is working…", collapsed=False)
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(box)
        chat.scroll_end(animate=False)
        self._working("The Chief Scientist is working…")
        self.run_worker(partial(self._chief, line, box, steps), thread=True, group="chief", exit_on_error=False)

    def _chief(self, line: str, box: Collapsible, steps: Static) -> None:
        done: list[str] = []

        def observer(call, outcome) -> None:
            mark = "✗" if outcome.is_error else "✓"
            first = outcome.content.strip().splitlines()[0][:200] if outcome.is_error and outcome.content else ""
            done.append(f"{mark} {call.name}" + (f": {first}" if first else ""))
            self.call_from_thread(steps.update, Text("\n".join(done)))
            self.call_from_thread(self._working, f"The Chief Scientist is working… {len(done)} steps")

        try:
            reply, error = self.session.handle(line, observer=observer), None
        except Exception as exc:  # show it and keep the conversation usable; nothing half-recorded remains
            reply, error = None, f"{type(exc).__name__}: {exc}"
        self.call_from_thread(self._answered, box, len(done), reply, error)

    def _answered(self, box: Collapsible, steps: int, reply: str | None, error: str | None) -> None:
        if steps:
            box.title = f"Chief Scientist: {steps} step{'s' if steps > 1 else ''}"
            box.collapsed = True
        else:
            box.remove()
        chat = self.query_one("#chat", VerticalScroll)
        if error is not None:
            chat.mount(Static(Text(error), classes="error"))
        else:
            chat.mount(Static("Chief Scientist ›", classes="speaker"))
            chat.mount(Markdown(reply or "(no reply)", open_links=False))
        chat.scroll_end(animate=False)
        self.chief_busy = False
        self._working("")
        self.refresh_state()
        if self.waiting:
            self._ask(self.waiting.popleft())

    def _working(self, text: str) -> None:
        self._working_text = text
        working = self.query_one("#working", Static)
        queued = f" ({len(self.waiting)} message{'s' if len(self.waiting) > 1 else ''} waiting)" if self.waiting else ""
        working.update(Text(text + queued if text else ""))
        working.set_class(bool(text), "busy")

    # -- the live panel ---------------------------------------------------------------------------

    def refresh_state(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        self.run_worker(self._snapshot_thread, thread=True, group="refresh", exit_on_error=False)

    def _snapshot_thread(self) -> None:
        try:
            snapshot, error = self._snapshot(), None
        except Exception as exc:  # show it in the status line; the next refresh tries again
            snapshot, error = None, f"Could not read the record: {type(exc).__name__}: {exc}"
        self.call_from_thread(self._apply, snapshot, error)

    def _snapshot(self) -> _Snapshot:
        now = self.clock()
        started = self.session.ensure_dispatcher()  # restarts one that stopped while work remains
        if started:
            self.call_from_thread(self.say, started, "notice")
        activities = current_activity(self.workspace)
        automatic = self.session.auto_dispatch
        dispatcher, alarm = dispatcher_status(heartbeat_age(self.workspace, now), activities, now, automatic)
        running = not alarm and not (automatic and dispatcher.startswith("no dispatcher"))
        investigation_id = self.session.investigation_id
        settings = self.session.clients.settings
        if investigation_id is None:
            return _Snapshot("No investigation yet: describe the discrepancy to the Chief Scientist.", dispatcher,
                             alarm, subagent_status(None, activities, now, running, automatic),
                             model_line(settings, None))
        version = self.workspace.store.current_version(investigation_id)
        changed = version != self._version
        if changed:
            self._events = self.workspace.store.read_stream(investigation_id)
            self._state = replay(self._events)
            self._version = version
        state = self._state
        return _Snapshot(status_line(state, now), f"{work_summary(state)} · {dispatcher}", alarm,
                         subagent_status(state, activities, now, running, automatic), model_line(settings, state),
                         swarm_rows(state, self._events, activities, now),
                         self.session.notices() if changed else [], self._new_results(state) if changed else [],
                         state.round)

    def _new_results(self, state: InvestigationState) -> list[str]:
        """New evidence and assessments since the last look; nothing on the first look."""
        results = {}
        for record in state.evidence.values():
            ev = record.evidence
            value = f" = {ev.value.value!r} {ev.value.unit}" if ev.value else ""
            results[("evidence", record.evidence_id)] = (f"Evidence {record.evidence_id}: {ev.quality.value}, "
                                                        f"{ev.observable.name}{value}")
        for hyp in state.hypotheses.values():
            for entry in hyp.assessments:
                results[("assessment", hyp.hypothesis_id, str(entry.recorded_at))] = (
                    f"{hyp.hypothesis_id} assessed {entry.assessment.value} ({entry.completeness.value.lower()})")
        first_look = self._seen_results is None
        fresh = [text for key, text in results.items() if not first_look and key not in self._seen_results]
        self._seen_results = set(results)
        return fresh

    def _apply(self, snapshot: _Snapshot | None, error: str | None) -> None:
        self._refreshing = False
        status = self.query_one("#status", Static)
        if snapshot is None:
            status.update(Text(error))
            status.set_class(True, "alarm")
            return
        status.update(Text(f"{snapshot.status}\n{snapshot.summary}"))
        status.set_class(snapshot.alarm, "alarm")
        self.query_one("#subagents", Static).update(Text(snapshot.subagents))
        self.query_one("#models", Static).update(Text(snapshot.models))
        self._round = snapshot.round
        if snapshot.rows != self._rows:
            self._rows = snapshot.rows
            table = self.query_one(DataTable)
            cursor = table.cursor_row
            table.clear()
            for row in snapshot.rows:
                lifecycle = row.lifecycle + (f", {row.priority.replace('_', ' ')}" if row.priority != "normal" else "")
                table.add_row(row.hypothesis_id, lifecycle, row.agent, row.jobs, row.evidence, row.assessment,
                              key=row.hypothesis_id)
            if snapshot.rows:
                table.move_cursor(row=min(cursor, len(snapshot.rows) - 1))
            self._fit_panel()
        for notice in snapshot.notices:
            if notice.review:
                chat = self.query_one("#chat", VerticalScroll)
                chat.mount(Static("Chief Scientist (review of the agents' reports) ›", classes="speaker"))
                chat.mount(Markdown(notice.text, open_links=False))
                chat.scroll_end(animate=False)
            else:
                self.say(notice.text, "notice")
        for alert in snapshot.alerts:
            self.notify(alert, title="New result", timeout=10, markup=False)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_page(f"hypotheses/{event.row_key.value}.md")

    # -- actions ----------------------------------------------------------------------------------

    def action_toggle_panel(self) -> None:
        shown = self._panel_choice if self._panel_choice is not None else self.size.width >= NARROW_COLUMNS
        self._panel_choice = not shown
        self._fit_panel()

    def action_round_page(self) -> None:
        self.action_page(round_file(self._round))

    def action_page(self, page: str) -> None:
        investigation_id = self.session.investigation_id
        if investigation_id is None:
            self.notify("No investigation is open yet.", severity="warning")
            return
        self.run_worker(partial(self._open_page, investigation_id, page), thread=True, group="notebook",
                        exit_on_error=False)

    def _open_page(self, investigation_id: str, page: str) -> None:
        try:
            update = update_notebook(self.workspace, investigation_id)
        except Exception as exc:  # the notebook is a view; report the failure and keep the conversation
            self.call_from_thread(self.say, f"Could not update the notebook: {type(exc).__name__}: {exc}", "error")
            return
        if not (update.directory / page).is_file():
            self.call_from_thread(self.notify, f"No page {page} yet.", severity="warning")
            return
        self.call_from_thread(self.push_screen, NotebookScreen(update.directory, page, self.workspace.artifacts.root))


def run_tui(session: ChatSession, started: str | None = None) -> None:
    """Run the interface; ``started`` says that a dispatcher was started for it in the background."""
    Sim2RealApp(session, started=started).run()
