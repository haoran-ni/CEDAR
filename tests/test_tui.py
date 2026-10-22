"""The full-screen interface, driven headless by Textual's test pilot.

ScriptedModel and FakeScheduler TEST DOUBLES on a SYNTHETIC site; no model or job runs.
"""

import asyncio
import re
import threading
from datetime import timedelta

import pytest
from factories import add_hypothesis
from sites import aims_output
from test_orchestration import (
    ENERGY,
    HYP,
    INV,
    NOW,
    ROLE,
    agent_reads,
    agent_requests,
    latest_calc,
    make_system,
    open_and_delegate,
    open_and_start,
    review_inputs,
    reviewer_decides,
)
from textual.color import Color
from textual.widgets import Collapsible, DataTable, Footer, Markdown, OptionList, Static, TextArea

from cedar.chat import ChatSession
from cedar.contracts import Assessment, AssessmentRecorded, Completeness, EventDraft, SchedulerState
from cedar.dashboard import short_id
from cedar.dispatch import JUDGE
from cedar.models import TextPart
from cedar.models.scripted import reply
from cedar.swarm import clock_time, review_deadline
from cedar.tui import (
    AGENT_COLORS,
    LOGO,
    MENU_COMMANDS,
    TAGLINE,
    Composer,
    FileScreen,
    NotebookScreen,
    PanelTop,
    CedarApp,
    ThinScrollBar,
)


@pytest.fixture
def system(tmp_path, site):
    return make_system(tmp_path, site)


def drive(app, script, size=(180, 50)) -> None:
    async def main():
        async with app.run_test(size=size) as pilot:
            await script(pilot)

    asyncio.run(main())


async def settle(pilot) -> None:
    """Wait until background threads have finished and their results are on screen."""
    app = pilot.app
    for _ in range(200):
        await pilot.pause()
        if not app.chief_busy and not app._refreshing and not any(w.is_running for w in app.workers):
            await pilot.pause()
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the interface did not settle")


async def send(pilot, text: str) -> None:
    composer = pilot.app.query_one(Composer)
    composer.focus()
    composer.value = text
    await pilot.press("enter")
    await settle(pilot)


async def refresh(pilot) -> None:
    pilot.app.refresh_state()
    await settle(pilot)


def conversation(app) -> str:
    parts = []
    for widget in app.query_one("#chat").query("*"):
        if isinstance(widget, Markdown):
            parts.append(widget.source)
        elif isinstance(widget, Static):
            parts.append(str(widget.content))
    return "\n".join(parts)


def panel(app) -> list[list[str]]:
    """The hypotheses table."""
    table = app.query_one("#swarm", DataTable)
    return [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]


def agents(app) -> dict[str, list[str]]:
    """The agents table, by agent: role, status, for, task."""
    table = app.query_one("#agents", DataTable)
    rows = [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]
    return {row[0]: row[1:] for row in rows}


def status(app) -> str:
    return str(app.query_one("#status", Static).content)


def awaiting(app) -> str:
    """The panel's highlighted line of what waits for the researcher."""
    return str(app.query_one("#awaiting", Static).content)


def state(system):
    return system["workspace"].ledger.load(INV)


def proposal_id(system) -> str:
    return state(system).pending_proposal.proposal_id


def validated_run(system) -> str:
    """Open, start the agent, approve and finish one calculation without the interface; returns its id."""
    chat, clients, dispatcher, scheduler = system["chat"], system["clients"], system["dispatcher"], system["scheduler"]
    open_and_start(system)
    clients.add("hypothesis_agent", *agent_requests(system))
    dispatcher.tick()
    review_inputs(system)
    calc = latest_calc(system["workspace"])
    chat.handle(f"/approve submit {calc}")
    output = aims_output(energy=ENERGY)
    scheduler.on_submit = lambda job, workdir: (workdir / "aims.out").write_text(output)
    dispatcher.tick()
    scheduler.set_state(next(iter(scheduler.jobs)), SchedulerState.COMPLETED)
    clients.add("hypothesis_agent", *agent_reads(system, output))
    dispatcher.tick()
    return calc


def test_conversation_live_panel_and_approval(system):
    clients, dispatcher = system["clients"], system["dispatcher"]
    clients.add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="Opened and **delegated**.")))
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        assert "No investigation yet" in status(app) and "no dispatcher has run" in status(app)
        await send(pilot, "PBE puts CO in the fcc hollow on Cu(111); experiments see atop.")
        text = conversation(app)
        assert "you › PBE puts CO in the fcc hollow" in text and "Opened and **delegated**." in text
        steps = app.query_one(Collapsible)
        assert steps.collapsed and re.fullmatch(r"Chief Scientist: \d+ steps", steps.title)

        await refresh(pilot)
        (row,) = panel(app)
        assert row[0] == HYP and row[1] == "ACTIVE"
        assert agents(app)[f"agent-{HYP}"] == [ROLE, "proposed · starts in 10 min", "0 s",
                                               "Compute a PBE baseline total energy for CO atop."]
        assert status(app).startswith(f"{INV} · INVESTIGATING · round 1 · plan v1 · 0 agents")
        assert "1 agent proposed: start in 10 min unless you reply (/agents)" in status(app)
        assert awaiting(app) == "⚑ Waiting for you: proposed agents: /approve agents · /pending for details"
        (block,) = app.query_one("#chat").query(".approvals")
        assert str(block.content).splitlines() == [
            "⚑ Waiting for your approval",
            f"  /approve agents  1 hypothesis agent proposed ({proposal_id(system)}); they start at "
            f"{clock_time(review_deadline(state(system), state(system).pending_proposal))} unless you reply",
            "  /pending shows the details"]
        assert [n.title for n in app._notifications] == ["Waiting for your approval"]

        await send(pilot, "/agents")
        assert f"{HYP}: {ROLE}" in conversation(app) and "first task: Compute a PBE baseline" in conversation(app)
        await send(pilot, "/approve agents")
        assert f"Started agent-{HYP} on {HYP}" in conversation(app)
        await refresh(pilot)
        assert agents(app)[f"A1 agent-{HYP}"][1] == "queued" and "proposed" not in status(app)

        # The dispatcher works in its own thread, as a separate process would.
        clients.add("hypothesis_agent", *agent_requests(system))
        await asyncio.to_thread(dispatcher.tick)
        await refresh(pilot)
        calc = latest_calc(system["workspace"])
        assert panel(app)[0][2] == f"{short_id(calc)} with the input reviewer"
        clients.add("input_reviewer", *reviewer_decides(system))
        await asyncio.to_thread(dispatcher.tick)
        await refresh(pilot)
        assert agents(app)["Input reviewer"][1] == "idle"
        assert agents(app)["Input reviewer"][3] == f"last: review the inputs of {short_id(calc)}"
        (row,) = panel(app)
        assert row[2] == f"{short_id(calc)} needs /approve submit" and row[3] == "none" and row[4] == "NOT_TESTED"
        assert agents(app)[f"A1 agent-{HYP}"][1] == "waiting for you" and "dispatcher seen 0 s ago" in status(app)
        assert agents(app)[f"A1 agent-{HYP}"][3] == f"/approve submit {short_id(calc)}"
        assert awaiting(app) == (f"⚑ Waiting for you: 1 calculation to run (A1: {short_id(calc)}): /approve submit "
                                 f"{short_id(calc)} · /pending for details")
        assert f"  /approve submit {short_id(calc)}  A1 agent-{HYP} on {HYP}; " in conversation(app)

        # The agent's message is set apart: a bar and its name in its colour, how its task ended, what it said.
        message = next(w for w in app.query_one("#chat").query(".agent") if f"A1 agent-{HYP}" in str(w.content))
        assert str(message.content) == f"A1 agent-{HYP}  waiting for calculations\nRequested the PBE baseline; " \
                                       "waiting for it."
        assert message.styles.border_left[1] == Color.parse(AGENT_COLORS[0])
        assert any(span.style == f"bold {AGENT_COLORS[0]}" for span in message.content.spans)
        reviewer = next(w for w in app.query_one("#chat").query(".agent") if "input reviewer" in str(w.content))
        assert str(reviewer.content).startswith("input reviewer  finished\n")

        await send(pilot, f"/approve submit {short_id(calc)}")
        assert "Approved 1 calculation to run" in conversation(app) and calc in conversation(app)
        await refresh(pilot)
        assert panel(app)[0][2] == f"{short_id(calc)} authorized, not yet submitted"
        assert awaiting(app) == "" and not app.query_one("#awaiting").display

    drive(app, script)
    calc = latest_calc(system["workspace"])
    authorization = system["workspace"].ledger.load(INV).interventions["msg-3"]
    assert authorization.text == f"/approve submit {short_id(calc)}"
    assert authorization.authorizations[0].target_ids == (calc,)


def test_commands_answer_while_the_chief_works_and_messages_wait_their_turn(system):
    started, release = threading.Event(), threading.Event()

    def slow(_request):
        started.set()
        assert release.wait(10)
        return reply(TextPart(text="First answer."))

    system["clients"].add("chief_scientist", slow, reply(TextPart(text="Second answer.")))
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        app.query_one(Composer).value = "First question"
        await pilot.press("enter")
        assert await asyncio.to_thread(started.wait, 10)
        for line in ("Second question", "/status"):
            app.query_one(Composer).value = line
            await pilot.press("enter")
        for _ in range(100):  # the command answers from its own thread while the Chief is still busy
            await pilot.pause()
            if "No investigation is open yet" in conversation(app):
                break
            await asyncio.sleep(0.02)
        text = conversation(app)
        assert "No investigation is open yet" in text and "First answer." not in text and app.chief_busy
        assert "waiting until the Chief Scientist has answered the previous message" in text
        assert "1 message waiting" in str(app.query_one("#working", Static).content)
        release.set()
        await settle(pilot)
        text = conversation(app)
        assert text.index("First answer.") < text.index("Second answer.")
        assert not app.query_one("#working").has_class("busy")

    drive(app, script)


def test_notebook_pages_and_cited_output_open_in_place(system):
    calc = validated_run(system)
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        assert panel(app)[0][3] == "2 VALIDATED"  # a cited reading and an analysis result
        app.query_one("#swarm", DataTable).focus()
        await pilot.press("enter")
        await settle(pilot)
        page = app.screen
        assert isinstance(page, NotebookScreen) and page.history[-1].name == f"{HYP}.md"
        document = page.query_one(Markdown)
        assert calc in document.source and repr(ENERGY) in document.source

        cited = re.search(r"\]\((\.\./\.\./\.\./artifacts/[^)#]+#L(\d+))\)", document.source)
        document.post_message(Markdown.LinkClicked(document, cited.group(1)))
        await settle(pilot)
        assert isinstance(app.screen, FileScreen)
        area = app.screen.query_one(TextArea)
        assert area.cursor_location[0] + 1 == int(cited.group(2))
        assert "Total energy of the DFT" in area.document.get_line(area.cursor_location[0])
        await pilot.press("escape")
        await settle(pilot)

        document.post_message(Markdown.LinkClicked(document, "../index.md"))
        await settle(pilot)
        assert page.history[-1].name == "index.md"
        await pilot.press("backspace")
        await settle(pilot)
        assert page.history[-1].name == f"{HYP}.md"
        # Nothing outside the notebook and the stored files opens, whatever a page links to.
        (system["workspace"].root / "outside.md").write_text("# Not part of the notebook\n")
        for href in ("../../../outside.md", "../../../events.sqlite"):
            document.post_message(Markdown.LinkClicked(document, href))
            await settle(pilot)
            assert app.screen is page and page.history[-1].name == f"{HYP}.md"
        await pilot.press("escape")
        await send(pilot, "/timeline")
        assert isinstance(app.screen, NotebookScreen) and app.screen.history[-1].name == "timeline.md"

    drive(app, script)


def test_new_results_raise_an_alert_once(system):
    validated_run(system)
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)
    alerts = []
    app.notify = lambda message, **_: alerts.append(message)

    async def script(pilot):
        await settle(pilot)
        assert alerts == []  # what was already recorded is not news
        ledger = system["workspace"].ledger
        evidence = next(iter(ledger.load(INV).evidence))
        ledger.record(INV, [EventDraft(actor=JUDGE, payload=AssessmentRecorded(
            hypothesis_id=HYP, assessment=Assessment.INCONCLUSIVE, completeness=Completeness.PARTIAL,
            evidence_ids=(evidence,), rationale="One energy cannot settle a site ordering."))])
        await refresh(pilot)
        await refresh(pilot)
        assert alerts == [f"{HYP} assessed INCONCLUSIVE (partial)"]

    drive(app, script)


def test_narrow_terminals_start_with_the_panel_hidden_and_a_summary_shown(system):
    validated_run(system)
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        tables = ("#agents", "#agents-title", "#swarm", "#swarm-title")
        assert not any(app.query_one(t).display for t in tables) and app.query_one("#subagents").display
        summary = status(app).splitlines()[1]
        assert summary.startswith("1 hypothesis · jobs: none active · evidence: 2 VALIDATED")
        await send(pilot, "/panel")
        assert all(app.query_one(t).display for t in tables) and not app.query_one("#subagents").display
        await send(pilot, "/panel")
        assert not any(app.query_one(t).display for t in tables)

    drive(app, script, size=(90, 30))


def test_the_chief_scientists_reviews_appear_as_its_messages(system):
    clients, dispatcher = system["clients"], system["dispatcher"]
    open_and_start(system)
    clients.add("hypothesis_agent", reply(TextPart(text="Nothing to compute before the baseline is approved.")))
    dispatcher.tick()
    dispatcher.chief_reviews = True
    clients.add("chief_scientist", reply(TextPart(text="The agent has **not requested** any calculation yet.")))
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        assert "Nothing to compute before the baseline" in conversation(app)
        await asyncio.to_thread(dispatcher.tick)
        await refresh(pilot)
        text = conversation(app)
        assert "Chief Scientist (review of the agents' reports) ›" in text
        assert "The agent has **not requested** any calculation yet." in [
            w.source for w in app.query_one("#chat").query(Markdown)]

    drive(app, script)


def test_a_reopened_investigation_shows_the_conversation_so_far(system, monkeypatch):
    clients, chat = system["clients"], system["chat"]
    clients.add("chief_scientist", reply(TextPart(text="Which site do the experiments see?")),
                open_and_delegate(system["site"]), reply(TextPart(text="Opened and **delegated**.")))
    chat.handle("PBE puts CO in the fcc hollow on Cu(111).")
    chat.handle("Atop, at low coverage.")  # the investigation opens: both messages are recorded only now
    chat.handle("/approve agents")
    clients.add("hypothesis_agent", reply(TextPart(text="Nothing to compute before the baseline is approved.")))
    system["dispatcher"].tick()
    clients.add("chief_scientist", reply(TextPart(text="Noted.")))
    chat.handle("Thanks.")

    def reopened() -> CedarApp:
        session = ChatSession(system["workspace"], system["profiles"], clients, investigation_id=INV,
                              backend=system["scheduler"], clock=system["clock"])
        return CedarApp(session, refresh_seconds=3600, clock=lambda: NOW)

    app = reopened()

    async def script(pilot):
        await settle(pilot)
        text = conversation(app)
        shown = ["you › PBE puts CO in the fcc hollow", "Which site do the experiments see?",
                 "you › Atop, at low coverage.", "Opened and **delegated**.", "› /approve agents",
                 "Nothing to compute before the baseline", "you › Thanks.", "Noted.",
                 f"Investigation {INV}, continued: the conversation so far is above."]
        assert [text.index(line) for line in shown] == sorted(text.index(line) for line in shown)
        assert text.count("Nothing to compute before the baseline") == 1  # not shown again as news
        view = app.query_one("#chat")
        assert view.max_scroll_y > 0 and view.scroll_y == view.max_scroll_y

    drive(app, script, size=(100, 30))
    monkeypatch.setattr("cedar.tui.HISTORY_SHOWN", 2)
    app = reopened()

    async def shortened(pilot):
        await settle(pilot)
        text = conversation(app)
        assert "earlier messages and notices are not shown" in text and "Noted." in text
        assert "you › Thanks." in text and "Opened and **delegated**." not in text

    drive(app, shortened)


def test_the_panel_is_always_there_with_the_logo_across_its_top_border(system):
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        top = app.query_one(PanelTop)
        above, border = top.render().plain.split("\n")
        assert above == "   " + LOGO[0].plain
        assert border.startswith("╭─ " + LOGO[1].plain + " ─") and border.endswith("─╮")
        assert len(border) == top.size.width == app.size.width
        assert str(app.query_one("#tagline", Static).content) == TAGLINE
        assert str(app.query_one("#models", Static).content).startswith("Models: Chief Scientist "
                                                                        "scripted-chief_scientist (scripted)")
        assert not app.query_one("#swarm").display  # no hypotheses yet: the panel stays, without their table
        assert app.query_one("#agents").display and list(agents(app)) == ["Chief Scientist", "Evidence Judge"]

    drive(app, script)


def test_on_narrow_terminals_the_panel_stays_and_its_table_can_be_shown(system):
    validated_run(system)
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        assert app.query_one("#panel").display and not app.query_one("#swarm").display
        assert app.query_one(PanelTop).render().plain.split("\n")[1].endswith("╮")
        await send(pilot, "/panel")
        assert app.query_one("#swarm").display and app.query_one("#agents").display

    drive(app, script, size=(80, 40))


def test_a_slash_at_the_start_of_a_line_lists_the_matching_commands(system):
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    def shown() -> list[str]:
        menu = app.query_one("#commands", OptionList)
        if not app.menu_open:
            return []
        return [menu.get_option_at_index(i).prompt.plain.split("  ")[0] for i in range(menu.option_count)]

    async def script(pilot):
        await settle(pilot)
        composer = app.query_one(Composer)
        await pilot.press("slash", "e")
        assert shown() == ["/exit"]
        await pilot.press("x", "a")  # "/exa" matches nothing, so the menu goes
        assert shown() == []
        await pilot.press("backspace", "backspace", "backspace")
        assert shown() == [c.usage for c in MENU_COMMANDS]
        for line in ("hello /e", " /e", "e/"):  # only a line that starts with "/" gets suggestions
            composer.value = line
            await settle(pilot)
            assert shown() == []

        composer.value = "/a"
        await settle(pilot)
        assert shown() == ["/agents", "/approve agents", "/approve submit <calc>", "/approve submit all",
                           "/approve submit <agent>", "/approve close", "/attach <path> <note>"]
        await pilot.press("up")  # the arrows wrap around the menu
        assert app.query_one("#commands", OptionList).highlighted == 6
        await pilot.press("down", "down", "down", "tab")
        submits = ["/approve submit <calc>", "/approve submit all", "/approve submit <agent>"]
        assert composer.value == "/approve submit " and shown() == submits
        await pilot.press("escape")  # closed until the line changes
        assert shown() == [] and app.focused is composer
        await pilot.press("backspace")
        assert shown() == submits

        # Enter completes a partial command; Enter on the complete command runs it.
        composer.value = "/sta"
        await settle(pilot)
        await pilot.press("enter")
        await settle(pilot)
        assert composer.value == "/status" and "No investigation is open yet" not in conversation(app)
        await pilot.press("enter")
        await settle(pilot)
        assert "No investigation is open yet" in conversation(app) and composer.value == "" and shown() == []

        composer.value = "/ti"  # choosing with the mouse completes too
        await settle(pilot)
        app.query_one("#commands", OptionList).action_select()
        await settle(pilot)
        assert composer.value == "/timeline"

        composer.value = ""  # with the menu closed, Tab leaves the input as before
        await settle(pilot)
        assert shown() == []
        await pilot.press("tab")
        assert app.focused is not composer

    drive(app, script)


def test_each_table_shows_at_most_three_rows_and_scrolls_for_the_rest(system):
    open_and_start(system)
    for n in range(14):
        add_hypothesis(system["workspace"].ledger, INV, f"hyp-{n:02d}")
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        await refresh(pilot)
        for name, rows in (("#agents", 17), ("#swarm", 15)):
            table = app.query_one(name, DataTable)
            assert table.row_count == rows and table.size.height == 4  # its header and three rows
            assert table.max_scroll_y > 0 and table.virtual_size.width <= table.size.width  # scrolls, not sideways
        assert app.query_one("#chat").size.height >= 10

    drive(app, script, size=(120, 40))


def test_refreshing_the_tables_keeps_where_they_are_scrolled(system):
    validated_run(system)
    now = [NOW]
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: now[0])

    async def script(pilot):
        await settle(pilot)
        await send(pilot, "/panel")
        table, agents_table = app.query_one("#swarm", DataTable), app.query_one("#agents", DataTable)
        assert table.max_scroll_x > 0 and agents_table.max_scroll_y > 0
        table.scroll_to(x=table.max_scroll_x, animate=False)
        agents_table.scroll_to(y=agents_table.max_scroll_y, animate=False)
        await pilot.pause()
        x, y = table.scroll_x, agents_table.scroll_y
        assert x > 0 and y > 0

        # A cell changes: the table stays where it was scrolled to (DataTable.clear scrolls to the start).
        ledger = system["workspace"].ledger
        ledger.record(INV, [EventDraft(actor=JUDGE, payload=AssessmentRecorded(
            hypothesis_id=HYP, assessment=Assessment.INCONCLUSIVE, completeness=Completeness.PARTIAL,
            evidence_ids=tuple(ledger.load(INV).evidence)[:1], rationale="One energy cannot settle a site ordering."))])
        now[0] += timedelta(minutes=5)  # and every agent's "For"
        await refresh(pilot)
        assert panel(app)[0][4] == "INCONCLUSIVE (partial)" and table.scroll_x == x
        assert agents(app)[f"A1 agent-{HYP}"][2] == "5 min" and agents_table.scroll_y == y

        # A row comes: the table is refilled, still scrolled sideways, its cursor on the same row.
        table.move_cursor(row=0)
        add_hypothesis(ledger, INV, "hyp-more")
        await refresh(pilot)
        assert [row[0] for row in panel(app)] == [HYP, "hyp-more"] and table.scroll_x == x and table.cursor_row == 0

    drive(app, script, size=(80, 40))


def test_scrollbars_are_lines_thinner_than_a_row(system):
    def bar(position, vertical):
        drawn = ThinScrollBar.render_bar(size=10, virtual_size=40, window_size=10, position=position,
                                         vertical=vertical)
        return "".join(segment.text for segment in drawn.segments if segment.text != "\n")

    # A heavy thumb along a light track, its ends on half cells; a click before or after it pages, on it drags.
    assert bar(0, vertical=False) == "━━╾───────" and bar(10, vertical=False) == "──╼━━─────"
    assert bar(30, vertical=True) == "│││││││╽┃┃"
    drawn = ThinScrollBar.render_bar(size=10, virtual_size=40, window_size=10, position=10, vertical=False)
    assert [s.style.meta["@mouse.down"] for s in drawn.segments if s.style][1:4] == ["scroll_up", "grab", "grab"]

    validated_run(system)
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        await send(pilot, "/panel")
        table = app.query_one("#swarm", DataTable)
        assert table.show_horizontal_scrollbar and isinstance(table.horizontal_scrollbar.render(), ThinScrollBar)
        assert table.horizontal_scrollbar.size.height == 1
        for name in ("#agents", "#swarm", "#chat", "#composer"):
            assert app.query_one(name).styles.scrollbar_size_vertical == 1

    drive(app, script, size=(80, 40))


def test_new_messages_follow_at_the_bottom_until_the_researcher_scrolls_up(system):
    answer = "\n\n".join(f"Paragraph {i} of a long answer." for i in range(40))
    system["clients"].add("chief_scientist", reply(TextPart(text=answer)), reply(TextPart(text="Short answer.")))
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        chat = app.query_one("#chat")
        bar = chat.vertical_scrollbar

        def at_bottom() -> bool:
            return chat.max_scroll_y > 0 and chat.scroll_y == chat.max_scroll_y

        async def drag(start: int, end: int) -> None:
            await pilot.mouse_down(bar, offset=(0, start))
            await pilot.hover(bar, offset=(0, end))
            await pilot.mouse_up(bar, offset=(0, end))
            await settle(pilot)

        await send(pilot, "Explain")
        assert at_bottom()  # the whole answer, to its last paragraph

        # Scrolled up, the conversation stays where it is while messages come.
        await drag(bar.size.height - 1, 0)
        y, height = chat.scroll_y, chat.virtual_size.height
        assert y < chat.max_scroll_y
        for i in range(5):
            app.say(f"Notice {i}", "notice")
            await pilot.pause()
        await settle(pilot)
        assert chat.virtual_size.height > height and chat.scroll_y == y

        # Back at the bottom, it follows them again.
        await drag(0, bar.size.height - 1)
        assert at_bottom()
        app.say("Another notice", "notice")
        await settle(pilot)
        assert at_bottom()

        # The researcher's own message brings the conversation back to the bottom, and the answer follows.
        await drag(bar.size.height - 1, 0)
        assert not at_bottom()
        await send(pilot, "And briefly?")
        assert at_bottom() and "Short answer." in conversation(app)

    drive(app, script, size=(100, 40))


def test_the_panel_shows_the_mode_and_web_access_on_their_own_line(system):
    system["clients"].add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="Opened.")))
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    def mode():
        widget = app.query_one("#mode", Static)
        return str(widget.content), widget.has_class("automated")

    async def script(pilot):
        await settle(pilot)
        assert mode() == ("Mode: human-in-the-loop · Web: off", False)  # before any investigation: new ones start so
        await send(pilot, "/web only arxiv.org")
        await refresh(pilot)
        assert mode() == ("Mode: human-in-the-loop · Web: only arxiv.org", False)  # held until it opens
        await send(pilot, "PBE puts CO in the fcc hollow on Cu(111); experiments see atop.")
        await refresh(pilot)
        assert mode() == ("Mode: human-in-the-loop · Web: only arxiv.org", False)
        await send(pilot, "/mode full-automation 100 1000000")
        await refresh(pilot)
        assert mode() == ("Mode: full-automation · Web: only arxiv.org", True)
        await send(pilot, "/web off")
        await send(pilot, "/mode human-in-the-loop")
        await refresh(pilot)
        assert mode() == ("Mode: human-in-the-loop · Web: off", False)

    drive(app, script)


def test_the_input_box_wraps_grows_and_takes_clicks(system):
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        composer = app.query_one(Composer)
        assert composer.outer_size.height == 3  # one line inside its border, like the old input line
        composer.value = "PBE puts CO in the fcc hollow on Cu(111) while experiments see atop. " * 3
        await pilot.pause()
        assert composer.wrapped_document.height > 1 and composer.outer_size.height == composer.wrapped_document.height + 2
        composer.value = "\n".join(f"line {i}" for i in range(20))
        await pilot.pause()
        assert composer.outer_size.height == 8  # six lines at most; the rest scrolls

        composer.value = "hello wide world"
        await pilot.pause()  # the box shrinks back to one line before the click
        await pilot.click(Composer, offset=(2 + 6, 1))  # inside the border and padding: before "wide"
        assert composer.cursor_location == (0, 6)
        composer.value = "first line\nsecond line"
        await pilot.press("up")  # with the menu closed, the arrows move between lines
        assert composer.cursor_location[0] == 0 and not app.menu_open
        await pilot.press("enter")  # sends instead of adding a line
        await settle(pilot)
        assert composer.value == "" and "you › first line\nsecond line" in conversation(app)

    drive(app, script, size=(80, 40))


def test_the_conversation_has_no_shortcut_keys(system):
    validated_run(system)
    app = CedarApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        main = app.screen
        assert not main.query(Footer)
        shown = app.query_one("#swarm", DataTable).display
        for key in ("f2", "f3", "f4", "f8", "ctrl+p"):  # commands are typed with "/" only
            await pilot.press(key)
            await settle(pilot)
            assert app.screen is main and app.query_one("#swarm", DataTable).display == shown
        await send(pilot, "/notebook")
        assert isinstance(app.screen, NotebookScreen)

    drive(app, script)


def test_the_window_starts_a_dispatcher_when_work_waits_and_none_runs(system):
    clients = system["clients"]
    clients.add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="Proposed.")))
    system["chat"].handle("Investigate CO on Cu(111).")  # the proposal now waits out its review window
    started = []
    session = system["chat"]
    session.auto_dispatch, session.start_dispatcher = True, lambda workspace: started.append(workspace) or 4242
    app = CedarApp(session, refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        assert len(started) == 1 and "[dispatcher] Started in the background (process 4242)" in conversation(app)
        assert "one starts by itself when there is work" in status(app) and "start `cedar dispatch`" not in status(app)
        await refresh(pilot)
        assert len(started) == 1  # not restarted within a minute

    drive(app, script)
