"""The full-screen interface, driven headless by Textual's test pilot.

ScriptedModel and FakeScheduler TEST DOUBLES on a SYNTHETIC site; no model or job runs.
"""

import asyncio
import re
import threading

import pytest
from sites import aims_output
from test_orchestration import (
    ENERGY,
    HYP,
    INV,
    NOW,
    agent_reads,
    agent_requests,
    latest_calc,
    make_system,
    open_and_delegate,
)
from textual.widgets import Collapsible, DataTable, Input, Markdown, Static, TextArea

from sim2real.contracts import Assessment, AssessmentRecorded, Completeness, EventDraft, SchedulerState
from sim2real.dashboard import short_id
from sim2real.dispatch import JUDGE
from sim2real.models import TextPart
from sim2real.models.scripted import reply
from sim2real.tui import FileScreen, NotebookScreen, Sim2RealApp


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
    composer = pilot.app.query_one(Input)
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
    table = app.query_one(DataTable)
    return [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]


def status(app) -> str:
    return str(app.query_one("#status", Static).content)


def validated_run(system) -> str:
    """Open, delegate, approve and finish one calculation without the interface; returns its id."""
    chat, clients, dispatcher, scheduler = system["chat"], system["clients"], system["dispatcher"], system["scheduler"]
    clients.add("chief_scientist", open_and_delegate(system["site"]), reply(TextPart(text="Opened.")))
    chat.handle("Investigate CO on Cu(111).")
    clients.add("hypothesis_agent", *agent_requests(system))
    dispatcher.tick()
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
    app = Sim2RealApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

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
        assert row[0] == HYP and row[1] == "ACTIVE" and row[2].startswith("agent queued")
        assert status(app).startswith(f"{INV} · INVESTIGATING · round 1 · plan v1")

        # The dispatcher works in its own thread, as a separate process would.
        clients.add("hypothesis_agent", *agent_requests(system))
        await asyncio.to_thread(dispatcher.tick)
        await refresh(pilot)
        calc = latest_calc(system["workspace"])
        (row,) = panel(app)
        assert row[3] == f"{short_id(calc)} needs /approve submit" and row[4] == "none" and row[5] == "NOT_TESTED"
        assert row[2] == "agent waiting for jobs" and "dispatcher seen 0 s ago" in status(app)
        assert "1 waiting for you (/pending)" in status(app)
        assert f"[waiting for you] /approve submit {calc}" in conversation(app)
        assert "Requested the PBE baseline; waiting for it." in conversation(app)

        await send(pilot, f"/approve submit {short_id(calc)}")
        assert f"Submission of {calc} authorized" in conversation(app)
        await refresh(pilot)
        assert panel(app)[0][3] == f"{short_id(calc)} authorized, not yet submitted"
        assert "waiting for you" not in status(app)

    drive(app, script)
    calc = latest_calc(system["workspace"])
    authorization = system["workspace"].ledger.load(INV).interventions["msg-2"]
    assert authorization.text == f"/approve submit {short_id(calc)}"
    assert authorization.authorizations[0].target_ids == (calc,)


def test_commands_answer_while_the_chief_works_and_messages_wait_their_turn(system):
    started, release = threading.Event(), threading.Event()

    def slow(_request):
        started.set()
        assert release.wait(10)
        return reply(TextPart(text="First answer."))

    system["clients"].add("chief_scientist", slow, reply(TextPart(text="Second answer.")))
    app = Sim2RealApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        app.query_one(Input).value = "First question"
        await pilot.press("enter")
        assert await asyncio.to_thread(started.wait, 10)
        for line in ("Second question", "/status"):
            app.query_one(Input).value = line
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
    app = Sim2RealApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        assert panel(app)[0][4] == "2 VALIDATED"  # a cited reading and an analysis result
        app.query_one(DataTable).focus()
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
    app = Sim2RealApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)
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
    app = Sim2RealApp(system["chat"], refresh_seconds=3600, clock=lambda: NOW)

    async def script(pilot):
        await settle(pilot)
        assert not app.query_one(DataTable).display
        summary = status(app).splitlines()[1]
        assert summary.startswith("1 hypothesis · jobs: none active · evidence: 2 VALIDATED")
        await pilot.press("f8")
        assert app.query_one(DataTable).display
        await send(pilot, "/panel")
        assert not app.query_one(DataTable).display

    drive(app, script, size=(90, 30))
