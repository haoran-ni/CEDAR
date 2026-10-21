"""The researcher's /web and /attach: the record they leave, and what each agent gets from them.

ScriptedModel and FakeScheduler TEST DOUBLES on a SYNTHETIC site; no model, web page or job is reached.
"""

import io

import pytest
from factories import CHIEF, USER, draft, start_investigation
from PIL import Image
from pydantic import ValidationError
from test_orchestration import INV as OPENED, call, make_system, open_and_delegate
from test_swarm import INV, agents_answer, investigation, propose, says

from sim2real.contracts import (
    AgentTaskQueued,
    InvestigationConstraints,
    InvestigationStatus,
    InvestigationStatusChanged,
    MaterialAttached,
    RetrievalPolicy,
    RetrievalPolicyChanged,
    UserInterventionRecorded,
)
from sim2real.errors import RuleViolation
from sim2real.models import WebAccess
from sim2real.models.scripted import reply
from sim2real.notebook import render_notebook, round_file
from sim2real.paths import config_dir


@pytest.fixture
def system(tmp_path, site):
    return make_system(tmp_path, site)


def png(path):
    Image.new("RGB", (40, 20), "white").save(path, format="PNG")
    return path


def tool_names(model_call) -> set[str]:
    return {t.name for t in model_call.tools}


# -- the record ------------------------------------------------------------------------------------

def test_only_the_researcher_sets_web_access_and_attaches_materials(ledger, artifacts, tmp_path):
    start_investigation(ledger, "inv-1")
    ledger.record("inv-1", [draft(USER, UserInterventionRecorded(intervention_id="msg-1", text="/web on"))])
    on = RetrievalPolicyChanged(policy=RetrievalPolicy.ENABLED, intervention_id="msg-1")
    with pytest.raises(RuleViolation, match="may not record"):
        ledger.record("inv-1", [draft(CHIEF, on)])
    with pytest.raises(RuleViolation, match="cites their recorded command"):
        ledger.record("inv-1", [draft(USER, on.model_copy(update={"intervention_id": "msg-9"}))])
    assert ledger.load("inv-1").retrieval.policy is RetrievalPolicy.DISABLED  # how every investigation starts
    ledger.record("inv-1", [draft(USER, on)])
    assert ledger.load("inv-1").retrieval.policy is RetrievalPolicy.ENABLED

    material = MaterialAttached(material_id="mat-1", name="Fig 3.png", note="tilted phase", intervention_id="msg-1",
                                artifact=artifacts.put_file(png(tmp_path / "Fig 3.png"), media_type="image/png"))
    with pytest.raises(RuleViolation, match="may not record"):
        ledger.record("inv-1", [draft(CHIEF, material)])
    ledger.record("inv-1", [draft(USER, material)])
    with pytest.raises(RuleViolation, match="already exists"):
        ledger.record("inv-1", [draft(USER, material)])
    assert ledger.load("inv-1").materials["mat-1"].note == "tilted phase"

    ledger.record("inv-1", [draft(USER, InvestigationStatusChanged(status=InvestigationStatus.CANCELLED,
                                                                    reason="test"))])
    with pytest.raises(RuleViolation, match="CANCELLED"):
        ledger.record("inv-1", [draft(USER, on)])


def test_domains_go_with_curated_only_and_are_plain_host_names():
    assert RetrievalPolicyChanged(policy=RetrievalPolicy.CURATED_ONLY, allowed_domains=("ArXiv.org",),
                                  intervention_id="msg-1").allowed_domains == ("arxiv.org",)
    for policy, domains in ((RetrievalPolicy.CURATED_ONLY, ()), (RetrievalPolicy.ENABLED, ("arxiv.org",)),
                            (RetrievalPolicy.CURATED_ONLY, ("arxiv.org/abs",)),
                            (RetrievalPolicy.CURATED_ONLY, ("localhost",))):
        with pytest.raises(ValidationError):
            RetrievalPolicyChanged(policy=policy, allowed_domains=domains, intervention_id="msg-1")
    with pytest.raises(ValidationError):
        InvestigationConstraints(external_retrieval=RetrievalPolicy.CURATED_ONLY)


# -- the researcher's commands --------------------------------------------------------------------

def test_web_sets_the_agents_access_and_records_the_command(system):
    chat = investigation(system, "hyp-a")
    ledger = system["workspace"].ledger
    assert chat.handle("/web").startswith("Web access for the agents: off.")
    said = chat.handle("/web only https://arxiv.org/ DOI.org")
    assert said.startswith("The Chief Scientist and the hypothesis agents may use the web from their next runs: "
                           "only arxiv.org, doi.org (msg-1).")
    state = ledger.load(INV)
    assert (state.retrieval.policy, state.retrieval.allowed_domains) == (RetrievalPolicy.CURATED_ONLY,
                                                                         ("arxiv.org", "doi.org"))
    assert state.interventions["msg-1"].text == "/web only https://arxiv.org/ DOI.org"
    assert chat.handle("/web only arxiv.org/abs").startswith("Give plain domain names")
    assert chat.handle("/web sometimes").startswith("Usage: /web")
    assert "only arxiv.org, doi.org" in chat.handle("/status")
    chat.handle("/web on")
    chat.handle("/web off")
    assert [r.policy for r in ledger.load(INV).retrieval_changes] == [
        RetrievalPolicy.CURATED_ONLY, RetrievalPolicy.ENABLED, RetrievalPolicy.DISABLED]


def test_attach_stores_a_copy_and_records_it(system, tmp_path):
    chat = investigation(system, "hyp-a")
    ledger, artifacts = system["workspace"].ledger, system["workspace"].artifacts
    figure = png(tmp_path / "STM image.png")
    said = chat.handle(f"/attach '{figure}' Fig. 3: the tilted phase")
    assert said.startswith("Attached mat-1: STM image.png (image/png, ") and "(msg-1)" in said
    material = ledger.load(INV).materials["mat-1"]
    assert (material.name, material.note) == ("STM image.png", "Fig. 3: the tilted phase")
    assert artifacts.read(material.artifact) == figure.read_bytes()
    (tmp_path / "spectrum.csv").write_text("energy,intensity\n1.0,2.0\n")
    chat.handle(f"/attach {tmp_path / 'spectrum.csv'}")
    assert chat.handle("/materials").splitlines()[1].startswith("mat-2: spectrum.csv (text/csv, ")

    (tmp_path / "restart.bin").write_bytes(b"\x00\x01binary")
    assert "no agent could view it" in chat.handle(f"/attach {tmp_path / 'restart.bin'}")
    assert chat.handle(f"/attach {tmp_path / 'missing.png'}").startswith("No file")
    keys = config_dir() / "credentials.toml"
    keys.parent.mkdir(parents=True, exist_ok=True)
    keys.write_text("[anthropic]\napi_key = 'x'\n")
    assert "cannot be attached" in chat.handle(f"/attach {keys}")
    assert len(ledger.load(INV).materials) == 2


def test_before_the_investigation_opens_web_and_materials_wait_for_it(system, tmp_path):
    chat, clients, ledger = system["chat"], system["clients"], system["workspace"].ledger
    assert "recorded when the investigation opens" in chat.handle("/web on")
    assert "recorded when the investigation opens" in chat.handle(f"/attach {png(tmp_path / 'fig.png')} the STM")
    clients.add("chief_scientist", open_and_delegate(system["site"]), says("Opened."))
    chat.handle("Investigate CO on Cu(111); my figure shows atop CO.")

    first = clients.models["chief_scientist"].calls[0]
    assert first.web == WebAccess() and {"list_materials", "view_material"} <= tool_names(first)
    assert first.messages[-1].text.startswith("[record: the researcher attached mat-1: fig.png (image/png, ")
    state = ledger.load(OPENED)
    texts = [i.text for i in state.interventions.values()]
    assert texts[:3] == ["/web on", f"/attach {tmp_path / 'fig.png'} the STM",
                         "Investigate CO on Cu(111); my figure shows atop CO."]
    assert state.retrieval.policy is RetrievalPolicy.ENABLED and state.retrieval.intervention_id == "msg-1"
    assert state.materials["mat-1"].intervention_id == "msg-2"

    chat.handle("/web off")
    clients.add("chief_scientist", says("No web now."))
    chat.handle("Anything new?")
    later = clients.models["chief_scientist"].calls[-1]
    assert later.web is None and not later.messages[-1].text.startswith("[record: the researcher attached")


# -- what each agent gets --------------------------------------------------------------------------

def test_only_the_chief_scientist_and_hypothesis_agents_get_the_web(tmp_path, site):
    system = make_system(tmp_path, site, reviews=True)
    ledger, clients, dispatcher = system["workspace"].ledger, system["clients"], system["dispatcher"]
    chat = investigation(system, "hyp-a")
    chat.handle("/web only arxiv.org")
    chat.handle(f"/attach {png(tmp_path / 'fig.png')}")
    chat.handle("/mode full-automation 100 1000000")
    propose(ledger, "p-1", "hyp-a")
    clients.add("approver", reply(call("approve", target_id="p-1", reason="Worth a test.")), says("Approved."))
    router = agents_answer(system, **{"hyp-a": [says("Planned a test.")]})
    ledger.record(INV, [draft(CHIEF, AgentTaskQueued(task_id="t-judge", role="evidence_judge", hypothesis_id="hyp-a",
                                                     instruction="Assess hyp-a."))])
    clients.add("evidence_judge", says("Nothing to assess yet."))
    clients.add("chief_scientist", says("Reviewed: one test planned."))
    for _ in range(4):  # the approver starts the agent; the Judge, the agent, then the review run
        dispatcher.tick()

    curated = WebAccess(allowed_domains=("arxiv.org",))
    agent_call = router.models["agent-hyp-a"].calls[0]
    assert agent_call.web == curated and {"list_materials", "view_material", "view_file"} <= tool_names(agent_call)
    (review,) = clients.models["chief_scientist"].calls
    assert review.web == curated and "view_material" in tool_names(review)
    for role in ("approver", "evidence_judge"):
        (role_call, *_) = clients.models[role].calls
        assert role_call.web is None and {"list_materials", "view_material"} <= tool_names(role_call)


def test_the_notebook_shows_web_access_and_materials(system, tmp_path):
    chat = investigation(system, "hyp-a")
    chat.handle("/web only arxiv.org")
    chat.handle(f"/attach {png(tmp_path / 'fig.png')} Fig. 3")
    events = system["workspace"].store.read_stream(INV)
    pages = render_notebook(events)
    index = pages["index.md"]
    assert "## Web access and materials" in index and "agents: **only arxiv.org**" in index
    assert "- Set only arxiv.org by the researcher's msg-1" in index
    assert "- **mat-1** [fig.png](../../artifacts/objects/" in index and "msg-2): “Fig. 3”" in index
    round_page = pages[round_file(events[-1].round)]
    assert "Web access for the Chief Scientist and the hypothesis agents set only arxiv.org by" in round_page
    assert "attached mat-1, [fig.png](" in round_page


def test_agents_view_materials_by_id(artifacts, tmp_path):
    from sim2real.agents.toolkits.materials import Material, material_tools

    figure = artifacts.put_file(png(tmp_path / "fig.png"), media_type="image/png")
    table = artifacts.put_bytes(b"energy,intensity\n" + b"".join(f"{i},{i * i}\n".encode() for i in range(300)),
                                label="spectrum.csv", media_type="text/csv")
    materials = [Material("mat-1", "fig.png", figure, "Fig. 3"), Material("mat-2", "spectrum.csv", table)]
    tools = {t.name: t for t in material_tools(lambda: materials, artifacts)}

    assert tools["list_materials"].invoke({}).content.splitlines() == [
        f"mat-1: fig.png (image/png, {figure.size_bytes:,} bytes): Fig. 3",
        f"mat-2: spectrum.csv (text/csv, {table.size_bytes:,} bytes)"]
    shown = tools["view_material"].invoke({"material_id": "mat-1"})
    assert shown.media[0].media_type == "image/png" and "context, not evidence" in shown.content
    lines = tools["view_material"].invoke({"material_id": "mat-2", "start_line": 300, "max_lines": 5}).content
    assert lines.splitlines() == ["300: 298,88804", "301: 299,89401", "[mat-2 spectrum.csv: lines 300-301 of 301]"]
    for args, problem in (({"material_id": "mat-9"}, "attached: mat-1, mat-2"),
                          ({"material_id": "mat-2", "pages": "1"}, "only to PDFs"),
                          ({"material_id": "mat-1", "start_line": 3}, "only to text files")):
        refused = tools["view_material"].invoke(args)
        assert refused.is_error and problem in refused.content
