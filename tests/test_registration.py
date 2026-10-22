"""Runs computed outside CEDAR, registered so that readings and analyses of them can become evidence.

SYNTHETIC run directories on a SYNTHETIC site; the FakeScheduler TEST DOUBLE runs nothing. Analysis scripts
really run, under their guards.
"""

import json

import pytest
from factories import CHIEF, USER, VALIDATOR, add_hypothesis, agent_for, draft, sha256_of
from sites import ELEMENTS, aims_output, co_on_cu111, geometry_text
from test_orchestration import call, make_system
from test_swarm import INV, investigation, says
from worlds import World

from cedar.agents.toolkits import ResearchContext, file_tools, research_tools
from cedar.agents.toolkits import agent_access
from cedar.codes.fhiaims.control import render_control
from cedar.contracts import (
    CalculationState,
    EvidenceKind,
    EvidenceObject,
    EvidenceQuality,
    EvidenceRecorded,
    ExaminationVerdict,
    Observable,
    OutputCitation,
    PredictionRegistered,
    Prediction,
    QualityCheck,
    Quantity,
    UserInterventionRecorded,
)
from cedar.engine import RunRegistrar
from cedar.errors import RuleViolation
from cedar.models.scripted import reply
from cedar.notebook import render_notebook
from cedar.paths import config_dir
from cedar.protocols import TOTAL_ENERGY
from cedar.state import replay
from cedar.state.projection import hypothesis_evidence

ENERGY = -1234.567890123
EIGENVALUES = "# state occupation eigenvalue_eV\n1 2.0 -9.1\n2 0.0 -1.9\n"
SETTINGS = {"xc": "pbe", "spin": "none", "charge": "0", "relativistic": "atomic_zora scalar", "k_grid": "4 4 1",
            "occupation_type": "gaussian 0.1", "sc_accuracy_rho": "1E-5"}


@pytest.fixture
def world(ledger, store, artifacts, site):
    return World(ledger, store, artifacts, site)


def control_text(site, level="light", species: dict[str, str] | None = None, **keywords) -> str:
    """A SYNTHETIC control.in with species blocks copied from the site's species defaults."""
    root = site.codes["fhi-aims"].data_paths["species_defaults"]
    settings = {**SETTINGS, **keywords}
    blocks = [(species or {}).get(e) or open(f"{root}/{level}/{ELEMENTS[e]:02d}_{e}_default").read()
              for e in ("Cu", "C", "O")]
    return render_control(["SYNTHETIC control.in of an earlier run"], [(k, v) for k, v in settings.items() if v],
                          blocks)


def run_directory(path, site, *, output=None, control=None, extra=None):
    """A finished run's directory as the researcher left it (SYNTHETIC)."""
    path.mkdir(parents=True)
    slab, bottom = co_on_cu111()
    (path / "geometry.in").write_text(geometry_text(slab, fixed=bottom))
    (path / "control.in").write_text(control or control_text(site))
    (path / "aims.out").write_text(aims_output(energy=ENERGY) if output is None else output)
    for name, text in (extra or {}).items():
        (path / name).write_text(text)
    return path


def researcher_registers(world, directory, note="the researcher's PBE atop run", **kwargs):
    message = f"msg-{len(world.state().interventions) + 1}"
    world.ledger.record(world.inv, [draft(USER, UserInterventionRecorded(intervention_id=message,
                                                                         text=f"/register {directory} {note}"))])
    registrar = RunRegistrar(world.ledger, world.artifacts, profile=world.site)
    return registrar.register(world.inv, directory, requested_by=USER, description=note, intervention_id=message,
                              **kwargs)


def agent_tools(world, hypothesis, tmp_path):
    files = agent_access(deny=[tmp_path / "config"])
    (tmp_path / "data" / "workspaces" / hypothesis).mkdir(parents=True, exist_ok=True)
    registrar = RunRegistrar(world.ledger, world.artifacts, profile=world.site,
                             readable=lambda path: files.resolve(str(path)))
    ctx = ResearchContext(ledger=world.ledger, investigation_id=world.inv, hypothesis_id=hypothesis,
                          actor=agent_for(hypothesis), registry=world.registry, artifacts=world.artifacts,
                          engine=world.engine, validator=world.validator, files=files, analysis=world.analysis,
                          workspace=tmp_path / "data" / "workspaces" / hypothesis, registrar=registrar)
    return {t.name: t for t in research_tools(ctx) + file_tools(files)}


def reading(calculation_id, output, file="aims.out"):
    lines = output.splitlines()
    line = next(i for i, text in enumerate(lines, start=1) if "Total energy of the DFT" in text)
    return {"calculation_id": calculation_id, "observable_name": "total energy",
            "observable_definition": "final total energy printed by FHI-aims", "value": ENERGY, "unit": "eV",
            "summary": "the earlier run's total energy",
            "citations": [{"file": file, "line_start": line, "line_end": line, "quote": lines[line - 1].strip()}]}


# -- what registering records ---------------------------------------------------------------------

def test_a_registered_run_records_its_files_settings_and_health(world, tmp_path):
    directory = run_directory(tmp_path / "earlier" / "co-atop", world.site)
    outcome = researcher_registers(world, directory)
    assert outcome.calculation_id and not outcome.problems
    calc = world.calc(outcome.calculation_id)
    assert calc.registered and calc.shared and calc.hypothesis_ids == []
    assert calc.state is CalculationState.QUALITY_ASSESSED and calc.scheduler_state is None
    assert calc.run_check.healthy and {c.name for c in calc.run_check.checks} == {"normal_termination",
                                                                                  "scf_converged"}
    assert [r.name for r in calc.run_check.readings] == ["total energy", "zero-broadening total energy"]
    registration = calc.registration
    assert registration.source == str(directory) and registration.code_version == "test-1.0"
    assert registration.requested_by == USER
    assert world.state().interventions[registration.intervention_id].text.startswith(f"/register {directory}")
    # The settings the inputs used, classified as a task declares them.
    task = calc.task
    assert task.controls == {"xc": "pbe", "spin": "none", "charge": 0, "relativistic": "atomic_zora scalar"}
    assert task.numerical_settings == {"k_grid": [4, 4, 1], "occupation_type": "gaussian 0.1",
                                       "sc_accuracy_rho": 1e-5, "species_defaults": "light"}
    assert task.structure.sha256 == sha256_of((directory / "geometry.in").read_text())
    assert set(calc.input_files) == {"control.in", "geometry.in"} and set(calc.run_check.outputs) == {"aims.out"}
    assert calc.run_check.method["origin"] == "registered"
    assert any("Computed outside CEDAR" in text for text in calc.run_check.limitations)
    # The stored files are the files that were checked, whatever happens to the directory later.
    (directory / "aims.out").write_text("changed\n")
    assert b"Have a nice day." in world.artifacts.read(calc.run_check.outputs["aims.out"])
    assert replay(world.store.read_stream(world.inv)).calculations[outcome.calculation_id].registration == registration


def test_settings_the_protocol_does_not_classify_are_still_recorded(world, tmp_path):
    odd_cu = "  species Cu\n    nucleus 29\n    l_hartree 8\n"
    control = control_text(world.site, species={"Cu": odd_cu}, use_hartree_fock=" ", postprocess_anyway=" ",
                           mixer="pulay")
    directory = run_directory(tmp_path / "odd", world.site, control=control)
    calc = world.calc(researcher_registers(world, directory).calculation_id)
    assert calc.task.controls["postprocess_anyway"] is True and calc.task.controls["use_hartree_fock"] is True
    assert "mixer" not in calc.task.controls and "mixer" not in calc.task.numerical_settings
    species = calc.task.numerical_settings["species_defaults"]
    assert species["C"] == species["O"] == "light" and species["Cu"].startswith("sha256:")
    codes = {n.code for n in calc.registration.notes}
    assert {"forbidden_keyword", "unclassified_keyword", "species_not_a_default"} <= codes


def test_runs_that_crashed_or_did_not_converge_give_no_validated_evidence(world, tmp_path):
    crashed = researcher_registers(world, run_directory(tmp_path / "crashed", world.site,
                                                        output=aims_output(normal_end=False)))
    relaxing = control_text(world.site, relax_geometry="bfgs 1e-2")
    unconverged = researcher_registers(world, run_directory(tmp_path / "relax", world.site, control=relaxing,
                                                            output=aims_output(relax=True, geometry_converged=False)))
    assert world.calc(crashed.calculation_id).registration.failure_quality is EvidenceQuality.INVALID
    relax = world.calc(unconverged.calculation_id).registration
    assert relax.failure_quality is EvidenceQuality.INCONCLUSIVE
    assert [c.name for c in relax.checks if not c.passed] == ["geometry_converged"]


def test_a_directory_without_the_run_files_registers_nothing(world, tmp_path):
    directory = run_directory(tmp_path / "no-output", world.site)
    (directory / "aims.out").unlink()
    registrar = RunRegistrar(world.ledger, world.artifacts)
    outcome = registrar.register(world.inv, directory, requested_by=CHIEF, description="d")
    assert outcome.calculation_id is None and "has no aims.out" in outcome.problems[0]
    assert registrar.register(world.inv, "relative/dir", requested_by=CHIEF, description="d").problems
    (directory / "run.out").write_text(aims_output())
    named = registrar.register(world.inv, directory, requested_by=CHIEF, description="d", output_file="run.out")
    assert set(world.calc(named.calculation_id).run_check.outputs) == {"run.out"}
    assert len(world.state().calculations) == 1


def test_registering_the_same_run_again_returns_it(world, tmp_path):
    directory = run_directory(tmp_path / "again", world.site)
    first = researcher_registers(world, directory)
    second = researcher_registers(world, directory)
    assert second.reused and second.calculation_id == first.calculation_id
    (directory / "aims.out").write_text(aims_output(energy=ENERGY + 1))
    assert researcher_registers(world, directory).calculation_id != first.calculation_id


def test_who_may_register_what(world, tmp_path):
    directory = run_directory(tmp_path / "who", world.site)
    registrar = RunRegistrar(world.ledger, world.artifacts)
    with pytest.raises(RuleViolation, match="cites their recorded command"):
        registrar.register(world.inv, directory, requested_by=USER, description="d", intervention_id="msg-9")
    with pytest.raises(RuleViolation, match="is not the active agent of hyp-b"):
        add_hypothesis(world.ledger, world.inv, "hyp-b")
        registrar.register(world.inv, directory, requested_by=agent_for(world.hyp), description="d",
                           hypothesis_id="hyp-b")
    with pytest.raises(ValueError, match="an agent registers runs for its own hypothesis"):
        registrar.register(world.inv, directory, requested_by=agent_for(world.hyp), description="d")


# -- evidence from registered runs ----------------------------------------------------------------

def test_readings_of_a_registered_run_are_archived_evidence(world, tmp_path):
    directory = run_directory(tmp_path / "earlier", world.site)
    calc_id = researcher_registers(world, directory).calculation_id
    tools = agent_tools(world, world.hyp, tmp_path)
    shown = tools["run_outputs"].invoke({"calculation_id": calc_id}).content
    assert f"registered run from {directory}" in shown and "Stored inputs: control.in, geometry.in" in shown
    assert not tools["read_file"].invoke({"path": str(directory / "aims.out")}).is_error
    outcome = tools["submit_reading"].invoke(reading(calc_id, aims_output(energy=ENERGY)))
    assert "VALIDATED" in outcome.content, outcome.content
    (record,) = world.state().evidence.values()
    ev = record.evidence
    assert ev.kind is EvidenceKind.ARCHIVED_COMPUTATION and ev.calculation_id == calc_id
    assert any("Computed outside CEDAR" in text for text in ev.limitations)
    assert {s.sha256 for s in ev.sources} >= {ref.sha256 for ref in world.calc(calc_id).input_files.values()}


def test_a_reading_of_a_registered_run_is_never_presented_as_newly_computed(world, tmp_path):
    calc_id = researcher_registers(world, run_directory(tmp_path / "kind", world.site)).calculation_id
    calc = world.calc(calc_id)
    output = calc.run_check.outputs["aims.out"]
    reading_line = next(r for r in calc.run_check.readings if r.name == "total energy")
    evidence = EvidenceObject(
        kind=EvidenceKind.DIRECT_COMPUTATION, summary="s", observable=TOTAL_ENERGY,
        value=Quantity(value=ENERGY, unit="eV"), calculation_id=calc_id, quality=EvidenceQuality.VALIDATED,
        quality_checks=(QualityCheck(name="run_healthy", passed=True), QualityCheck(name="citations_verified",
                                                                                     passed=True)),
        sources=(output,), citations=(OutputCitation(artifact=output, line_start=reading_line.line_start,
                                                     line_end=reading_line.line_end, quote=reading_line.quote),))
    with pytest.raises(RuleViolation, match="archived computation evidence: it is a registered run"):
        world.ledger.record(world.inv, [draft(VALIDATOR, EvidenceRecorded(evidence_id="ev-x", evidence=evidence))])


def test_analysis_of_a_registered_run_stages_its_files(world, tmp_path):
    directory = run_directory(tmp_path / "with-eigenvalues", world.site, extra={"eigenvalues.dat": EIGENVALUES})
    calc_id = researcher_registers(world, directory).calculation_id
    tools = agent_tools(world, world.hyp, tmp_path)
    script = (f"import json\nrows = [l.split() for l in open('inputs/{calc_id}/eigenvalues.dat') if l[0] != '#']\n"
              "gap = float(rows[1][2]) - float(rows[0][2])\nprint('gap', gap)\n"
              "json.dump({'results': {'gap': {'value': gap, 'unit': 'eV'}}}, open('results.json', 'w'))\n")
    outcome = tools["run_analysis"].invoke({"purpose": "gap", "script": script, "inputs": [
        {"calculation_id": calc_id, "files": ["aims.out", "eigenvalues.dat"]}], "save_products": ["results.json"]})
    assert not outcome.is_error, outcome.content
    analysis_id = next(iter(world.state().analyses))
    saved = tmp_path / "data" / "workspaces" / world.hyp / "analyses" / analysis_id / "results.json"
    assert json.loads(saved.read_text())["results"]["gap"]["value"] == pytest.approx(7.2)
    recorded = tools["record_result"].invoke({"analysis_id": analysis_id, "result_name": "gap",
                                              "observable_name": "gap", "observable_definition": "LUMO - HOMO",
                                              "summary": "from the earlier run"})
    assert "VALIDATED" in recorded.content, recorded.content
    ev = next(iter(world.state().evidence.values())).evidence
    assert ev.kind is EvidenceKind.ANALYSIS and any("Computed outside CEDAR" in t for t in ev.limitations)


def test_a_registered_run_cannot_be_predicted(world, tmp_path):
    calc_id = researcher_registers(world, run_directory(tmp_path / "known", world.site)).calculation_id
    tools = agent_tools(world, world.hyp, tmp_path)
    outcome = tools["register_existing_run"].invoke({"directory": str(tmp_path / "known"),
                                                    "description": "my own copy"})
    own = next(c for c in world.state().calculations if c != calc_id)
    assert not outcome.is_error and world.calc(own).hypothesis_ids == [world.hyp]
    with pytest.raises(RuleViolation, match="a prediction names calculations before they are submitted"):
        world.ledger.record(world.inv, [draft(agent_for(world.hyp), PredictionRegistered(
            prediction_id="p-late", hypothesis_id=world.hyp, prediction=Prediction(
                statement="s", observable=Observable(name="e", definition="e"), expected_outcome="x",
                falsified_if="y", calculation_ids=(own,))))])


# -- who sees what ---------------------------------------------------------------------------------

def test_an_agent_registers_runs_for_its_own_hypothesis_and_every_agent_may_use_them(world, tmp_path):
    other = add_hypothesis(world.ledger, world.inv, "hyp-b")
    directory = run_directory(tmp_path / "mine", world.site)
    mine, theirs = agent_tools(world, world.hyp, tmp_path), agent_tools(world, other, tmp_path)
    outcome = mine["register_existing_run"].invoke({"directory": str(directory), "description": "k-grid check"})
    assert not outcome.is_error, outcome.content
    calc_id = next(iter(world.state().calculations))
    assert world.calc(calc_id).hypothesis_ids == [world.hyp] and not world.calc(calc_id).shared
    assert calc_id not in theirs["investigation_overview"].invoke({}).content
    assert calc_id in mine["investigation_overview"].invoke({}).content
    assert calc_id in theirs["other_agents"].invoke({}).content
    assert not theirs["run_outputs"].invoke({"calculation_id": calc_id}).is_error
    assert "VALIDATED" in theirs["submit_reading"].invoke(reading(calc_id, aims_output(energy=ENERGY))).content
    assert len(hypothesis_evidence(world.state(), other)) == 1 and hypothesis_evidence(world.state(), world.hyp) == []
    # Nothing in CEDAR's own records, or in secrets, can be registered.
    config = tmp_path / "config"
    run_directory(config / "run", world.site)
    refused = mine["register_existing_run"].invoke({"directory": str(config / "run"), "description": "keys"})
    assert refused.is_error and "not readable by agents" in refused.content


def test_shared_runs_serve_every_agent_without_sharing_readings(world, tmp_path):
    other = add_hypothesis(world.ledger, world.inv, "hyp-b")
    calc_id = researcher_registers(world, run_directory(tmp_path / "baseline", world.site)).calculation_id
    tools_a, tools_b = agent_tools(world, world.hyp, tmp_path), agent_tools(world, other, tmp_path)
    assert "VALIDATED" in tools_b["submit_reading"].invoke(reading(calc_id, aims_output(energy=ENERGY))).content
    (b_reading,) = world.state().evidence
    overview_a = tools_a["investigation_overview"].invoke({}).content
    assert "Registered runs shared with every agent" in overview_a and calc_id in overview_a
    assert f"{b_reading} for {other}" in overview_a  # seen by every agent, but B's
    assert "VALIDATED" in tools_a["submit_reading"].invoke(reading(calc_id, aims_output(energy=ENERGY))).content
    state = world.state()
    assert [r.evidence_id for r in hypothesis_evidence(state, other)] == [b_reading]
    assert len(hypothesis_evidence(state, world.hyp)) == 1


def test_identical_inputs_to_a_registered_run_it_can_see_are_not_run_again(world, tmp_path):
    directory = run_directory(tmp_path / "already", world.site)
    calc_id = researcher_registers(world, directory).calculation_id
    control = world.artifacts.put_bytes((directory / "control.in").read_bytes(), label="control.in")
    structure = world.artifacts.put_bytes((directory / "geometry.in").read_bytes(), label="geometry.in")
    world.structure = structure
    again = world.request(authored={"control.in": control}, version="0.2",
                          numerical={"sc_accuracy_eev": None, "sc_accuracy_etot": None})
    world.engine.examine(world.inv, again)
    exam = world.calc(again).examination
    assert exam.verdict is ExaminationVerdict.REJECTED
    (finding,) = [f for f in exam.findings if f.code == "duplicate_inputs"]
    assert f"identical to registered run {calc_id}" in finding.message


# -- the researcher's /register, the Chief Scientist's tool, and the notebook ---------------------------

def test_the_researcher_registers_shared_runs_from_the_chat(tmp_path, site):
    system = make_system(tmp_path, site)
    chat = investigation(system, "hyp-a")
    directory = run_directory(tmp_path / "earlier" / "fcc", site)
    said = chat.handle(f'/register {directory} "my PBE fcc run, the one that disagrees"')
    assert said.startswith("Registered (msg-1); every agent can read and analyse it"), said
    state = system["workspace"].ledger.load(INV)
    (calc,) = state.calculations.values()
    assert calc.shared and calc.registration.description == "my PBE fcc run, the one that disagrees"
    assert calc.registration.intervention_id == "msg-1" and state.interventions["msg-1"].text.startswith("/register")
    assert chat.handle(f"/register {directory} again").startswith("That run is already registered")
    assert "holds neither an FHI-aims run (control.in, geometry.in) nor a VASP run" in chat.handle(
        f"/register {tmp_path}")
    refused = chat.handle(f"/register {run_directory(config_dir() / 'run', site)} keys")
    assert "configuration directory" in refused

    # The Chief Scientist registers runs too, shared or for one hypothesis.
    other = run_directory(tmp_path / "earlier" / "atop", site)
    system["clients"].add("chief_scientist", reply_registering(other), says("Registered the atop run."))
    chat.handle("Please register my atop run for hyp-a.")
    state = system["workspace"].ledger.load(INV)
    atop = next(c for c in state.calculations.values() if c.registration.source == str(other))
    assert atop.hypothesis_ids == ["hyp-a"] and atop.registration.requested_by.kind.value == "chief_scientist"

    pages = render_notebook(system["workspace"].store.read_stream(INV))
    assert f"| {calc.calculation_id} | every agent | " in pages["index.md"]
    assert "registered for every agent from" in pages["timeline.md"]
    assert f"**{atop.calculation_id}** for" in pages["hypotheses/hyp-a.md"]
    assert "computed outside CEDAR" in pages["hypotheses/hyp-a.md"]


def reply_registering(directory):
    return reply(call("register_existing_run", directory=str(directory), description="the atop run",
                      hypothesis_id="hyp-a"))
