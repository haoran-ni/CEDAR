"""Analysis scripts: examination, guarded execution on staged inputs, and results as evidence.

Scripts really run here, in a child Python process, against SYNTHETIC outputs from the
FakeScheduler TEST DOUBLE.
"""

import json

import pytest
from factories import AGENT
from sites import aims_output
from worlds import World

from sim2real.contracts import AnalysisOutcome, EvidenceKind, EvidenceQuality, Observable, SchedulerState
from sim2real.engine.analysis import AnalysisLimits, AnalysisRunner, examine_script
from sim2real.engine.results import ANALYSIS_LIMITATION
from sim2real.errors import RuleViolation
from sim2real.state import replay

ENERGY = -1234.567890123
EIGENVALUES = "# state occupation eigenvalue_eV\n1 2.0 -9.1\n2 2.0 -8.4\n3 0.0 -1.9\n4 0.0 -1.9\n"
GAP_SCRIPT = """
import json, re
rows = [line.split() for line in open("inputs/CALC/eigenvalues.dat") if not line.startswith("#")]
occupied = [float(r[2]) for r in rows if float(r[1]) > 0]
empty = [float(r[2]) for r in rows if float(r[1]) == 0]
electrons = sum(float(r[1]) for r in rows)
energy = None
for line in open("inputs/CALC/aims.out"):
    if "Total energy of the DFT" in line:
        energy = float(re.findall(r"[-+]?\\d+\\.\\d+", line)[-1])
print("HOMO", max(occupied), "LUMO", min(empty), "energy", energy, "electrons", electrons)
lumo_states = empty.count(min(empty))
json.dump({
    "results": {
        "gap": {"value": min(empty) - max(occupied), "unit": "eV", "description": "LUMO minus HOMO",
                "checks": {"lumo_degeneracy": {"passed": lumo_states == LUMO_STATES,
                                               "detail": f"{lumo_states} states at the LUMO"}}},
        "total_energy": {"value": energy, "unit": "eV"},
    },
    "checks": {"electron_count": {"passed": electrons == ELECTRONS, "detail": f"{electrons} electrons"}},
}, open("results.json", "w"))
open("notes.txt", "w").write("done")
"""


def gap_script(calc: str, electrons: float = 4.0, lumo_states: int = 2) -> str:
    """The gap script for ``calc``, expecting this many electrons and LUMO states (the file has 4 and 2)."""
    return (GAP_SCRIPT.replace("CALC", calc).replace("ELECTRONS", repr(electrons))
            .replace("LUMO_STATES", str(lumo_states)))


def results_json(results: dict | None = None, checks: dict | None = None) -> str:
    """A script that writes the given results and checks to results.json."""
    content = {key: value for key, value in (("results", results), ("checks", checks)) if value is not None}
    return f"import json\njson.dump({content!r}, open('results.json', 'w'))\n"


@pytest.fixture
def world(ledger, store, artifacts, site):
    return World(ledger, store, artifacts, site)


@pytest.fixture
def checked(world):
    cid = world.request()
    world.checked(cid, output=aims_output(energy=ENERGY), extra={"eigenvalues.dat": EIGENVALUES})
    return cid


def run(world, script, inputs, **kwargs):
    return world.analysis.run(world.inv, world.hyp, purpose="test analysis", script=script, inputs=inputs, **kwargs)


@pytest.mark.parametrize("script, problem", [
    ("import subprocess\nsubprocess.run(['true'])", "imports subprocess"),
    ("from urllib.request import urlopen", "imports from urllib.request"),
    ("import os\nos.system('true')", "uses os.system"),
    ("import os as o\no.execv('/bin/true', [])", "uses os.execv"),
    ("from os import popen", "imports os.popen"),
    ("import ctypes", "imports ctypes"),
    ("eval('1+1')", "uses eval"),
    ("def f(:\n  pass", "line 1"),
])
def test_scripts_that_start_programs_or_load_code_are_refused(script, problem):
    problems = examine_script(script)
    assert problems and problem in problems[0]


def test_ordinary_analysis_scripts_pass_examination():
    script = ("import json, math, os, re\nimport numpy as np\nfrom pathlib import Path\n"
              "print(os.path.join('a', 'b'), np.mean([1, 2]))\njson.dump({}, open('results.json', 'w'))\n")
    assert examine_script(script) == []


def test_a_script_runs_on_staged_inputs_and_its_results_are_recorded(world, checked):
    report = run(world, gap_script(checked), {checked: ["aims.out", "eigenvalues.dat"]})
    assert report.problems == () and report.analysis_id is not None
    run_ = report.run
    assert run_.outcome is AnalysisOutcome.COMPLETED and run_.exit_code == 0
    assert {r.name: r.value.value for r in run_.results} == {"gap": pytest.approx(6.5), "total_energy": ENERGY}
    assert "HOMO -8.4 LUMO -1.9" in report.shown_output
    assert set(run_.products) == {"results.json", "notes.txt"}
    (staged,) = run_.inputs
    stored = world.calc(checked).run_check.outputs["aims.out"]
    assert staged.files["aims.out"] == stored  # the checked output, by hash
    assert world.artifacts.read(staged.files["eigenvalues.dat"]) == EIGENVALUES.encode()
    assert world.artifacts.read(run_.script).decode() == gap_script(checked)
    assert [(c.name, c.passed) for c in run_.checks] == [("electron_count", True)]
    assert [(c.name, c.passed) for c in run_.result("gap").checks] == [("lumo_degeneracy", True)]
    state = world.state()
    assert state.analyses[report.analysis_id].hypothesis_id == world.hyp
    assert replay(world.store.read_stream(world.inv)) == state
    assert not (world.runs.parent / "analyses" / report.analysis_id).exists()  # the working copy is removed


def test_the_guards_keep_scripts_to_their_inputs(world, checked, tmp_path, monkeypatch):
    secret = tmp_path / "home" / ".config" / "secret.txt"
    secret.parent.mkdir(parents=True)
    secret.write_text("not for scripts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-reach-a-script")
    script = f"""
import os
attempts = {{
    "read outside": lambda: open({str(secret)!r}).read(),
    "read another run": lambda: open({str(world.runs)!r} + "/x").read(),
    "write outside": lambda: open({str(tmp_path / 'dropped.txt')!r}, "w"),
    "list outside": lambda: os.listdir({str(tmp_path)!r}),
    "start a program": lambda: getattr(os, "sys" + "tem")("true"),
    "delete outside": lambda: os.remove({str(secret)!r}),
}}
for name, attempt in attempts.items():
    try:
        attempt()
        print("ALLOWED", name)
    except PermissionError as exc:
        print("refused", name)
print("key visible:", any("sk-ant" in value for value in os.environ.values()))
"""
    report = run(world, script, {checked: None})
    assert report.run.outcome is AnalysisOutcome.COMPLETED, report.shown_output
    assert "ALLOWED" not in report.shown_output and report.shown_output.count("refused") == 6
    assert "key visible: False" in report.shown_output
    assert secret.read_text() == "not for scripts" and not (tmp_path / "dropped.txt").exists()


def test_scientific_libraries_work_under_the_guards(world, checked):
    script = (f"import numpy as np, ase.io\natoms = ase.io.read('inputs/{checked}/geometry.in', format='aims')\n"
              "d = atoms.get_distance(-2, -1)\nprint('atoms', len(atoms))\n"
              "import json\njson.dump({'results': {'last_bond': {'value': float(d), 'unit': 'Angstrom'}}}, "
              "open('results.json', 'w'))\n")
    report = run(world, script, {checked: None})
    assert report.run.outcome is AnalysisOutcome.COMPLETED, report.shown_output
    assert report.run.results[0].value.unit == "Angstrom" and report.run.results[0].value.value > 0.5


def test_failures_timeouts_and_bad_results_are_recorded_without_results(world, checked):
    failed = run(world, results_json({"x": {"value": 1, "unit": "eV"}}) + "raise SystemExit('stopped on purpose')",
                 {checked: None})
    assert failed.run.outcome is AnalysisOutcome.FAILED and failed.run.results == ()
    assert "stopped on purpose" in failed.shown_output

    bad = run(world, results_json({"ok": {"value": 1.5, "unit": "eV"}, "nan": {"value": "x", "unit": "eV"},
                                   "bad name!": {"value": 1, "unit": "eV"}}), {checked: None})
    assert [r.name for r in bad.run.results] == ["ok"] and len(bad.run.problems) == 2

    quick = AnalysisRunner(world.ledger, world.artifacts, world.runs, world.runs.parent / "analyses",
                           limits=AnalysisLimits(wall_seconds=2))
    slow = quick.run(world.inv, world.hyp, purpose="spin", script="while True:\n    pass\n", inputs={checked: None})
    assert slow.run.outcome is AnalysisOutcome.TIMED_OUT and "did not finish within 2 s" in slow.problems[0]
    assert len(world.state().analyses) == 3


def test_scripts_are_refused_before_anything_runs_or_is_recorded(world, checked):
    assert run(world, "import socket", {checked: None}).analysis_id is None
    unchecked = world.request(interventions={"xc": "pbe0"})
    report = run(world, "print(1)", {unchecked: None})
    assert report.analysis_id is None and "has not been checked" in report.problems[0]
    missing = run(world, "print(1)", {checked: ["../../etc/passwd"]})
    assert missing.analysis_id is None and "relative to the run directory" in missing.problems[0]
    assert world.state().analyses == {}


TOTAL = Observable(name="total energy", definition="final total energy printed by FHI-aims", unit="eV")
GAP = Observable(name="Kohn-Sham gap", definition="lowest empty minus highest occupied eigenvalue")


class TestResultsAsEvidence:
    def test_a_reported_result_becomes_validated_evidence(self, world, checked):
        report = run(world, gap_script(checked), {checked: ["aims.out", "eigenvalues.dat"]})
        outcome = world.validator.record_analysis_result(world.inv, report.analysis_id, "gap", observable=GAP,
                                                         summary="gap from the eigenvalue table", reader=AGENT)
        state = outcome.recorded.state
        (evidence_id,) = state.calculations[checked].evidence_ids
        ev = state.evidence[evidence_id].evidence
        assert ev.kind is EvidenceKind.ANALYSIS and ev.quality is EvidenceQuality.VALIDATED
        assert ev.value.value == pytest.approx(6.5) and ev.analysis_id == report.analysis_id
        assert ev.limitations[0] == ANALYSIS_LIMITATION
        assert report.run.script in ev.sources and ev.interpreted_by == AGENT
        assert ev.analysis_result == "gap"
        assert {c.name for c in ev.quality_checks} == {
            "analysis_completed", f"run_healthy:{checked}", "script:electron_count", "script:lumo_degeneracy"}

    def test_unknown_results_and_mismatched_units_are_refused(self, world, checked):
        report = run(world, gap_script(checked), {checked: ["aims.out", "eigenvalues.dat"]})
        actor = AGENT
        missing = world.validator.record_analysis_result(world.inv, report.analysis_id, "work_function",
                                                         observable=TOTAL, summary="s", reader=actor)
        assert missing.recorded is None and "reported no result 'work_function'" in missing.problems[0]
        wrong_unit = TOTAL.model_copy(update={"unit": "Ha"})
        mismatch = world.validator.record_analysis_result(world.inv, report.analysis_id, "total_energy",
                                                          observable=wrong_unit, summary="s", reader=actor)
        assert mismatch.recorded is None and "unit" in mismatch.problems[0]
        with pytest.raises(RuleViolation, match="unknown analysis"):
            world.validator.record_analysis_result(world.inv, "an_missing", "gap", observable=GAP, summary="s",
                                                   reader=actor)

    def test_results_from_an_unhealthy_run_are_invalid(self, world):
        cid = world.request()
        world.checked(cid, output=aims_output(energy=ENERGY), final=SchedulerState.FAILED)
        report = run(world, results_json({"n": {"value": 1.0, "unit": "eV"}}), {cid: None})
        state = world.validator.record_analysis_result(world.inv, report.analysis_id, "n", observable=TOTAL,
                                                       summary="s", reader=AGENT).recorded.state
        (record,) = state.evidence.values()
        assert record.evidence.quality is EvidenceQuality.INVALID
        assert json.loads(world.artifacts.read(report.run.products["results.json"])) == {
            "results": {"n": {"value": 1.0, "unit": "eV"}}}

    def test_a_failed_check_leaves_the_results_it_applies_to_inconclusive(self, world, checked):
        def record(report, name):
            outcome = world.validator.record_analysis_result(world.inv, report.analysis_id, name, observable=GAP,
                                                             summary="s", reader=AGENT)
            (event,) = outcome.recorded.events
            return event.payload.evidence

        wrong_count = run(world, gap_script(checked, electrons=6.0), {checked: ["aims.out", "eigenvalues.dat"]})
        assert not wrong_count.run.checks[0].passed
        assert record(wrong_count, "gap").quality is EvidenceQuality.INCONCLUSIVE
        assert record(wrong_count, "total_energy").quality is EvidenceQuality.INCONCLUSIVE  # applies to every result

        not_degenerate = run(world, gap_script(checked, lumo_states=1), {checked: ["aims.out", "eigenvalues.dat"]})
        gap = record(not_degenerate, "gap")
        assert gap.quality is EvidenceQuality.INCONCLUSIVE
        assert next(c for c in gap.quality_checks if c.name == "script:lumo_degeneracy").detail == "2 states at the LUMO"
        assert record(not_degenerate, "total_energy").quality is EvidenceQuality.VALIDATED  # the gap's check only


def test_results_are_withheld_when_their_checks_are_malformed(world, checked):
    value = {"value": 1.0, "unit": "eV"}
    bad_shared = run(world, results_json({"a": value}, {"count": {"passed": "yes"}}), {checked: None})
    assert bad_shared.run.results == () and "check 'count' is malformed" in bad_shared.problems[0]
    bad_own = run(world, results_json({"a": {**value, "checks": {"x": {"detail": "no verdict"}}}, "b": value}),
                  {checked: None})
    assert [r.name for r in bad_own.run.results] == ["b"]
    assert any("result 'a' not taken: one of its checks is malformed" in p for p in bad_own.problems)
    flat = run(world, "import json\njson.dump({'a': {'value': 1.0, 'unit': 'eV'}}, open('results.json', 'w'))",
               {checked: None})
    assert flat.run.results == () and flat.problems[0].startswith('results.json must be {"results"')
