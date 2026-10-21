import re

import pytest
from ase.build import molecule
from sites import aims_output, co_on_cu111, declared, geometry_text

from cedar.contracts import (
    ExaminationVerdict,
    FindingSeverity,
    Observable,
    ProtocolRef,
    SchedulerState,
    ScientificTaskSpec,
    verdict_for,
)
from cedar.contracts.status import EvidenceQuality
from cedar.protocols import FhiAimsRun

PROTOCOL = ProtocolRef(name="fhiaims.run", version="0.1")
WORK_FUNCTION = Observable(name="work function", definition="vacuum level minus Fermi level", unit="eV")


def task(observable=WORK_FUNCTION, protocol=PROTOCOL, **overrides) -> ScientificTaskSpec:
    return ScientificTaskSpec(protocol=protocol, observable=observable, estimand="work function of CO/Cu(111) atop",
                              system_label="CO/Cu(111) atop (synthetic test)", **declared(**overrides))


@pytest.fixture
def protocol(site):
    return FhiAimsRun(site)


@pytest.fixture
def slab_geometry():
    slab, bottom = co_on_cu111()
    return geometry_text(slab, fixed=bottom).encode()


def generated(protocol, spec, geometry):
    files, findings = protocol.generate(spec, {"geometry.in": geometry})
    return {"geometry.in": geometry, **files}, findings


def blocking(findings):
    return {f.code for f in findings if f.severity in (FindingSeverity.ERROR, FindingSeverity.REVIEW)}


def codes(findings, severity):
    return {f.code for f in findings if f.severity is severity}


def test_generated_inputs_pass_their_own_examination(protocol, slab_geometry):
    spec = task()
    files, findings = generated(protocol, spec, slab_geometry)
    assert set(files) == {"geometry.in", "control.in", "job.sh"}
    assert codes(findings, FindingSeverity.INFO) == {"generated"}
    examination = protocol.examine(spec, files)
    assert blocking(examination) == set(), examination
    assert verdict_for(examination) is ExaminationVerdict.APPROVED
    control = files["control.in"].decode()
    for line in ("xc  pbe", "k_grid  4 4 1", "occupation_type  gaussian 0.1", "relativistic  atomic_zora scalar"):
        assert " ".join(line.split()) in {" ".join(l.split()) for l in control.splitlines()}
    assert control.count("SYNTHETIC light test species file") == 3


def test_generation_is_deterministic(protocol, slab_geometry):
    assert generated(protocol, task(), slab_geometry)[0] == generated(protocol, task(), slab_geometry)[0]


class TestDeclarations:
    def test_every_basic_setting_must_be_declared(self, protocol, slab_geometry):
        spec = task(controls={"relativistic": None}, numerical={"occupation_type": None})
        findings = protocol.examine(spec, {"geometry.in": slab_geometry})
        missing = [f.message for f in findings if f.code == "missing_variable"]
        assert any("relativistic" in m for m in missing) and any("occupation_type" in m for m in missing)

    def test_numerical_settings_cannot_pose_as_controls(self, protocol, slab_geometry):
        spec = task(numerical={"k_grid": None}, controls={"k_grid": [4, 4, 1]})
        assert "misclassified_variable" in codes(protocol.examine(spec, {"geometry.in": slab_geometry}),
                                                 FindingSeverity.ERROR)

    def test_physics_cannot_hide_among_numerical_settings(self, protocol, slab_geometry):
        spec = task(numerical={"vdw_ts": True})
        assert "misclassified_variable" in codes(protocol.examine(spec, {"geometry.in": slab_geometry}),
                                                 FindingSeverity.ERROR)

    def test_unknown_variables_cannot_be_applied(self, protocol, slab_geometry):
        spec = task(controls={"temperature": "300 K"})
        assert "unknown_variable" in codes(protocol.examine(spec, {"geometry.in": slab_geometry}),
                                           FindingSeverity.ERROR)

    def test_species_level_must_exist_in_the_profile(self, protocol, slab_geometry):
        spec = task(numerical={"species_defaults": "../light"})
        assert "unknown_species_level" in codes(protocol.examine(spec, {"geometry.in": slab_geometry}),
                                                FindingSeverity.ERROR)


class TestAgentWrittenControl:
    """Agents may write control.in themselves; the examiner ties it to the declared task."""

    @pytest.fixture
    def files(self, protocol, slab_geometry):
        return generated(protocol, task(), slab_geometry)[0]

    def edit(self, files, old, new):
        """Replace the keyword line ``old`` (matched whitespace-insensitively) or literal text."""
        control = files["control.in"].decode()
        pattern = re.compile(r"^" + r"\s+".join(map(re.escape, old.split())) + r"[ \t]*$", re.MULTILINE)
        edited, count = pattern.subn(new, control, count=1)
        if not count:
            assert old in control
            edited = control.replace(old, new, 1)
        return {**files, "control.in": edited.encode()}

    def test_contradicting_the_declaration_is_rejected(self, protocol, files):
        findings = protocol.examine(task(), self.edit(files, "xc  pbe", "xc  pbe0"))
        assert "contradicts_task" in codes(findings, FindingSeverity.ERROR)

    def test_undeclared_physics_is_rejected(self, protocol, files):
        findings = protocol.examine(task(), self.edit(files, "xc  pbe", "xc  pbe\nvdw_ts"))
        assert "undeclared_setting" in codes(findings, FindingSeverity.ERROR)

    def test_declared_physics_must_appear(self, protocol, files):
        findings = protocol.examine(task(controls={"vdw_ts": True}), files)
        assert "missing_keyword" in codes(findings, FindingSeverity.ERROR)

    def test_hiding_scf_failure_is_forbidden(self, protocol, files):
        findings = protocol.examine(task(), self.edit(files, "xc  pbe", "xc  pbe\npostprocess_anyway .true."))
        assert "forbidden_keyword" in codes(findings, FindingSeverity.ERROR)

    def test_unknown_keywords_need_review(self, protocol, files):
        findings = protocol.examine(task(), self.edit(files, "xc  pbe", "xc  pbe\nsome_new_keyword 3"))
        assert blocking(findings) == {"unknown_keyword"}
        assert verdict_for(findings) is ExaminationVerdict.NEEDS_REVIEW

    def test_convergence_aids_need_no_declaration(self, protocol, files):
        edited = self.edit(files, "xc  pbe", "xc  pbe\nmixer pulay\ncharge_mix_param 0.05\nsc_iter_limit 200")
        assert blocking(protocol.examine(task(), edited)) == set()

    def test_numbers_compare_by_value(self, protocol, files):
        assert blocking(protocol.examine(task(), self.edit(files, "sc_accuracy_rho  1e-05", "sc_accuracy_rho 1.0d-5"))) == set()

    def test_modified_species_block_needs_review(self, protocol, files):
        findings = protocol.examine(task(), self.edit(files, "#     hydro 4 f 7.4", "     hydro 4 f 7.4"))
        assert "modified_species" in codes(findings, FindingSeverity.REVIEW)

    def test_missing_species_block_is_rejected(self, protocol, files):
        control = files["control.in"].decode()
        without_oxygen = control[:control.index("################################################################################\n#  SYNTHETIC light test species file for O")]
        findings = protocol.examine(task(), {**files, "control.in": without_oxygen.encode()})
        assert "missing_species" in codes(findings, FindingSeverity.ERROR)


class TestGeometryChecks:
    def test_k_points_across_vacuum_are_rejected(self, protocol, slab_geometry):
        findings = protocol.examine(task(numerical={"k_grid": [4, 4, 4]}), {"geometry.in": slab_geometry})
        assert "k_points_across_vacuum" in codes(findings, FindingSeverity.ERROR)

    def test_overlapping_atoms_are_rejected(self, protocol):
        text = "lattice_vector 10 0 0\nlattice_vector 0 10 0\nlattice_vector 0 0 10\natom 0 0 0 C\natom 0 0 0.2 O\n"
        findings = protocol.examine(task(numerical={"k_grid": [1, 1, 1]}), {"geometry.in": text.encode()})
        assert "overlapping_atoms" in codes(findings, FindingSeverity.ERROR)

    def test_molecule_without_cell_must_not_declare_k_points(self, protocol):
        geometry = geometry_text(molecule("CO")).encode()
        assert "k_points_without_cell" in codes(protocol.examine(task(), {"geometry.in": geometry}),
                                                FindingSeverity.ERROR)
        files, _ = generated(protocol, task(numerical={"k_grid": None}), geometry)
        assert blocking(protocol.examine(task(numerical={"k_grid": None}), files)) == set()

    def test_moments_need_spin(self, protocol):
        slab, bottom = co_on_cu111()
        geometry = geometry_text(slab, fixed=bottom, moments={0: 1.0}).encode()
        assert "moments_without_spin" in codes(protocol.examine(task(), {"geometry.in": geometry}),
                                               FindingSeverity.ERROR)

    def test_relaxing_slab_without_constraints_is_flagged(self, protocol):
        slab, _ = co_on_cu111()
        spec = task(controls={"relax_geometry": "bfgs 0.01"})
        findings = protocol.examine(spec, {"geometry.in": geometry_text(slab).encode()})
        assert "unconstrained_slab" in codes(findings, FindingSeverity.WARNING)


def test_budget_limits_the_job_script(protocol, slab_geometry):
    spec = task().model_copy(update={"budget": task().budget.model_copy(update={"max_core_hours": 1.0})})
    files, _ = generated(protocol, spec, slab_geometry)
    assert b"--time=00:01:00" in files["job.sh"]
    hungry = {**files, "job.sh": files["job.sh"].replace(b"--time=00:01:00", b"--time=02:00:00")}
    assert "over_budget" in codes(protocol.examine(spec, hungry), FindingSeverity.ERROR)


def test_the_agent_declares_the_observable(protocol):
    assert "Observable: declare your own" in protocol.describe()


def test_profile_without_fhi_aims_cannot_host_the_protocol(site):
    with pytest.raises(ValueError, match="no 'fhi-aims' installation"):
        FhiAimsRun(site.model_copy(update={"codes": {"other": site.codes["fhi-aims"]}}))


class TestRunChecks:
    def check(self, protocol, text, state=SchedulerState.COMPLETED, **overrides):
        outputs = {} if text is None else {"aims.out": text.encode()}
        return protocol.check_run(task(**overrides), outputs, state)

    def test_healthy_run_reads_the_final_energies_with_their_lines(self, protocol):
        report = self.check(protocol, aims_output(energy=-42.123456789))
        assert all(c.passed for c in report.checks)
        total, zero_broadening = report.readings
        assert (total.name, total.value.value, total.file) == ("total energy", -42.123456789, "aims.out")
        assert "Total energy of the DFT" in total.quote
        assert zero_broadening.name == "zero-broadening total energy"
        assert zero_broadening.value.value == pytest.approx(-42.123843789)
        assert report.method["code_version"] == "test-1.0" and report.method["xc"] == "pbe"

    @pytest.mark.parametrize("text, state, failed", [
        (aims_output(version="other"), SchedulerState.COMPLETED, {"code_version"}),
        (aims_output(normal_end=False), SchedulerState.TIMEOUT, {"scheduler_completed", "normal_termination"}),
        (aims_output(scf_failure=True), SchedulerState.FAILED, {"scheduler_completed", "normal_termination", "scf_converged"}),
        (None, SchedulerState.FAILED, {"scheduler_completed", "output_present"}),
    ])
    def test_failures_are_named_and_invalid(self, protocol, text, state, failed):
        report = self.check(protocol, text, state)
        assert {c.name for c in report.checks if not c.passed} == failed
        assert report.failure_quality is EvidenceQuality.INVALID

    def test_unconverged_relaxation_is_inconclusive(self, protocol):
        report = self.check(protocol, aims_output(relax=True, geometry_converged=False),
                            controls={"relax_geometry": "bfgs 0.01"})
        assert {c.name for c in report.checks if not c.passed} == {"geometry_converged"}
        assert report.failure_quality is EvidenceQuality.INCONCLUSIVE
