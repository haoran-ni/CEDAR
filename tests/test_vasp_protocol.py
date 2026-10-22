"""``vasp.run`` 0.1: what the examination generates, rules on and records for the input reviewer, how a run's health is
checked, and VASP runs registered from outside CEDAR.

SYNTHETIC site, inputs and outputs (see sites.py); the FakeScheduler TEST DOUBLE runs nothing.
"""

import pytest
from factories import CHIEF, agent_for
from sites import (
    VASP_VERSION,
    add_vasp,
    co_on_cu111,
    dataset_text,
    incar_text,
    kpoints_text,
    poscar_text,
    vasp_outcar,
)
from worlds import World

from cedar.contracts import (
    CalculationState,
    ExaminationVerdict,
    FindingSeverity,
    ProtocolRef,
    SchedulerState,
    ScientificTaskSpec,
    verdict_for,
)
from cedar.contracts.status import EvidenceQuality
from cedar.engine import RunRegistrar
from cedar.engine.inputs import DUPLICATE_INPUTS
from cedar.hpc.jobscript import render_job_script
from cedar.hpc.profile import CodeInstallation, ComputeProfile, check_profile
from cedar.protocols import FREE_ENERGY, SIGMA0_ENERGY, VaspRun
from cedar.workspace import registry_for

DATASETS = ["Cu", "C", "O"]


@pytest.fixture
def vasp_site(site, tmp_path):
    return add_vasp(site, tmp_path / "site")


@pytest.fixture
def protocol(vasp_site):
    return VaspRun(vasp_site)


@pytest.fixture
def slab_poscar():
    slab, bottom = co_on_cu111()
    return poscar_text(slab, fixed=bottom).encode()


def task(protocol=VaspRun.ref, potcar=DATASETS, **update) -> ScientificTaskSpec:
    controls = {"xc": "PBE"} | ({"potcar": potcar} if potcar is not None else {})
    return ScientificTaskSpec(protocol=protocol, observable=FREE_ENERGY, estimand="free energy of CO/Cu(111) atop",
                              system_label="CO/Cu(111) atop (synthetic)", controls=controls,
                              numerical_settings={"ENCUT": 400, "k-points": [4, 4, 1]}).model_copy(update=update)


def inputs(poscar: bytes, **incar) -> dict[str, bytes]:
    return {"POSCAR": poscar, "INCAR": incar_text(**incar).encode(), "KPOINTS": kpoints_text().encode()}


def complete(protocol, spec, files):
    generated, findings = protocol.generate(spec, files)
    return {**files, **generated}, findings


def codes(findings, severity=None):
    return {f.code for f in findings if severity is None or f.severity is severity}


def messages(findings, code):
    return [f.message for f in findings if f.code == code]


# -- generation and examination --------------------------------------------------------------------


def test_generated_inputs_pass_their_own_examination(protocol, vasp_site, slab_poscar):
    spec = task()
    files, findings = complete(protocol, spec, inputs(slab_poscar))
    assert set(files) == {"POSCAR", "INCAR", "KPOINTS", "POTCAR", "job.sh"}
    assert codes(findings) == {"generated"}
    vasp = vasp_site.codes["vasp"]
    script = files["job.sh"].decode()
    assert "vasp_env.sh" in script and f"{vasp.binary} > vasp.out" in script  # VASP's own template
    examination = protocol.examine(spec, files)
    assert verdict_for(examination) is ExaminationVerdict.APPROVED, examination
    assert codes(examination, FindingSeverity.WARNING) == set()
    library = vasp.data_paths["potcars"]
    assert messages(examination, "potcar_dataset") == [
        f"PAW_PBE {name} 01Jan2000, ZVAL {z}: equals {library}/{name}/POTCAR" for name, z in (("Cu", 11), ("C", 4),
                                                                                             ("O", 6))]
    (structure,) = messages(examination, "structure")
    assert structure.startswith("14 atoms (12 Cu, 1 C, 1 O); 4 with directions fixed by selective dynamics")
    assert "Å of empty space along the third lattice vector" in structure
    assert messages(examination, "build") == [f"launches the standard build, {vasp.binary}"]
    settings = protocol.input_settings(files)
    assert settings["ENCUT"] == "400" and settings["KPOINTS"] == "Γ-centred 4×4×1 mesh"
    assert settings["POTCAR"] == "PAW_PBE Cu 01Jan2000; PAW_PBE C 01Jan2000; PAW_PBE O 01Jan2000"


def test_the_fhi_aims_template_stays_the_profiles(vasp_site):
    from cedar.protocols import FhiAimsRunJobChecked

    files, _ = FhiAimsRunJobChecked(vasp_site).generate(task(protocol=ProtocolRef(name="fhiaims.run", version="0.4")),
                                                        {})
    assert "vasp_env" not in files["job.sh"].decode() and "aims.out" in files["job.sh"].decode()


def test_non_collinear_inputs_run_on_the_ncl_build(protocol, vasp_site, slab_poscar):
    vasp = vasp_site.codes["vasp"]
    files, _ = complete(protocol, task(), inputs(slab_poscar, LSORBIT=".TRUE."))
    assert f"{vasp.variants['ncl']} > vasp.out" in files["job.sh"].decode()
    assert verdict_for(protocol.examine(task(), files)) is ExaminationVerdict.APPROVED
    collinear, _ = complete(protocol, task(), inputs(slab_poscar))
    files["job.sh"] = collinear["job.sh"]  # the agent's own script, launching the standard build
    findings = protocol.examine(task(), files)
    assert messages(findings, "wrong_build") == [f"INCAR sets LSORBIT, which needs the ncl build "
                                                 f"({vasp.variants['ncl']}); the standard build stops on non-collinear "
                                                 "input"]


def test_without_an_ncl_build_a_non_collinear_run_is_only_flagged(site, tmp_path, slab_poscar):
    protocol = VaspRun(add_vasp(site, tmp_path / "site", variants=False))
    files, _ = complete(protocol, task(), inputs(slab_poscar, LNONCOLLINEAR=".TRUE."))
    findings = protocol.examine(task(), files)
    assert "no_ncl_build" in codes(findings, FindingSeverity.WARNING) and verdict_for(findings) is not \
        ExaminationVerdict.REJECTED


def test_the_gamma_build_is_flagged_for_a_mesh(protocol, vasp_site, slab_poscar):
    vasp = vasp_site.codes["vasp"]
    files, _ = complete(protocol, task(), inputs(slab_poscar))
    resources = protocol.requested_resources(files)
    files["job.sh"] = render_job_script(vasp_site, vasp, resources, job_name="cedar", output_file="vasp.out",
                                        binary=vasp.variants["gam"]).encode()
    assert "gamma_build" in codes(protocol.examine(task(), files), FindingSeverity.WARNING)
    files["KPOINTS"] = kpoints_text((1, 1, 1)).encode()
    findings = protocol.examine(task(), files)
    assert "gamma_build" not in codes(findings) and messages(findings, "build")[0].startswith("launches the gam build")


class TestPotcar:
    def test_potcar_must_hold_poscars_species_in_order(self, protocol, slab_poscar):
        files, _ = complete(protocol, task(potcar=["C", "Cu", "O"]), inputs(slab_poscar))
        mismatches = messages(protocol.examine(task(), files), "potcar_mismatch")
        assert mismatches[0] == ("species 1 of POSCAR is Cu, but dataset 1 of POTCAR is PAW_PBE C 01Jan2000: VASP "
                                 "would put the C potential on the Cu atoms")
        files, _ = complete(protocol, task(potcar=["Cu", "C"]), inputs(slab_poscar))
        assert messages(protocol.examine(task(), files), "potcar_mismatch") == [
            "POTCAR has 2 datasets (Cu, C) but POSCAR 3 species; VASP needs one dataset per species, in POSCAR's order"]

    def test_a_poscar_naming_no_species_takes_them_from_potcar(self, protocol):
        slab, _ = co_on_cu111()
        files, _ = complete(protocol, task(), inputs(poscar_text(slab, names=False).encode()))
        findings = protocol.examine(task(), files)
        assert messages(findings, "species_from_potcar") == ["POSCAR does not name its species; VASP takes them from "
                                                             "POTCAR: Cu, C, O"]
        assert verdict_for(findings) is ExaminationVerdict.APPROVED

    def test_potcar_is_given_or_its_datasets_named(self, protocol, slab_poscar):
        files, _ = complete(protocol, task(potcar=None), inputs(slab_poscar))
        (missing,) = [f for f in protocol.examine(task(potcar=None), files) if f.code == "missing_file"]
        assert missing.file == "POTCAR" and "declare potcar, the PAW datasets of POSCAR's species" in missing.message
        _, findings = protocol.generate(task(potcar=["Cu_sv", "C", "O"]), inputs(slab_poscar))
        (unknown,) = messages(findings, "unknown_dataset")
        assert unknown.endswith("has no dataset Cu_sv; its datasets for Cu: Cu, Cu_pv")
        _, findings = protocol.generate(task(potcar=3), inputs(slab_poscar))
        assert "invalid_value" in codes(findings, FindingSeverity.ERROR)

    def test_a_potcar_of_the_agents_own_is_compared_with_the_library(self, protocol, slab_poscar):
        given = (dataset_text("Cu_pv") + dataset_text("C") + dataset_text("O").replace("synthetic", "edited")).encode()
        files, _ = complete(protocol, task(potcar=None), {**inputs(slab_poscar), "POTCAR": given})
        facts = messages(protocol.examine(task(potcar=None), files), "potcar_dataset")
        assert facts[0].endswith("Cu_pv/POTCAR") and "equals" in facts[0] and "differs from" in facts[2]


class TestIncarFacts:
    def test_tags_vasp_would_not_read_are_pointed_out(self, protocol, slab_poscar):
        files, _ = complete(protocol, task(), inputs(slab_poscar, ENCUT=None, ENCUTT=400))
        files["INCAR"] += b"Electronic relaxation\nEDIFFG -0.01\n"
        findings = protocol.examine(task(), files)
        (unread,) = messages(findings, "unread_tag")
        assert unread.startswith("ENCUTT is named nowhere in the installed VASP's source")
        heading, missing = [f for f in findings if f.code == "ignored_line"]
        assert heading.severity is FindingSeverity.INFO and heading.message.endswith("Electronic relaxation")
        assert missing.severity is FindingSeverity.WARNING and missing.message == (
            "VASP ignores this line, which sets no tag: EDIFFG -0.01; EDIFFG is a tag it reads, but only after '='")
        assert verdict_for(findings) is ExaminationVerdict.APPROVED  # facts for the reviewer, not rules

    def test_without_the_source_unread_tags_are_left_to_the_reviewer(self, site, tmp_path, slab_poscar):
        protocol = VaspRun(add_vasp(site, tmp_path / "site", source=False))
        files, _ = complete(protocol, task(), inputs(slab_poscar, ENCUTT=400))
        assert "unread_tag" not in codes(protocol.examine(task(), files))

    def test_kspacing_may_replace_kpoints(self, protocol, slab_poscar):
        files, _ = complete(protocol, task(), inputs(slab_poscar, KSPACING=0.3))
        del files["KPOINTS"]
        findings = protocol.examine(task(), files)
        assert messages(findings, "kpoints") == ["no KPOINTS: VASP generates the mesh from KSPACING (INCAR sets 0.3)"]
        assert verdict_for(findings) is ExaminationVerdict.APPROVED

    def test_restart_files_vasp_would_rewrite_are_flagged(self, protocol, slab_poscar):
        files, _ = complete(protocol, task(), inputs(slab_poscar, LWAVE=None))
        files["WAVECAR"] = b"\x00binary wavefunctions"
        findings = protocol.examine(task(), files)
        assert codes(findings, FindingSeverity.WARNING) == {"rewritten_input"}
        assert "other_file" in codes(findings) and verdict_for(findings) is ExaminationVerdict.APPROVED
        files["INCAR"] = incar_text().encode()  # LWAVE = .FALSE.
        assert "rewritten_input" not in codes(protocol.examine(task(), files))

    def test_an_input_cannot_take_an_outputs_name(self, protocol, slab_poscar):
        files, _ = complete(protocol, task(), inputs(slab_poscar))
        files["OUTCAR"] = b"old"
        assert "output_name" in codes(protocol.examine(task(), files), FindingSeverity.ERROR)


def test_describe_names_the_library_and_the_builds(protocol, vasp_site):
    text = protocol.describe()
    vasp = vasp_site.codes["vasp"]
    assert f"Code: VASP {VASP_VERSION}; builds: std {vasp.binary}, gam {vasp.variants['gam']}" in text
    assert f"the library {vasp.data_paths['potcars']}" in text


def test_a_profile_without_vasp_cannot_host_the_protocol(site):
    with pytest.raises(ValueError, match="no 'vasp' installation"):
        VaspRun(site)
    assert [r.name for r in registry_for(site).refs()] == ["fhiaims.run"] * 4


# -- run checks ------------------------------------------------------------------------------------


class TestRunChecks:
    def report(self, protocol, outcar=None, stdout=None, state=SchedulerState.COMPLETED):
        outputs = {name: text.encode() for name, text in (("OUTCAR", outcar), ("vasp.out", stdout)) if text}
        return protocol.check_run(task(), outputs, state)

    def test_healthy_run_reads_the_final_energies_with_their_lines(self, protocol):
        outcar = vasp_outcar(energies=(-120.5, -123.25))
        report = self.report(protocol, outcar)
        assert all(c.passed for c in report.checks), report.checks
        assert {c.name for c in report.checks} == {"scheduler_completed", "code_version", "normal_termination",
                                                   "scf_converged"}
        lines = outcar.splitlines()
        free, sigma0 = report.readings
        assert (free.name, free.value.value, free.value.unit) == (FREE_ENERGY.name, -123.25, "eV")
        assert lines[free.line_start - 1].strip() == free.quote and "free  energy   TOTEN" in free.quote
        assert (sigma0.name, sigma0.value.value) == (SIGMA0_ENERGY.name, pytest.approx(-123.2495))
        assert report.method["code"] == "VASP" and report.method["code_version"] == VASP_VERSION

    @pytest.mark.parametrize("outcar, failed", [
        (vasp_outcar(energies=(-1.0,), unconverged=(1,)), "scf_converged"),
        (vasp_outcar(normal_end=False), "normal_termination"),
        (vasp_outcar(version="5.4.4"), "code_version"),
    ])
    def test_failures_are_named_and_invalid(self, protocol, outcar, failed):
        report = self.report(protocol, outcar)
        assert [c.name for c in report.checks if not c.passed] == [failed]
        assert report.failure_quality is EvidenceQuality.INVALID

    def test_an_unconverged_relaxation_is_inconclusive(self, protocol):
        report = self.report(protocol, vasp_outcar(energies=(-1.0, -1.1), relax=True, geometry_converged=False))
        (check,) = [c for c in report.checks if not c.passed]
        assert check.name == "geometry_converged" and check.detail == "not converged after 2 ionic steps (NSW = 5)"
        assert report.failure_quality is EvidenceQuality.INCONCLUSIVE
        converged = self.report(protocol, vasp_outcar(energies=(-1.0, -1.1), relax=True))
        assert all(c.passed for c in converged.checks)

    def test_a_crash_says_what_vasp_said(self, protocol):
        report = self.report(protocol, stdout=" DAV: 1\n VERY BAD NEWS! internal error in subroutine SGRCON\n",
                             state=SchedulerState.FAILED)
        (missing,) = [c for c in report.checks if c.name == "output_present"]
        assert missing.detail == ("OUTCAR is missing; its standard output says at line 2: VERY BAD NEWS! internal "
                                  "error in subroutine SGRCON")


# -- the compute profile ---------------------------------------------------------------------------


def test_a_codes_own_template_must_be_well_formed(vasp_site):
    vasp = vasp_site.codes["vasp"]
    with pytest.raises(ValueError, match="exactly one template line must launch"):
        CodeInstallation(binary=vasp.binary, version="1", job_template="#!/bin/bash\nmpirun {{binary}}\n")
    missing = vasp.model_copy(update={"variants": {**vasp.variants, "ncl": "/nonexistent/vasp_ncl"}})
    findings = check_profile(vasp_site.model_copy(update={"codes": {**vasp_site.codes, "vasp": missing}}))
    assert "vasp: /nonexistent/vasp_ncl does not exist" in [f.message for f in findings]


def test_profiles_approved_before_builds_and_code_templates_keep_their_hash(site):
    from cedar.contracts.base import canonical_json, sha256_hex

    data = site.model_dump(mode="json", exclude={"local"})
    for code in data["codes"].values():
        del code["variants"], code["job_template"]
    assert site.content_hash() == sha256_hex(canonical_json(data))
    assert ComputeProfile.model_validate(data).content_hash() == site.content_hash()


# -- through the engine ----------------------------------------------------------------------------


@pytest.fixture
def world(ledger, store, artifacts, vasp_site):
    world = World(ledger, store, artifacts, vasp_site)
    world.registry.register(VaspRun(vasp_site))
    return world


def request(world, slab_poscar, **incar) -> str:
    authored = {name: world.artifacts.put_bytes(data, label=name) for name, data in inputs(slab_poscar, **incar).items()
                if name != "POSCAR"}
    spec = task(structure=world.artifacts.put_bytes(slab_poscar, label="POSCAR"), authored_files=authored)
    cid = world.ledger.request_calculation(world.inv, agent_for(world.hyp), world.hyp, spec, "test").calculation_id
    world.engine.examine(world.inv, cid)
    return cid


def test_a_vasp_calculation_runs_and_its_energy_becomes_evidence(world, slab_poscar):
    cid = request(world, slab_poscar)
    calc = world.calc(cid)
    assert calc.state is CalculationState.SPECIFIED and calc.examination.agent_review
    material = world.reviewer_material(cid)
    vasp = world.site.codes["vasp"]
    assert f"Installed VASP {VASP_VERSION}: standard build {vasp.binary}" in material
    assert f"Source of this installation: {vasp.binary.rsplit('/bin/', 1)[0]}/src" in material
    assert "Settings the inputs make (classify every one of them): SYSTEM, ENCUT" in material
    assert "--- INCAR ---" in material and "--- POTCAR ---" in material
    assert not world.review(cid).is_error
    assert world.calc(cid).state is CalculationState.VALIDATED_INPUT
    world.approve_submission(cid)
    job = world.manager.submit(world.inv, cid)
    run = world.runs / world.calc(cid).workdir
    outcar = vasp_outcar(energies=(-123.25,))
    for name, text in (("OUTCAR", outcar), ("OSZICAR", "   1 F= -.12325000E+03 E0= -.12324950E+03\n"),
                       ("vasp.out", " running\n")):
        (run / name).write_text(text)
    world.scheduler.set_state(job, SchedulerState.COMPLETED)
    world.monitor.poll(world.inv)
    world.validator.check_run(world.inv, cid)
    check = world.calc(cid).run_check
    assert all(c.passed for c in check.checks), check.checks
    assert set(check.outputs) == {"OUTCAR", "OSZICAR", "vasp.out"}
    assert [r.name for r in check.readings] == [FREE_ENERGY.name, SIGMA0_ENERGY.name]


# -- runs computed outside CEDAR -------------------------------------------------------------------


def vasp_directory(path, site, slab_poscar, *, potcar=True, outcar=None):
    """A finished VASP run's directory as the researcher left it (SYNTHETIC)."""
    path.mkdir(parents=True)
    library = site.codes["vasp"].data_paths["potcars"]
    files = {**inputs(slab_poscar, NCORE=4, IVDW=11, ALGO="Fast"), "OUTCAR": (outcar or vasp_outcar()).encode(),
             "OSZICAR": b"   1 F= -.12345679E+03\n", "CONTCAR": slab_poscar}
    if potcar:
        files["POTCAR"] = "".join(open(f"{library}/{n}/POTCAR").read() for n in DATASETS).encode()
    for name, data in files.items():
        (path / name).write_bytes(data)
    return path


def test_a_registered_vasp_run_records_its_files_settings_and_health(world, tmp_path, slab_poscar):
    directory = vasp_directory(tmp_path / "earlier" / "co-atop", world.site, slab_poscar)
    registrar = RunRegistrar(world.ledger, world.artifacts, profile=world.site)
    outcome = registrar.register(world.inv, directory, requested_by=CHIEF, description="the PBE+D3 atop run")
    calc = world.calc(outcome.calculation_id)
    registration = calc.registration
    assert calc.task.protocol == VaspRun.ref and registration.code == "VASP"
    assert set(registration.inputs) == {"INCAR", "POSCAR", "KPOINTS", "POTCAR"}
    assert set(registration.outputs) == {"OUTCAR", "OSZICAR", "CONTCAR"}
    assert calc.task.controls == {"GGA": "PE", "IVDW": "11"}  # NCORE, ALGO and SYSTEM only change how it runs
    assert calc.task.numerical_settings == {
        "ENCUT": "400", "EDIFF": "1E-6", "ISMEAR": "1", "SIGMA": "0.1", "KPOINTS": "Γ-centred 4×4×1 mesh",
        "POTCAR": "PAW_PBE Cu 01Jan2000; PAW_PBE C 01Jan2000; PAW_PBE O 01Jan2000"}
    assert all(c.passed for c in registration.checks) and [r.name for r in registration.readings] == [
        FREE_ENERGY.name, SIGMA0_ENERGY.name]
    assert [n.message for n in registration.notes if n.code == "unclassified_tag"] == []

    # The same inputs requested again are that run, which already has outputs.
    cid = request(world, slab_poscar, NCORE=4, IVDW=11, ALGO="Fast")
    duplicate = [f.message for f in world.calc(cid).examination.findings if f.code == DUPLICATE_INPUTS]
    assert duplicate and duplicate[0].startswith(f"these inputs are identical to registered run {calc.calculation_id}")


def test_a_registered_vasp_run_without_its_potcar_names_the_datasets_from_outcar(world, tmp_path, slab_poscar):
    directory = vasp_directory(tmp_path / "run", world.site, slab_poscar, potcar=False,
                               outcar=vasp_outcar(potcars=("Cu_pv", "C", "O")))
    outcome = RunRegistrar(world.ledger, world.artifacts, profile=world.site).register(
        world.inv, directory, requested_by=CHIEF, description="an older run")
    calc = world.calc(outcome.calculation_id)
    assert calc.task.numerical_settings["POTCAR"] == ("PAW_PBE Cu_pv 01Jan2000; PAW_PBE C 01Jan2000; PAW_PBE O "
                                                      "01Jan2000")
    assert "POTCAR" not in calc.registration.inputs


def test_a_directory_must_hold_a_run_of_a_known_code(world, tmp_path, slab_poscar):
    registrar = RunRegistrar(world.ledger, world.artifacts, profile=world.site)
    empty = tmp_path / "empty"
    empty.mkdir()
    (problem,) = registrar.register(world.inv, empty, requested_by=CHIEF, description="x").problems
    assert problem.endswith("holds neither an FHI-aims run (control.in, geometry.in) nor a VASP run (INCAR, POSCAR, "
                            "OUTCAR)")
    directory = vasp_directory(tmp_path / "unfinished", world.site, slab_poscar)
    (directory / "OUTCAR").unlink()
    (problem,) = registrar.register(world.inv, directory, requested_by=CHIEF, description="x").problems
    assert "has no OUTCAR; a registered VASP run needs its INCAR, POSCAR and output file" in problem
