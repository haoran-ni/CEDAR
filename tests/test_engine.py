"""End-to-end path from an agent's request to recorded evidence.

Uses the FakeScheduler TEST DOUBLE and SYNTHETIC outputs: no job runs, no numbers are real.
"""

import os

import pytest
from factories import AGENT, authorize, draft
from sites import aims_output
from worlds import World

from cedar.contracts import (
    CalculationState,
    CalculationStateChanged,
    EvidenceQuality,
    ExaminationVerdict,
    FileCitation,
    Observable,
    OutputReading,
    PermissionAction,
    ProtocolRef,
    Quantity,
    SchedulerState,
)
from cedar.engine import InputsChanged, job_name_for
from cedar.engine.jobs import JOB_MANAGER
from cedar.errors import RuleViolation
from cedar.state import replay

ENERGY = -98765.432101234


@pytest.fixture
def world(ledger, store, artifacts, site):
    return World(ledger, store, artifacts, site)


def energy_line(text: str) -> tuple[int, str]:
    for number, line in enumerate(text.splitlines(), start=1):
        if "Total energy of the DFT" in line:
            return number, line.strip()
    raise AssertionError("no final energy line")


def reading(text: str, value: float = ENERGY, *, line_offset: int = 0, quote: str | None = None) -> OutputReading:
    number, line = energy_line(text)
    return OutputReading(
        reader=AGENT, value=Quantity(value=value, unit="eV"), summary="Final total energy read from aims.out",
        citations=(FileCitation(file="aims.out", line_start=number + line_offset, line_end=number + line_offset,
                                quote=quote or line),),
    )


class TestExamination:
    def test_generated_inputs_are_approved_and_written_read_only(self, world):
        cid = world.request()
        world.engine.examine(world.inv, cid)
        calc = world.calc(cid)
        assert calc.state is CalculationState.VALIDATED_INPUT
        assert calc.examination.verdict is ExaminationVerdict.APPROVED
        assert set(calc.examination.files) == {"control.in", "geometry.in", "job.sh"}
        assert calc.examination.execution_files == ("job.sh",)
        directory = world.runs / calc.workdir
        for name, ref in calc.examination.files.items():
            assert (directory / name).read_bytes() == world.artifacts.read(ref)
            assert not os.access(directory / name, os.W_OK)

    def test_agent_written_control_is_examined_unchanged(self, world):
        first = world.request()
        world.engine.examine(world.inv, first)
        control = world.artifacts.read(world.calc(first).examination.files["control.in"])
        rewritten = control.replace(b"xc                pbe", b"xc pbe0")
        ref = world.artifacts.put_bytes(rewritten, label="control.in")
        cid = world.request(authored={"control.in": ref})
        world.engine.examine(world.inv, cid)
        calc = world.calc(cid)
        assert calc.examination.files["control.in"].sha256 == ref.sha256
        assert calc.state is CalculationState.REJECTED
        (finding,) = [f for f in calc.examination.findings if f.code == "contradicts_task"]
        assert finding.message == "xc pbe0 does not match the declared xc pbe"
        assert not (world.runs / calc.workdir).exists()

    def test_identical_inputs_from_another_request_are_rejected_as_duplicates(self, world):
        first = world.request()
        world.engine.examine(world.inv, first)
        control = world.calc(first).examination.files["control.in"]
        second = world.request(authored={"control.in": control})
        world.engine.examine(world.inv, second)
        calc = world.calc(second)
        assert calc.state is CalculationState.REJECTED
        assert "duplicate_inputs" in {f.code for f in calc.examination.findings}

    def test_inputs_needing_review_wait_for_the_user(self, world):
        first = world.request()
        world.engine.examine(world.inv, first)
        control = world.artifacts.read(world.calc(first).examination.files["control.in"])
        unusual = world.artifacts.put_bytes(control.replace(b"\nspin", b"\nsome_new_keyword 1\nspin", 1))
        cid = world.request(authored={"control.in": unusual})
        world.engine.examine(world.inv, cid)
        assert world.calc(cid).state is CalculationState.SPECIFIED
        assert world.calc(cid).examination.verdict is ExaminationVerdict.NEEDS_REVIEW
        with pytest.raises(RuleViolation, match="no user approval"):
            world.engine.approve(world.inv, cid)
        authorize(world.ledger, world.inv, PermissionAction.APPROVE_INPUTS, cid, intervention_id="looked-at-it")
        world.engine.approve(world.inv, cid)
        assert world.calc(cid).state is CalculationState.VALIDATED_INPUT

    def test_unknown_protocol_is_rejected_with_a_reason(self, world):
        spec = world.task().model_copy(update={"protocol": ProtocolRef(name="vasp.energy", version="1")})
        cid = world.ledger.request_calculation(world.inv, AGENT, world.hyp, spec, "test").calculation_id
        world.engine.examine(world.inv, cid)
        assert world.calc(cid).state is CalculationState.REJECTED

    def test_examination_happens_once(self, world):
        cid = world.request()
        world.engine.examine(world.inv, cid)
        with pytest.raises(RuleViolation, match="already been examined"):
            world.engine.examine(world.inv, cid)


class TestSubmission:
    def test_submission_needs_user_authorization_before_the_scheduler_is_called(self, world):
        cid = world.request()
        world.engine.examine(world.inv, cid)
        with pytest.raises(RuleViolation, match="authorization"):
            world.manager.submit(world.inv, cid)
        assert world.scheduler.submissions == []
        assert world.calc(cid).state is CalculationState.VALIDATED_INPUT

    def test_submission_uses_the_examined_script_and_is_idempotent(self, world):
        cid = world.request()
        world.engine.examine(world.inv, cid)
        world.approve_submission(cid)
        job = world.manager.submit(world.inv, cid)
        assert world.manager.submit(world.inv, cid) == job
        (script, workdir, name), = world.scheduler.submissions
        assert script == world.runs / world.calc(cid).workdir / "job.sh" and name == job_name_for(cid)
        assert world.calc(cid).scheduler_job_id == job

    def test_tampered_run_directory_is_not_submitted(self, world):
        cid = world.request()
        world.engine.examine(world.inv, cid)
        world.approve_submission(cid)
        path = world.runs / world.calc(cid).workdir / "control.in"
        os.chmod(path, 0o644)
        path.write_text(path.read_text() + "\npostprocess_anyway .true.\n")
        with pytest.raises(InputsChanged):
            world.manager.submit(world.inv, cid)
        assert world.scheduler.submissions == []

    def test_crash_after_submission_is_reconciled_without_a_second_job(self, world):
        cid = world.request()
        world.engine.examine(world.inv, cid)
        world.approve_submission(cid)
        world.ledger.record(world.inv, [draft(JOB_MANAGER, CalculationStateChanged(
            calculation_id=cid, state=CalculationState.SUBMISSION_PENDING))])
        directory = world.runs / world.calc(cid).workdir
        orphan = world.scheduler.submit(directory / "job.sh", directory, job_name_for(cid))  # then the process "died"
        assert world.manager.submit(world.inv, cid) == orphan
        assert len(world.scheduler.submissions) == 1

    def test_reconcile_reports_pending_submissions_with_no_job(self, world):
        cid = world.request()
        world.engine.examine(world.inv, cid)
        world.ledger.record(world.inv, [draft(JOB_MANAGER, CalculationStateChanged(
            calculation_id=cid, state=CalculationState.SUBMISSION_PENDING))])
        assert world.manager.reconcile(world.inv) == {cid: None}

    def test_cancel_needs_authorization_and_stops_the_job(self, world):
        cid = world.request()
        world.engine.examine(world.inv, cid)
        world.approve_submission(cid)
        job = world.manager.submit(world.inv, cid)
        with pytest.raises(RuleViolation, match="authorization"):
            world.manager.cancel(world.inv, cid, "not needed")
        assert world.scheduler.jobs[job]["state"] is SchedulerState.PENDING
        authorize(world.ledger, world.inv, PermissionAction.CANCEL, cid, intervention_id=f"cancel-{cid}")
        world.manager.cancel(world.inv, cid, "user asked")
        assert world.scheduler.jobs[job]["state"] is SchedulerState.CANCELLED
        assert world.calc(cid).state is CalculationState.CANCELLED


class TestMonitoring:
    def test_states_follow_the_scheduler_and_repeat_polls_record_nothing(self, world):
        cid = world.request()
        world.engine.examine(world.inv, cid)
        world.approve_submission(cid)
        job = world.manager.submit(world.inv, cid)
        assert [e.payload.state for e in world.monitor.poll(world.inv).recorded] == [CalculationState.QUEUED]
        assert world.monitor.poll(world.inv).recorded == []
        world.scheduler.set_state(job, SchedulerState.TIMEOUT)
        world.monitor.poll(world.inv)
        calc = world.calc(cid)
        assert (calc.state, calc.scheduler_state) == (CalculationState.SCHEDULER_TERMINAL, SchedulerState.TIMEOUT)

    def test_jobs_the_scheduler_no_longer_reports_are_left_alone(self, world):
        cid = world.request()
        world.engine.examine(world.inv, cid)
        world.approve_submission(cid)
        job = world.manager.submit(world.inv, cid)
        world.scheduler.forget(job)
        result = world.monitor.poll(world.inv)
        assert result.unknown == [cid] and result.recorded == []
        assert world.calc(cid).state is CalculationState.SUBMITTED


class TestRunChecks:
    def test_finished_run_is_checked_once_and_its_outputs_stored(self, world):
        output = aims_output(energy=ENERGY)
        cid = world.request()
        world.run(cid, output=output)
        assert set(world.validator.output_paths(world.inv, cid)) == {"aims.out"}
        state = world.validator.check_run(world.inv, cid).state
        calc = state.calculations[cid]
        assert calc.state is CalculationState.QUALITY_ASSESSED and calc.run_check.healthy and not calc.evidence_ids
        assert {c.name for c in calc.run_check.checks} >= {
            "scheduler_completed", "normal_termination", "scf_converged", "code_version", "inputs_unchanged"}
        assert world.artifacts.read(calc.run_check.outputs["aims.out"]) == output.encode()
        total = next(r for r in calc.run_check.readings if r.name == "total energy")
        assert (total.value.value, total.line_start) == (ENERGY, energy_line(output)[0])
        assert world.validator.check_run(world.inv, cid).deduplicated
        assert replay(world.store.read_stream(world.inv)) == world.state()

    def test_failed_run_is_checked_with_its_failures_named(self, world):
        cid = world.request()
        world.run(cid, output=aims_output(normal_end=False), final=SchedulerState.TIMEOUT)
        check = world.validator.check_run(world.inv, cid).state.calculations[cid].run_check
        assert not check.healthy and check.failure_quality is EvidenceQuality.INVALID
        assert {c.name for c in check.checks if not c.passed} >= {"scheduler_completed", "normal_termination"}

    def test_runs_that_have_not_ended_cannot_be_checked(self, world):
        cid = world.request()
        world.engine.examine(world.inv, cid)
        with pytest.raises(RuleViolation, match="has not ended"):
            world.validator.check_run(world.inv, cid)


TOTAL = Observable(name="total energy", definition="final total energy printed by FHI-aims", unit="eV")


class TestReadings:
    def test_cited_reading_becomes_validated_evidence(self, world):
        output = aims_output(energy=ENERGY)
        cid = world.request()
        world.checked(cid, output=output)
        outcome = world.validator.record_reading(world.inv, cid, reading(output), TOTAL)
        assert outcome.problems == ()
        state = outcome.recorded.state
        (evidence_id,) = state.calculations[cid].evidence_ids
        evidence = state.evidence[evidence_id].evidence
        assert evidence.quality is EvidenceQuality.VALIDATED and evidence.observable == TOTAL
        assert evidence.value.value == ENERGY and evidence.interpreted_by == AGENT
        assert world.artifacts.read(evidence.citations[0].artifact) == output.encode()
        assert {c.name for c in evidence.quality_checks} == {f"run_healthy:{cid}", "citations_verified"}
        assert replay(world.store.read_stream(world.inv)) == state

    def test_one_run_gives_several_readings(self, world):
        output = aims_output(energy=ENERGY)
        cid = world.request()
        world.checked(cid, output=output)
        number, line = next((n, l.strip()) for n, l in enumerate(output.splitlines(), start=1)
                            if "Total energy corrected" in l)
        corrected = OutputReading(
            reader=AGENT, value=Quantity(value=float(line.split(":")[1].split()[0]), unit="eV"),
            summary="corrected energy",
            citations=(FileCitation(file="aims.out", line_start=number, line_end=number, quote=line),))
        world.validator.record_reading(world.inv, cid, reading(output), TOTAL)
        other = Observable(name="corrected total energy", definition="'Total energy corrected' line", unit="eV")
        state = world.validator.record_reading(world.inv, cid, corrected, other).recorded.state
        assert [state.evidence[e].evidence.observable.name for e in state.calculations[cid].evidence_ids] == [
            "total energy", "corrected total energy"]

    @pytest.mark.parametrize("bad", [
        {"line_offset": 1},
        {"quote": "| Total energy of the DFT / Hartree-Fock s.c.f. calculation : -1.0 eV"},
        {"value": ENERGY + 1.0},
    ])
    def test_unverifiable_readings_are_refused_and_nothing_is_recorded(self, world, bad):
        output = aims_output(energy=ENERGY)
        cid = world.request()
        world.checked(cid, output=output)
        before = world.state().version
        value = bad.pop("value", ENERGY)
        outcome = world.validator.record_reading(world.inv, cid, reading(output, value, **bad), TOTAL)
        assert outcome.recorded is None and outcome.problems
        assert world.state().version == before

    def test_readings_wait_for_the_run_check_and_follow_its_health(self, world):
        output = aims_output(energy=ENERGY)  # a complete output, but the scheduler says the job timed out
        cid = world.request()
        world.run(cid, output=output, final=SchedulerState.TIMEOUT)
        with pytest.raises(RuleViolation, match="has not been checked"):
            world.validator.record_reading(world.inv, cid, reading(output), TOTAL)
        world.validator.check_run(world.inv, cid)
        state = world.validator.record_reading(world.inv, cid, reading(output), TOTAL).recorded.state
        (record,) = state.evidence.values()
        assert record.evidence.quality is EvidenceQuality.INVALID
        assert not next(c for c in record.evidence.quality_checks if c.name.startswith("run_healthy")).passed
