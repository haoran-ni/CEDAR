from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from factories import RAW_OUTPUT, SITE_ENERGY, alignment, computed_evidence, hypothesis_spec, task
from cedar.contracts import (
    Actor,
    ActorKind,
    AlignmentStatus,
    ArtifactRef,
    CalculationRequested,
    EvidenceKind,
    EvidenceObject,
    EvidenceQuality,
    ExternalSource,
    FidelityTier,
    HypothesisSpec,
    HypothesisStatus,
    HypothesisStatusChanged,
    QualityCheck,
    Quantity,
    RecordedEvent,
    ResourceBudget,
    ScientificTaskSpec,
    Uncertainty,
    UncertaintyKind,
    discrepancy,
)


@pytest.mark.parametrize("bad", ["../x", "a/b", ".hidden", "", "x" * 200])
def test_identifiers_cannot_be_paths(bad):
    with pytest.raises(ValidationError):
        Actor(kind=ActorKind.USER, id=bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_quantities_must_be_finite(value):
    with pytest.raises(ValidationError):
        Quantity(value=value, unit="eV")


def test_free_form_settings_reject_non_finite_numbers():
    with pytest.raises(ValidationError, match="finite"):
        ScientificTaskSpec.model_validate({**task().model_dump(), "numerical_settings": {"tol": float("nan")}})


@pytest.mark.parametrize("bad", ["../x", "/abs/path", "a/../b", "a//b", "./a", "a\\b"])
def test_artifact_paths_must_stay_inside_the_store(bad):
    with pytest.raises(ValidationError):
        ArtifactRef(path=bad, sha256="a" * 64, size_bytes=1)


def test_contracts_are_immutable_and_strict():
    spec = hypothesis_spec()
    with pytest.raises(ValidationError):
        spec.title = "changed"
    with pytest.raises(ValidationError):
        HypothesisSpec.model_validate({**spec.model_dump(), "confidence": 0.9})


def test_hypothesis_must_be_falsifiable():
    with pytest.raises(ValidationError):
        HypothesisSpec(title="t", statement="s", rationale="r", falsifiers=())


def test_task_variable_is_classified_exactly_once():
    with pytest.raises(ValidationError, match="classified as both"):
        ScientificTaskSpec.model_validate({**task().model_dump(), "controls": {"xc": "PBE"}})
    with pytest.raises(ValidationError, match="reserved"):
        ScientificTaskSpec.model_validate({**task().model_dump(), "controls": {"structure": "x"}})


def test_task_identity_ignores_labels_tier_and_budget():
    base = task()
    relabelled = base.model_copy(update={
        "system_label": "another label",
        "fidelity": FidelityTier.REFERENCE,
        "budget": ResourceBudget(max_core_hours=10),
    })
    assert relabelled.identity_hash() == base.identity_hash()
    assert task(xc="PBE0").identity_hash() != base.identity_hash()
    assert task(k_grid=(6, 6, 1)).identity_hash() != base.identity_hash()


def test_task_identity_uses_structure_content_not_its_label():
    a = task(structure=RAW_OUTPUT)
    b = task(structure=RAW_OUTPUT.model_copy(update={"label": "renamed.in"}))
    c = task(structure=RAW_OUTPUT.model_copy(update={"sha256": "b" * 64}))
    assert a.identity_hash() == b.identity_hash() != c.identity_hash()


def test_calculation_request_hash_must_match_task():
    with pytest.raises(ValidationError, match="task_hash"):
        CalculationRequested(calculation_id="c1", hypothesis_id="h1", task=task(),
                             task_hash="0" * 64, rationale="r")


def test_waiting_status_names_the_awaited_calculations():
    with pytest.raises(ValidationError):
        HypothesisStatusChanged(hypothesis_id="h", status=HypothesisStatus.WAITING_FOR_JOB, reason="r")
    with pytest.raises(ValidationError):
        HypothesisStatusChanged(hypothesis_id="h", status=HypothesisStatus.ANALYZING, reason="r",
                                awaiting_calculation_ids=("c1",))


class TestEvidenceRules:
    def test_validated_needs_recorded_checks(self):
        with pytest.raises(ValidationError, match="at least one recorded quality check"):
            computed_evidence("c1", checks=())

    def test_validated_cannot_have_failed_checks(self):
        with pytest.raises(ValidationError, match="failed quality checks"):
            computed_evidence("c1", checks=(QualityCheck(name="scf_converged", passed=False),))

    def test_failed_checks_allowed_on_lower_quality(self):
        ev = computed_evidence("c1", quality=EvidenceQuality.INVALID,
                               checks=(QualityCheck(name="scf_converged", passed=False),))
        assert ev.quality is EvidenceQuality.INVALID

    def test_direct_computation_needs_calculation_and_raw_output(self):
        base = computed_evidence("c1").model_dump()
        with pytest.raises(ValidationError, match="calculation_id"):
            EvidenceObject.model_validate({**base, "calculation_id": None})
        external = {"kind": "external", "citation": "Some paper"}
        with pytest.raises(ValidationError, match="raw output artifact"):
            EvidenceObject.model_validate({**base, "sources": [external]})

    def test_published_experiment_cites_external_source(self):
        with pytest.raises(ValidationError, match="external source"):
            EvidenceObject(kind=EvidenceKind.PUBLISHED_EXPERIMENT, summary="s", observable=SITE_ENERGY,
                           quality=EvidenceQuality.PROVISIONAL, sources=(RAW_OUTPUT,))
        EvidenceObject(kind=EvidenceKind.PUBLISHED_EXPERIMENT, summary="s", observable=SITE_ENERGY,
                       quality=EvidenceQuality.PROVISIONAL, sources=(ExternalSource(citation="A paper"),))

    def test_inference_states_assumptions(self):
        with pytest.raises(ValidationError, match="assumptions"):
            EvidenceObject(kind=EvidenceKind.INFERENCE, summary="s", observable=SITE_ENERGY,
                           quality=EvidenceQuality.PROVISIONAL, sources=(ExternalSource(citation="A paper"),))

    def test_evidence_needs_a_source(self):
        with pytest.raises(ValidationError):
            EvidenceObject(kind=EvidenceKind.INDIRECT_CONSTRAINT, summary="s", observable=SITE_ENERGY,
                           quality=EvidenceQuality.PROVISIONAL, sources=())


class TestDiscrepancy:
    aligned = alignment(AlignmentStatus.ALIGNED, "same quantity on both sides")

    def test_unaligned_observables_are_not_comparable(self):
        result = discrepancy(Quantity(value=1.0, unit="eV"), Quantity(value=0.5, unit="eV"), alignment())
        assert not result.comparable and result.difference is None
        assert "UNASSESSED" in result.reasons[0]

    def test_alignment_status_needs_a_rationale(self):
        with pytest.raises(ValidationError, match="rationale"):
            alignment(AlignmentStatus.ALIGNED)

    def test_units_are_never_silently_converted(self):
        result = discrepancy(Quantity(value=1.0, unit="eV"), Quantity(value=96.0, unit="kJ/mol"), self.aligned)
        assert not result.comparable
        assert any("units differ" in r for r in result.reasons)

    def test_references_must_match(self):
        result = discrepancy(Quantity(value=1.0, unit="eV", reference="vacuum level"),
                             Quantity(value=0.5, unit="eV", reference="Fermi level"), self.aligned)
        assert any("references differ" in r for r in result.reasons)

    def test_comparable_values_give_a_difference_without_invented_uncertainty(self):
        sim = Quantity(value=1.0, unit="eV", uncertainties=(
            Uncertainty(kind=UncertaintyKind.NUMERICAL, value=0.05, basis="k-grid test"),))
        exp = Quantity(value=0.75, unit="eV")
        result = discrepancy(sim, exp, self.aligned)
        assert result.comparable
        assert result.difference.value == pytest.approx(0.25)
        assert result.difference.uncertainties == ()


def test_recorded_event_round_trips_through_json():
    event = RecordedEvent(
        event_id="evt-1", investigation_id="inv-1", stream_version=2,
        occurred_at=datetime(2026, 10, 21, tzinfo=timezone.utc),
        actor=Actor(kind=ActorKind.VALIDATOR, id="v"), round=0, plan_version=1,
        payload={"type": "evidence_recorded", "evidence_id": "ev-1",
                 "evidence": computed_evidence("c1").model_dump(mode="json")},
    )
    assert RecordedEvent.model_validate_json(event.model_dump_json()) == event
    assert event.type == "evidence_recorded"


def test_naive_timestamps_are_rejected():
    with pytest.raises(ValidationError):
        RecordedEvent(event_id="e", investigation_id="i", stream_version=1, occurred_at=datetime(2026, 10, 21),
                      actor=Actor(kind=ActorKind.USER, id="u"), round=0, plan_version=1,
                      payload={"type": "round_started", "round": 1, "reason": "r"})


class TestExaminationContract:
    def test_verdict_must_follow_from_findings(self):
        from factories import examination
        from cedar.contracts import ExaminationVerdict, Finding, FindingSeverity, InputExamination

        exam = examination("c1")
        with pytest.raises(ValidationError, match="does not follow"):
            InputExamination.model_validate({**exam.model_dump(), "findings": [
                Finding(severity=FindingSeverity.ERROR, code="x", message="m").model_dump()]})
        assert exam.verdict is ExaminationVerdict.APPROVED

    def test_input_hash_must_match_files(self):
        from factories import examination
        from cedar.contracts import InputExamination

        with pytest.raises(ValidationError, match="input_hash"):
            InputExamination.model_validate({**examination("c1").model_dump(), "input_hash": "0" * 64})

    def test_job_scripts_do_not_change_scientific_identity(self):
        from cedar.contracts import ProtocolRef, scientific_input_hash

        protocol = ProtocolRef(name="p", version="1")
        control = RAW_OUTPUT
        script_a = RAW_OUTPUT.model_copy(update={"sha256": "b" * 64})
        script_b = RAW_OUTPUT.model_copy(update={"sha256": "c" * 64})
        a = scientific_input_hash(protocol, "v", {"control.in": control, "job.sh": script_a}, ["job.sh"])
        b = scientific_input_hash(protocol, "v", {"control.in": control, "job.sh": script_b}, ["job.sh"])
        assert a == b != scientific_input_hash(protocol, "v2", {"control.in": control}, [])


class TestCitations:
    def test_computed_values_must_be_cited(self):
        base = computed_evidence("c1").model_dump()
        with pytest.raises(ValidationError, match="cite the output lines"):
            EvidenceObject.model_validate({**base, "citations": []})

    def test_cited_output_must_be_a_source(self):
        base = computed_evidence("c1").model_dump()
        other = RAW_OUTPUT.model_copy(update={"sha256": "b" * 64}).model_dump()
        citation = {**base["citations"][0], "artifact": other}
        with pytest.raises(ValidationError, match="among the sources"):
            EvidenceObject.model_validate({**base, "citations": [citation]})

    def test_validated_citations_need_verification(self):
        with pytest.raises(ValidationError, match="citations_verified"):
            computed_evidence("c1", checks=(QualityCheck(name="scf_converged", passed=True),))

    def test_line_range_is_ordered(self):
        from cedar.contracts import OutputCitation

        with pytest.raises(ValidationError):
            OutputCitation(artifact=RAW_OUTPUT, line_start=5, line_end=4, quote="x")


def test_agent_written_files_are_part_of_task_identity():
    control = RAW_OUTPUT.model_copy(update={"label": "control.in"})
    written = task(authored_files={"control.in": control})
    rewritten = task(authored_files={"control.in": control.model_copy(update={"sha256": "b" * 64})})
    assert written.identity_hash() != task().identity_hash()
    assert written.identity_hash() != rewritten.identity_hash()
    assert written.identity_hash() == task(authored_files={"control.in": control.model_copy(
        update={"label": "renamed"})}).identity_hash()
