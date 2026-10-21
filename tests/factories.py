"""Builders for test data. All numbers here are synthetic and carry no scientific meaning."""

from sim2real.contracts import (
    Actor,
    ActorKind,
    AlignmentStatus,
    ArtifactRef,
    Authorization,
    CalculationState,
    CalculationStateChanged,
    EventDraft,
    EvidenceKind,
    EvidenceObject,
    EvidenceQuality,
    EvidenceRecorded,
    ExperimentProvenance,
    HypothesisProposed,
    HypothesisSpec,
    HypothesisStatus,
    HypothesisStatusChanged,
    InvestigationConstraints,
    InvestigationSpec,
    InvestigationStatus,
    InvestigationStatusChanged,
    Observable,
    ObservableAlignment,
    PermissionAction,
    ProtocolRef,
    QualityCheck,
    Quantity,
    ResearchPlan,
    SchedulerState,
    ScientificTaskSpec,
    SimulationProvenance,
    UserInterventionRecorded,
)
from sim2real.state import ResearchLedger

USER = Actor(kind=ActorKind.USER, id="researcher")
CHIEF = Actor(kind=ActorKind.CHIEF_SCIENTIST, id="chief")
AGENT = Actor(kind=ActorKind.HYPOTHESIS_AGENT, id="agent-1")
JUDGE = Actor(kind=ActorKind.EVIDENCE_JUDGE, id="judge")
ENGINE = Actor(kind=ActorKind.PROTOCOL_ENGINE, id="protocol-engine")
MANAGER = Actor(kind=ActorKind.JOB_MANAGER, id="job-manager")
MONITOR = Actor(kind=ActorKind.JOB_MONITOR, id="job-monitor")
VALIDATOR = Actor(kind=ActorKind.VALIDATOR, id="validator")

SITE_ENERGY = Observable(
    name="site energy difference",
    definition="E_total(CO at atop) - E_total(CO at fcc hollow), same slab, cell and coverage",
    unit="eV",
)
PREFERRED_SITE = Observable(
    name="preferred adsorption site",
    definition="site occupied at low coverage, as identified by the measurement",
)
RAW_OUTPUT = ArtifactRef(path="objects/aa/" + "a" * 64, sha256="a" * 64, size_bytes=123, label="aims.out")


def alignment(status: AlignmentStatus = AlignmentStatus.UNASSESSED, *rationale: str) -> ObservableAlignment:
    return ObservableAlignment(simulated=SITE_ENERGY, measured=PREFERRED_SITE, status=status, rationale=rationale)


def investigation_spec(**constraints) -> InvestigationSpec:
    return InvestigationSpec(
        title="CO/Cu(111) adsorption site (test fixture)",
        question="Why does the computed site preference disagree with the observed site?",
        target_claim="The disagreement is explained by identifiable, tested causes.",
        simulations=(SimulationProvenance(label="semilocal slab calculation", code="FHI-aims", xc="PBE"),),
        experiments=(ExperimentProvenance(
            label="low-coverage measurement", observable=PREFERRED_SITE, technique="placeholder technique",
        ),),
        alignment=alignment(),
        constraints=InvestigationConstraints(**constraints),
    )


def plan(**priorities) -> ResearchPlan:
    return ResearchPlan(objective="Explain the site-preference discrepancy", hypothesis_priorities=priorities)


def hypothesis_spec(title: str = "XC error in CO frontier orbitals") -> HypothesisSpec:
    return HypothesisSpec(
        title=title,
        statement="The semilocal functional misplaces CO frontier orbitals, favouring hollow sites.",
        rationale="Site preference depends on CO-metal orbital mixing.",
        falsifiers=("Correcting the frontier-orbital energies leaves the site ordering unchanged.",),
    )


def task(
    *,
    xc: str = "PBE",
    k_grid: tuple[int, int, int] = (4, 4, 1),
    spin: str = "none",
    estimand: str = "E(atop) - E(fcc) at fixed geometries",
    structure: ArtifactRef | None = None,
    **extra,
) -> ScientificTaskSpec:
    return ScientificTaskSpec(
        protocol=ProtocolRef(name="test.adsorption_site_energy", version="0.1"),
        observable=SITE_ENERGY,
        estimand=estimand,
        system_label="CO/Cu(111) p(2x2)",
        structure=structure,
        interventions={"xc": xc},
        numerical_settings={"k_grid": list(k_grid), "species_defaults": "tight"},
        controls={"spin": spin, "coverage": "0.25 ML"},
        **extra,
    )


def draft(actor: Actor, payload, **kwargs) -> EventDraft:
    return EventDraft(actor=actor, payload=payload, **kwargs)


def start_investigation(ledger: ResearchLedger, investigation_id: str = "inv-1", **constraints) -> str:
    ledger.create_investigation(USER, investigation_spec(**constraints), plan(), investigation_id=investigation_id)
    for status in (
        InvestigationStatus.ALIGNING_OBSERVABLES,
        InvestigationStatus.PLANNING,
        InvestigationStatus.INVESTIGATING,
    ):
        ledger.record(investigation_id, [draft(CHIEF, InvestigationStatusChanged(status=status, reason="test"))])
    return investigation_id


def add_hypothesis(ledger: ResearchLedger, investigation_id: str, hypothesis_id: str = "hyp-xc") -> str:
    """Propose a hypothesis and bring it to PLANNING_TEST."""
    ledger.record(investigation_id, [
        draft(CHIEF, HypothesisProposed(hypothesis_id=hypothesis_id, spec=hypothesis_spec(hypothesis_id))),
    ])
    set_hypothesis_status(ledger, investigation_id, hypothesis_id,
                          HypothesisStatus.SCREENED, HypothesisStatus.ACTIVE, HypothesisStatus.PLANNING_TEST)
    return hypothesis_id


def set_hypothesis_status(ledger, investigation_id, hypothesis_id, *statuses, awaiting=()):
    for status in statuses:
        ledger.record(investigation_id, [draft(AGENT, HypothesisStatusChanged(
            hypothesis_id=hypothesis_id, status=status, reason="test",
            awaiting_calculation_ids=awaiting if status is HypothesisStatus.WAITING_FOR_JOB else (),
        ))])


def authorize(ledger, investigation_id, action: PermissionAction, *targets: str, intervention_id: str) -> str:
    ledger.record(investigation_id, [draft(USER, UserInterventionRecorded(
        intervention_id=intervention_id,
        text=f"I authorize {action} of {', '.join(targets)}.",
        authorizations=(Authorization(action=action, target_ids=targets),),
    ))])
    return intervention_id


def move(ledger, investigation_id, calculation_id, actor, state, **kwargs):
    return ledger.record(investigation_id, [
        draft(actor, CalculationStateChanged(calculation_id=calculation_id, state=state, **kwargs)),
    ])


def submit(ledger, investigation_id, calculation_id, job_id: str = "1001"):
    """Prepare and submit a calculation, with the user authorization submission needs."""
    auth = authorize(ledger, investigation_id, PermissionAction.SUBMIT, calculation_id,
                     intervention_id=f"approve-{calculation_id}")
    move(ledger, investigation_id, calculation_id, ENGINE, CalculationState.VALIDATED_INPUT)
    move(ledger, investigation_id, calculation_id, MANAGER, CalculationState.SUBMISSION_PENDING)
    move(ledger, investigation_id, calculation_id, MANAGER, CalculationState.SUBMITTED,
         scheduler_job_id=job_id, authorization_id=auth)


def run_to_parsing(ledger, investigation_id, calculation_id, *,
                   scheduler_state: SchedulerState = SchedulerState.COMPLETED, job_id: str = "1001"):
    submit(ledger, investigation_id, calculation_id, job_id)
    move(ledger, investigation_id, calculation_id, MONITOR, CalculationState.RUNNING)
    move(ledger, investigation_id, calculation_id, MONITOR, CalculationState.SCHEDULER_TERMINAL,
         scheduler_state=scheduler_state)
    move(ledger, investigation_id, calculation_id, VALIDATOR, CalculationState.PARSING)


def computed_evidence(calculation_id: str, *, quality: EvidenceQuality = EvidenceQuality.VALIDATED,
                      checks=(QualityCheck(name="scf_converged", passed=True),),
                      observable: Observable = SITE_ENERGY) -> EvidenceObject:
    return EvidenceObject(
        kind=EvidenceKind.DIRECT_COMPUTATION,
        summary="Synthetic test result",
        observable=observable,
        value=Quantity(value=0.1, unit="eV"),
        calculation_id=calculation_id,
        quality=quality,
        quality_checks=checks,
        sources=(RAW_OUTPUT,),
    )


def record_evidence(ledger, investigation_id, calculation_id, evidence_id: str = "ev-1", **kwargs):
    """Record a calculation's evidence and close its quality assessment in one atomic batch."""
    return ledger.record(investigation_id, [
        draft(VALIDATOR, EvidenceRecorded(evidence_id=evidence_id,
                                          evidence=computed_evidence(calculation_id, **kwargs))),
        draft(VALIDATOR, CalculationStateChanged(calculation_id=calculation_id,
                                                 state=CalculationState.QUALITY_ASSESSED)),
    ])
