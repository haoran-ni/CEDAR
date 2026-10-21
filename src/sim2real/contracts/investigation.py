"""The problem statement of an investigation: what disagrees, and under which conditions."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, BeforeValidator, Field, JsonValue, StringConstraints, model_validator

from sim2real.contracts.base import (
    Contract,
    Identifier,
    NonEmptyStr,
    Observable,
    Quantity,
    SourceRef,
)
from sim2real.contracts.task import ProtocolRef, ResourceBudget


class SimulationProvenance(Contract):
    label: NonEmptyStr
    quantity: Quantity | None = None
    code: str | None = None
    code_version: str | None = None
    method: str | None = None
    xc: str | None = None
    dispersion: str | None = None
    basis: str | None = None
    k_sampling: str | None = None
    charge: float | None = None
    spin: str | None = None
    coverage: str | None = None
    boundary_conditions: str | None = None
    numerical_settings: dict[str, JsonValue] = {}
    conditions: dict[str, JsonValue] = {}
    sources: tuple[SourceRef, ...] = ()


class ExperimentProvenance(Contract):
    label: NonEmptyStr
    observable: Observable
    technique: NonEmptyStr
    quantity: Quantity | None = None
    calibration_reference: str | None = None
    temperature: Quantity | None = None
    pressure: Quantity | None = None
    preparation: str | None = None
    composition: str | None = None
    coverage: str | None = None
    environment: str | None = None
    conditions: dict[str, JsonValue] = {}
    sources: tuple[SourceRef, ...] = ()


class AlignmentStatus(StrEnum):
    UNASSESSED = "UNASSESSED"
    ALIGNED = "ALIGNED"
    REQUIRES_FORWARD_MODEL = "REQUIRES_FORWARD_MODEL"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class ObservableAlignment(Contract):
    """Whether the simulated and measured quantities are the same thing."""

    simulated: Observable
    measured: Observable
    status: AlignmentStatus = AlignmentStatus.UNASSESSED
    forward_model: str | None = None
    """How the simulated quantity maps onto the measured signal, if a mapping is needed."""
    rationale: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _assessed_status_needs_rationale(self) -> "ObservableAlignment":
        if self.status is not AlignmentStatus.UNASSESSED and not self.rationale:
            raise ValueError(f"alignment status {self.status} needs a rationale")
        return self


class RetrievalPolicy(StrEnum):
    """Whether the investigation's agents may use the web: not at all, only some domains, or all of it."""

    DISABLED = "disabled"
    CURATED_ONLY = "curated_only"
    ENABLED = "enabled"


Domain = Annotated[str, BeforeValidator(lambda v: v.lower() if isinstance(v, str) else v), StringConstraints(
    max_length=253, pattern=r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,62}$")]
"""A plain host name such as ``arxiv.org``, which also covers its subdomains."""


def check_retrieval(policy: RetrievalPolicy, domains: Sequence[str]) -> None:
    if (policy is RetrievalPolicy.CURATED_ONLY) != bool(domains):
        raise ValueError("web access limited to some domains names them; no other setting takes domains")


class InvestigationConstraints(Contract):
    compute_profile: Identifier | None = None
    """The user-approved compute profile calculations run with."""
    allowed_protocols: tuple[ProtocolRef, ...] = ()
    budget: ResourceBudget | None = None
    deadline: AwareDatetime | None = None
    clusters: tuple[NonEmptyStr, ...] = ()
    external_retrieval: RetrievalPolicy = RetrievalPolicy.DISABLED
    """Web access at the start; the researcher changes it with ``retrieval_policy_changed``."""
    allowed_domains: tuple[Domain, ...] = ()
    """With ``curated_only``: the domains agents may search and fetch from."""
    submit_requires_approval: bool = True
    """Cancelling a submitted job always requires user authorization, independent of this."""
    model_token_budget: int | None = Field(default=None, gt=0)
    """Tokens the investigation's agents may process in total (input, cached input and output) when it opens;
    none unless set. The researcher changes it with ``token_budget_changed``."""
    agent_review_minutes: int = Field(default=10, ge=0)
    """How long proposed hypothesis agents wait for the researcher's reply before they start."""

    @model_validator(mode="after")
    def _domains_with_curated_only(self) -> "InvestigationConstraints":
        check_retrieval(self.external_retrieval, self.allowed_domains)
        return self


class InvestigationSpec(Contract):
    title: NonEmptyStr
    question: NonEmptyStr
    target_claim: NonEmptyStr
    simulations: tuple[SimulationProvenance, ...] = Field(min_length=1)
    experiments: tuple[ExperimentProvenance, ...] = Field(min_length=1)
    alignment: ObservableAlignment
    constraints: InvestigationConstraints = InvestigationConstraints()


class Discrepancy(Contract):
    comparable: bool
    difference: Quantity | None = None
    """Simulated minus measured. Uncertainties are not combined: their correlation is unknown."""
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> "Discrepancy":
        if self.comparable != (self.difference is not None):
            raise ValueError("a difference is given if and only if the values are comparable")
        if not self.comparable and not self.reasons:
            raise ValueError("a non-comparable discrepancy needs reasons")
        return self


def comparability_issues(a: Quantity, b: Quantity) -> list[str]:
    issues = []
    if a.unit != b.unit:
        issues.append(f"units differ: {a.unit!r} vs {b.unit!r}")
    if a.reference != b.reference:
        issues.append(f"references differ: {a.reference!r} vs {b.reference!r}")
    return issues


def discrepancy(
    simulated: Quantity | None,
    measured: Quantity | None,
    alignment: ObservableAlignment,
) -> Discrepancy:
    """Size of the disagreement, or NOT_COMPARABLE with reasons.

    A number is produced only when the observables are aligned and both values share a
    unit and reference. Units are never converted here.
    """
    reasons: list[str] = []
    if alignment.status is not AlignmentStatus.ALIGNED:
        reasons.append(f"observable alignment is {alignment.status}")
    if simulated is None:
        reasons.append("no simulated value")
    if measured is None:
        reasons.append("no measured value")
    if simulated is not None and measured is not None:
        reasons.extend(comparability_issues(simulated, measured))
    if reasons:
        return Discrepancy(comparable=False, reasons=tuple(reasons))
    return Discrepancy(
        comparable=True,
        difference=Quantity(value=simulated.value - measured.value, unit=simulated.unit),
    )
