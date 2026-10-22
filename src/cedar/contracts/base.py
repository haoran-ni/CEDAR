"""Shared building blocks for persisted contracts.

Every persisted object is an immutable Pydantic model that rejects unknown fields and
non-finite numbers, so a malformed record fails at the boundary instead of propagating
into evidence.
"""

import hashlib
import json
import uuid
from datetime import date, datetime, timezone
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints


class Contract(BaseModel):
    """Base class for persisted contracts: frozen, no unknown fields, no NaN/inf."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
"""Stable identifier. Safe to embed in file names: no slashes and no leading dot."""

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

FileName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
"""A plain file name such as ``control.in``: no directories, no leading dot."""


def _check_relative_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError("path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"path must be relative, normalized and free of '..': {value!r}")
    return value


RelativePath = Annotated[str, AfterValidator(_check_relative_path)]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> str:
    """Deterministic JSON used for hashing and equality of scientific content."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class ActorKind(StrEnum):
    USER = "user"
    CHIEF_SCIENTIST = "chief_scientist"
    HYPOTHESIS_AGENT = "hypothesis_agent"
    EVIDENCE_JUDGE = "evidence_judge"
    PROTOCOL_ENGINE = "protocol_engine"
    JOB_MANAGER = "job_manager"
    JOB_MONITOR = "job_monitor"
    VALIDATOR = "validator"
    ANALYSIS_RUNNER = "analysis_runner"
    APPROVER = "approver"
    """Decides on the researcher's behalf, in full automation only, whether proposed agents start and
    calculations are submitted."""
    WORKBENCH = "workbench"
    """Runs agents' commands in their workspaces, in a sandbox, and records them."""
    INPUT_REVIEWER = "input_reviewer"
    """Checks a calculation's inputs before they can run: that the code runs them as described, and that they are
    harmless."""
    SYSTEM = "system"


class Actor(Contract):
    kind: ActorKind
    id: Identifier


class Observable(Contract):
    """A precisely defined quantity, not just a label such as "gap"."""

    name: NonEmptyStr
    definition: NonEmptyStr
    unit: str | None = None


class UncertaintyKind(StrEnum):
    EXPERIMENTAL = "experimental"
    STATISTICAL = "statistical"
    NUMERICAL = "numerical"
    MODEL = "model"
    UNSPECIFIED = "unspecified"


class Uncertainty(Contract):
    """One uncertainty component, in the unit of the quantity it belongs to."""

    kind: UncertaintyKind
    value: float = Field(ge=0)
    basis: NonEmptyStr


class Quantity(Contract):
    value: float
    unit: NonEmptyStr
    reference: str | None = None
    """Zero or reference state, e.g. "vacuum level" or "CO(g) + clean slab"."""
    uncertainties: tuple[Uncertainty, ...] = ()


class ArtifactRef(Contract):
    """A file in the artifact store, addressed by content hash."""

    kind: Literal["artifact"] = "artifact"
    path: RelativePath
    sha256: Sha256Hex
    size_bytes: int = Field(ge=0)
    media_type: str | None = None
    label: str | None = None


class ExternalSource(Contract):
    """A citable source outside this system (paper, dataset, report)."""

    kind: Literal["external"] = "external"
    citation: NonEmptyStr
    doi: str | None = None
    url: str | None = None
    locator: str | None = None
    """Figure, table or page within the source."""
    accessed_on: date | None = None


SourceRef = Annotated[ArtifactRef | ExternalSource, Field(discriminator="kind")]
