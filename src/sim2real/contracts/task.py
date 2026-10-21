"""Typed scientific task requests and controlled-comparison checks.

Agents describe *what* to compute; a versioned protocol later decides *how*. Each input
variable is classified exactly once: an intervention (deliberately changed), a numerical
setting (controlled to reach a target accuracy), or a physical control (held fixed). That
classification lets a comparison state which of its differences are intended.
"""

from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any

from pydantic import Field, JsonValue, model_validator

from sim2real.contracts.base import (
    ArtifactRef,
    Contract,
    FileName,
    Identifier,
    NonEmptyStr,
    Observable,
    canonical_json,
    sha256_hex,
)

STRUCTURE = "structure"
"""Reserved variable name under which structure differences are reported."""


class ProtocolRef(Contract):
    name: Identifier
    version: NonEmptyStr


class FidelityTier(StrEnum):
    """Evidence expectations, not parameter bundles."""

    SCREENING = "screening"
    RESEARCH = "research"
    REFERENCE = "reference"


class ResourceBudget(Contract):
    max_core_hours: float | None = Field(default=None, gt=0)
    max_walltime_hours: float | None = Field(default=None, gt=0)
    max_jobs: int | None = Field(default=None, gt=0)


class ScientificTaskSpec(Contract):
    """A request for one calculation, expressed as scientific intent."""

    protocol: ProtocolRef
    observable: Observable
    estimand: NonEmptyStr
    """What the calculation estimates, e.g. "E(atop) - E(fcc) at fixed PBE geometries"."""
    system_label: NonEmptyStr
    structure: ArtifactRef | None = None
    interventions: dict[Identifier, JsonValue] = {}
    numerical_settings: dict[Identifier, JsonValue] = {}
    controls: dict[Identifier, JsonValue] = {}
    fidelity: FidelityTier = FidelityTier.SCREENING
    budget: ResourceBudget = ResourceBudget()
    authored_files: dict[FileName, ArtifactRef] = {}
    """Input files and job scripts written by an agent; the protocol generates any it needs
    that are not given. All of them are examined before anything runs."""

    @model_validator(mode="after")
    def _each_variable_classified_once(self) -> "ScientificTaskSpec":
        seen: dict[str, str] = {}
        for category, names in _categorized(self):
            for name in names:
                if name == STRUCTURE:
                    raise ValueError(f"{STRUCTURE!r} is reserved; use the structure field")
                if name in seen:
                    raise ValueError(f"variable {name!r} is classified as both {seen[name]} and {category}")
                seen[name] = category
        return self

    def identity_hash(self) -> str:
        """Hash of the scientific content, used to detect duplicate calculations.

        Descriptive labels, fidelity tier and budget are excluded; the structure and
        agent-written files count only through their content hashes.
        """
        return sha256_hex(canonical_json({
            "protocol": self.protocol.model_dump(mode="json"),
            "observable": self.observable.model_dump(mode="json"),
            "estimand": self.estimand,
            "structure_sha256": self.structure.sha256 if self.structure else None,
            "interventions": self.interventions,
            "numerical_settings": self.numerical_settings,
            "controls": self.controls,
            "authored_files": {name: ref.sha256 for name, ref in self.authored_files.items()},
        }))


def _categorized(task: ScientificTaskSpec) -> tuple[tuple[str, dict[str, Any]], ...]:
    return (
        ("intervention", task.interventions),
        ("numerical", task.numerical_settings),
        ("control", task.controls),
    )


class DifferenceKind(StrEnum):
    DECLARED = "declared"
    """Deliberately varied by the comparison."""
    NUMERICAL = "numerical"
    """Undeclared numerical-setting difference; needs a convergence justification."""
    UNCONTROLLED = "uncontrolled"


class VariableDifference(Contract):
    variable: str
    category: str
    values: tuple[JsonValue, ...]
    """One entry per compared task; ``None`` where the variable is absent."""
    kind: DifferenceKind


class ComparisonReport(Contract):
    declared: tuple[str, ...]
    differences: tuple[VariableDifference, ...]
    unused_declarations: tuple[str, ...]
    """Declared as varied but identical across all tasks."""

    @property
    def uncontrolled(self) -> tuple[VariableDifference, ...]:
        return tuple(d for d in self.differences if d.kind is DifferenceKind.UNCONTROLLED)

    @property
    def needs_numerical_justification(self) -> tuple[VariableDifference, ...]:
        return tuple(d for d in self.differences if d.kind is DifferenceKind.NUMERICAL)

    @property
    def is_controlled(self) -> bool:
        return not self.uncontrolled


def _all_equal(values: Sequence[Any]) -> bool:
    return len({canonical_json(v) for v in values}) == 1


def compare_tasks(tasks: Sequence[ScientificTaskSpec], varied: Iterable[str]) -> ComparisonReport:
    """Classify every difference between tasks meant to form one controlled comparison.

    Differences in observable or estimand are always uncontrolled: comparing, say, a
    fixed-geometry energy difference with an independently relaxed one mixes two
    estimands. Undeclared numerical differences are allowed but flagged, because equal
    accuracy does not always mean equal settings. A variable classified differently
    across tasks is uncontrolled.
    """
    if len(tasks) < 2:
        raise ValueError("a comparison needs at least two tasks")
    declared = frozenset(varied)
    differences: list[VariableDifference] = []

    fixed_aspects = (
        ("protocol", lambda t: t.protocol.model_dump(mode="json"), True),
        ("observable", lambda t: t.observable.model_dump(mode="json"), False),
        ("estimand", lambda t: t.estimand, False),
        (STRUCTURE, lambda t: t.structure.sha256 if t.structure else None, True),
    )
    for name, value_of, declarable in fixed_aspects:
        values = tuple(value_of(t) for t in tasks)
        if not _all_equal(values):
            kind = DifferenceKind.DECLARED if declarable and name in declared else DifferenceKind.UNCONTROLLED
            differences.append(VariableDifference(variable=name, category=name, values=values, kind=kind))

    per_task = [
        {name: (category, value) for category, names in _categorized(t) for name, value in names.items()}
        for t in tasks
    ]
    for name in sorted(set().union(*per_task)):
        entries = [variables.get(name) for variables in per_task]
        categories = {entry[0] for entry in entries if entry is not None}
        values = tuple(entry[1] if entry is not None else None for entry in entries)
        if len(categories) > 1:
            differences.append(VariableDifference(
                variable=name, category="mixed", values=values, kind=DifferenceKind.UNCONTROLLED,
            ))
            continue
        (category,) = categories
        if all(entry is not None for entry in entries) and _all_equal(values):
            continue
        if name in declared:
            kind = DifferenceKind.DECLARED
        elif category == "numerical":
            kind = DifferenceKind.NUMERICAL
        else:
            kind = DifferenceKind.UNCONTROLLED
        differences.append(VariableDifference(variable=name, category=category, values=values, kind=kind))

    differing = {d.variable for d in differences}
    return ComparisonReport(
        declared=tuple(sorted(declared)),
        differences=tuple(differences),
        unused_declarations=tuple(sorted(declared - differing)),
    )
