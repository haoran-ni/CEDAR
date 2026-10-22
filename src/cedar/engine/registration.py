"""Registering runs computed outside CEDAR, so that readings and analyses of them can become evidence.

The researcher's starting calculations, and earlier runs an agent finds, usually sit in
directories CEDAR did not create. Registering one runs nothing. The validator:

1. stores the run's ``control.in``, ``geometry.in`` and output by content hash, so evidence
   always points to exactly the files that were checked;
2. checks the run's health from its output with the protocol's own markers (normal end, SCF
   and geometry convergence); there is no scheduler state, and the code version the output
   reports is recorded, not required;
3. records the settings the inputs actually used as the run's task, classified as a task
   declares them, so a comparison of runs sees what differs.

A registered run is marked as computed outside CEDAR: its inputs were not examined before it
ran, and its outputs existed before any prediction about it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cedar.contracts.base import Actor, ArtifactRef, Observable, new_id
from cedar.contracts.events import EventDraft, RunRegistered
from cedar.contracts.evidence import RunReading
from cedar.contracts.examination import scientific_input_hash
from cedar.contracts.task import ScientificTaskSpec
from cedar.engine.results import VALIDATOR, verify_quote
from cedar.hpc.profile import ComputeProfile
from cedar.protocols.fhiaims_run import (
    CODE,
    REGISTERED_ESTIMAND,
    REGISTERED_OBSERVABLE,
    SPECIES_ROLE,
    FhiAimsRun,
    registered_report,
    registered_settings,
)
from cedar.state.ledger import ResearchLedger
from cedar.state.projection import InvestigationState
from cedar.store.artifacts import ArtifactStore

INPUT_FILES = ("control.in", "geometry.in")
MAX_INPUT_BYTES = 50 * 1024 ** 2
"""Largest control.in or geometry.in registered; real ones are far smaller."""


def registered_limitation(source: str) -> str:
    return (f"Computed outside CEDAR and registered from {source}: its inputs were not examined before it ran, and "
            "its outputs existed before any prediction recorded about it.")


@dataclass(frozen=True)
class RegistrationOutcome:
    calculation_id: str | None
    """None when nothing was registered; ``problems`` says why."""
    problems: tuple[str, ...] = ()
    reused: bool = False
    """The same files from the same directory were already registered for the same hypothesis (or shared)."""


class RunRegistrar:
    def __init__(self, ledger: ResearchLedger, artifacts: ArtifactStore, *, profile: ComputeProfile | None = None,
                 actor: Actor = VALIDATOR, readable: Callable[[Path], Path] | None = None) -> None:
        """``profile``, if the investigation has one, names the species-defaults levels and the code version runs
        are compared with. ``readable`` resolves a directory the requester may read, raising if it may not."""
        self._ledger = ledger
        self._artifacts = artifacts
        self._profile = profile
        self._actor = actor
        self._readable = readable

    def register(self, investigation_id: str, directory: str | Path, *, requested_by: Actor, description: str,
                 hypothesis_id: str | None = None, output_file: str = "aims.out", system_label: str | None = None,
                 observable: Observable | None = None, estimand: str | None = None,
                 intervention_id: str | None = None) -> RegistrationOutcome:
        path = Path(directory).expanduser()
        if not path.is_absolute():
            return RegistrationOutcome(None, (f"give the run directory as an absolute path, not {directory}",))
        source = self._readable(path) if self._readable is not None else path.resolve()
        if not source.is_dir():
            return RegistrationOutcome(None, (f"{source} is not a directory",))
        if self._profile is not None and source.is_relative_to(Path(self._profile.runs_root).resolve()):
            return RegistrationOutcome(None, (f"{source} is a run directory of CEDAR's own calculations; use that "
                                              "calculation directly instead of registering it as a run computed "
                                              "outside CEDAR",))
        names = (*INPUT_FILES, output_file)
        missing = [name for name in names if not (source / name).is_file()]
        if missing:
            return RegistrationOutcome(None, (f"{source} has no {', '.join(missing)}; a registered FHI-aims run needs "
                                              "its control.in, geometry.in and output file (name it with "
                                              "output_file if it is not aims.out)",))
        large = [name for name in INPUT_FILES if (source / name).stat().st_size > MAX_INPUT_BYTES]
        if large:
            return RegistrationOutcome(None, (f"{', '.join(large)} in {source} exceed {MAX_INPUT_BYTES} bytes",))
        if output_file in INPUT_FILES:
            return RegistrationOutcome(None, (f"{output_file} is an input file, not the run's output",))

        stored = {name: self._artifacts.put_file(source / name) for name in names}
        texts = {name: self._artifacts.read(stored[name]).decode(errors="replace") for name in INPUT_FILES}
        code = self._profile.codes.get(CODE) if self._profile is not None else None
        species_root = Path(code.data_paths[SPECIES_ROLE]) if code and SPECIES_ROLE in code.data_paths else None
        controls, numerical, notes = registered_settings(texts["control.in"], texts["geometry.in"], species_root)
        method = {"code": "FHI-aims", "protocol": f"{FhiAimsRun.ref.name} {FhiAimsRun.ref.version}",
                  "origin": "registered", "source": str(source), **controls, **numerical}
        limitations = (registered_limitation(str(source)),
                       "One run at the settings its inputs used; convergence of the quantities taken from it (and of "
                       "differences between runs) with respect to k-points, basis, slab thickness and vacuum must "
                       "be established by a comparison of runs.")
        output = self._artifacts.read(stored[output_file])
        report = registered_report(output, controls, output_file=output_file, method=method, limitations=limitations)
        version = report.method.get("code_version") or "unknown"
        if code is not None and version != code.version:
            limitations += (f"{output_file} reports FHI-aims {version}; the compute profile's FHI-aims is "
                            f"{code.version}.",)
        inputs = {name: stored[name] for name in INPUT_FILES}
        task = ScientificTaskSpec(
            protocol=FhiAimsRun.ref, observable=observable or REGISTERED_OBSERVABLE,
            estimand=estimand or REGISTERED_ESTIMAND, system_label=system_label or source.name,
            structure=stored["geometry.in"], controls=controls, numerical_settings=numerical,
            authored_files={"control.in": stored["control.in"]})
        input_hash = scientific_input_hash(task.protocol, version, inputs)

        state = self._ledger.load(investigation_id)
        existing = self._existing(state, str(source), input_hash, stored[output_file], hypothesis_id)
        if existing is not None:
            return RegistrationOutcome(existing, reused=True)
        readings = tuple(
            RunReading(name=r.name, value=r.value, file=r.file, line_start=r.line_start, line_end=r.line_end,
                       quote=r.quote)
            for r in report.readings if verify_quote(output, r.line_start, r.line_end, r.quote) is None)
        calculation_id = new_id("calc")
        self._ledger.record(investigation_id, [EventDraft(actor=self._actor, payload=RunRegistered(
            calculation_id=calculation_id, hypothesis_id=hypothesis_id, requested_by=requested_by,
            description=description, source=str(source), task=task, task_hash=task.identity_hash(),
            code_version=version, inputs=inputs, input_hash=input_hash, notes=tuple(notes),
            outputs={output_file: stored[output_file]}, checks=report.checks,
            failure_quality=report.failure_quality, readings=readings, limitations=limitations,
            method=report.method, intervention_id=intervention_id))])
        return RegistrationOutcome(calculation_id)

    @staticmethod
    def _existing(state: InvestigationState, source: str, input_hash: str, output: ArtifactRef,
                  hypothesis_id: str | None) -> str | None:
        for calc in state.calculations.values():
            registration = calc.registration
            if (registration is not None and registration.source == source and registration.input_hash == input_hash
                    and output.sha256 in {ref.sha256 for ref in registration.outputs.values()}
                    and registration.hypothesis_id == hypothesis_id):
                return calc.calculation_id
        return None
