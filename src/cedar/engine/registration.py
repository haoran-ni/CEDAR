"""Registering runs computed outside CEDAR, so that readings and analyses of them can become evidence.

The researcher's starting calculations, and earlier runs an agent finds, usually sit in
directories CEDAR did not create. Registering one runs nothing. The code is recognized from the
inputs: ``control.in`` and ``geometry.in`` for FHI-aims, ``INCAR`` and ``POSCAR`` for VASP.
The validator:

1. stores the run's inputs and output by content hash (for FHI-aims ``control.in``,
   ``geometry.in`` and ``aims.out``; for VASP ``INCAR``, ``POSCAR``, ``KPOINTS`` and ``POTCAR``
   where kept, ``OUTCAR``, and ``OSZICAR``, ``CONTCAR``, ``vasprun.xml`` and ``vasp.out`` where
   kept and not too large), so evidence always points to exactly the files that were checked;
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

from cedar.codes.vasp.potcar import parse_potcar
from cedar.contracts.base import Actor, ArtifactRef, Observable, new_id
from cedar.contracts.events import EventDraft, RunRegistered
from cedar.contracts.evidence import RunReading
from cedar.contracts.examination import Finding, FindingSeverity, scientific_input_hash
from cedar.contracts.task import ProtocolRef, ScientificTaskSpec
from cedar.engine.results import VALIDATOR, verify_quote
from cedar.hpc.profile import ComputeProfile
from cedar.protocols import CODES, fhiaims_run, vasp_run
from cedar.protocols.fhiaims_run import REGISTERED_ESTIMAND, SPECIES_ROLE, FhiAimsRun
from cedar.protocols.vasp_run import VaspRun, outcar_potcars, outcar_report
from cedar.state.ledger import ResearchLedger
from cedar.state.projection import InvestigationState
from cedar.store.artifacts import ArtifactStore

MAX_INPUT_BYTES = 50 * 1024 ** 2
"""Largest input file registered; real ones are far smaller."""
MAX_EXTRA_OUTPUT_BYTES = 100 * 1024 ** 2
"""Largest output kept besides the one the run's health is checked from; a larger one stays in its directory."""


@dataclass(frozen=True)
class RunFiles:
    """What a registered run of one code is made of."""

    code: str
    """The code's name, as people write it."""
    protocol: ProtocolRef
    """The protocol its settings and health are recorded under."""
    inputs: tuple[str, ...]
    structure: str
    output: str
    """The output its health is checked from, unless another is named."""
    optional_inputs: tuple[str, ...] = ()
    optional_outputs: tuple[str, ...] = ()
    limitation: str = ""


FHI_AIMS = RunFiles("FHI-aims", FhiAimsRun.ref, ("control.in", "geometry.in"), "geometry.in", "aims.out",
                    limitation="One run at the settings its inputs used; convergence of the quantities taken from it "
                               "(and of differences between runs) with respect to k-points, basis, slab thickness and "
                               "vacuum must be established by a comparison of runs.")
VASP = RunFiles("VASP", VaspRun.ref, ("INCAR", "POSCAR"), "POSCAR", "OUTCAR", ("KPOINTS", "POTCAR"),
                ("OSZICAR", "CONTCAR", "vasprun.xml", "vasp.out"),
                limitation="One run at the settings its inputs used; convergence of the quantities taken from it (and "
                           "of differences between runs) with respect to k-points, plane-wave cutoff, slab thickness "
                           "and vacuum must be established by a comparison of runs.")


def run_files(directory: Path) -> RunFiles | None:
    """Which code's run ``directory`` holds, from its inputs."""
    if any((directory / name).is_file() for name in FHI_AIMS.inputs):
        return FHI_AIMS
    if any((directory / name).is_file() for name in VASP.inputs):
        return VASP
    return None


def registration_problem(directory: Path, output_file: str | None = None) -> str | None:
    """Why ``directory`` cannot be registered as a run, if its files already say: None if they do not."""
    kind = run_files(directory)
    if kind is None:
        return (f"{directory} holds neither an FHI-aims run (control.in, geometry.in) nor a VASP run (INCAR, POSCAR, "
                "OUTCAR)")
    output = output_file or kind.output
    missing = [name for name in (*kind.inputs, output) if not (directory / name).is_file()]
    if missing:
        return (f"{directory} has no {', '.join(missing)}; a registered {kind.code} run needs its "
                f"{', '.join(kind.inputs)} and output file (name it with output_file if it is not {kind.output})")
    if output in (*kind.inputs, *kind.optional_inputs):
        return f"{output} is an input file, not the run's output"
    return None


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
                 hypothesis_id: str | None = None, output_file: str | None = None, system_label: str | None = None,
                 observable: Observable | None = None, estimand: str | None = None,
                 intervention_id: str | None = None) -> RegistrationOutcome:
        """``output_file`` is the output the run's health is checked from: the code's usual one (aims.out, OUTCAR)
        when None."""
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
        problem = registration_problem(source, output_file)
        if problem is not None:
            return RegistrationOutcome(None, (problem,))
        kind = run_files(source)
        output_file = output_file or kind.output
        inputs = (*kind.inputs, *(name for name in kind.optional_inputs if (source / name).is_file()))
        large = [name for name in inputs if (source / name).stat().st_size > MAX_INPUT_BYTES]
        if large:
            return RegistrationOutcome(None, (f"{', '.join(large)} in {source} exceed {MAX_INPUT_BYTES} bytes",))
        notes: list[Finding] = []
        extra = []
        for name in kind.optional_outputs:
            path = source / name
            if name == output_file or not path.is_file():
                continue
            if path.stat().st_size > MAX_EXTRA_OUTPUT_BYTES:
                notes.append(Finding(severity=FindingSeverity.INFO, code="output_not_stored", file=name,
                                     message=f"{name} ({path.stat().st_size} bytes) is not stored; read it in "
                                             f"{source}"))
            else:
                extra.append(name)
        outputs = (output_file, *extra)

        stored = {name: self._artifacts.put_file(source / name) for name in (*inputs, *outputs)}
        texts = {name: self._artifacts.read(stored[name]).decode(errors="replace") for name in inputs}
        output = self._artifacts.read(stored[output_file])
        code = self._profile.codes.get(CODES[kind.protocol.name]) if self._profile is not None else None
        limitations = (registered_limitation(str(source)), kind.limitation)
        method = {"code": kind.code, "protocol": f"{kind.protocol.name} {kind.protocol.version}",
                  "origin": "registered", "source": str(source)}
        if kind is VASP:
            titles = ([d.title for d in parse_potcar(texts["POTCAR"]).datasets] if "POTCAR" in texts else
                      outcar_potcars(output.decode(errors="replace")))
            controls, numerical, settings_notes = vasp_run.registered_settings(texts["INCAR"], texts.get("KPOINTS"),
                                                                                titles)
            method |= {**controls, **numerical}
            stdout = self._artifacts.read(stored["vasp.out"]) if "vasp.out" in stored else None
            report = outcar_report(output, stdout, method=method, output_file=output_file, limitations=limitations)
            registered_observable = vasp_run.REGISTERED_OBSERVABLE
        else:
            species_root = Path(code.data_paths[SPECIES_ROLE]) if code and SPECIES_ROLE in code.data_paths else None
            controls, numerical, settings_notes = fhiaims_run.registered_settings(texts["control.in"],
                                                                                   texts["geometry.in"], species_root)
            method |= {**controls, **numerical}
            report = fhiaims_run.registered_report(output, controls, output_file=output_file, method=method,
                                                   limitations=limitations)
            registered_observable = fhiaims_run.REGISTERED_OBSERVABLE
        notes = settings_notes + notes
        version = report.method.get("code_version") or "unknown"
        if code is not None and version != code.version:
            limitations += (f"{output_file} reports {kind.code} {version}; the compute profile's {kind.code} is "
                            f"{code.version}.",)
        input_refs = {name: stored[name] for name in inputs}
        task = ScientificTaskSpec(
            protocol=kind.protocol, observable=observable or registered_observable,
            estimand=estimand or REGISTERED_ESTIMAND, system_label=system_label or source.name,
            structure=stored[kind.structure], controls=controls, numerical_settings=numerical,
            authored_files={name: ref for name, ref in input_refs.items() if name != kind.structure})
        input_hash = scientific_input_hash(task.protocol, version, input_refs)

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
            code_version=version, inputs=input_refs, input_hash=input_hash, notes=tuple(notes),
            outputs={name: stored[name] for name in outputs}, checks=report.checks,
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
