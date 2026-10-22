"""One FHI-aims run of one structure, at fixed geometry or relaxing it: ``fhiaims.run`` 0.4 (``FhiAimsRunJobChecked``).

The agent writes ``control.in`` and ``geometry.in``, and may give any other file the run
reads; the protocol generates only the job script, from the approved template, when the
agent does not write one. The run goes through Slurm (``job.sh``) or runs on the interactive
node CEDAR runs on (``execution: local``; ``local.sh``, the template without its Slurm
directives), never on a login node.

Before the run, the examination checks deterministically only what runs on the machine: that
``control.in``, ``geometry.in`` and the job script are there, and the job script against the
approved template, its resources against the profile's limits and the task's budget and, for
a local run, the allocation. It enforces nothing about the DFT settings or the other files:
it records what ``control.in`` sets and facts about the structure and the species blocks, and
the input reviewer, an agent, checks that FHI-aims will run the files as the task describes
and that nothing in them is harmful. After the run, the protocol checks its health from
FHI-aims's own markers and reads the final total energies; what else to take from a healthy
run is the agent's decision.

Versions 0.1 to 0.3 stay for the calculations recorded under them and accept no new
requests: 0.1 and 0.2 (``FhiAimsRunSlurmOnly``, ``FhiAimsRun``) held ``control.in`` to a
fixed list of keywords and to the task's declared settings, and 0.3 (``FhiAimsRunReviewed``)
to the declared settings.
"""

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import JsonValue

from cedar.codes.fhiaims.control import (
    ControlFile,
    normalize_lines,
    parse_control,
    render_control,
    species_default_path,
)
from cedar.codes.fhiaims.geometry import Geometry, parse_geometry
from cedar.codes.fhiaims.output import parse_output
from cedar.codes.structure import minimum_distance, vacuum_along_c
from cedar.contracts.base import Observable, Quantity, sha256_hex
from cedar.contracts.evidence import QualityCheck
from cedar.contracts.examination import Finding, FindingSeverity
from cedar.contracts.status import EvidenceQuality, SchedulerState
from cedar.contracts.task import ProtocolRef, ScientificTaskSpec
from cedar.protocols.base import Protocol, Reading, RunReport
from cedar.protocols.runs import CodeRun

_E, _R, _W, _I = FindingSeverity.ERROR, FindingSeverity.REVIEW, FindingSeverity.WARNING, FindingSeverity.INFO

CODE = "fhi-aims"
"""Key of the FHI-aims installation in a compute profile's ``codes``."""
SPECIES_ROLE = "species_defaults"
"""Key of the species-defaults directory in the installation's ``data_paths``."""
SPECIES_DEFAULTS = "species_defaults"

TOTAL_ENERGY = Observable(
    name="total energy",
    definition="FHI-aims 'Total energy of the DFT / Hartree-Fock s.c.f. calculation' in the final output "
               "block, for the final geometry",
    unit="eV",
)
ZERO_BROADENING_ENERGY = Observable(
    name="zero-broadening total energy",
    definition="FHI-aims 'Final zero-broadening corrected energy' in the final output block, for the final "
               "geometry; meaningful for metals with occupation broadening",
    unit="eV",
)

PHYSICAL_KEYWORDS = frozenset({
    "xc", "hse_unit", "hybrid_xc_coeff", "spin", "charge", "relativistic", "vdw_ts", "vdw_correction_hirshfeld",
    "many_body_dispersion", "many_body_dispersion_nl", "use_dipole_correction", "relax_geometry", "relax_unit_cell",
    "default_initial_moment", "fixed_spin_moment",
})
"""Keywords that change the physical model or the estimand; declared as interventions or controls."""
NUMERICAL_KEYWORDS = frozenset({
    "k_grid", "k_offset", "occupation_type", "sc_accuracy_rho", "sc_accuracy_eev", "sc_accuracy_etot",
    "sc_accuracy_forces", "sc_accuracy_stress", "basis_threshold",
})
"""Keywords that set numerical accuracy; declared as numerical settings."""
CONVERGENCE_PATH_KEYWORDS = frozenset({
    "mixer", "charge_mix_param", "n_max_pulay", "preconditioner", "ini_linear_mixing", "ini_linear_mix_param",
    "sc_iter_limit", "sc_init_iter", "sc_abandon_etot", "ks_method", "compute_forces", "final_forces_cleaned",
    "hessian_to_restart_geometry", "output_level", "output", "compensate_multipole_errors",
})
"""Keywords that change how a result is reached or reported, not what it is; need not be declared."""
FORBIDDEN_KEYWORDS = {
    "postprocess_anyway": "lets FHI-aims carry on after an unconverged SCF cycle, hiding the failure",
}
ALWAYS_DECLARED = ("xc", "spin", "charge", "relativistic", "occupation_type", SPECIES_DEFAULTS)
"""Stated explicitly by every task, so no record depends on an unstated code default."""
RUN_LIMITATION = (
    "One run at the declared numerical settings; convergence of the quantities taken from it (and of differences "
    "between runs) with respect to k-points, basis, slab thickness and vacuum must be established by a declared "
    "convergence comparison."
)
REGISTERED_OBSERVABLE = Observable(
    name="registered run",
    definition="the outputs of an FHI-aims run computed outside CEDAR, as registered; what is taken from them is "
               "declared with each reading or analysis",
)
REGISTERED_ESTIMAND = "what the registered run's outputs show, at the settings its inputs used"
_KEYWORD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SLAB_VACUUM = 5.0
"""Å of empty space along c above which a periodic structure is treated as a slab."""
MIN_DISTANCE = 0.5


def keyword_args(value: object) -> tuple[str, ...] | None:
    """A declared value as control.in arguments; None means the keyword is absent."""
    if value is True:
        return ()
    if value is False or value is None:
        return None
    if isinstance(value, (int, float)):
        return (f"{value:g}" if isinstance(value, float) else str(value),)
    if isinstance(value, str):
        return tuple(value.split())
    if isinstance(value, list) and all(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in value):
        return tuple(f"{v:g}" if isinstance(v, float) else str(v) for v in value)
    raise ValueError(f"cannot express {value!r} as control.in arguments")


def _args(values: Mapping[str, object], name: str) -> tuple[str, ...] | None:
    """Declared arguments of ``name``; None if undeclared, declared absent, or not expressible."""
    try:
        return keyword_args(values.get(name))
    except ValueError:
        return None


def _setting(name: str, args: tuple[str, ...]) -> str:
    return " ".join((name, *args))


def _same_args(expected: tuple[str, ...], actual: tuple[str, ...]) -> bool:
    if len(expected) != len(actual):
        return False
    for e, a in zip(expected, actual):
        try:
            if not math.isclose(float(e.replace("d", "e").replace("D", "E")),
                                float(a.replace("d", "e").replace("D", "E")), rel_tol=1e-9):
                return False
        except ValueError:
            if e.lower() != a.lower():
                return False
    return True


class FhiAimsRun(CodeRun):
    """``fhiaims.run`` 0.2, kept for the calculations recorded under it.

    Task variables are ``control.in`` keywords, named as in ``control.in``, plus the pseudo-variable
    ``species_defaults``: a level directory (such as ``light``) under the profile's species defaults. The protocol
    knows what each keyword in its lists means (physical model, numerical accuracy, or how convergence is reached),
    and the examiner checks that ``control.in`` sets exactly what was declared. A run goes through Slurm or on the
    interactive node CEDAR runs on.
    """

    ref = ProtocolRef(name="fhiaims.run", version="0.2")
    input_files = ("control.in", "geometry.in", "job.sh", "local.sh")
    structure_file = "geometry.in"
    execution_files = ("job.sh", "local.sh")
    output_files = ("aims.out",)
    code = CODE
    run_output = "aims.out"
    superseded_by = "0.4"
    generates_control = True
    """Whether a missing control.in is written from the declared keywords; from 0.4 on the agent writes it."""

    def input_settings(self, files: Mapping[str, bytes]) -> dict[str, str]:
        """Every keyword control.in sets, with its arguments as written; repeats are joined by '; '."""
        if "control.in" not in files:
            return {}
        settings: dict[str, list[str]] = {}
        for keyword in parse_control(files["control.in"].decode(errors="replace")).keywords:
            settings.setdefault(keyword.name, []).append(" ".join(keyword.args))
        return {name: "; ".join(values) for name, values in settings.items()}

    @property
    def _species_root(self) -> Path | None:
        path = self._code.data_paths.get(SPECIES_ROLE)
        return Path(path) if path else None

    def describe(self) -> str:
        root = self._species_root
        levels = sorted(d.name for d in root.iterdir() if d.is_dir()) if root and root.is_dir() else []
        return "\n".join([
            super().describe(),
            f"Code: FHI-aims {self.code_version}",
            "Variables are control.in keywords, named as in control.in, with their arguments as the value "
            "(a string such as \"atomic_zora scalar\", a number, a list such as [4, 4, 1], true for a flag, "
            "false for absent).",
            f"  Physical (declare as intervention or control): {', '.join(sorted(PHYSICAL_KEYWORDS))}",
            f"  Numerical (declare under numerical_settings): {', '.join(sorted(NUMERICAL_KEYWORDS))}, "
            f"{SPECIES_DEFAULTS}",
            f"  Convergence aids (may appear in control.in undeclared): {', '.join(sorted(CONVERGENCE_PATH_KEYWORDS))}",
            f"  Forbidden: {', '.join(f'{k} ({v})' for k, v in FORBIDDEN_KEYWORDS.items())}",
            f"Always declare: {', '.join(ALWAYS_DECLARED)}; and k_grid for a periodic structure.",
            f"{SPECIES_DEFAULTS} levels available on this system: {', '.join(levels) or 'none found'}",
            "If you write control.in yourself it must set exactly the declared physical and numerical keywords, "
            "with species blocks copied unchanged from the declared level (changed blocks need user review). "
            "Keywords the examiner does not know also need user review.",
            *self._where(),
        ])

    # -- declared variables -----------------------------------------------------------

    def resolve(self, task: ScientificTaskSpec) -> tuple[dict[str, object], list[Finding]]:
        """Declared variables checked against keyword meanings; returns {name: value}."""
        findings: list[Finding] = []
        values: dict[str, object] = {}
        declared_names: set[str] = set()
        for category, declared in (("intervention", task.interventions), ("control", task.controls),
                                   ("numerical", task.numerical_settings)):
            for name, value in declared.items():
                declared_names.add(name)
                physical = name in PHYSICAL_KEYWORDS
                numerical = name in NUMERICAL_KEYWORDS or name in CONVERGENCE_PATH_KEYWORDS or name == SPECIES_DEFAULTS
                if name in FORBIDDEN_KEYWORDS:
                    findings.append(Finding(severity=_E, code="forbidden_keyword",
                                            message=f"{name} {FORBIDDEN_KEYWORDS[name]}"))
                elif not physical and not numerical:
                    findings.append(Finding(severity=_E, code="unknown_variable",
                                            message=f"{name!r} is not a keyword {self.ref.name} {self.ref.version} "
                                                    "knows how to apply"))
                elif physical and category == "numerical":
                    findings.append(Finding(severity=_E, code="misclassified_variable",
                                            message=f"{name} changes the physical model; declare it as an "
                                                    "intervention or control"))
                elif numerical and category != "numerical":
                    findings.append(Finding(severity=_E, code="misclassified_variable",
                                            message=f"{name} is a numerical setting; declare it under "
                                                    "numerical_settings"))
                else:
                    values[name] = value
        for name in ALWAYS_DECLARED:
            if name not in declared_names:
                findings.append(Finding(severity=_E, code="missing_variable",
                                        message=f"the task must declare {name} explicitly"))
        for name, value in values.items():
            if name == SPECIES_DEFAULTS:
                findings += self._check_species_level(value)
                continue
            try:
                keyword_args(value)
            except ValueError as exc:
                findings.append(Finding(severity=_E, code="invalid_value", message=f"{name}: {exc}"))
        if "k_grid" in values:
            args = keyword_args(values["k_grid"]) or ()
            if len(args) != 3 or not all(a.isdigit() and int(a) >= 1 for a in args):
                findings.append(Finding(severity=_E, code="invalid_value", message="k_grid needs three positive integers"))
        return values, findings

    def _check_species_level(self, level: object) -> list[Finding]:
        root = self._species_root
        if root is None:
            return [Finding(severity=_E, code="no_species_defaults",
                            message=f"the compute profile records no {SPECIES_ROLE} directory for {CODE}")]
        if not isinstance(level, str) or "/" in level or level.startswith(".") or not (root / level).is_dir():
            return [Finding(severity=_E, code="unknown_species_level",
                            message=f"species_defaults={level!r} is not a directory under {root}")]
        return []

    @staticmethod
    def expected_keywords(values: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
        expected = {}
        for name, value in values.items():
            if name == SPECIES_DEFAULTS:
                continue
            args = keyword_args(value)
            if args is not None:
                expected[name] = args
        return expected

    @staticmethod
    def _relaxes(values: Mapping[str, object]) -> bool:
        return _args(values, "relax_geometry") is not None

    # -- generation -------------------------------------------------------------------

    def generate(self, task: ScientificTaskSpec, present: Mapping[str, bytes]) -> tuple[dict[str, bytes], list[Finding]]:
        generated: dict[str, bytes] = {}
        findings: list[Finding] = []
        if self.generates_control and "control.in" not in present and "geometry.in" in present:
            values, problems = self.resolve(task)
            if SPECIES_DEFAULTS not in values and not any(f.severity is _E for f in problems):
                findings.append(Finding(severity=_E, code="missing_variable", file="control.in",
                                        message=f"declare {SPECIES_DEFAULTS} (a level directory of the species "
                                                "defaults) so control.in can be generated, or write control.in"))
            elif not any(f.severity is _E for f in problems):
                geometry = parse_geometry(present["geometry.in"].decode(errors="replace"))
                generated["control.in"] = self._render_control(values, geometry).encode()
        script, problems = self._generate_script(task, present)
        generated |= script
        findings += problems + self._generated(generated)
        return generated, findings

    def _render_control(self, values: Mapping[str, object], geometry: Geometry) -> str:
        level = str(values[SPECIES_DEFAULTS])
        species = []
        for element in geometry.elements:
            path = species_default_path(self._species_root, level, element)
            if path is not None:
                species.append(path.read_text())
        keywords = [(name, " ".join(args)) for name, args in self.expected_keywords(values).items()]
        header = [f"Generated by CEDAR protocol {self.ref.name} {self.ref.version}",
                  f"species_defaults: {level}"]
        return render_control(header, keywords, species)

    # -- examination ------------------------------------------------------------------

    def examine(self, task: ScientificTaskSpec, files: Mapping[str, bytes]) -> list[Finding]:
        findings: list[Finding] = []
        values, problems = self.resolve(task)
        findings += problems
        if task.execution == "local":
            findings += self._examine_where()
        for name in ("control.in", "geometry.in", self._script(task)):
            if name not in files:
                findings.append(Finding(severity=_E, code="missing_file", file=name, message=f"{name} is missing"))
        geometry = None
        if "geometry.in" in files:
            geometry = parse_geometry(files["geometry.in"].decode(errors="replace"))
            findings += self._examine_geometry(geometry, values)
        if "control.in" in files and geometry is not None and not any(f.severity is _E for f in problems):
            findings += self._examine_control(files["control.in"].decode(errors="replace"), values, geometry)
        return findings + self._examine_scripts(task, files)

    def _examine_geometry(self, geometry: Geometry, values: Mapping[str, object]) -> list[Finding]:
        f = "geometry.in"
        findings = [Finding(severity=_E, code="malformed_geometry", file=f, line=line or None,
                            message=f"cannot read: {text}") for line, text in geometry.malformed_lines]
        findings += [Finding(severity=_R, code="unknown_geometry_keyword", file=f, line=line,
                             message=f"not recognized by the examiner: {text}") for line, text in geometry.unknown_lines]
        if geometry.malformed_lines:
            return findings
        if not geometry.atoms:
            findings.append(Finding(severity=_E, code="no_atoms", file=f, message="the structure has no atoms"))
            return findings
        atoms = geometry.to_atoms()
        shortest = minimum_distance(atoms)
        if shortest < MIN_DISTANCE:
            findings.append(Finding(severity=_E, code="overlapping_atoms", file=f,
                                    message=f"two atoms are {shortest:.3f} Å apart"))
        spin = _args(values, "spin")
        if spin and spin[0].lower() == "none" and any(a.initial_moment for a in geometry.atoms):
            findings.append(Finding(severity=_E, code="moments_without_spin", file=f,
                                    message="initial_moment is set but the task declares spin none"))
        relax = self._relaxes(values)
        fixed = sum(a.constrained for a in geometry.atoms)
        if fixed and not relax:
            findings.append(Finding(severity=_I, code="constraints_unused", file=f,
                                    message=f"{fixed} constrained atoms; constraints have no effect without relaxation"))
        k_grid = _args(values, "k_grid")
        if geometry.periodic:
            vacuum = vacuum_along_c(atoms)
            slab = vacuum >= SLAB_VACUUM
            findings.append(Finding(severity=_I, code="vacuum", file=f,
                                    message=f"{vacuum:.2f} Å of vacuum along c; treated as "
                                            f"{'a slab' if slab else 'periodic in all directions'}"))
            if slab and relax and not fixed:
                findings.append(Finding(severity=_W, code="unconstrained_slab", file=f,
                                        message="no atoms are constrained; the whole slab will relax"))
            if k_grid is None:
                findings.append(Finding(severity=_E, code="missing_variable",
                                        message="a periodic structure needs a declared k_grid"))
            elif slab and k_grid[-1] != "1":
                findings.append(Finding(severity=_E, code="k_points_across_vacuum",
                                        message=f"{vacuum:.1f} Å of vacuum along c needs a single k-point along c"))
        elif k_grid is not None:
            findings.append(Finding(severity=_E, code="k_points_without_cell",
                                    message="k_grid is declared but the structure is not periodic"))
        return findings

    def _examine_control(self, text: str, values: Mapping[str, object], geometry: Geometry) -> list[Finding]:
        f = "control.in"
        control = parse_control(text)
        expected = self.expected_keywords(values)
        findings: list[Finding] = []
        seen: set[str] = set()
        for keyword in control.keywords:
            name, where = keyword.name, {"file": f, "line": keyword.line}
            if name in seen and name != "output":
                findings.append(Finding(severity=_E, code="repeated_keyword", message=f"{name} is set twice", **where))
                continue
            seen.add(name)
            if name in FORBIDDEN_KEYWORDS:
                findings.append(Finding(severity=_E, code="forbidden_keyword",
                                        message=f"{name} {FORBIDDEN_KEYWORDS[name]}", **where))
            elif name in expected:
                if not _same_args(expected[name], keyword.args):
                    findings.append(Finding(
                        severity=_E, code="contradicts_task",
                        message=f"{_setting(name, keyword.args)} does not match the declared "
                                f"{_setting(name, expected[name])}", **where))
            elif name in PHYSICAL_KEYWORDS or name in NUMERICAL_KEYWORDS:
                findings.append(Finding(severity=_E, code="undeclared_setting",
                                        message=f"{_setting(name, keyword.args)} changes the calculation, but "
                                                "the task does not declare it", **where))
            elif name not in CONVERGENCE_PATH_KEYWORDS:
                findings.append(Finding(severity=_R, code="unknown_keyword",
                                        message=f"{name} is not known to {self.ref.name} {self.ref.version}; its "
                                                "effect cannot be verified", **where))
        for name, args in expected.items():
            if name not in seen:
                findings.append(Finding(severity=_E, code="missing_keyword", file=f,
                                        message=f"the task declares {_setting(name, args)}, but control.in "
                                                "does not set it"))
        for name in ("sc_accuracy_rho", "sc_accuracy_eev", "sc_accuracy_etot"):
            if name not in expected:
                findings.append(Finding(severity=_W, code="code_default", file=f,
                                        message=f"{name} is not declared; FHI-aims {self.code_version} uses its own "
                                                "default"))
        findings += self._examine_species(control, values, geometry)
        return findings

    def _examine_species(self, control, values: Mapping[str, object], geometry: Geometry) -> list[Finding]:
        f = "control.in"
        level = str(values[SPECIES_DEFAULTS])
        findings: list[Finding] = []
        for element in geometry.elements:
            block = control.species_block(element)
            if block is None:
                findings.append(Finding(severity=_E, code="missing_species", file=f,
                                        message=f"no species block for {element}"))
                continue
            default = species_default_path(self._species_root, level, element)
            if default is None:
                findings.append(Finding(severity=_E, code="no_species_default", file=f, line=block.line,
                                        message=f"no {level} species default for {element}"))
            elif block.normalized() != normalize_lines(default.read_text().splitlines()):
                findings.append(Finding(severity=_R, code="modified_species", file=f, line=block.line,
                                        message=f"species block for {element} differs from {default}, so "
                                                f"species_defaults={level} does not describe it"))
        for block in control.species:
            if block.species not in geometry.elements:
                findings.append(Finding(severity=_W, code="unused_species", file=f, line=block.line,
                                        message=f"species {block.species} does not occur in the structure"))
        return findings

    # -- run checks -------------------------------------------------------------------

    def check_run(self, task: ScientificTaskSpec, outputs: Mapping[str, bytes],
                  scheduler_state: SchedulerState | None) -> RunReport:
        values, _ = self.resolve(task)
        method: dict = {"code": "FHI-aims", "protocol": f"{self.ref.name} {self.ref.version}", **values}
        scheduler = QualityCheck(name="scheduler_completed", passed=scheduler_state is SchedulerState.COMPLETED,
                                 detail=f"scheduler state {scheduler_state}")
        return output_report(outputs.get("aims.out"), relax=self._relaxes(values), method=method,
                             checks=(scheduler,), expected_version=self.code_version)


def output_report(raw: bytes | None, *, relax: bool, method: dict, checks: Sequence[QualityCheck] = (),
                  expected_version: str | None = None, output_file: str = "aims.out",
                  limitations: tuple[str, ...] = (RUN_LIMITATION,)) -> RunReport:
    """The health of one FHI-aims run from its output, after any ``checks`` made before; ``expected_version`` is the
    version the run must report, if one is required."""
    checks = list(checks)
    if raw is None:
        checks.append(QualityCheck(name="output_present", passed=False, detail=f"{output_file} is missing"))
        return RunReport(tuple(checks), EvidenceQuality.INVALID, (), limitations, method)
    out = parse_output(raw.decode(errors="replace"))
    method["code_version"] = out.version_string
    end = out.normal_termination
    if expected_version is not None:
        checks.append(QualityCheck(name="code_version", passed=out.version_string == expected_version,
                                   detail=f"{output_file} reports {out.version_string}; profile records "
                                          f"{expected_version}"))
    checks += [
        QualityCheck(name="normal_termination", passed=end is not None,
                     detail=f"'Have a nice day.' at line {end.line}" if end else
                     (f"killed (line {out.killed.line})" if out.killed else "no normal end of run")),
        QualityCheck(name="scf_converged", passed=bool(out.scf_converged) and not out.scf_failures,
                     detail=f"'Self-consistency cycle converged.' appears {len(out.scf_converged)} time(s)"
                            + (f"; failure at line {out.scf_failures[0].line}" if out.scf_failures else "")),
    ]
    if relax:
        converged = out.geometry_converged is not None and out.optimization_aborted is None
        checks.append(QualityCheck(
            name="geometry_converged", passed=converged,
            detail=f"converged at line {out.geometry_converged.line} after {out.relaxation_steps} steps"
            if converged else f"not converged after {out.relaxation_steps} steps"))
    readings = tuple(
        Reading(observable.name, Quantity(value=energy.value, unit="eV"), output_file, energy.line, energy.line,
                energy.text.strip())
        for observable, energy in ((TOTAL_ENERGY, out.final_total_energy),
                                   (ZERO_BROADENING_ENERGY, out.final_zero_broadening_energy))
        if energy is not None)
    only_geometry_failed = all(c.passed for c in checks if c.name != "geometry_converged")
    failure = EvidenceQuality.INCONCLUSIVE if relax and only_geometry_failed else EvidenceQuality.INVALID
    return RunReport(tuple(checks), failure, readings, limitations, method)


class FhiAimsRunSlurmOnly(FhiAimsRun):
    """``fhiaims.run`` 0.1: the same examination and checks, through Slurm only."""

    ref = ProtocolRef(name="fhiaims.run", version="0.1")
    input_files = ("control.in", "geometry.in", "job.sh")
    execution_files = ("job.sh",)
    runs_locally = False


REVIEWED_LIMITATION = (
    "Its inputs were checked deterministically only for what runs on the machine (the job script against the approved "
    "template, its resources and the task's budget) and for consistency with the task's own declarations; their "
    "scientific content was judged by the input reviewer, an agent, or by the researcher, not verified by rules.")


class FhiAimsRunReviewed(FhiAimsRun):
    """``fhiaims.run`` 0.3: the input reviewer, an agent, judges the scientific content of the inputs.

    The examination keeps only what needs no knowledge of FHI-aims keywords: the job script against the approved
    template, its resources, the budget and, for a local run, the allocation; that control.in sets every declared
    keyword as declared; that the files can be read and every element has a species block; and that nothing hides
    an unconverged SCF cycle from the run check. Everything else it observes (keywords the task does not declare,
    species blocks that differ from the declared level, distances, vacuum) is recorded as a fact for the reviewer,
    which reads the files, the installed code's manual and source, and the documentation on the web.
    """

    ref = ProtocolRef(name="fhiaims.run", version="0.3")
    reviewed_by_agent = True
    superseded_by = "0.4"
    limitation = REVIEWED_LIMITATION
    """What a run's evidence must say about how its inputs were checked."""

    def describe(self) -> str:
        root = self._species_root
        levels = sorted(d.name for d in root.iterdir() if d.is_dir()) if root and root.is_dir() else []
        return "\n".join([
            Protocol.describe(self),
            f"Code: FHI-aims {self.code_version}",
            "Declare every setting your test depends on, as control.in keywords named as in control.in, with their "
            "arguments as the value (a string such as \"atomic_zora scalar\", a number, a list such as [4, 4, 1], true "
            "for a flag, false for absent): what you vary as interventions, physical conditions you hold fixed as "
            "controls, accuracy settings as numerical_settings, and species_defaults, the level directory of the "
            "species blocks. control.in must set each declared keyword exactly as declared.",
            f"{SPECIES_DEFAULTS} levels available on this system: {', '.join(levels) or 'none found'}",
            "No keyword list limits what you may set. The input reviewer, an agent, reads your files with the "
            "installed FHI-aims manual and source and judges whether they make sense for your test: it approves them "
            "(the researcher then authorizes submission), warns (the researcher looks at them first) or rejects them "
            "(you are woken with its reasons). Declare what matters: undeclared settings are shown to it as such.",
            f"Forbidden: {', '.join(f'{k} ({v})' for k, v in FORBIDDEN_KEYWORDS.items())}",
            "The job script is checked against the researcher's approved template, its limits and your budget.",
            *self._where(),
        ])

    def resolve(self, task: ScientificTaskSpec) -> tuple[dict[str, object], list[Finding]]:
        """Every declared variable, whatever its name; only what cannot be written into control.in is refused."""
        findings: list[Finding] = []
        values: dict[str, object] = {}
        for declared in (task.interventions, task.controls, task.numerical_settings):
            values.update(declared)
        for name, value in values.items():
            if name in FORBIDDEN_KEYWORDS:
                findings.append(Finding(severity=_E, code="forbidden_keyword",
                                        message=f"{name} {FORBIDDEN_KEYWORDS[name]}"))
            elif name == SPECIES_DEFAULTS:
                findings += self._check_species_level(value)
            elif not _KEYWORD_NAME.fullmatch(name):
                findings.append(Finding(severity=_E, code="invalid_variable",
                                        message=f"{name!r} cannot be a control.in keyword"))
            else:
                try:
                    keyword_args(value)
                except ValueError as exc:
                    findings.append(Finding(severity=_E, code="invalid_value", message=f"{name}: {exc}"))
        return values, findings

    def _examine_geometry(self, geometry: Geometry, values: Mapping[str, object]) -> list[Finding]:
        f = "geometry.in"
        findings = [Finding(severity=_E, code="malformed_geometry", file=f, line=line or None,
                            message=f"cannot read: {text}") for line, text in geometry.malformed_lines]
        if geometry.malformed_lines:
            return findings
        if not geometry.atoms:
            return findings + [Finding(severity=_E, code="no_atoms", file=f, message="the structure has no atoms")]
        findings += [Finding(severity=_I, code="geometry_keyword", file=f, line=line,
                             message=f"not interpreted by the examiner: {text}")
                     for line, text in geometry.unknown_lines]
        atoms = geometry.to_atoms()
        fixed = sum(a.constrained for a in geometry.atoms)
        moments = sum(1 for a in geometry.atoms if a.initial_moment)
        facts = [f"{len(geometry.atoms)} atoms ({', '.join(geometry.elements)})",
                 f"{fixed} constrained" if fixed else "none constrained",
                 f"{moments} with an initial moment" if moments else "no initial moments"]
        if len(geometry.atoms) > 1:
            facts.append(f"shortest interatomic distance {minimum_distance(atoms):.3f} Å")
        if geometry.periodic:
            facts.append(f"periodic; {vacuum_along_c(atoms):.2f} Å of empty space along the third lattice vector")
        else:
            facts.append("not periodic")
        findings.append(Finding(severity=_I, code="structure", file=f, message="; ".join(facts)))
        return findings

    def _examine_control(self, text: str, values: Mapping[str, object], geometry: Geometry) -> list[Finding]:
        f = "control.in"
        control = parse_control(text)
        expected = self.expected_keywords(values)
        findings: list[Finding] = []
        seen: dict[str, int] = {}
        for keyword in control.keywords:
            name, where = keyword.name, {"file": f, "line": keyword.line}
            if name in FORBIDDEN_KEYWORDS:
                findings.append(Finding(severity=_E, code="forbidden_keyword",
                                        message=f"{name} {FORBIDDEN_KEYWORDS[name]}", **where))
            elif name in expected:
                if name in seen:
                    findings.append(Finding(severity=_E, code="repeated_keyword", message=f"the declared {name} is "
                                            f"set again (first at line {seen[name]})", **where))
                elif not _same_args(expected[name], keyword.args):
                    findings.append(Finding(
                        severity=_E, code="contradicts_task",
                        message=f"{_setting(name, keyword.args)} does not match the declared "
                                f"{_setting(name, expected[name])}", **where))
            else:
                findings.append(Finding(severity=_I, code="undeclared_setting",
                                        message=f"{_setting(name, keyword.args)} is set but not declared by the task",
                                        **where))
            seen.setdefault(name, keyword.line)
        for name, args in expected.items():
            if name not in seen:
                findings.append(Finding(severity=_E, code="missing_keyword", file=f,
                                        message=f"the task declares {_setting(name, args)}, but control.in "
                                                "does not set it"))
        level = values.get(SPECIES_DEFAULTS)
        for element in geometry.elements:
            block = control.species_block(element)
            if block is None:
                findings.append(Finding(severity=_E, code="missing_species", file=f,
                                        message=f"no species block for {element}"))
                continue
            default = species_default_path(self._species_root, str(level), element) if level else None
            if level and default is None:
                findings.append(Finding(severity=_I, code="species_level", file=f, line=block.line,
                                        message=f"no {level} species default for {element} to compare with"))
            elif default is not None and block.normalized() != normalize_lines(default.read_text().splitlines()):
                findings.append(Finding(severity=_I, code="species_differs", file=f, line=block.line,
                                        message=f"species block for {element} differs from {default}"))
        findings += [Finding(severity=_I, code="unused_species", file=f, line=block.line,
                             message=f"species {block.species} does not occur in the structure")
                     for block in control.species if block.species not in geometry.elements]
        return findings

    def check_run(self, task: ScientificTaskSpec, outputs: Mapping[str, bytes],
                  scheduler_state: SchedulerState | None) -> RunReport:
        """As 0.2, except that a relaxation is recognized from the output too: control.in may relax the geometry
        without the task declaring it."""
        values, _ = self.resolve(task)
        raw = outputs.get("aims.out")
        shown = parse_output(raw.decode(errors="replace")) if raw is not None else None
        relax = self._relaxes(values) or (shown is not None and (
            shown.relaxation_steps > 0 or shown.geometry_converged is not None
            or shown.optimization_aborted is not None))
        method: dict = {"code": "FHI-aims", "protocol": f"{self.ref.name} {self.ref.version}", **values}
        scheduler = QualityCheck(name="scheduler_completed", passed=scheduler_state is SchedulerState.COMPLETED,
                                 detail=f"scheduler state {scheduler_state}")
        return output_report(raw, relax=relax, method=method, checks=(scheduler,),
                             expected_version=self.code_version, limitations=(RUN_LIMITATION, self.limitation))


JOB_CHECKED_LIMITATION = (
    "Its inputs were checked deterministically only for what runs on the machine (the job script against the approved "
    "template, its resources and the task's budget). Whether FHI-aims runs them as the task describes was judged by "
    "the input reviewer, an agent, or by the researcher, not verified by rules; the settings the task declares are "
    "its own description, and what control.in sets is recorded from the file.")


class FhiAimsRunJobChecked(FhiAimsRunReviewed):
    """``fhiaims.run`` 0.4: nothing about the DFT settings is enforced by rules; the input reviewer checks the inputs.

    The examination checks only what runs on the machine: that control.in, geometry.in and the job script are there,
    the job script against the approved template, its resources, the budget and, for a local run, the allocation. A
    fixed list of rules cannot know a code's inputs and raises false alarms, so the task's declared settings are the
    agent's own description of the calculation, in whatever terms suit it, and are not compared with control.in.
    What control.in sets, the structure and which species defaults the blocks equal are recorded as facts for the
    input reviewer, which checks that FHI-aims will run the files as the task describes and that nothing in them is
    harmful. The agent writes control.in itself, and may give any other file the run reads under any plain name but
    the output's; CEDAR stores such files without reading them, and the input reviewer checks them. Only the job
    script is generated.
    """

    ref = ProtocolRef(name="fhiaims.run", version="0.4")
    superseded_by = None
    generates_control = False
    checks_declarations = False
    other_files = True
    limitation = JOB_CHECKED_LIMITATION

    def describe(self) -> str:
        root = self._species_root
        levels = sorted(d.name for d in root.iterdir() if d.is_dir()) if root and root.is_dir() else []
        return "\n".join([
            Protocol.describe(self),
            f"Code: FHI-aims {self.code_version}",
            "Write control.in and geometry.in yourself, for example by copying the researcher's files and changing "
            "what your test changes. Only the job script is generated, from the researcher's approved template, if "
            "you do not write it.",
            "Declare what the calculation is, in whatever terms describe it: what you vary as interventions, physical "
            "conditions you hold fixed as controls, accuracy settings as numerical_settings, e.g. {\"dispersion\": "
            "\"MBD-NL\"}, {\"adsorption_site\": \"top\", \"n_layers\": 4}, {\"k_grid\": [4, 4, 1]}. These describe "
            "the calculation for the record, the input reviewer and the comparisons; CEDAR does not check them "
            "against the files, and records what control.in actually sets from the file.",
            f"{SPECIES_DEFAULTS} levels available on this system: {', '.join(levels) or 'none found'}",
            "Other files the run reads, such as a restart file, may be given too, under any plain file name but the "
            "output's (aims.out); give files that are not text by path. CEDAR stores them without reading them. "
            "Every input file is read-only while the run runs and must be unchanged after it: a run that rewrites one "
            "fails its check.",
            "The input reviewer, an agent, checks that FHI-aims will run your files as you describe and that nothing "
            "in them is harmful; your scientific choices are yours. If it finds a problem, you are woken with its "
            "reasons. Then the researcher approves the run, the last step before it starts.",
            "The job script is checked against the researcher's approved template, its limits and your budget.",
            *self._where(),
        ])

    def resolve(self, task: ScientificTaskSpec) -> tuple[dict[str, object], list[Finding]]:
        """Every declared variable as declared; none is refused, since none is written into control.in."""
        return {**task.interventions, **task.controls, **task.numerical_settings}, []

    def examine(self, task: ScientificTaskSpec, files: Mapping[str, bytes]) -> list[Finding]:
        findings: list[Finding] = self._examine_where() if task.execution == "local" else []
        for name in ("control.in", "geometry.in", self._script(task)):
            if name not in files:
                findings.append(Finding(severity=_E, code="missing_file", file=name, message=f"{name} is missing"
                                        + ("; write it, this protocol generates only the job script"
                                           if name == "control.in" else "")))
        findings += self._other_files(files)
        if "geometry.in" in files:
            geometry = parse_geometry(files["geometry.in"].decode(errors="replace"))
            findings += self._structure_facts(geometry)
            if "control.in" in files and not geometry.malformed_lines:
                findings += self._species_facts(parse_control(files["control.in"].decode(errors="replace")), geometry)
        return findings + self._examine_scripts(task, files)

    def _structure_facts(self, geometry: Geometry) -> list[Finding]:
        """What the structure is, and what CEDAR's reader could not read, as facts: FHI-aims is the judge of its
        own input."""
        f = "geometry.in"
        unread = [Finding(severity=_I, code="geometry_unread", file=f, line=line or None,
                          message=f"CEDAR's reader cannot read this line: {text}")
                  for line, text in geometry.malformed_lines]
        if unread:
            return unread
        if not geometry.atoms:
            return [Finding(severity=_I, code="structure", file=f, message="CEDAR's reader finds no atoms")]
        return super()._examine_geometry(geometry, {})

    def _species_facts(self, control: ControlFile, geometry: Geometry) -> list[Finding]:
        """Which species-defaults level each element's block equals, or that it has none."""
        notes: list[Finding] = []
        levels = _species_levels(control, geometry, self._species_root, notes)
        facts = [Finding(severity=_I, code=n.code, file=n.file, line=n.line, message=n.message) for n in notes]
        if isinstance(levels, str):
            facts.append(Finding(severity=_I, code="species_level", file="control.in",
                                 message=f"every species block equals the {levels} species defaults"))
        else:
            facts.append(Finding(severity=_I, code="species_level", file="control.in",
                                 message="species blocks: " + ", ".join(f"{element} {level}"
                                                                         for element, level in levels.items())))
        facts += [Finding(severity=_I, code="unused_species", file="control.in", line=block.line,
                          message=f"species {block.species} does not occur in the structure")
                  for block in control.species if block.species not in geometry.elements]
        return facts


# -- runs computed outside CEDAR ----------------------------------------------------------


def declared_value(args: tuple[str, ...]) -> JsonValue:
    """control.in arguments as a task declares them: true for a flag, a number, a list of numbers, or the text."""
    if not args:
        return True
    numbers: list[int | float] = []
    for arg in args:
        try:
            numbers.append(int(arg))
            continue
        except ValueError:
            pass
        try:
            number = float(arg.replace("d", "e").replace("D", "E"))
        except ValueError:
            return " ".join(args)
        if not math.isfinite(number):
            return " ".join(args)
        numbers.append(number)
    return numbers[0] if len(numbers) == 1 else numbers


def registered_settings(control_text: str, geometry_text: str, species_root: Path | None
                        ) -> tuple[dict[str, JsonValue], dict[str, JsonValue], list[Finding]]:
    """The settings a run's inputs used, classified as a task declares them: (controls, numerical settings, notes).

    Physical keywords, and keywords this protocol does not know, become controls, so that a comparison of runs
    sees every difference; numerical keywords become numerical settings; keywords that only change how a result
    is reached are left out, as they are for a task. ``species_defaults`` names the species-defaults level the
    species blocks equal, per element where they differ, or a hash of a block that equals none.
    """
    control = parse_control(control_text)
    geometry = parse_geometry(geometry_text)
    controls: dict[str, JsonValue] = {}
    numerical: dict[str, JsonValue] = {}
    notes: list[Finding] = []
    grouped: dict[str, list] = {}
    for keyword in control.keywords:
        grouped.setdefault(keyword.name, []).append(keyword)
    for name, occurrences in grouped.items():
        if name in CONVERGENCE_PATH_KEYWORDS:
            continue
        where = {"file": "control.in", "line": occurrences[0].line}
        if not _KEYWORD_NAME.fullmatch(name) or name in (SPECIES_DEFAULTS, "structure"):
            notes.append(Finding(severity=_W, code="unrecorded_keyword", message=f"{name} cannot be recorded as a "
                                 "setting; compare runs that use it by their files", **where))
            continue
        value = (declared_value(occurrences[0].args) if len(occurrences) == 1
                 else [" ".join(k.args) for k in occurrences])
        if name in NUMERICAL_KEYWORDS:
            numerical[name] = value
            continue
        controls[name] = value
        if name in FORBIDDEN_KEYWORDS:
            notes.append(Finding(severity=_W, code="forbidden_keyword", message=f"{name} {FORBIDDEN_KEYWORDS[name]}",
                                 **where))
        elif name not in PHYSICAL_KEYWORDS:
            notes.append(Finding(severity=_I, code="unclassified_keyword",
                                 message=f"{name} is not known to {FhiAimsRun.ref.name} {FhiAimsRun.ref.version}; "
                                         "recorded among the controls", **where))
    notes += [Finding(severity=_W, code="malformed_geometry", file="geometry.in", line=line or None,
                      message=f"cannot read: {text}") for line, text in geometry.malformed_lines]
    numerical[SPECIES_DEFAULTS] = _species_levels(control, geometry, species_root, notes)
    return controls, numerical, notes


def _species_levels(control: ControlFile, geometry: Geometry, root: Path | None, notes: list[Finding]) -> JsonValue:
    levels = sorted(d.name for d in root.iterdir() if d.is_dir()) if root is not None and root.is_dir() else []
    found: dict[str, str] = {}
    for element in geometry.elements:
        block = control.species_block(element)
        if block is None:
            notes.append(Finding(severity=_W, code="missing_species", file="control.in",
                                 message=f"no species block for {element}"))
            found[element] = "missing"
            continue
        normalized = block.normalized()
        found[element] = next(
            (level for level in levels if (path := species_default_path(root, level, element)) is not None
             and normalize_lines(path.read_text().splitlines()) == normalized),
            "sha256:" + sha256_hex("\n".join(normalized))[:16])
    unmatched = sorted(e for e, level in found.items() if level.startswith("sha256:"))
    if unmatched:
        notes.append(Finding(severity=_I, code="species_not_a_default", file="control.in",
                             message=f"species blocks of {', '.join(unmatched)} equal no species-defaults level "
                                     + ("of the compute profile" if levels else "(no compute profile to compare "
                                                                                "with)") + "; recorded by hash"))
    distinct = set(found.values())
    if len(distinct) == 1 and not unmatched and "missing" not in distinct:
        return distinct.pop()
    return found


def registered_report(raw: bytes, controls: Mapping[str, JsonValue], *, output_file: str, method: dict,
                      limitations: tuple[str, ...]) -> RunReport:
    """The health of a registered run from its output: no scheduler state, and the version it reports is recorded,
    not required."""
    relax = _args(controls, "relax_geometry") is not None
    return output_report(raw, relax=relax, method=method, output_file=output_file, limitations=limitations)
