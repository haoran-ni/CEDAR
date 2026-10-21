"""One FHI-aims run of one structure, at fixed geometry or relaxing it (``fhiaims.run`` 0.1).

The protocol examines the inputs before the run and checks the run's health after it. What
to take from a healthy run is declared by the agent, who reads it from the outputs or
computes it with an analysis script; the protocol itself reads the final total energies,
which every run prints.

Task variables are FHI-aims ``control.in`` keywords, named as in ``control.in``, plus the
pseudo-variable ``species_defaults``: a level directory (such as ``light``) under the
species-defaults directory of the approved compute profile. The protocol knows what each
keyword means: whether it changes the physical model, sets numerical accuracy, or only
affects how convergence is reached. It holds no site paths and no preferred values;
agents declare every setting and the examiner checks that ``control.in`` implements
exactly what was declared.
"""

import math
from collections.abc import Mapping
from pathlib import Path

from sim2real.codes.fhiaims.control import normalize_lines, parse_control, render_control, species_default_path
from sim2real.codes.fhiaims.geometry import Geometry, minimum_distance, parse_geometry, vacuum_along_c
from sim2real.codes.fhiaims.output import parse_output
from sim2real.contracts.base import Observable, Quantity
from sim2real.contracts.evidence import QualityCheck
from sim2real.contracts.examination import Finding, FindingSeverity
from sim2real.contracts.status import EvidenceQuality, SchedulerState
from sim2real.contracts.task import ProtocolRef, ScientificTaskSpec
from sim2real.hpc.jobscript import JobResources, examine_job_script, render_job_script
from sim2real.hpc.profile import ComputeProfile
from sim2real.protocols.base import Protocol, Reading, RunReport

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


class FhiAimsRun(Protocol):
    ref = ProtocolRef(name="fhiaims.run", version="0.1")
    input_files = ("control.in", "geometry.in", "job.sh")
    structure_file = "geometry.in"
    execution_files = ("job.sh",)
    output_files = ("aims.out",)

    def __init__(self, profile: ComputeProfile) -> None:
        if CODE not in profile.codes:
            raise ValueError(f"compute profile {profile.name} has no {CODE!r} installation")
        self._profile = profile
        self._code = profile.codes[CODE]

    @property
    def code_version(self) -> str:
        return self._code.version

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
        if "control.in" not in present and "geometry.in" in present:
            values, problems = self.resolve(task)
            if not any(f.severity is _E for f in problems):
                geometry = parse_geometry(present["geometry.in"].decode(errors="replace"))
                generated["control.in"] = self._render_control(values, geometry).encode()
        if "job.sh" not in present:
            resources = self._default_resources(task)
            if resources is None:
                findings.append(Finding(severity=_E, code="budget_too_small",
                                        message="the task budget does not allow one minute at the profile's default "
                                                "task count"))
            else:
                generated["job.sh"] = render_job_script(self._profile, self._code, resources, job_name="sim2real",
                                                        output_file="aims.out").encode()
        for name in sorted(generated):
            findings.append(Finding(severity=_I, code="generated", file=name,
                                    message=f"{name} was generated by {self.ref.name} {self.ref.version}"))
        return generated, findings

    def _render_control(self, values: Mapping[str, object], geometry: Geometry) -> str:
        level = str(values[SPECIES_DEFAULTS])
        species = []
        for element in geometry.elements:
            path = species_default_path(self._species_root, level, element)
            if path is not None:
                species.append(path.read_text())
        keywords = [(name, " ".join(args)) for name, args in self.expected_keywords(values).items()]
        header = [f"Generated by sim2real protocol {self.ref.name} {self.ref.version}",
                  f"species_defaults: {level}"]
        return render_control(header, keywords, species)

    def _default_resources(self, task: ScientificTaskSpec) -> JobResources | None:
        limits = self._profile.limits
        minutes = limits.default_walltime_minutes
        if task.budget.max_walltime_hours is not None:
            minutes = min(minutes, math.floor(task.budget.max_walltime_hours * 60))
        if task.budget.max_core_hours is not None:
            minutes = min(minutes, math.floor(task.budget.max_core_hours * 60 / limits.default_ntasks))
        if minutes < 1:
            return None
        nodes = math.ceil(limits.default_ntasks / limits.cores_per_node)
        return JobResources(limits.default_partition, nodes, limits.default_ntasks, minutes)

    # -- examination ------------------------------------------------------------------

    def examine(self, task: ScientificTaskSpec, files: Mapping[str, bytes]) -> list[Finding]:
        findings: list[Finding] = []
        values, problems = self.resolve(task)
        findings += problems
        for name in self.input_files:
            if name not in files:
                findings.append(Finding(severity=_E, code="missing_file", file=name, message=f"{name} is missing"))
        geometry = None
        if "geometry.in" in files:
            geometry = parse_geometry(files["geometry.in"].decode(errors="replace"))
            findings += self._examine_geometry(geometry, values)
        if "control.in" in files and geometry is not None and not any(f.severity is _E for f in problems):
            findings += self._examine_control(files["control.in"].decode(errors="replace"), values, geometry)
        if "job.sh" in files:
            script = examine_job_script(files["job.sh"].decode(errors="replace"), self._profile, self._code,
                                        output_file="aims.out")
            findings += script.findings
            if script.resources is not None:
                findings += self._examine_budget(task, script.resources)
        return findings

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

    @staticmethod
    def _examine_budget(task: ScientificTaskSpec, resources: JobResources) -> list[Finding]:
        findings = []
        budget = task.budget
        if budget.max_core_hours is not None and resources.core_hours > budget.max_core_hours:
            findings.append(Finding(severity=_E, code="over_budget", file="job.sh",
                                    message=f"{resources.core_hours:g} core-hours requested; the task allows "
                                            f"{budget.max_core_hours:g}"))
        if budget.max_walltime_hours is not None and resources.walltime_minutes > budget.max_walltime_hours * 60:
            findings.append(Finding(severity=_E, code="over_budget", file="job.sh",
                                    message=f"{resources.walltime_minutes} minutes requested; the task allows "
                                            f"{budget.max_walltime_hours:g} h"))
        return findings

    # -- run checks -------------------------------------------------------------------

    def check_run(self, task: ScientificTaskSpec, outputs: Mapping[str, bytes],
                  scheduler_state: SchedulerState | None) -> RunReport:
        values, _ = self.resolve(task)
        relax = self._relaxes(values)
        limitations = (
            "One run at the declared numerical settings; convergence of the quantities taken from it (and of "
            "differences between runs) with respect to k-points, basis, slab thickness and vacuum must be "
            "established by a declared convergence comparison.",
        )
        method: dict = {"code": "FHI-aims", "protocol": f"{self.ref.name} {self.ref.version}", **values}
        checks = [QualityCheck(name="scheduler_completed", passed=scheduler_state is SchedulerState.COMPLETED,
                               detail=f"scheduler state {scheduler_state}")]
        raw = outputs.get("aims.out")
        if raw is None:
            checks.append(QualityCheck(name="output_present", passed=False, detail="aims.out is missing"))
            return RunReport(tuple(checks), EvidenceQuality.INVALID, (), limitations, method)
        out = parse_output(raw.decode(errors="replace"))
        method["code_version"] = out.version_string
        end = out.normal_termination
        checks += [
            QualityCheck(name="code_version", passed=out.version_string == self.code_version,
                         detail=f"aims.out reports {out.version_string}; profile records {self.code_version}"),
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
            Reading(observable.name, Quantity(value=energy.value, unit="eV"), "aims.out", energy.line, energy.line,
                    energy.text.strip())
            for observable, energy in ((TOTAL_ENERGY, out.final_total_energy),
                                       (ZERO_BROADENING_ENERGY, out.final_zero_broadening_energy))
            if energy is not None)
        only_geometry_failed = all(c.passed for c in checks if c.name != "geometry_converged")
        failure = EvidenceQuality.INCONCLUSIVE if relax and only_geometry_failed else EvidenceQuality.INVALID
        return RunReport(tuple(checks), failure, readings, limitations, method)
