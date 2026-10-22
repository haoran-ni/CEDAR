"""One VASP run of one structure, at fixed geometry or relaxing it: ``vasp.run`` 0.1 (``VaspRun``).

The agent writes ``INCAR``, ``POSCAR`` and, unless INCAR's ``KSPACING`` is to set the
k-points, ``KPOINTS``. It gives ``POTCAR`` by path, or declares ``potcar``, the PAW datasets
of POSCAR's species in order, and the protocol writes POTCAR from the compute profile's
library. It may give any other file the run reads. The job script is generated from the
approved template when the agent does not write one, launching the build the inputs need:
the profile's ``ncl`` build when INCAR sets LNONCOLLINEAR or LSORBIT, the standard one
otherwise. The run goes through Slurm (``job.sh``) or runs on the interactive node CEDAR
runs on (``execution: local``; ``local.sh``), never on a login node.

Before the run, the examination checks deterministically what runs on the machine, as
``fhiaims.run`` 0.4 does: that INCAR, POSCAR, POTCAR and the job script are there, and the
job script against the approved template, its resources against the profile's limits and the
task's budget and, for a local run, the allocation. Two things about the inputs are rules too,
because VASP would otherwise stop at once or silently compute something else: POTCAR has one
dataset per species of POSCAR, for the elements POSCAR names, in its order; and a
non-collinear INCAR runs on the ncl build. Nothing about the DFT settings is enforced: the
examination records what INCAR sets, the k-point set, which library dataset each part of
POTCAR equals and facts about the structure, and points out INCAR tags the installed VASP
never reads and lines it ignores, for the input reviewer, an agent, which checks that VASP
will run the files as the task describes and that nothing in them is harmful. After the run,
the protocol checks its health from OUTCAR's markers and reads the final energies; what else
to take from a healthy run is the agent's decision.
"""

import functools
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import JsonValue

from cedar.codes.structure import minimum_distance, vacuum_along_c
from cedar.codes.vasp.incar import Incar, parse_incar
from cedar.codes.vasp.kpoints import parse_kpoints
from cedar.codes.vasp.output import fatal_error, parse_outcar
from cedar.codes.vasp.poscar import Poscar, element_of, parse_poscar
from cedar.codes.vasp.potcar import Potcar, library_dataset, parse_potcar, render_potcar
from cedar.contracts.base import Observable, Quantity
from cedar.contracts.evidence import QualityCheck
from cedar.contracts.examination import Finding, FindingSeverity
from cedar.contracts.status import EvidenceQuality, SchedulerState
from cedar.contracts.task import ProtocolRef, ScientificTaskSpec
from cedar.protocols.base import Protocol, Reading, RunReport
from cedar.protocols.runs import CodeRun

_E, _W, _I = FindingSeverity.ERROR, FindingSeverity.WARNING, FindingSeverity.INFO

CODE = "vasp"
"""Key of the VASP installation in a compute profile's ``codes``."""
POTCARS = "potcars"
"""Key of the PAW library (e.g. a ``potpaw_PBE`` directory) in the installation's ``data_paths``."""
SOURCE = "source"
"""Key of the VASP source directory (its ``src``) in the installation's ``data_paths``; found next to the binary's
directory when not given."""
NCL, GAM = "ncl", "gam"
"""Names of the non-collinear and the Γ-only builds among the installation's ``variants``."""
POTCAR_SETTING = "potcar"
"""The declared setting naming the PAW datasets of POSCAR's species, from which POTCAR is written."""
NONCOLLINEAR_TAGS = ("LNONCOLLINEAR", "LSORBIT")
RESTART_FILES = {"WAVECAR": "LWAVE", "CHGCAR": "LCHARG"}
"""Files a run may read to restart, which VASP writes again at the end unless the tag is .FALSE."""

FREE_ENERGY = Observable(
    name="free energy TOTEN",
    definition="VASP 'free  energy   TOTEN' at the end of the last ionic step in OUTCAR: the electronic free energy, "
               "including the smearing term -TS",
    unit="eV",
)
SIGMA0_ENERGY = Observable(
    name="energy(sigma->0)",
    definition="VASP 'energy(sigma->0)' at the end of the last ionic step in OUTCAR: the total energy extrapolated "
               "to zero smearing",
    unit="eV",
)
RUN_LIMITATION = (
    "One run at the declared numerical settings; convergence of the quantities taken from it (and of differences "
    "between runs) with respect to k-points, plane-wave cutoff, slab thickness and vacuum must be established by a "
    "declared convergence comparison."
)
LIMITATION = (
    "Its inputs were checked deterministically only for what runs on the machine (the job script against the approved "
    "template, its resources and the task's budget, the build against INCAR) and for POTCAR matching POSCAR's "
    "species. Whether VASP runs them as the task describes was judged by the input reviewer, an agent, or by the "
    "researcher, not verified by rules; the settings the task declares are its own description, and what INCAR sets "
    "is recorded from the file.")
REGISTERED_OBSERVABLE = Observable(
    name="registered run",
    definition="the outputs of a VASP run computed outside CEDAR, as registered; what is taken from them is declared "
               "with each reading or analysis",
)
_QUOTED = re.compile(r"""['"]([A-Za-z][A-Za-z0-9_]*)['"]""")
_POTCAR_LINE = re.compile(r"^\s*POTCAR:\s+(.+?)\s*$")


def declared_datasets(task: ScientificTaskSpec) -> list[str] | None:
    """The PAW datasets the task declares as ``potcar``, in order; None if it declares none or not as names."""
    for declared in (task.controls, task.numerical_settings, task.interventions):
        value = declared.get(POTCAR_SETTING)
        if value is None:
            continue
        if isinstance(value, str):
            return value.split() or None
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            return list(value)
        return None
    return None


@functools.lru_cache(maxsize=8)
def _source_names(directory: str, modified: float) -> frozenset[str]:
    """Every quoted name in the Fortran sources of ``directory``, in upper case: the INCAR tags it can read are among
    them. ``modified`` is the directory's modification time, so a rebuilt tree is read again."""
    names: set[str] = set()
    for path in Path(directory).glob("*.F"):
        try:
            names.update(n.upper() for n in _QUOTED.findall(path.read_text(errors="replace")))
        except OSError:
            continue
    return frozenset(names)


class VaspRun(CodeRun):
    """``vasp.run`` 0.1: the agent writes the inputs, or names the PAW datasets for POTCAR; the input reviewer checks
    them.

    Like ``fhiaims.run`` 0.4, the task's declared settings are the agent's own description of the calculation and are
    not compared with the files; the examination records what the files set, for the input reviewer, and rules on what
    runs on the machine, on POTCAR against POSCAR, and on the build a non-collinear INCAR needs.
    """

    ref = ProtocolRef(name="vasp.run", version="0.1")
    input_files = ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "job.sh", "local.sh")
    structure_file = "POSCAR"
    execution_files = ("job.sh", "local.sh")
    output_files = ("vasp.out", "OUTCAR", "OSZICAR", "CONTCAR", "vasprun.xml")
    code = CODE
    run_output = "vasp.out"
    reviewed_by_agent = True
    checks_declarations = False
    other_files = True
    limitation = LIMITATION

    @property
    def _library(self) -> Path | None:
        path = self._code.data_paths.get(POTCARS)
        return Path(path) if path else None

    @property
    def source(self) -> Path | None:
        """The installed VASP's source directory, if the profile names it or it lies next to the binary's."""
        if SOURCE in self._code.data_paths:
            return Path(self._code.data_paths[SOURCE])
        for binary in self._code.binaries:
            candidate = Path(binary).parent.parent / "src"
            if candidate.is_dir() and any(candidate.glob("*.F")):
                return candidate
        return None

    def describe(self) -> str:
        library = self._library
        builds = ", ".join([f"std {self._code.binary}", *(f"{n} {p}" for n, p in self._code.variants.items())])
        lines = [
            Protocol.describe(self),
            f"Code: VASP {self.code_version}; builds: {builds}",
            "Write INCAR, and KPOINTS unless INCAR's KSPACING is to set the k-points; give POSCAR, naming its species "
            "(VASP 5 format), as the structure. Copy the researcher's files where you can and change what your test "
            "changes.",
            "POTCAR: give your own by path (input_paths), or declare potcar, the PAW datasets of POSCAR's species in "
            "its order, e.g. {\"potcar\": [\"Te\", \"Pb_d\"]} among the controls, and CEDAR writes POTCAR from "
            + (f"the library {library}, whose directory names are its datasets (e.g. Fe, Fe_pv, Fe_sv); list it with "
               "the file tools to choose." if library else "the profile's PAW library, which this profile does not "
               "name: give POTCAR yourself.")
            + " POTCAR must have one dataset per species of POSCAR, for the same elements in the same order, or the "
            "request is rejected: VASP would stop, or put the wrong potentials on the atoms.",
            f"The generated job script launches the build the inputs need: {NCL} when INCAR sets LNONCOLLINEAR or "
            "LSORBIT, the standard build otherwise"
            + (f"; to run the Γ-only build ({GAM}), write the job script yourself launching it." if GAM in
               self._code.variants else "."),
            "Declare what the calculation is, in whatever terms describe it: what you vary as interventions, physical "
            "conditions you hold fixed as controls, accuracy settings as numerical_settings, e.g. {\"xc\": "
            "\"PBE+D3\"}, {\"magnetic order\": \"AFM\"}, {\"ENCUT\": 520, \"k-points\": [6, 6, 1]}. These describe "
            "the calculation for the record, the input reviewer and the comparisons; CEDAR does not check them "
            "against the files (potcar only names what to write), and records what INCAR actually sets from the file.",
            "Other files the run reads, such as WAVECAR or CHGCAR to restart from, may be given too, under any plain "
            f"file name but the outputs' ({', '.join(self.output_files)}); give files that are not text by path. "
            "CEDAR stores them without reading them. Every input file is read-only while the run runs and must be "
            "unchanged after it: VASP writes WAVECAR and CHGCAR at the end of a run unless INCAR sets LWAVE = .FALSE. "
            "and LCHARG = .FALSE., so set those when you give the files.",
            f"After the run, {', '.join(self.output_files)} are stored and OUTCAR's final energies read ("
            f"{FREE_ENERGY.name}, {SIGMA0_ENERGY.name}); other files stay in the run directory.",
            "The input reviewer, an agent, checks that VASP will run your files as you describe and that nothing in "
            "them is harmful; your scientific choices are yours. If it finds a problem, you are woken with its "
            "reasons. Then the researcher approves the run, the last step before it starts.",
            "The job script is checked against the researcher's approved template, its limits and your budget.",
            *self._where(),
        ]
        return "\n".join(lines)

    def resolve(self, task: ScientificTaskSpec) -> tuple[dict[str, object], list[Finding]]:
        """Every declared variable as declared; none is refused, since none is written into INCAR."""
        return {**task.interventions, **task.controls, **task.numerical_settings}, []

    def input_settings(self, files: Mapping[str, bytes]) -> dict[str, str]:
        """Every tag INCAR sets, as written (repeats joined by '; '), the k-point set KPOINTS gives and the datasets
        POTCAR holds."""
        settings: dict[str, list[str]] = {}
        if "INCAR" in files:
            for tag in parse_incar(files["INCAR"].decode(errors="replace")).tags:
                settings.setdefault(tag.name, []).append(tag.value)
        joined = {name: "; ".join(values) for name, values in settings.items()}
        if "KPOINTS" in files:
            joined["KPOINTS"] = parse_kpoints(files["KPOINTS"].decode(errors="replace")).summary
        if "POTCAR" in files:
            potcar = parse_potcar(files["POTCAR"].decode(errors="replace"))
            joined["POTCAR"] = "; ".join(d.title for d in potcar.datasets)
        return joined

    # -- generation -------------------------------------------------------------------

    def generate(self, task: ScientificTaskSpec, present: Mapping[str, bytes]
                 ) -> tuple[dict[str, bytes], list[Finding]]:
        generated: dict[str, bytes] = {}
        findings: list[Finding] = []
        names = declared_datasets(task)
        if "POTCAR" not in present and names is not None:
            library = self._library
            if library is None:
                findings.append(Finding(severity=_E, code="no_potcar_library", file="POTCAR",
                                        message=f"the compute profile names no PAW library ({POTCARS}) for {CODE}; "
                                                "give POTCAR yourself"))
            else:
                missing = [name for name in names if library_dataset(library, name) is None]
                if missing:
                    findings += [Finding(severity=_E, code="unknown_dataset", file="POTCAR",
                                         message=f"{library} has no dataset {name}"
                                                 + self._alternatives(library, name)) for name in missing]
                else:
                    generated["POTCAR"] = render_potcar(library, names).encode()
        elif "POTCAR" not in present and any(POTCAR_SETTING in d for d in (task.controls, task.numerical_settings,
                                                                            task.interventions)):
            findings.append(Finding(severity=_E, code="invalid_value", file="POTCAR",
                                    message=f"{POTCAR_SETTING} names the PAW datasets, as a list such as [\"Te\", "
                                            "\"Pb_d\"]"))
        incar = parse_incar(present["INCAR"].decode(errors="replace")) if "INCAR" in present else None
        binary = self._code.variants.get(NCL) if incar is not None and self._noncollinear(incar) else None
        script, problems = self._generate_script(task, present, binary)
        generated |= script
        findings += problems + self._generated(generated)
        return generated, findings

    @staticmethod
    def _alternatives(library: Path, name: str) -> str:
        element = element_of(name)
        if element is None:
            return ""
        options = sorted(p.name for p in library.glob(f"{element}*")
                         if element_of(p.name) == element and (p / "POTCAR").is_file())
        return f"; its datasets for {element}: {', '.join(options)}" if options else f"; it has none for {element}"

    @staticmethod
    def _noncollinear(incar: Incar) -> bool:
        return any(incar.flag(tag) for tag in NONCOLLINEAR_TAGS)

    # -- examination ------------------------------------------------------------------

    def examine(self, task: ScientificTaskSpec, files: Mapping[str, bytes]) -> list[Finding]:
        findings: list[Finding] = self._examine_where() if task.execution == "local" else []
        hints = {"INCAR": "; write it, this protocol generates only POTCAR and the job script",
                 "POTCAR": f"; give it by path, or declare {POTCAR_SETTING}, the PAW datasets of POSCAR's species in "
                           "order, for CEDAR to write it"}
        for name in ("INCAR", "POSCAR", "POTCAR", self._script(task)):
            if name not in files:
                findings.append(Finding(severity=_E, code="missing_file", file=name,
                                        message=f"{name} is missing{hints.get(name, '')}"))
        findings += self._other_files(files)
        incar = parse_incar(files["INCAR"].decode(errors="replace")) if "INCAR" in files else None
        if incar is not None:
            findings += self._incar_facts(incar) + self._restart_facts(incar, files)
        findings += self._kpoints_facts(files, incar)
        poscar = parse_poscar(files["POSCAR"].decode(errors="replace")) if "POSCAR" in files else None
        potcar = parse_potcar(files["POTCAR"].decode(errors="replace")) if "POTCAR" in files else None
        if potcar is not None:
            findings += self._potcar_facts(potcar)
        if poscar is not None:
            findings += self._species_rules(poscar, potcar) + self._structure_facts(poscar, potcar)
        if incar is not None:
            findings += self._build_facts(incar, files)
        return findings + self._examine_scripts(task, files)

    def _incar_facts(self, incar: Incar) -> list[Finding]:
        """Lines VASP ignores, tags set more than once, and tags the installed VASP never reads. An ignored line is
        a warning only when it starts with a tag the installed VASP reads, as when '=' was left out; most are
        headings."""
        f = "INCAR"
        source = self.source
        known = _source_names(str(source), source.stat().st_mtime) if source is not None and source.is_dir() else None
        findings = []
        for line, text in incar.ignored_lines:
            word = text.split()[0].upper()
            if known and word in known:
                findings.append(Finding(severity=_W, code="ignored_line", file=f, line=line,
                                        message=f"VASP ignores this line, which sets no tag: {text}; {word} is a tag "
                                                "it reads, but only after '='"))
            else:
                findings.append(Finding(severity=_I, code="ignored_line", file=f, line=line,
                                        message=f"VASP ignores this line, which sets no tag (a tag needs '='): {text}"))
        lines: dict[str, list[int]] = {}
        for tag in incar.tags:
            lines.setdefault(tag.name, []).append(tag.line)
        findings += [Finding(severity=_I, code="repeated_tag", file=f, line=at[1],
                             message=f"{name} is set {len(at)} times, at lines {', '.join(map(str, at))}")
                     for name, at in lines.items() if len(at) > 1]
        if known:
            findings += [Finding(severity=_W, code="unread_tag", file=f, line=at[0],
                                 message=f"{name} is named nowhere in the installed VASP's source ({source}); VASP "
                                         "silently ignores tags it does not read")
                         for name, at in lines.items() if name not in known]
        return findings

    @staticmethod
    def _restart_facts(incar: Incar, files: Mapping[str, bytes]) -> list[Finding]:
        return [Finding(severity=_W, code="rewritten_input", file=name,
                        message=f"{name} is an input, so read-only during the run, but VASP writes {name} at the end "
                                f"of a run unless INCAR sets {tag} = .FALSE.; if it does, the run fails")
                for name, tag in RESTART_FILES.items() if name in files and incar.flag(tag) is not False]

    @staticmethod
    def _kpoints_facts(files: Mapping[str, bytes], incar: Incar | None) -> list[Finding]:
        if "KPOINTS" not in files:
            spacing = incar.get("KSPACING") if incar is not None else None
            return [Finding(severity=_I, code="kpoints", message="no KPOINTS: VASP generates the mesh from KSPACING ("
                            + (f"INCAR sets {spacing.value})" if spacing else "not set in INCAR, so VASP's default)"))]
        kpoints = parse_kpoints(files["KPOINTS"].decode(errors="replace"))
        findings = [Finding(severity=_I, code="kpoints", file="KPOINTS", message=kpoints.summary)]
        findings += [Finding(severity=_I, code="kpoints_unread", file="KPOINTS", line=line or None,
                             message=f"CEDAR's reader cannot read this line: {text}")
                     for line, text in kpoints.problems]
        return findings

    def _potcar_facts(self, potcar: Potcar) -> list[Finding]:
        """The datasets POTCAR holds, and which of them equal the library's."""
        f = "POTCAR"
        findings = [Finding(severity=_I, code="potcar_unread", file=f, line=line, message=text)
                    for line, text in potcar.problems]
        library = self._library
        for dataset in potcar.datasets:
            what = f"{dataset.title}" + (f", ZVAL {dataset.valence:g}" if dataset.valence is not None else "")
            path = library_dataset(library, dataset.name) if library is not None else None
            if library is None:
                where = "the compute profile names no PAW library to compare it with"
            elif path is None:
                where = f"{library} has no dataset {dataset.name}"
            elif dataset.same_as(path.read_text(errors="replace")):
                where = f"equals {path}"
            else:
                where = f"differs from {path}"
            findings.append(Finding(severity=_I, code="potcar_dataset", file=f, line=dataset.line,
                                    message=f"{what}: {where}"))
        return findings

    @staticmethod
    def _species_rules(poscar: Poscar, potcar: Potcar | None) -> list[Finding]:
        """POTCAR must hold one dataset per species of POSCAR, for the elements POSCAR names, in its order: otherwise
        VASP stops, or puts the wrong potential on the atoms without a word."""
        if potcar is None or not poscar.counts or not potcar.datasets:
            return []
        datasets = potcar.datasets
        if len(datasets) != len(poscar.counts):
            return [Finding(severity=_E, code="potcar_mismatch", file="POTCAR",
                            message=f"POTCAR has {len(datasets)} datasets ({', '.join(d.name for d in datasets)}) but "
                                    f"POSCAR {len(poscar.counts)} species; VASP needs one dataset per species, in "
                                    "POSCAR's order")]
        if poscar.species is None:
            return [Finding(severity=_I, code="species_from_potcar", file="POSCAR",
                            message="POSCAR does not name its species; VASP takes them from POTCAR: "
                                    + ", ".join(d.element or d.name for d in datasets))]
        findings = []
        for number, (name, dataset) in enumerate(zip(poscar.species, datasets), start=1):
            element = element_of(name)
            if element is not None and dataset.element is not None and element != dataset.element:
                findings.append(Finding(severity=_E, code="potcar_mismatch", file="POTCAR", line=dataset.line,
                                        message=f"species {number} of POSCAR is {name}, but dataset {number} of POTCAR "
                                                f"is {dataset.title}: VASP would put the {dataset.element} potential "
                                                f"on the {element} atoms"))
        return findings

    @staticmethod
    def _structure_facts(poscar: Poscar, potcar: Potcar | None) -> list[Finding]:
        """What the structure is, and what CEDAR's reader could not read, as facts: VASP is the judge of its own
        input."""
        f = "POSCAR"
        findings = [Finding(severity=_I, code="poscar_unread", file=f, line=line or None,
                            message=f"CEDAR's reader cannot read this: {text}") for line, text in poscar.problems]
        if not poscar.atoms:
            return findings + [Finding(severity=_I, code="structure", file=f, message="CEDAR's reader finds no atoms")]
        names = poscar.species or (tuple(d.name for d in potcar.datasets) if potcar is not None else None)
        symbols = [element_of(n) for n in names] if names and len(names) == len(poscar.counts) else None
        groups = ", ".join(f"{count} {names[i] if names and i < len(names) else f'species {i + 1}'}"
                           for i, count in enumerate(poscar.counts))
        fixed = sum(any(a.fixed) for a in poscar.atoms)
        facts = [f"{len(poscar.atoms)} atoms ({groups})",
                 f"{fixed} with directions fixed by selective dynamics" if fixed else
                 ("selective dynamics, nothing fixed" if poscar.selective else "no selective dynamics")]
        if symbols is not None and None not in symbols and len(poscar.atoms) == sum(poscar.counts):
            atoms = poscar.to_atoms(symbols)
            if len(atoms) > 1:
                facts.append(f"shortest interatomic distance {minimum_distance(atoms):.3f} Å")
            facts.append(f"{vacuum_along_c(atoms):.2f} Å of empty space along the third lattice vector")
        findings.append(Finding(severity=_I, code="structure", file=f, message="; ".join(facts)))
        return findings

    def _build_facts(self, incar: Incar, files: Mapping[str, bytes]) -> list[Finding]:
        """Which build the job script launches; a non-collinear INCAR must run on the ncl build, if the profile has
        one, since VASP's other builds stop on it."""
        examined = self._examined_script(files)
        if examined is None or examined[1].binary is None:
            return []
        script, binary = examined[0], examined[1].binary
        builds = {path: name for name, path in self._code.variants.items()}
        build = builds.get(binary, "standard")
        findings = [Finding(severity=_I, code="build", file=script, message=f"launches the {build} build, {binary}")]
        ncl = self._code.variants.get(NCL)
        set_tags = [tag for tag in NONCOLLINEAR_TAGS if incar.flag(tag)]
        if set_tags and ncl is not None and binary != ncl:
            findings.append(Finding(severity=_E, code="wrong_build", file=script,
                                    message=f"INCAR sets {' and '.join(set_tags)}, which needs the {NCL} build "
                                            f"({ncl}); the {build} build stops on non-collinear input"))
        elif set_tags and ncl is None:
            findings.append(Finding(severity=_W, code="no_ncl_build", file=script,
                                    message=f"INCAR sets {' and '.join(set_tags)}, and the compute profile records no "
                                            f"{NCL} build; VASP's standard build stops on non-collinear input unless "
                                            "it was compiled for it"))
        elif not set_tags and binary == ncl:
            findings.append(Finding(severity=_I, code="build", file=script,
                                    message="the ncl build runs it, but INCAR sets neither LNONCOLLINEAR nor LSORBIT"))
        if build == GAM and "KPOINTS" in files:
            kpoints = parse_kpoints(files["KPOINTS"].decode(errors="replace"))
            if not kpoints.gamma_only:
                findings.append(Finding(severity=_W, code="gamma_build", file=script,
                                        message=f"the {GAM} build computes the Γ point only, but KPOINTS gives "
                                                f"{kpoints.summary}"))
        return findings

    # -- run checks -------------------------------------------------------------------

    def check_run(self, task: ScientificTaskSpec, outputs: Mapping[str, bytes],
                  scheduler_state: SchedulerState | None) -> RunReport:
        values, _ = self.resolve(task)
        method: dict = {"code": "VASP", "protocol": f"{self.ref.name} {self.ref.version}", **values}
        scheduler = QualityCheck(name="scheduler_completed", passed=scheduler_state is SchedulerState.COMPLETED,
                                 detail=f"scheduler state {scheduler_state}")
        return outcar_report(outputs.get("OUTCAR"), outputs.get(self.run_output), method=method, checks=(scheduler,),
                             expected_version=self.code_version, limitations=(RUN_LIMITATION, self.limitation))


def outcar_report(raw: bytes | None, stdout: bytes | None, *, method: dict, checks: Sequence[QualityCheck] = (),
                  expected_version: str | None = None, output_file: str = "OUTCAR",
                  limitations: tuple[str, ...] = (RUN_LIMITATION,)) -> RunReport:
    """The health of one VASP run from its OUTCAR and, where there is one, its standard output, after any ``checks``
    made before; ``expected_version`` is the version the run must report, if one is required."""
    checks = list(checks)
    fatal = fatal_error(stdout.decode(errors="replace")) if stdout is not None else None
    said = f"; its standard output says at line {fatal.line}: {fatal.text.strip()}" if fatal else ""
    if raw is None:
        checks.append(QualityCheck(name="output_present", passed=False, detail=f"{output_file} is missing{said}"))
        return RunReport(tuple(checks), EvidenceQuality.INVALID, (), limitations, method)
    out = parse_outcar(raw.decode(errors="replace"))
    method["code_version"] = out.version_string
    if expected_version is not None:
        checks.append(QualityCheck(name="code_version", passed=out.version_string == expected_version,
                                   detail=f"{output_file} reports {out.version_string}; profile records "
                                          f"{expected_version}"))
    end = out.normal_termination
    unconverged = out.unconverged_steps
    steps = len(out.electronic_steps)
    nelm = out.parameters.get("NELM")
    checks += [
        QualityCheck(name="normal_termination", passed=end is not None,
                     detail=f"'General timing and accounting' at line {end.line}" if end else
                     f"no normal end of run{said}"),
        QualityCheck(name="scf_converged", passed=steps > 0 and not unconverged and not out.scf_failures,
                     detail=f"EDIFF reached in {steps - len(unconverged)} of {steps} ionic step(s)"
                            + (f"; not in step(s) {', '.join(map(str, unconverged[:10]))}"
                               + (" …" if len(unconverged) > 10 else "") + (f" (NELM = {nelm})" if nelm else "")
                               if unconverged else "")),
    ]
    relax = out.relaxes
    if relax:
        ionic = len(out.free_energies)
        converged = out.geometry_converged is not None
        checks.append(QualityCheck(
            name="geometry_converged", passed=converged,
            detail=f"'reached required accuracy' at line {out.geometry_converged.line} after {ionic} ionic steps"
            if converged else f"not converged after {ionic} ionic steps (NSW = {out.parameters.get('NSW')})"))
    final = out.free_energies[-1] if out.free_energies else None
    readings = tuple(
        Reading(observable.name, Quantity(value=energy.value, unit="eV"), output_file, energy.line, energy.line,
                energy.text.strip())
        for observable, energy in ((FREE_ENERGY, final), (SIGMA0_ENERGY, out.final_sigma0)) if energy is not None)
    only_geometry_failed = all(c.passed for c in checks if c.name != "geometry_converged")
    failure = EvidenceQuality.INCONCLUSIVE if relax and only_geometry_failed else EvidenceQuality.INVALID
    return RunReport(tuple(checks), failure, readings, limitations, method)


# -- runs computed outside CEDAR ----------------------------------------------------------

PHYSICAL_TAGS = frozenset({
    "GGA", "METAGGA", "LHFCALC", "HFSCREEN", "AEXX", "AGGAX", "AGGAC", "ALDAC", "IVDW", "LUSE_VDW", "ISPIN", "MAGMOM",
    "LNONCOLLINEAR", "LSORBIT", "SAXIS", "NUPDOWN", "LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "NELECT", "IBRION",
    "ISIF", "NSW", "ICHARG", "IDIPOL", "LDIPOL", "DIPOL", "EFIELD", "GGA_COMPAT", "ISYM", "LASPH", "IMAGES", "TEBEG",
    "TEEND", "SMASS",
})
"""Tags that change the physical model or what the run computes; recorded among the controls."""
NUMERICAL_TAGS = frozenset({
    "ENCUT", "ENAUG", "PREC", "EDIFF", "EDIFFG", "ISMEAR", "SIGMA", "KSPACING", "KGAMMA", "LREAL", "ADDGRID", "NGX",
    "NGY", "NGZ", "NGXF", "NGYF", "NGZF", "LMAXMIX", "NBANDS", "ROPT", "PRECFOCK", "NKRED", "NKREDX", "NKREDY",
    "NKREDZ", "ENCUTFOCK", "POTIM", "NFREE",
})
"""Tags that set numerical accuracy; recorded among the numerical settings."""
CONVERGENCE_PATH_TAGS = frozenset({
    "SYSTEM", "ALGO", "IALGO", "NELM", "NELMIN", "NELMDL", "AMIX", "BMIX", "AMIX_MAG", "BMIX_MAG", "AMIN", "IMIX",
    "MAXMIX", "WC", "INIMIX", "MIXPRE", "TIME", "NSIM", "NPAR", "NCORE", "KPAR", "LPLANE", "NWRITE", "LWAVE",
    "LCHARG", "LVTOT", "LVHAR", "LORBIT", "NEDOS", "EMIN", "EMAX", "LELF", "LAECHG", "ISTART", "NBLOCK", "KBLOCK",
    "LSCALAPACK", "LSCALU", "NELMALL",
})
"""Tags that change how a result is reached, run or reported, not what it is; left out."""


def registered_settings(incar_text: str, kpoints_text: str | None, potcar_titles: Sequence[str]
                        ) -> tuple[dict[str, JsonValue], dict[str, JsonValue], list[Finding]]:
    """The settings a VASP run's inputs used, classified as a task declares them: (controls, numerical settings,
    notes). Tags this protocol does not classify become controls, so that a comparison of runs sees every difference;
    the k-point set and the PAW datasets are numerical settings, as FHI-aims's species defaults are."""
    incar = parse_incar(incar_text)
    controls: dict[str, JsonValue] = {}
    numerical: dict[str, JsonValue] = {}
    notes: list[Finding] = [Finding(severity=_W, code="ignored_line", file="INCAR", line=line,
                                    message=f"VASP ignores this line, which sets no tag: {text}")
                            for line, text in incar.ignored_lines]
    grouped: dict[str, list[str]] = {}
    for tag in incar.tags:
        grouped.setdefault(tag.name, []).append(tag.value)
    for name, values in grouped.items():
        if name in CONVERGENCE_PATH_TAGS:
            continue
        value: JsonValue = values[0] if len(values) == 1 else values
        if name in NUMERICAL_TAGS:
            numerical[name] = value
            continue
        controls[name] = value
        if name not in PHYSICAL_TAGS:
            notes.append(Finding(severity=_I, code="unclassified_tag", file="INCAR",
                                 line=next(t.line for t in incar.tags if t.name == name),
                                 message=f"{name} is not classified by {VaspRun.ref.name} {VaspRun.ref.version}; "
                                         "recorded among the controls"))
    numerical["KPOINTS"] = (parse_kpoints(kpoints_text).summary if kpoints_text is not None else
                            "no KPOINTS: generated from KSPACING")
    numerical["POTCAR"] = "; ".join(potcar_titles) if potcar_titles else "unknown"
    return controls, numerical, notes


def outcar_potcars(outcar_text: str) -> list[str]:
    """The PAW datasets an OUTCAR names at its top, in order, for a run whose POTCAR is not kept."""
    titles: list[str] = []
    for line in outcar_text.splitlines()[:200]:
        match = _POTCAR_LINE.match(line)
        if match:
            titles.append(match.group(1))
        elif titles:
            break
    return titles
