"""Run-health markers in FHI-aims ``aims.out``, with the line each was found on.

Only markers whose wording was checked against the FHI-aims source (versions 240507 and
260331.1) and real outputs are recognized. This is not a general parser: agents may read
anything else in the output themselves and cite it.
"""

import re
from dataclasses import dataclass

_NUMBER = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?"

_VERSION = re.compile(r"^\s*FHI-aims version\s*:\s*(\S+)")
_TASKS = re.compile(r"^\s*Using\s+(\d+)\s+parallel tasks")
_SCF_CONVERGED = re.compile(r"^\s*Self-consistency cycle converged\.")
_SCF_FAILED = re.compile(r"SCF cycle not converged\.|SELF-CONSISTENCY CYCLE DID NOT CONVERGE")
_GEOMETRY_CONVERGED = re.compile(r"^\s*Present geometry is converged\.")
_RELAXATION_STEP = re.compile(r"^\s*Relaxation step number\s+(\d+)")
_ABORTED = re.compile(r"\*\s*Aborting optimization")
_FINAL_TOTAL = re.compile(rf"^\s*\|\s*Total energy of the DFT / Hartree-Fock s\.c\.f\. calculation\s*:\s*({_NUMBER})\s+eV")
_FINAL_ZERO_BROADENING = re.compile(rf"^\s*\|\s*Final zero-broadening corrected energy.*:\s*({_NUMBER})\s+eV")
_NORMAL_END = re.compile(r"^\s*Have a nice day\.")
_KILLED = re.compile(r"BAD TERMINATION|KILLED BY SIGNAL")


def parse_number(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def numbers_in(text: str) -> list[float]:
    return [parse_number(m) for m in re.findall(_NUMBER, text)]


@dataclass(frozen=True)
class Marker:
    line: int
    text: str


@dataclass(frozen=True)
class EnergyLine:
    value: float
    line: int
    text: str


@dataclass(frozen=True)
class AimsOutput:
    version: Marker | None
    parallel_tasks: int | None
    scf_converged: tuple[Marker, ...]
    scf_failures: tuple[Marker, ...]
    relaxation_steps: int
    geometry_converged: Marker | None
    optimization_aborted: Marker | None
    final_total_energy: EnergyLine | None
    final_zero_broadening_energy: EnergyLine | None
    normal_termination: Marker | None
    killed: Marker | None

    @property
    def version_string(self) -> str | None:
        if self.version is None:
            return None
        match = _VERSION.match(self.version.text)
        return match.group(1) if match else None


def parse_output(text: str) -> AimsOutput:
    version = killed = geometry_converged = aborted = normal_end = None
    final_total = final_zero = None
    tasks = None
    scf_ok: list[Marker] = []
    scf_bad: list[Marker] = []
    steps = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if version is None and _VERSION.match(line):
            version = Marker(number, line)
        elif tasks is None and (m := _TASKS.match(line)):
            tasks = int(m.group(1))
        elif _SCF_CONVERGED.match(line):
            scf_ok.append(Marker(number, line))
        elif _SCF_FAILED.search(line):
            scf_bad.append(Marker(number, line))
        elif _GEOMETRY_CONVERGED.match(line):
            geometry_converged = Marker(number, line)
        elif m := _RELAXATION_STEP.match(line):
            steps = max(steps, int(m.group(1)))
        elif aborted is None and _ABORTED.search(line):
            aborted = Marker(number, line)
        elif m := _FINAL_TOTAL.match(line):
            final_total = EnergyLine(parse_number(m.group(1)), number, line)
        elif m := _FINAL_ZERO_BROADENING.match(line):
            final_zero = EnergyLine(parse_number(m.group(1)), number, line)
        elif _NORMAL_END.match(line):
            normal_end = Marker(number, line)
        elif killed is None and _KILLED.search(line):
            killed = Marker(number, line)
    return AimsOutput(
        version=version,
        parallel_tasks=tasks,
        scf_converged=tuple(scf_ok),
        scf_failures=tuple(scf_bad),
        relaxation_steps=steps,
        geometry_converged=geometry_converged,
        optimization_aborted=aborted,
        final_total_energy=final_total,
        final_zero_broadening_energy=final_zero,
        normal_termination=normal_end,
        killed=killed,
    )
