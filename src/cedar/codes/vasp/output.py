"""Run-health markers in VASP ``OUTCAR``, with the line each was found on, and fatal errors in its standard output.

Only markers whose wording was checked against real output of VASP 5.4.4, and against VASP
6's output format where it differs, are recognized. Every electronic step starts a line
``Iteration  N(  k)``, for ionic step N; an electronic loop that converges ends with
``aborting loop because EDIFF is reached`` (VASP 6 says ``EDIFF was not reached
(unconverged)`` when it does not; VASP 5 just goes on after NELM steps). Each ionic step ends
with a block that gives ``free  energy   TOTEN`` and ``energy(sigma->0)``, spaced differently
from the lines of the electronic steps. This is not a general parser: agents may read anything
else in the outputs themselves and cite it.
"""

import re
from dataclasses import dataclass

from cedar.codes.fhiaims.output import EnergyLine, Marker, parse_number

_NUMBER = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?"

_VERSION = re.compile(r"^\s*vasp\.(\S+)")
_PARAMETER = {name: re.compile(rf"^\s*{name}\s*=\s*(-?\d+)") for name in ("NELM", "NSW", "IBRION")}
_ITERATION = re.compile(r"Iteration\s+(\d+)\(\s*(\d+)\)")
_EDIFF_REACHED = re.compile(r"aborting loop because EDIFF is reached")
_EDIFF_NOT_REACHED = re.compile(r"aborting loop EDIFF was not reached")
_FREE_ENERGY = re.compile(rf"^\s*free  energy   TOTEN\s*=\s*({_NUMBER})\s*eV")
_SIGMA0 = re.compile(rf"^\s*energy  without entropy=\s*({_NUMBER})\s+energy\(sigma->0\)\s*=\s*({_NUMBER})")
_GEOMETRY_CONVERGED = re.compile(r"reached required accuracy - stopping structural energy minimisation")
_NORMAL_END = re.compile(r"^\s*General timing and accounting informations? for this job")
_FATAL = re.compile(r"VERY BAD NEWS|\bERROR\b|BAD TERMINATION|KILLED BY SIGNAL|forrtl: severe|Segmentation fault")
RELAXING_IBRION = (1, 2, 3)
"""IBRION values that minimize the energy with respect to the ionic positions (and, with ISIF, the cell)."""


@dataclass(frozen=True)
class VaspOutput:
    version: Marker | None
    parameters: dict[str, int]
    """NELM, NSW and IBRION, as the run used them."""
    electronic_steps: dict[int, int]
    """Electronic steps of each ionic step that started."""
    scf_converged: frozenset[int]
    """Ionic steps whose electronic loop reached EDIFF."""
    scf_failures: tuple[Marker, ...]
    free_energies: tuple[EnergyLine, ...]
    """``free  energy   TOTEN`` at the end of each ionic step."""
    final_sigma0: EnergyLine | None
    geometry_converged: Marker | None
    normal_termination: Marker | None

    @property
    def version_string(self) -> str | None:
        if self.version is None:
            return None
        match = _VERSION.match(self.version.text)
        return match.group(1) if match else None

    @property
    def relaxes(self) -> bool:
        """The run minimizes the energy with respect to the ionic positions."""
        return self.parameters.get("IBRION") in RELAXING_IBRION and self.parameters.get("NSW", 0) > 0

    @property
    def unconverged_steps(self) -> list[int]:
        """Ionic steps whose electronic loop did not reach EDIFF."""
        return sorted(set(self.electronic_steps) - self.scf_converged)


def parse_outcar(text: str) -> VaspOutput:
    version = geometry = normal_end = sigma0 = None
    parameters: dict[str, int] = {}
    steps: dict[int, int] = {}
    converged: set[int] = set()
    failures: list[Marker] = []
    energies: list[EnergyLine] = []
    ionic = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if m := _ITERATION.search(line):
            ionic = int(m.group(1))
            steps[ionic] = max(steps.get(ionic, 0), int(m.group(2)))
        elif _EDIFF_REACHED.search(line):
            converged.add(ionic)
        elif _EDIFF_NOT_REACHED.search(line):
            failures.append(Marker(number, line))
        elif m := _FREE_ENERGY.match(line):
            energies.append(EnergyLine(parse_number(m.group(1)), number, line))
        elif m := _SIGMA0.match(line):
            sigma0 = EnergyLine(parse_number(m.group(2)), number, line)
        elif _GEOMETRY_CONVERGED.search(line):
            geometry = Marker(number, line)
        elif _NORMAL_END.match(line):
            normal_end = Marker(number, line)
        elif version is None and _VERSION.match(line):
            version = Marker(number, line)
        else:
            for name, pattern in _PARAMETER.items():
                if name not in parameters and (m := pattern.match(line)):
                    parameters[name] = int(m.group(1))
    return VaspOutput(version, parameters, steps, frozenset(converged), tuple(failures), tuple(energies), sigma0,
                      geometry, normal_end)


def fatal_error(text: str) -> Marker | None:
    """The first line of VASP's standard output that reports a fatal error, if any."""
    for number, line in enumerate(text.splitlines(), start=1):
        if _FATAL.search(line):
            return Marker(number, line)
    return None
