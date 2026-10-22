"""Reading FHI-aims ``geometry.in`` files; what the examiner measures in a structure is in ``cedar.codes.structure``."""

from dataclasses import dataclass, field, replace

import numpy as np
from ase import Atoms

_ATOM_MODIFIERS = {"constrain_relaxation", "initial_moment", "initial_charge"}


@dataclass(frozen=True)
class GeometryAtom:
    symbol: str
    position: tuple[float, float, float]
    fractional: bool
    line: int
    constrained: bool = False
    initial_moment: float | None = None


@dataclass(frozen=True)
class Geometry:
    lattice: tuple[tuple[float, float, float], ...]
    atoms: tuple[GeometryAtom, ...]
    unknown_lines: tuple[tuple[int, str], ...] = field(default=())
    malformed_lines: tuple[tuple[int, str], ...] = field(default=())

    @property
    def periodic(self) -> bool:
        return len(self.lattice) == 3

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(a.symbol for a in self.atoms))

    def to_atoms(self) -> Atoms:
        cell = np.array(self.lattice) if self.periodic else None
        cartesian = []
        for atom in self.atoms:
            position = np.array(atom.position)
            cartesian.append(position @ cell if atom.fractional and cell is not None else position)
        return Atoms(
            symbols=[a.symbol for a in self.atoms],
            positions=cartesian,
            cell=cell,
            pbc=self.periodic,
        )


def _floats(tokens: list[str], count: int) -> tuple[float, ...] | None:
    try:
        values = tuple(float(t) for t in tokens[:count])
    except ValueError:
        return None
    return values if len(values) == count else None


def parse_geometry(text: str) -> Geometry:
    lattice: list[tuple[float, float, float]] = []
    atoms: list[GeometryAtom] = []
    unknown: list[tuple[int, str]] = []
    malformed: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        tokens = raw.split("#", 1)[0].split()
        if not tokens:
            continue
        keyword, args = tokens[0].lower(), tokens[1:]
        if keyword == "lattice_vector":
            vector = _floats(args, 3)
            if vector is None or len(args) != 3:
                malformed.append((number, raw.strip()))
            else:
                lattice.append(vector)
        elif keyword in ("atom", "atom_frac"):
            position = _floats(args, 3)
            if position is None or len(args) != 4:
                malformed.append((number, raw.strip()))
            else:
                atoms.append(GeometryAtom(args[3], position, keyword == "atom_frac", number))
        elif keyword in _ATOM_MODIFIERS:
            if not atoms:
                malformed.append((number, raw.strip()))
                continue
            last = atoms[-1]
            if keyword == "constrain_relaxation":
                atoms[-1] = replace(last, constrained=bool(args) and args[0].lower() != ".false.")
            elif keyword == "initial_moment":
                moment = _floats(args, 1)
                if moment is None:
                    malformed.append((number, raw.strip()))
                else:
                    atoms[-1] = replace(last, initial_moment=moment[0])
        else:
            unknown.append((number, raw.strip()))
    if len(lattice) not in (0, 3):
        malformed.append((0, f"{len(lattice)} lattice vectors; expected 0 or 3"))
    return Geometry(tuple(lattice), tuple(atoms), tuple(unknown), tuple(malformed))
