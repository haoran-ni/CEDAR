"""Reading VASP ``POSCAR`` files: the cell, the species and their counts, and the positions.

The format is positional: a comment; a scaling factor (one number, negative for a volume, or
three numbers, one per Cartesian direction); three lattice vectors; the species names (from
VASP 5 on; VASP 4 files take them from POTCAR); the number of atoms of each species; an
optional ``Selective dynamics`` line; ``Direct`` or ``Cartesian``; then one line per atom,
with three T/F flags after the coordinates under selective dynamics. Anything after the
positions (velocities, predictor-corrector data) is not read.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms

_ELEMENT = re.compile(r"^([A-Z][a-z]?)")


def element_of(name: str) -> str | None:
    """The element a species or dataset name stands for: ``Fe`` for ``Fe``, ``Fe_pv`` or ``Fe_pv/1a2b``; ``H`` for
    ``H.75``."""
    match = _ELEMENT.match(name)
    return match.group(1) if match else None


@dataclass(frozen=True)
class PoscarAtom:
    species: int
    """Index of its species, in the order of the species counts."""
    position: tuple[float, float, float]
    """Cartesian, in Å."""
    line: int
    fixed: tuple[bool, bool, bool] = (False, False, False)
    """Directions held fixed by selective dynamics (F flags)."""


@dataclass(frozen=True)
class Poscar:
    lattice: tuple[tuple[float, float, float], ...]
    """Lattice vectors in Å, scaled."""
    species: tuple[str, ...] | None
    """The species names, or None for a file in the VASP 4 format, which names none."""
    counts: tuple[int, ...]
    atoms: tuple[PoscarAtom, ...]
    selective: bool = False
    problems: tuple[tuple[int, str], ...] = ()
    """Lines that cannot be read, with what is wrong; line 0 for the file as a whole."""

    def to_atoms(self, symbols: Sequence[str]) -> Atoms:
        """The structure, with ``symbols`` naming the elements of the species in order."""
        return Atoms(symbols=[symbols[a.species] for a in self.atoms], positions=[a.position for a in self.atoms],
                     cell=np.array(self.lattice), pbc=True)


def _floats(tokens: Sequence[str]) -> list[float] | None:
    try:
        return [float(t) for t in tokens]
    except ValueError:
        return None


def _flag(token: str) -> bool | None:
    upper = token.upper()
    return True if upper == "T" else False if upper == "F" else None


def parse_poscar(text: str) -> Poscar:
    lines = text.splitlines()
    problems: list[tuple[int, str]] = []

    def bad(number: int, message: str) -> Poscar:
        return Poscar((), None, (), (), problems=(*problems, (number, message)))

    if len(lines) < 7:
        return bad(0, f"{len(lines)} lines; a POSCAR has at least 7 before the positions")
    tokens = lines[1].split()
    scale = _floats(tokens[:3]) if len(tokens) >= 3 and _floats(tokens[:3]) else _floats(tokens[:1])
    if not scale or (len(scale) == 1 and scale[0] == 0):
        return bad(2, f"cannot read the scaling factor: {lines[1].strip()}")
    vectors = []
    for number in (3, 4, 5):
        vector = _floats(lines[number - 1].split()[:3])
        if vector is None or len(vector) != 3:
            return bad(number, f"cannot read the lattice vector: {lines[number - 1].strip()}")
        vectors.append(vector)
    lattice = np.array(vectors)
    if len(scale) == 3:
        lattice = lattice * np.array(scale)
        factor = 1.0
    elif scale[0] < 0:
        volume = abs(np.linalg.det(lattice))
        if volume == 0:
            return bad(3, "the lattice vectors span no volume")
        factor = (-scale[0] / volume) ** (1 / 3)
        lattice = lattice * factor
    else:
        factor = scale[0]
        lattice = lattice * factor

    at = 6
    species: tuple[str, ...] | None = None
    first = lines[at - 1].split()
    if first and _floats(first[:1]) is None:
        species = tuple(first)
        at += 1
    count_tokens = lines[at - 1].split() if at <= len(lines) else []
    counts = []
    for token in count_tokens:
        if not token.isdigit():
            break
        counts.append(int(token))
    if not counts:
        return bad(at, f"cannot read the numbers of atoms: {lines[at - 1].strip() if at <= len(lines) else ''}")
    if species is not None and len(species) != len(counts):
        problems.append((at, f"{len(species)} species names but {len(counts)} counts"))
    at += 1
    selective = at <= len(lines) and lines[at - 1].strip()[:1] in ("s", "S")
    if selective:
        at += 1
    if at > len(lines):
        return bad(at, "no Direct or Cartesian line")
    cartesian = lines[at - 1].strip()[:1] in ("c", "C", "k", "K")

    atoms: list[PoscarAtom] = []
    index = [i for i, count in enumerate(counts) for _ in range(count)]
    for n, kind in enumerate(index):
        number = at + 1 + n
        if number > len(lines):
            problems.append((0, f"{len(index)} atoms are counted, but only {n} positions follow"))
            break
        tokens = lines[number - 1].split()
        coordinates = _floats(tokens[:3])
        if coordinates is None or len(coordinates) != 3:
            problems.append((number, f"cannot read the position: {lines[number - 1].strip()}"))
            continue
        fixed = (False, False, False)
        if selective:
            flags = [_flag(t) for t in tokens[3:6]]
            if len(flags) != 3 or None in flags:
                problems.append((number, f"selective dynamics needs three T/F flags: {lines[number - 1].strip()}"))
            else:
                fixed = tuple(not f for f in flags)
        position = (np.array(coordinates) * (np.array(scale) if len(scale) == 3 else factor) if cartesian
                    else np.array(coordinates) @ lattice)
        atoms.append(PoscarAtom(kind, tuple(float(x) for x in position), number, fixed))
    return Poscar(tuple(tuple(float(x) for x in v) for v in lattice), species, tuple(counts), tuple(atoms),
                  selective, tuple(problems))
