"""Reading VASP ``POTCAR`` files, and writing them from a library of PAW datasets.

A POTCAR is the datasets of the structure's species, in POSCAR's order, one after another;
each ends with a line ``End of Dataset`` and names itself in its ``TITEL`` line, e.g.
``PAW_PBE Pb_d 06Sep2000``. A library (such as ``potpaw_PBE``) holds one directory per
dataset, named as in ``TITEL`` (``Pb_d``), with that dataset alone in its ``POTCAR``.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cedar.codes.vasp.poscar import element_of

_TITEL = re.compile(r"^\s*TITEL\s*=\s*(.+?)\s*$")
_ZVAL = re.compile(r"ZVAL\s*=\s*([-+]?\d+(?:\.\d*)?)")
_END = "End of Dataset"
_DATASET_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.+-]*$")


@dataclass(frozen=True)
class Dataset:
    title: str
    """Its ``TITEL``, e.g. ``PAW_PBE Pb_d 06Sep2000``; its first line if it has none."""
    name: str
    """Its name in a library, e.g. ``Pb_d``."""
    element: str | None
    valence: float | None
    """``ZVAL``, the number of valence electrons it treats."""
    line: int
    """Its first line in the file."""
    text: str

    def same_as(self, other: str) -> bool:
        """Whether ``other`` is this dataset's text, but for trailing whitespace."""
        return _normalized(self.text) == _normalized(other)


@dataclass(frozen=True)
class Potcar:
    datasets: tuple[Dataset, ...]
    problems: tuple[tuple[int, str], ...] = ()


def _normalized(text: str) -> list[str]:
    return [line.rstrip() for line in text.strip().splitlines()]


def parse_potcar(text: str) -> Potcar:
    datasets: list[Dataset] = []
    problems: list[tuple[int, str]] = []
    lines = text.splitlines()
    start = 0
    for index, line in enumerate(lines):
        if line.strip() != _END:
            continue
        block = lines[start:index + 1]
        datasets.append(_dataset(block, start + 1))
        start = index + 1
    if any(line.strip() for line in lines[start:]):
        problems.append((start + 1, f"text after the last '{_END}' line, or a dataset without one"))
    return Potcar(tuple(datasets), tuple(problems))


def _dataset(block: list[str], first: int) -> Dataset:
    title = next((m.group(1) for line in block if (m := _TITEL.match(line))), None)
    title = title or next((line.strip() for line in block if line.strip()), "")
    tokens = title.split()
    name = tokens[1] if len(tokens) >= 2 else (tokens[0] if tokens else "")
    valence = next((float(m.group(1)) for line in block if (m := _ZVAL.search(line))), None)
    return Dataset(title, name, element_of(name), valence, first, "\n".join(block) + "\n")


def library_dataset(library: Path, name: str) -> Path | None:
    """The ``POTCAR`` of dataset ``name`` in ``library``, if it has one."""
    if not _DATASET_NAME.match(name) or ".." in name:
        return None
    path = library / name / "POTCAR"
    return path if path.is_file() else None


def render_potcar(library: Path, names: Sequence[str]) -> str:
    """The datasets ``names`` from ``library``, in order; raises LookupError for one it does not have."""
    texts = []
    for name in names:
        path = library_dataset(library, name)
        if path is None:
            raise LookupError(name)
        text = path.read_text(errors="replace")
        texts.append(text if text.endswith("\n") else text + "\n")
    return "".join(texts)
