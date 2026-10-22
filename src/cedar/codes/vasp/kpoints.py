"""Reading VASP ``KPOINTS`` files, far enough to say what k-point set they give.

The second line is the number of points: 0 for a mesh VASP generates (``Gamma`` or
``Monkhorst-Pack`` subdivisions, or ``Auto`` with a length), more for an explicit list or,
when the third line says ``Line-mode``, the points per segment of a band path.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Kpoints:
    mode: str
    """gamma, monkhorst-pack, auto, line, explicit or unknown."""
    summary: str
    mesh: tuple[int, ...] = ()
    shift: tuple[float, ...] = ()
    points: tuple[tuple[float, float, float], ...] = ()
    """An explicit list's points, in its coordinates."""
    problems: tuple[tuple[int, str], ...] = ()

    @property
    def gamma_only(self) -> bool:
        """Only the Γ point."""
        if self.mode in ("gamma", "monkhorst-pack"):
            return self.mesh == (1, 1, 1) and not any(self.shift)
        return self.mode == "explicit" and len(self.points) == 1 and not any(self.points[0])


def _numbers(text: str, count: int, kind=float) -> tuple | None:
    try:
        values = tuple(kind(t) for t in text.split()[:count])
    except ValueError:
        return None
    return values if len(values) == count else None


def parse_kpoints(text: str) -> Kpoints:
    lines = text.splitlines() + [""] * 6
    count = _numbers(lines[1], 1, int)
    style = lines[2].strip()[:1].lower()
    if count is None or count[0] < 0:
        return Kpoints("unknown", "cannot read the number of points", problems=((2, lines[1].strip()),))
    n = count[0]
    if n == 0 and style in ("g", "m"):
        mode = "gamma" if style == "g" else "monkhorst-pack"
        mesh = _numbers(lines[3], 3, int)
        if mesh is None:
            return Kpoints(mode, "cannot read the subdivisions", problems=((4, lines[3].strip()),))
        shift = _numbers(lines[4], 3) or ()
        name = "Γ-centred" if mode == "gamma" else "Monkhorst-Pack"
        shifted = f", shifted by {' '.join(f'{s:g}' for s in shift)}" if any(shift) else ""
        return Kpoints(mode, f"{name} {'×'.join(map(str, mesh))} mesh{shifted}", mesh, shift)
    if n == 0 and style == "a":
        length = _numbers(lines[3], 1)
        if length is None:
            return Kpoints("auto", "cannot read the length", problems=((4, lines[3].strip()),))
        return Kpoints("auto", f"fully automatic mesh, length {length[0]:g}")
    if n == 0:
        return Kpoints("unknown", f"a generated mesh of a kind not read here ({lines[2].strip()})")
    if style == "l":
        segments = sum(1 for line in lines[4:] if _numbers(line, 3) is not None) // 2
        return Kpoints("line", f"band path in line mode, {n} points per segment, {segments} segments")
    points = []
    for number in range(4, 4 + n):
        point = _numbers(lines[number - 1], 3)
        if point is None:
            return Kpoints("explicit", f"{n} explicit k-points", problems=((number, lines[number - 1].strip()),))
        points.append(point)
    where = "Cartesian" if style in ("c", "k") else "reciprocal"
    return Kpoints("explicit", f"{n} explicit k-points ({where} coordinates)", points=tuple(points))
