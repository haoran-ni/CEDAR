"""Reading and writing FHI-aims ``control.in`` files.

Global keywords come first; everything from the first ``species`` line on belongs to
species blocks. Keywords are compared in lower case; the original text is kept for
messages.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Keyword:
    name: str
    args: tuple[str, ...]
    line: int


@dataclass(frozen=True)
class SpeciesBlock:
    species: str
    line: int
    lines: tuple[str, ...]

    def normalized(self) -> tuple[str, ...]:
        return normalize_lines(self.lines)


@dataclass(frozen=True)
class ControlFile:
    keywords: tuple[Keyword, ...]
    species: tuple[SpeciesBlock, ...]

    def find(self, name: str) -> list[Keyword]:
        return [k for k in self.keywords if k.name == name]

    def species_block(self, element: str) -> SpeciesBlock | None:
        return next((b for b in self.species if b.species == element), None)


def _content(line: str) -> str:
    return line.split("#", 1)[0].strip()


def normalize_lines(lines: Sequence[str]) -> tuple[str, ...]:
    """Lines without comments, blank lines or repeated whitespace, for comparing blocks."""
    return tuple(" ".join(_content(line).split()) for line in lines if _content(line))


def parse_control(text: str) -> ControlFile:
    keywords: list[Keyword] = []
    blocks: list[SpeciesBlock] = []
    current: tuple[str, int, list[str]] | None = None
    for number, raw in enumerate(text.splitlines(), start=1):
        tokens = _content(raw).split()
        if tokens and tokens[0].lower() == "species":
            if current is not None:
                blocks.append(SpeciesBlock(current[0], current[1], tuple(current[2])))
            current = (tokens[1] if len(tokens) > 1 else "", number, [raw])
        elif current is not None:
            current[2].append(raw)
        elif tokens:
            keywords.append(Keyword(tokens[0].lower(), tuple(tokens[1:]), number))
    if current is not None:
        blocks.append(SpeciesBlock(current[0], current[1], tuple(current[2])))
    return ControlFile(tuple(keywords), tuple(blocks))


def species_default_path(root: Path, level: str, element: str) -> Path | None:
    """The species-defaults file for ``element`` at ``level`` (e.g. ``light/29_Cu_default``)."""
    matches = sorted((root / level).glob(f"[0-9]*_{element}_default"))
    return matches[0] if len(matches) == 1 else None


def render_control(header: Sequence[str], keywords: Sequence[tuple[str, str]], species_texts: Sequence[str]) -> str:
    width = max(len(name) for name, _ in keywords)
    lines = [f"# {line}" for line in header]
    lines.append("")
    lines.extend(f"{name:<{width}}  {value}".rstrip() for name, value in keywords)
    for text in species_texts:
        lines.append("")
        lines.append(text.rstrip("\n"))
    return "\n".join(lines) + "\n"
