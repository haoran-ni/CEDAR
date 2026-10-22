"""Reading VASP ``INCAR`` files as VASP does: ``TAG = value`` statements, any number to a line.

Statements are separated by ``;``; ``#`` and ``!`` start a comment; a line ending in ``\\``
continues on the next, and so does a value in quotes that is not closed on its line. Tag
names are case-insensitive. A line with no ``=`` is ignored by VASP, so a tag written without
one (``ENCUT 520``) silently has no effect; such lines are kept to be reported.
"""

import re
from dataclasses import dataclass

_TAG_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Tag:
    name: str
    """Upper case, as VASP compares them."""
    value: str
    """As written, with its whitespace collapsed and any comment removed."""
    line: int


@dataclass(frozen=True)
class Incar:
    tags: tuple[Tag, ...]
    ignored_lines: tuple[tuple[int, str], ...] = ()
    """Lines with text but no tag, which VASP ignores."""

    def get(self, name: str) -> Tag | None:
        return next((t for t in self.tags if t.name == name.upper()), None)

    def flag(self, name: str) -> bool | None:
        """A logical tag's value; None if it is not set or not a logical."""
        tag = self.get(name)
        if tag is None or not tag.value:
            return None
        first = tag.value.split()[0].lstrip(".").upper()
        return True if first.startswith("T") else False if first.startswith("F") else None

    def number(self, name: str) -> float | None:
        tag = self.get(name)
        if tag is None or not tag.value:
            return None
        try:
            return float(tag.value.split()[0].replace("d", "e").replace("D", "E"))
        except ValueError:
            return None


def _statements(text: str) -> list[tuple[int, str]]:
    """(first line, statement) for every statement, comments removed, continuations joined."""
    statements: list[tuple[int, str]] = []
    current, start, quoted = "", 0, False
    for number, raw in enumerate(text.splitlines(), start=1):
        if not current:
            start = number
        line = raw
        if not quoted:
            line = line.rstrip()
        i = 0
        while i < len(line):
            char = line[i]
            if char == '"':
                quoted = not quoted
            elif not quoted and char in "#!":
                line = line[:i]
                break
            elif not quoted and char == ";":
                statements.append((start, current + line[:i]))
                current, line, i, start = "", line[i + 1:], -1, number
            i += 1
        if quoted:
            current += line + "\n"
        elif line.endswith("\\"):
            current += line[:-1] + " "
        else:
            statements.append((start, current + line))
            current = ""
    if current:
        statements.append((start, current))
    return statements


def parse_incar(text: str) -> Incar:
    tags: list[Tag] = []
    ignored: list[tuple[int, str]] = []
    for line, statement in _statements(text):
        name, equals, value = statement.partition("=")
        if not statement.strip():
            continue
        if not equals or not _TAG_NAME.match(name.strip()):
            ignored.append((line, " ".join(statement.split())))
            continue
        tags.append(Tag(name.strip().upper(), " ".join(value.split()), line))
    return Incar(tuple(tags), tuple(ignored))
