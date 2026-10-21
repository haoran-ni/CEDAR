"""Read-only file tools: list, find, read and search, with line numbers for citations, and
view images and PDFs.

Paths must resolve (after following symlinks) inside one of the allowed roots and outside
every denied directory, such as the one holding API keys. Nothing here writes. Roots can
be given as a function, so an agent's readable directories follow the research record,
e.g. the run directories of its own hypothesis's calculations.
"""

import fnmatch
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import Field

from cedar.agents.media import MAX_PDF_PAGES, load_media
from cedar.agents.tools import AgentTool, ToolFailure, ToolOutcome
from cedar.contracts.base import Contract

MAX_LINE = 400
MAX_LINES = 400
MAX_ENTRIES = 300
MAX_VIEW_BYTES = 100_000_000
"""Larger files are refused before they are read."""


class FileAccess:
    def __init__(self, roots: Sequence[str | Path] | Callable[[], Sequence[str | Path]],
                 deny: Sequence[str | Path] = ()) -> None:
        self._roots = roots
        self.deny = [Path(d).resolve() for d in deny]

    @property
    def roots(self) -> list[Path]:
        roots = self._roots() if callable(self._roots) else self._roots
        return [Path(r).resolve() for r in roots]

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            raise ToolFailure(f"use an absolute path; readable roots are {[str(r) for r in self.roots]}")
        resolved = candidate.resolve()
        if any(resolved == d or resolved.is_relative_to(d) for d in self.deny):
            raise PermissionError(f"{path} is not readable by agents")
        roots = self.roots
        if not any(resolved == r or resolved.is_relative_to(r) for r in roots):
            raise PermissionError(f"{path} is outside the readable roots {[str(r) for r in roots]}")
        return resolved

    def text_lines(self, path: str) -> tuple[Path, list[str]]:
        resolved = self.resolve(path)
        data = resolved.read_bytes()
        if b"\x00" in data[:8192]:
            raise ToolFailure(f"{path} looks like a binary file")
        return resolved, data.decode(errors="replace").splitlines()


def _clip(line: str) -> str:
    return line if len(line) <= MAX_LINE else line[:MAX_LINE] + f" …[{len(line) - MAX_LINE} more characters]"


class ListArgs(Contract):
    path: str = Field(description="Absolute path of a directory.")


class FindArgs(Contract):
    root: str = Field(description="Absolute path of the directory to search under.")
    name_pattern: str = Field(description="Shell-style file name pattern, e.g. 'aims.out' or '*.in'.")
    max_results: int = Field(default=50, ge=1, le=MAX_ENTRIES, description="Paths to return; narrow the root "
                                                                          "or pattern rather than raising this.")


class ReadArgs(Contract):
    path: str = Field(description="Absolute path of a text file.")
    start_line: int = Field(default=1, ge=1, description="First line to return (1-based).")
    max_lines: int = Field(default=200, ge=1, le=MAX_LINES, description="How many lines to return.")


class ViewArgs(Contract):
    path: str = Field(description="Absolute path of an image or a PDF.")
    pages: str | None = Field(default=None, description=f"PDF pages to view, e.g. '1-3, 7'; needed when the PDF "
                                                        f"has more than {MAX_PDF_PAGES} pages.")


class SearchArgs(Contract):
    path: str = Field(description="Absolute path of a text file.")
    pattern: str = Field(description="Python regular expression matched against each line.")
    max_matches: int = Field(default=50, ge=1, le=200)
    context_lines: int = Field(default=0, ge=0, le=5, description="Lines of context before and after each match.")


def file_tools(access: FileAccess, on_read: Callable[[Path], None] | None = None) -> list[AgentTool]:
    """The file tools; ``on_read`` is told about every file whose contents an agent reads."""

    def text_lines(path: str) -> list[str]:
        resolved, lines = access.text_lines(path)
        if on_read is not None:
            on_read(resolved)
        return lines

    def list_directory(args: ListArgs) -> str:
        directory = access.resolve(args.path)
        if not directory.is_dir():
            raise NotADirectoryError(f"{args.path} is not a directory")
        entries = sorted(directory.iterdir())
        lines = [f"{e.name}/" if e.is_dir() else f"{e.name}  ({e.stat().st_size} bytes)" for e in entries[:MAX_ENTRIES]]
        if len(entries) > MAX_ENTRIES:
            lines.append(f"... {len(entries) - MAX_ENTRIES} more entries")
        return "\n".join(lines) or "(empty directory)"

    def find_files(args: FindArgs) -> str:
        root = access.resolve(args.root)
        found, more = [], 0
        for path in root.rglob("*"):
            if fnmatch.fnmatch(path.name, args.name_pattern) and path.is_file():
                try:
                    resolved = str(access.resolve(str(path)))
                except PermissionError:
                    continue
                if len(found) < args.max_results:
                    found.append(resolved)
                else:
                    more += 1
                    if more >= 10_000:
                        break
        note = f"\n[{more}{'+' if more >= 10_000 else ''} more matches not shown]" if more else ""
        return ("\n".join(found) or "no matching files") + note

    def read_file(args: ReadArgs) -> str:
        lines = text_lines(args.path)
        first = args.start_line - 1
        chosen = lines[first:first + args.max_lines]
        body = "\n".join(f"{first + i + 1}: {_clip(line)}" for i, line in enumerate(chosen))
        tail = f"\n[lines {args.start_line}-{first + len(chosen)} of {len(lines)}]"
        return (body or "(no lines in this range)") + tail

    def search_file(args: SearchArgs) -> str:
        try:
            pattern = re.compile(args.pattern)
        except re.error as exc:
            raise ToolFailure(f"invalid regular expression: {exc}") from None
        lines = text_lines(args.path)
        shown: set[int] = set()
        out: list[str] = []
        matches = 0
        for index, line in enumerate(lines):
            if not pattern.search(line):
                continue
            matches += 1
            if matches > args.max_matches:
                break
            for j in range(max(0, index - args.context_lines), min(len(lines), index + args.context_lines + 1)):
                if j not in shown:
                    shown.add(j)
                    out.append(f"{j + 1}: {_clip(lines[j])}")
        note = f"\n[stopped after {args.max_matches} matches]" if matches > args.max_matches else ""
        return ("\n".join(out) or "no matching lines") + note + f"\n[file has {len(lines)} lines]"

    def view_file(args: ViewArgs) -> ToolOutcome:
        resolved = access.resolve(args.path)
        if not resolved.is_file():
            raise FileNotFoundError(f"{args.path} is not a file")
        if resolved.stat().st_size > MAX_VIEW_BYTES:
            raise ToolFailure(f"{args.path} is larger than {MAX_VIEW_BYTES:,} bytes")
        if on_read is not None:
            on_read(resolved)
        media, note = load_media(resolved.read_bytes(), resolved.name, args.pages)
        return ToolOutcome(f"Showing {note}. Values read from a picture are not citable lines; say where you read "
                           "them.", media=(media,))

    return [
        AgentTool("list_directory", "List a directory's entries with file sizes. Use it to see what a run "
                  "produced or what files exist before reading them.", ListArgs, list_directory),
        AgentTool("find_files", "Find files by name pattern under a directory.", FindArgs, find_files),
        AgentTool("read_file", "Read lines of a text file, each prefixed with its line number. Cite these "
                  "numbers when you report values from the file. Read large files in ranges.", ReadArgs, read_file),
        AgentTool("search_file", "Search a text file for lines matching a regular expression; returns matching "
                  "lines with their line numbers. Prefer this to reading whole large outputs.", SearchArgs,
                  search_file),
        AgentTool("view_file", "Look at an image (PNG, JPEG, GIF, WebP, or another format Pillow reads, converted) "
                  f"or a PDF (up to {MAX_PDF_PAGES} pages at a time). It stays in your conversation and costs tokens "
                  "on every later turn, so view only the pages you need.", ViewArgs, view_file),
    ]
