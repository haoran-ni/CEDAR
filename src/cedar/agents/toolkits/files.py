"""File tools: list, find, read and search, with line numbers for citations, and view images and
PDFs; and, for an agent's own workspace, write and delete.

Paths must resolve (after following symlinks) inside one of the allowed roots and outside
every denied directory, such as the one holding API keys, unless they lie inside an
explicitly allowed path. Roots, denied and allowed paths can be given as functions, so what
may be read can follow the research record. Agents read every file the researcher can,
except credentials (``agent_access``).

Text is read as a stream, so a very large file never has to fit in memory: reading stops
after the requested lines, and a search after ``MAX_SCAN_BYTES``. Only regular files are
read. Nothing but the workspace tools writes, and they write only inside the workspace.
"""

import fnmatch
import os
import re
import shutil
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

from pydantic import Field

from cedar.agents.media import MAX_PDF_PAGES, load_media
from cedar.agents.tools import AgentTool, ToolFailure, ToolOutcome
from cedar.contracts.base import Contract
from cedar.paths import workspace_path

MAX_LINE = 400
MAX_LINES = 400
MAX_ENTRIES = 300
MAX_VIEW_BYTES = 100_000_000
"""Larger files are refused before they are read."""
MAX_SCAN_BYTES = 2 * 1024 ** 3
"""A search stops after reading this much of a file."""
MAX_COUNT_BYTES = 256 * 1024 ** 2
"""Files up to this size have their lines counted in full when a range of them is read."""
MAX_VISITED = 200_000
"""A file search stops after visiting this many directory entries."""

PathsLike = Sequence[str | Path] | Callable[[], Sequence[str | Path]]

SECRET_PATHS = ("~/.ssh", "~/.gnupg", "~/.netrc", "~/.aws", "~/.config/gcloud", "~/.kube", "~/.docker")
"""Well-known stores of keys and passwords, never readable by agents."""
PSEUDO_FILESYSTEMS = ("/proc", "/sys", "/dev")
"""Not files: reading them could show a process's environment, or never end."""


def credential_paths(deny: Sequence[str | Path] = ()) -> list[Path]:
    """What no agent reads: ``deny`` (CEDAR's configuration directory, with the API keys), the well-known secret
    stores and the pseudo-filesystems."""
    return [*map(Path, deny), *(Path(p).expanduser() for p in SECRET_PATHS), *map(Path, PSEUDO_FILESYSTEMS)]


def agent_access(deny: Sequence[str | Path] = ()) -> "FileAccess":
    """What every agent may read: any file the researcher can, except credentials. CEDAR's own records, every run
    directory and every agent's workspace included."""
    return FileAccess(roots=["/"], deny=credential_paths(deny))


def _paths(value: PathsLike) -> list[Path]:
    paths = value() if callable(value) else value
    return [Path(p).resolve() for p in paths]


def _within(path: Path, base: Path) -> bool:
    return path == base or path.is_relative_to(base)


class FileAccess:
    def __init__(self, roots: PathsLike, deny: PathsLike = (), allow: PathsLike = ()) -> None:
        """``allow`` names paths readable even inside a denied directory."""
        self._roots = roots
        self._deny = deny
        self._allow = allow
        self._fixed = None if any(callable(p) for p in (roots, deny, allow)) else (
            _paths(roots), _paths(deny), _paths(allow))

    def snapshot(self) -> "FileAccess":
        """The same access with its paths worked out once, e.g. for one tool call that checks many paths."""
        return FileAccess(self.roots, self.deny, self.allow)

    @property
    def roots(self) -> list[Path]:
        return self._fixed[0] if self._fixed else _paths(self._roots)

    @property
    def deny(self) -> list[Path]:
        return self._fixed[1] if self._fixed else _paths(self._deny)

    @property
    def allow(self) -> list[Path]:
        return self._fixed[2] if self._fixed else _paths(self._allow)

    def readable(self, resolved: Path) -> bool:
        if any(_within(resolved, a) for a in self.allow):
            return True
        if any(_within(resolved, d) for d in self.deny):
            return False
        return any(_within(resolved, r) for r in self.roots)

    def leads_somewhere(self, resolved: Path) -> bool:
        """Whether a path is readable or holds an allowed path, so a search must look inside it."""
        return self.readable(resolved) or any(_within(a, resolved) for a in self.allow)

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            raise ToolFailure(f"use an absolute path; readable roots are {[str(r) for r in self.roots]}")
        resolved = candidate.resolve()
        allow, deny = self.allow, self.deny
        if any(_within(resolved, a) for a in allow):
            return resolved
        if any(_within(resolved, d) for d in deny):
            raise PermissionError(f"{path} is not readable by agents")
        roots = self.roots
        if not any(_within(resolved, r) for r in roots):
            raise PermissionError(f"{path} is outside the readable roots {[str(r) for r in roots]}")
        return resolved

    def regular_file(self, path: str) -> Path:
        resolved = self.resolve(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"{path} is not a regular file")
        return resolved

    def text_lines(self, path: str) -> tuple[Path, Iterator[str]]:
        """The file's lines as a stream, numbered exactly as the validator numbers them (``str.splitlines``)."""
        resolved = self.regular_file(path)
        with resolved.open("rb") as stream:
            if b"\x00" in stream.read(8192):
                raise ToolFailure(f"{path} looks like a binary file")
        return resolved, _stream_lines(resolved)


def _stream_lines(path: Path) -> Iterator[str]:
    # Splitting at b"\n" first and then with splitlines gives the same lines as splitlines on the whole text.
    with path.open("rb") as stream:
        for raw in stream:
            yield from raw.decode(errors="replace").splitlines()


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
    """The file tools; ``on_read`` is told about every file whose contents an agent reads. Tools that check many
    paths work out what may be read once, at the start of the call."""

    def text_lines(path: str) -> tuple[Path, Iterator[str]]:
        resolved, lines = access.text_lines(path)
        if on_read is not None:
            on_read(resolved)
        return resolved, lines

    def list_directory(args: ListArgs) -> str:
        view = access.snapshot()
        directory = view.resolve(args.path)
        if not directory.is_dir():
            raise NotADirectoryError(f"{args.path} is not a directory")
        entries = sorted(directory.iterdir())
        lines = []
        for entry in entries[:MAX_ENTRIES]:
            try:
                if not view.leads_somewhere(entry.resolve()):
                    lines.append(f"{entry.name}  (not readable by agents)")
                elif entry.is_dir():
                    lines.append(f"{entry.name}/")
                else:
                    lines.append(f"{entry.name}  ({entry.stat().st_size} bytes)")
            except OSError as exc:
                lines.append(f"{entry.name}  ({exc.strerror or 'unreadable'})")
        if len(entries) > MAX_ENTRIES:
            lines.append(f"... {len(entries) - MAX_ENTRIES} more entries")
        return "\n".join(lines) or "(empty directory)"

    def find_files(args: FindArgs) -> str:
        view = access.snapshot()
        root = view.resolve(args.root)
        found, more, visited, stopped = [], 0, 0, False
        for directory, subdirectories, names in os.walk(root):
            visited += len(subdirectories) + len(names)
            if visited > MAX_VISITED:
                stopped = True
                break
            here = Path(directory)
            subdirectories[:] = sorted(d for d in subdirectories
                                       if not (here / d).is_symlink() and view.leads_somewhere(here / d))
            for name in sorted(names):
                if not fnmatch.fnmatch(name, args.name_pattern):
                    continue
                path = here / name
                try:
                    resolved = view.resolve(str(path))
                except PermissionError:
                    continue
                if not resolved.is_file():
                    continue
                if len(found) < args.max_results:
                    found.append(str(resolved))
                else:
                    more += 1
        note = f"\n[{more} more matches not shown]" if more else ""
        if stopped:
            note += f"\n[stopped after visiting {MAX_VISITED} entries; search under a narrower root]"
        return ("\n".join(found) or "no matching files") + note

    def read_file(args: ReadArgs) -> str:
        resolved, lines = text_lines(args.path)
        first = args.start_line - 1
        chosen, total = [], 0
        countable = resolved.stat().st_size <= MAX_COUNT_BYTES
        for index, line in enumerate(lines):
            total = index + 1
            if first <= index < first + args.max_lines:
                chosen.append(f"{total}: {_clip(line)}")
            elif index >= first + args.max_lines and not countable:
                break
        body = "\n".join(chosen)
        last = first + len(chosen)
        if countable:
            tail = f"\n[lines {args.start_line}-{last} of {total}]"
        else:
            tail = f"\n[lines {args.start_line}-{last}; the file has {resolved.stat().st_size} bytes, too many to count]"
        return (body or "(no lines in this range)") + tail

    def search_file(args: SearchArgs) -> str:
        try:
            pattern = re.compile(args.pattern)
        except re.error as exc:
            raise ToolFailure(f"invalid regular expression: {exc}") from None
        resolved, lines = text_lines(args.path)
        before: deque[tuple[int, str]] = deque(maxlen=args.context_lines)
        out: list[str] = []
        matches = after = scanned = 0
        last_shown, count, note = -1, 0, ""
        for index, line in enumerate(lines):
            count = index + 1
            scanned += len(line) + 1
            if pattern.search(line):
                matches += 1
                if matches > args.max_matches:
                    note = f"\n[stopped after {args.max_matches} matches]"
                    break
                out += [f"{i + 1}: {_clip(text)}" for i, text in before if i > last_shown]
                out.append(f"{count}: {_clip(line)}")
                last_shown, after = index, args.context_lines
                before.clear()
            elif after:
                out.append(f"{count}: {_clip(line)}")
                last_shown, after = index, after - 1
            else:
                before.append((index, line))
            if scanned > MAX_SCAN_BYTES:
                note = f"\n[stopped after scanning {MAX_SCAN_BYTES} bytes, at line {count}; read further with " \
                       "read_file, or search a smaller file]"
                break
        else:
            note = f"\n[file has {count} lines]"
        return ("\n".join(out) or "no matching lines") + note

    def view_file(args: ViewArgs) -> ToolOutcome:
        resolved = access.regular_file(args.path)
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


# -- the agent's workspace ------------------------------------------------------------------


class WriteArgs(Contract):
    path: str = Field(description="Where in your workspace: a path relative to it, or an absolute path inside it. "
                                  "Missing directories are created.")
    text: str = Field(description="The file's full text.")
    overwrite: bool = Field(default=False, description="Replace the file if it exists.")


class DeleteArgs(Contract):
    path: str = Field(description="A file or directory in your workspace, relative to it or absolute.")


def workspace_tools(root: Path) -> list[AgentTool]:
    """Write and delete files in one workspace directory; nothing outside it."""

    def write_file(args: WriteArgs) -> str:
        target = workspace_path(root, args.path)
        if target == root.resolve() or target.is_dir():
            raise IsADirectoryError(f"{args.path} is a directory")
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target} exists; pass overwrite=true to replace it")
        target.parent.mkdir(parents=True, exist_ok=True)
        workspace_path(root, str(target))  # nothing created a link meanwhile
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(args.text)
        return f"Wrote {target} ({len(args.text.encode())} bytes)."

    def delete_file(args: DeleteArgs) -> str:
        target = workspace_path(root, args.path)
        if target == root.resolve():
            raise PermissionError("the workspace itself cannot be deleted")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
            return f"Deleted the directory {target}."
        if not target.exists() and not target.is_symlink():
            raise FileNotFoundError(f"{target} does not exist")
        target.unlink()
        return f"Deleted {target}."

    return [
        AgentTool("write_file", f"Write a text file in your workspace ({root}), e.g. a structure, input file or "
                  "script. You can read it back with the file tools and give its path to request_calculation.",
                  WriteArgs, write_file),
        AgentTool("delete_file", "Delete a file or directory in your workspace.", DeleteArgs, delete_file),
    ]
