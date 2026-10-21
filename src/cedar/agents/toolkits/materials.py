"""Tools to list and view the materials the researcher attached: figures, papers, measured data.

Every agent of an investigation has them. The materials are the researcher's, so sharing them
does not break the agents' independence from each other. They are context, not evidence.
Images and PDFs are shown to the model; text files come back as numbered lines, like the file
tools.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import Field

from cedar.agents.media import MAX_PDF_PAGES, load_media, viewable
from cedar.agents.tools import AgentTool, ToolFailure, ToolOutcome
from cedar.contracts.base import ArtifactRef, Contract, Identifier
from cedar.state.projection import InvestigationState
from cedar.store.artifacts import ArtifactStore

MAX_LINE = 400
MAX_LINES = 400


@dataclass(frozen=True)
class Material:
    material_id: str
    name: str
    artifact: ArtifactRef
    note: str | None = None


def recorded_materials(state: InvestigationState) -> list[Material]:
    return [Material(m.material_id, m.name, m.artifact, m.note) for m in state.materials.values()]


def material_line(material: Material) -> str:
    kind = material.artifact.media_type or "file"
    note = f": {material.note}" if material.note else ""
    return f"{material.material_id}: {material.name} ({kind}, {material.artifact.size_bytes:,} bytes){note}"


class NoArgs(Contract):
    pass


class ViewArgs(Contract):
    material_id: Identifier
    pages: str | None = Field(default=None, description=f"PDF pages to view, e.g. '1-3, 7'; needed when the PDF has "
                                                        f"more than {MAX_PDF_PAGES} pages.")
    start_line: int = Field(default=1, ge=1, description="Text files: first line to return (1-based).")
    max_lines: int = Field(default=200, ge=1, le=MAX_LINES, description="Text files: how many lines to return.")


def material_tools(materials: Callable[[], Sequence[Material]], artifacts: ArtifactStore) -> list[AgentTool]:
    """``materials`` gives the investigation's materials at the moment a tool is called."""

    def find(material_id: str) -> Material:
        known = {m.material_id: m for m in materials()}
        if material_id not in known:
            raise ToolFailure(f"no material {material_id}; attached: {', '.join(known) or 'none'}")
        return known[material_id]

    def list_materials(_: NoArgs) -> str:
        lines = [material_line(m) for m in materials()]
        return "\n".join(lines) or "The researcher has attached no materials."

    def view_material(args: ViewArgs) -> ToolOutcome:
        material = find(args.material_id)
        data = artifacts.read(material.artifact)
        if viewable(data):
            if args.start_line != 1:
                raise ToolFailure("start_line applies only to text files")
            media, note = load_media(data, material.name, args.pages)
            return ToolOutcome(f"Showing {material.material_id}, {note}. The researcher's material: context, not "
                               "evidence.", media=(media,))
        if b"\x00" in data[:8192]:
            raise ToolFailure(f"{material.material_id} is a binary file that is neither an image nor a PDF")
        if args.pages is not None:
            raise ToolFailure("pages applies only to PDFs")
        lines = data.decode(errors="replace").splitlines()
        first = args.start_line - 1
        chosen = lines[first:first + args.max_lines]
        body = "\n".join(f"{first + i + 1}: {line if len(line) <= MAX_LINE else line[:MAX_LINE] + ' …'}"
                         for i, line in enumerate(chosen))
        return ToolOutcome((body or "(no lines in this range)")
                           + f"\n[{material.material_id} {material.name}: lines {args.start_line}-"
                             f"{first + len(chosen)} of {len(lines)}]")

    return [
        AgentTool("list_materials", "List the files the researcher attached to the investigation (figures, papers, "
                  "measured data), with their notes.", NoArgs, list_materials),
        AgentTool("view_material", "Look at one attached material: an image, a PDF (up to "
                  f"{MAX_PDF_PAGES} pages at a time) or the lines of a text file. A picture or a PDF stays in your "
                  "conversation and costs tokens on every later turn, so view only what you need.", ViewArgs,
                  view_material),
    ]
