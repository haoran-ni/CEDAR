"""Workbench runs: shell commands and scripts an agent ran in its workspace, inside a sandbox.

A workbench run is recorded with what it ran, where, with what resources, how it ended,
what it printed, and the content hash of every file it created or changed in the workspace.
It is context for the agent's work, never evidence: evidence comes from readings and
analyses of checked or registered runs.
"""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from cedar.contracts.base import ArtifactRef, Contract, Identifier, NonEmptyStr, RelativePath, Sha256Hex


class WorkbenchOutcome(StrEnum):
    COMPLETED = "completed"
    """The command exited with status 0."""
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    STOPPED = "stopped"
    """Stopped because the agent's work was halted or the agent retired."""


class WorkbenchRun(Contract):
    command: NonEmptyStr
    purpose: NonEmptyStr
    directory: str = "."
    """Where it ran, relative to the workspace."""
    sandbox: Literal["bwrap", "python-guard"]
    """``bwrap``: any program, with no network, the filesystem read-only and secrets hidden; ``python-guard``: a
    Python script only, under an audit hook, where this machine has no bubblewrap."""
    host: NonEmptyStr
    allocation: Identifier | None = None
    """The Slurm allocation it ran in, on an interactive node."""
    cores: int = Field(ge=1)
    time_limit_seconds: int = Field(ge=1)
    outcome: WorkbenchOutcome
    exit_code: int | None = None
    duration_seconds: float = Field(ge=0)
    output: ArtifactRef | None = None
    """What it printed, standard output and errors together."""
    changed: dict[RelativePath, Sha256Hex] = {}
    """Files it created or changed in the workspace, by content hash."""
    deleted: tuple[RelativePath, ...] = ()
    problems: tuple[NonEmptyStr, ...] = ()
