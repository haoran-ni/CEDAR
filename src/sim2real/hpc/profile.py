"""Compute profiles: how codes run on one particular HPC system.

A profile is data. The environment-setup agent proposes it from local files or the
user's answers, ``check_profile`` verifies it, and it takes effect only once the user
approves that exact content. Nothing in this module knows about any specific site.

Job templates use ``{{name}}`` placeholders, which do not clash with shell ``${VAR}``.
"""

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, Field, model_validator

from sim2real.contracts.base import Contract, Identifier, NonEmptyStr, Sha256Hex, canonical_json, sha256_hex
from sim2real.contracts.examination import Finding, FindingSeverity

TEMPLATE_FIELDS = frozenset({"job_name", "partition", "nodes", "ntasks", "walltime", "binary", "output"})
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_]+)\s*\}\}")


def _check_absolute(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"expected a normalized absolute path: {value!r}")
    return value


AbsolutePath = Annotated[str, AfterValidator(_check_absolute)]


class ProfileSource(Contract):
    """Where a profile value came from."""

    kind: Literal["file", "user_answer", "command_output"]
    reference: NonEmptyStr
    """The file path, the question asked, or the command run."""
    sha256: Sha256Hex | None = None
    """Content hash of the file or command output when it was read."""
    note: str | None = None


class CodeInstallation(Contract):
    binary: AbsolutePath
    version: NonEmptyStr
    """The version string the code reports in its own output."""
    data_paths: dict[Identifier, AbsolutePath] = {}
    """Code data such as basis or species-default directories, by role."""


class SchedulerLimits(Contract):
    partitions: tuple[Identifier, ...] = Field(min_length=1)
    default_partition: Identifier
    cores_per_node: int = Field(gt=0)
    default_ntasks: int = Field(gt=0)
    max_nodes: int = Field(gt=0)
    default_walltime_minutes: int = Field(gt=0)
    max_walltime_minutes: int = Field(gt=0)

    @model_validator(mode="after")
    def _consistent(self) -> "SchedulerLimits":
        if self.default_partition not in self.partitions:
            raise ValueError("default_partition must be one of partitions")
        if self.default_walltime_minutes > self.max_walltime_minutes:
            raise ValueError("default walltime exceeds the maximum")
        if self.default_ntasks > self.cores_per_node * self.max_nodes:
            raise ValueError("default_ntasks exceeds what max_nodes can hold")
        return self


def template_fields(text: str) -> set[str]:
    return {m.group(1) for m in _PLACEHOLDER.finditer(text)}


def fill_template(text: str, values: dict[str, object]) -> str:
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)


class ComputeProfile(Contract):
    name: Identifier
    scheduler: Literal["slurm"]
    limits: SchedulerLimits
    codes: dict[Identifier, CodeInstallation] = Field(min_length=1)
    job_template: NonEmptyStr
    """Job script with ``{{placeholders}}``; exactly one line launches ``{{binary}}`` into ``{{output}}``."""
    runs_root: AbsolutePath
    """Directory for run directories, on a filesystem the compute nodes can reach."""
    sources: tuple[ProfileSource, ...] = ()

    @model_validator(mode="after")
    def _template_well_formed(self) -> "ComputeProfile":
        unknown = template_fields(self.job_template) - TEMPLATE_FIELDS
        if unknown:
            raise ValueError(f"unknown template placeholders: {sorted(unknown)}")
        if not self.job_template.startswith("#!"):
            raise ValueError("the job template must start with a shebang line")
        launch = [line for line in self.job_template.splitlines() if "binary" in template_fields(line)]
        if len(launch) != 1 or "output" not in template_fields(launch[0]):
            raise ValueError("exactly one template line must launch {{binary}} writing to {{output}}")
        return self

    def content_hash(self) -> str:
        return sha256_hex(canonical_json(self))


class ProfileApproval(Contract):
    profile_name: Identifier
    profile_hash: Sha256Hex
    approved_by: Identifier
    approved_at: AwareDatetime
    note: str | None = None


def check_profile(profile: ComputeProfile) -> list[Finding]:
    """Deterministic checks of a proposed profile against the filesystem it describes."""
    findings: list[Finding] = []

    def add(severity: FindingSeverity, code: str, message: str) -> None:
        findings.append(Finding(severity=severity, code=code, message=message))

    for name, code in profile.codes.items():
        binary = Path(code.binary)
        if not binary.is_file():
            add(FindingSeverity.ERROR, "missing_binary", f"{name}: {binary} does not exist")
        elif not os.access(binary, os.X_OK):
            add(FindingSeverity.ERROR, "binary_not_executable", f"{name}: {binary} is not executable")
        for role, path in code.data_paths.items():
            if not Path(path).is_dir():
                add(FindingSeverity.ERROR, "missing_data_path", f"{name}: {role} directory {path} does not exist")
    runs_root = Path(profile.runs_root)
    existing = next((p for p in (runs_root, *runs_root.parents) if p.exists()), None)
    if existing is None or not os.access(existing, os.W_OK):
        add(FindingSeverity.ERROR, "runs_root_not_writable", f"cannot create run directories under {runs_root}")
    if not profile.sources:
        add(FindingSeverity.WARNING, "no_sources", "the profile does not record where its values came from")
    return findings


class ProfileNotApproved(RuntimeError):
    pass


class ProfileStore:
    """Profiles and their approvals as JSON files in one directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _path(self, name: str, suffix: str) -> Path:
        return self.directory / f"{name}{suffix}"

    def save(self, profile: ComputeProfile) -> Path:
        """Store a proposed profile. Changing an approved profile withdraws its approval."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(profile.name, ".profile.json")
        path.write_text(json.dumps(profile.model_dump(mode="json"), indent=2) + "\n")
        return path

    def load(self, name: str) -> ComputeProfile:
        return ComputeProfile.model_validate_json(self._path(name, ".profile.json").read_text())

    def approve(self, name: str, user: str, at: datetime, note: str | None = None) -> ProfileApproval:
        profile = self.load(name)
        approval = ProfileApproval(profile_name=name, profile_hash=profile.content_hash(), approved_by=user,
                                   approved_at=at, note=note)
        self._path(name, ".approval.json").write_text(approval.model_dump_json(indent=2) + "\n")
        return approval

    def load_approved(self, name: str) -> ComputeProfile:
        profile = self.load(name)
        approval_path = self._path(name, ".approval.json")
        if not approval_path.exists():
            raise ProfileNotApproved(f"profile {name} has not been approved by the user")
        approval = ProfileApproval.model_validate_json(approval_path.read_text())
        if approval.profile_hash != profile.content_hash():
            raise ProfileNotApproved(f"profile {name} changed after it was approved")
        return profile


def file_source(path: str | Path, note: str | None = None) -> ProfileSource:
    """Record a local file the environment-setup agent read, with its content hash."""
    data = Path(path).read_bytes()
    return ProfileSource(kind="file", reference=str(path), sha256=hashlib.sha256(data).hexdigest(), note=note)
