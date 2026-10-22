"""Compute profiles: how codes run on one particular HPC system.

A profile is data. The environment-setup agent proposes it from local files or the
user's answers, ``check_profile`` verifies it, and it takes effect only once the user
approves that exact content. Nothing in this module knows about any specific site.

Job templates use ``{{name}}`` placeholders, which do not clash with shell ``${VAR}``. A code
whose jobs need a setup the profile's template does not have carries its own template.
"""

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import AfterValidator, AwareDatetime, Field, model_validator

from cedar.contracts.base import Contract, Identifier, NonEmptyStr, Sha256Hex, canonical_json, sha256_hex
from cedar.contracts.examination import Finding, FindingSeverity

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
    variants: dict[Identifier, AbsolutePath] = {}
    """Other builds of the same version, by name, which a job script may launch instead of ``binary``, e.g. VASP's
    ``gam`` and ``ncl``."""
    job_template: NonEmptyStr | None = None
    """This code's own job-script template, when it needs a setup the profile's does not have; the profile's when
    None."""

    @model_validator(mode="after")
    def _template_well_formed(self) -> "CodeInstallation":
        if self.job_template is not None:
            check_template(self.job_template)
        return self

    @property
    def binaries(self) -> tuple[str, ...]:
        """Every build a job script may launch: ``binary`` first."""
        return (self.binary, *self.variants.values())


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


class LocalExecution(Contract):
    """How the agents' workbench commands may use the machine CEDAR runs on.

    On a login node, shared with other users, each command gets few cores; on an interactive node, inside the
    researcher's Slurm allocation, commands share the allocation's cores with local calculations. Commands always
    have a time limit, because the agent waits for each one.
    """

    login_cores_per_command: int = Field(default=1, gt=0)
    login_max_commands: int = Field(default=2, gt=0)
    """Commands running at once on a login node, whichever agents started them."""
    default_command_minutes: int = Field(default=10, gt=0)
    login_max_command_minutes: int = Field(default=30, gt=0)
    node_max_command_minutes: int = Field(default=60, gt=0)
    python_environments: tuple[AbsolutePath, ...] = ()
    """Further Python environments (their prefixes), whose programs commands may run besides CEDAR's own."""
    hidden_paths: tuple[AbsolutePath, ...] = ()
    """Programs or directories commands must not see, besides the codes' binaries and the MPI launchers."""

    @model_validator(mode="after")
    def _consistent(self) -> "LocalExecution":
        if self.default_command_minutes > min(self.login_max_command_minutes, self.node_max_command_minutes):
            raise ValueError("the default command time exceeds a maximum")
        return self


def template_fields(text: str) -> set[str]:
    return {m.group(1) for m in _PLACEHOLDER.finditer(text)}


def fill_template(text: str, values: dict[str, object]) -> str:
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)


def check_template(text: str) -> None:
    """Raise ValueError unless ``text`` is a job template CEDAR can render and examine scripts against."""
    unknown = template_fields(text) - TEMPLATE_FIELDS
    if unknown:
        raise ValueError(f"unknown template placeholders: {sorted(unknown)}")
    if not text.startswith("#!"):
        raise ValueError("the job template must start with a shebang line")
    launch = [line for line in text.splitlines() if "binary" in template_fields(line)]
    if len(launch) != 1 or "output" not in template_fields(launch[0]):
        raise ValueError("exactly one template line must launch {{binary}} writing to {{output}}")


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
    local: LocalExecution | None = None
    """How workbench commands use this machine; the defaults of ``LocalExecution`` when not given."""

    @model_validator(mode="after")
    def _template_well_formed(self) -> "ComputeProfile":
        check_template(self.job_template)
        return self

    @property
    def local_execution(self) -> LocalExecution:
        return self.local or LocalExecution()

    def template_for(self, code: CodeInstallation) -> str:
        """The job template ``code`` runs with: its own, or the profile's."""
        return code.job_template or self.job_template

    def content_hash(self) -> str:
        # A profile approved before ``local``, or a code's ``variants`` and ``job_template``, existed keeps its hash,
        # and so its approval.
        data = self.model_dump(mode="json", exclude={"local"} if self.local is None else None)
        for code in data["codes"].values():
            if not code["variants"]:
                del code["variants"]
            if code["job_template"] is None:
                del code["job_template"]
        return sha256_hex(canonical_json(data))


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
        for binary in map(Path, code.binaries):
            if not binary.is_file():
                add(FindingSeverity.ERROR, "missing_binary", f"{name}: {binary} does not exist")
            elif not os.access(binary, os.X_OK):
                add(FindingSeverity.ERROR, "binary_not_executable", f"{name}: {binary} is not executable")
        for role, path in code.data_paths.items():
            if not Path(path).is_dir():
                add(FindingSeverity.ERROR, "missing_data_path", f"{name}: {role} directory {path} does not exist")
    for prefix in profile.local_execution.python_environments:
        if not (Path(prefix) / "bin").is_dir():
            add(FindingSeverity.ERROR, "missing_python_environment", f"{prefix} has no bin directory")
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

    def names(self) -> list[str]:
        return sorted(p.name.removesuffix(".profile.json") for p in self.directory.glob("*.profile.json"))

    def is_approved(self, name: str) -> bool:
        try:
            self.load_approved(name)
        except ProfileNotApproved:
            return False
        return True

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
