"""Tools for the environment-setup agent: discover how codes run here, then propose a profile.

The agent reads local files (with the file tools), runs read-only scheduler queries,
inspects paths without reading them, and asks the user. Every file it read, query it ran
and answer it received becomes a recorded source of the profile it proposes. A proposal
is saved but never approved here: the user approves it separately.
"""

import hashlib
import json
import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field, ValidationError

from cedar.agents.tools import AgentTool, ToolFailure, ToolOutcome
from cedar.contracts.base import Contract, Identifier, NonEmptyStr
from cedar.contracts.examination import FindingSeverity
from cedar.hpc.profile import (
    CodeInstallation,
    ComputeProfile,
    LocalExecution,
    ProfileSource,
    ProfileStore,
    SchedulerLimits,
    check_profile,
)
from cedar.hpc.queries import SchedulerQueries

MAX_ENTRIES = 100


@dataclass
class SetupSession:
    """What the agent consulted, kept as profile sources in the order first consulted."""

    sources: dict[str, ProfileSource] = field(default_factory=dict)

    def file_read(self, path: Path) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.sources[f"file:{path}"] = ProfileSource(kind="file", reference=str(path), sha256=digest)

    def query_run(self, command: str, output: str) -> None:
        self.sources[f"command:{command}"] = ProfileSource(
            kind="command_output", reference=command, sha256=hashlib.sha256(output.encode()).hexdigest())

    def answer(self, question: str, answer: str) -> None:
        self.sources[f"answer:{question}"] = ProfileSource(kind="user_answer", reference=question, note=answer)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({key: s.model_dump(mode="json") for key, s in self.sources.items()}, indent=1))
        os.chmod(path, 0o600)

    @classmethod
    def load(cls, path: Path) -> "SetupSession":
        data = json.loads(path.read_text()) if path.exists() else {}
        return cls({key: ProfileSource.model_validate(value) for key, value in data.items()})

    def with_notes(self, notes: dict[str, str]) -> tuple[ProfileSource, ...]:
        annotated = []
        for source in self.sources.values():
            note = notes.get(source.reference)
            if note and source.kind != "user_answer":
                source = source.model_copy(update={"note": note})
            annotated.append(source)
        return tuple(annotated)


class InspectArgs(Contract):
    path: str = Field(description="Absolute path to inspect.")


class QueryArgs(Contract):
    query: str = Field(description=f"One of: {', '.join(SchedulerQueries.QUERIES)}")
    argument: str | None = Field(default=None, description="Partition name, for the queries that take one.")


class AskArgs(Contract):
    question: NonEmptyStr = Field(description="One clear question; say why you need to know and offer options "
                                              "when there are obvious ones.")


class CodeArgs(Contract):
    binary: str
    version: NonEmptyStr = Field(description="The version string exactly as the code prints it in its output.")
    data_paths: dict[Identifier, str] = Field(default={}, description="e.g. {\"species_defaults\": \"/path\"} for "
                                                                      "FHI-aims, {\"potcars\": \"/path/potpaw_PBE\"} "
                                                                      "for VASP")
    variants: dict[Identifier, str] = Field(default={}, description="Other builds of the same version, by name, e.g. "
                                                                    "{\"gam\": \"/path/vasp_gam\", \"ncl\": "
                                                                    "\"/path/vasp_ncl\"} for VASP, whose binary is "
                                                                    "then vasp_std.")
    job_template: NonEmptyStr | None = Field(default=None, description="This code's own job template, with the same "
                                                                       "placeholders, when its jobs need a setup the "
                                                                       "profile's template does not have (other "
                                                                       "modules or environment scripts).")


class ProposeArgs(Contract):
    name: Identifier = Field(description="Short profile name, e.g. cluster-partition.")
    scheduler: str = Field(description="Scheduler type; only slurm is supported now.")
    partitions: list[Identifier] = Field(min_length=1, description="Partitions CEDAR may use.")
    default_partition: Identifier
    cores_per_node: int = Field(gt=0)
    default_ntasks: int = Field(gt=0, description="MPI tasks for generated job scripts.")
    max_nodes: int = Field(gt=0)
    default_walltime_minutes: int = Field(gt=0)
    max_walltime_minutes: int = Field(gt=0)
    codes: dict[Identifier, CodeArgs] = Field(min_length=1, description="Installations by code name: fhi-aims, "
                                                                         "vasp.")
    job_template: NonEmptyStr = Field(description="Job script with {{job_name}}, {{partition}}, {{nodes}}, "
                                                  "{{ntasks}}, {{walltime}}, {{binary}}, {{output}} placeholders.")
    runs_root: str = Field(description="Directory for run directories, reachable from compute nodes.")
    python_environments: list[str] = Field(default=[], description="Further Python environments (their prefixes) "
                                                                   "whose programs agents' workbench commands may "
                                                                   "run, if the researcher wants them.")
    hidden_paths: list[str] = Field(default=[], description="Further programs or directories workbench commands "
                                                            "must not see, e.g. other DFT installations; the codes' "
                                                            "binaries and the MPI launchers are always hidden.")
    source_notes: dict[str, str] = Field(default={}, description="For files and commands you consulted "
                                                                 "(by path or command): which values they support.")


def describe_path(path: Path) -> str:
    if not path.exists():
        return f"{path}: does not exist"
    info = path.stat()
    if path.is_dir():
        entries = sorted(e.name + ("/" if e.is_dir() else "") for e in path.iterdir())
        more = f" … {len(entries) - MAX_ENTRIES} more" if len(entries) > MAX_ENTRIES else ""
        return f"{path}: directory with {len(entries)} entries: {', '.join(entries[:MAX_ENTRIES])}{more}"
    executable = "executable" if os.access(path, os.X_OK) else "not executable"
    return f"{path}: file, {info.st_size} bytes, {stat.filemode(info.st_mode)}, {executable}"


def environment_tools(store: ProfileStore, queries: SchedulerQueries, session: SetupSession,
                      ask: Callable[[str], str], deny: Sequence[str | Path] = ()) -> list[AgentTool]:
    denied = [Path(d).resolve() for d in deny]

    def inspect_path(args: InspectArgs) -> str:
        path = Path(args.path)
        if not path.is_absolute():
            raise ToolFailure("use an absolute path")
        resolved = path.resolve()
        if any(resolved == d or resolved.is_relative_to(d) for d in denied):
            raise PermissionError(f"{args.path} is not inspectable by agents")
        return describe_path(path)

    def scheduler_query(args: QueryArgs) -> str:
        result = queries.run(args.query, args.argument)
        session.query_run(result.command, result.output)
        note = "\n[output truncated]" if result.truncated else ""
        return f"$ {result.command}\n{result.output}{note}"

    def ask_user(args: AskArgs) -> str:
        answer = ask(args.question).strip()
        session.answer(args.question, answer or "(no answer)")
        return answer or "The user gave no answer."

    def propose_profile(args: ProposeArgs) -> ToolOutcome:
        try:
            profile = ComputeProfile(
                name=args.name, scheduler=args.scheduler,
                limits=SchedulerLimits(
                    partitions=tuple(args.partitions), default_partition=args.default_partition,
                    cores_per_node=args.cores_per_node, default_ntasks=args.default_ntasks, max_nodes=args.max_nodes,
                    default_walltime_minutes=args.default_walltime_minutes,
                    max_walltime_minutes=args.max_walltime_minutes),
                codes={name: CodeInstallation(binary=code.binary, version=code.version, data_paths=code.data_paths,
                                              variants=code.variants, job_template=code.job_template)
                       for name, code in args.codes.items()},
                job_template=args.job_template, runs_root=args.runs_root,
                sources=session.with_notes(args.source_notes),
                local=LocalExecution(python_environments=tuple(args.python_environments),
                                     hidden_paths=tuple(args.hidden_paths))
                if args.python_environments or args.hidden_paths else None,
            )
        except ValidationError as exc:
            problems = "\n".join(f"- {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
            return ToolOutcome(f"The profile is not valid:\n{problems}", is_error=True)
        findings = check_profile(profile)
        listing = "\n".join(f"- [{f.severity.value}] {f.code}: {f.message}" for f in findings) or "- none"
        if any(f.severity is FindingSeverity.ERROR for f in findings):
            return ToolOutcome(f"Checks failed; nothing saved:\n{listing}", is_error=True)
        path = store.save(profile)
        return (f"Saved the proposal {profile.name} to {path} with {len(profile.sources)} recorded sources. "
                f"Check findings:\n{listing}\nIt is not in effect until the user approves it.")

    return [
        AgentTool("inspect_path", "Check whether a path exists, whether a file is executable, or what a directory "
                  "contains, without reading any file's contents. Works on any absolute path.", InspectArgs,
                  inspect_path),
        AgentTool("scheduler_query", "Run a read-only scheduler query: " + "; ".join(
            f"{k}: {v}" for k, v in SchedulerQueries.QUERIES.items()), QueryArgs, scheduler_query),
        AgentTool("ask_user", "Ask the researcher a question. Use it for choices only they can make (which "
                  "partitions CEDAR may use, default resources, where runs should go) and whenever local files "
                  "do not settle something. Ask one question at a time.", AskArgs, ask_user),
        AgentTool("propose_profile", "Check a complete compute profile and save it as a proposal for the user to "
                  "approve. Fix every error it reports and propose again.", ProposeArgs, propose_profile),
    ]
