"""Wiring: the research data directory, and the services an investigation's profile implies."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from cedar.agents.toolkits.files import PSEUDO_FILESYSTEMS, agent_access
from cedar.agents.toolkits.research import ResearchContext
from cedar.agents.transcripts import TranscriptStore
from cedar.contracts.base import Actor, ActorKind
from cedar.engine import JobManager, JobMonitor, ProtocolEngine, ResultValidator, RunRegistrar
from cedar.engine.analysis import AnalysisRunner
from cedar.engine.external import ExternalEvidenceRecorder
from cedar.engine.sandbox import SandboxView, program_paths
from cedar.engine.workbench import AgentBench, Workbench
from cedar.hpc.local import LocalBackend
from cedar.hpc.profile import ComputeProfile, LocalExecution, ProfileNotApproved, ProfileStore
from cedar.hpc.slurm import SchedulerBackend, SlurmBackend
from cedar.models import CredentialStore, ModelClient, ModelConfig, ModelSettings, create_client
from cedar.protocols import FhiAimsRun, FhiAimsRunReviewed, FhiAimsRunSlurmOnly, ProtocolRegistry
from cedar.protocols.fhiaims_run import current_host
from cedar.state.ledger import ResearchLedger
from cedar.state.projection import AgentRecord, CalculationRecord, InvestigationState
from cedar.store import ArtifactStore, SQLiteEventStore


@dataclass
class Workspace:
    """Everything stored under the research data directory."""

    root: Path
    store: SQLiteEventStore
    ledger: ResearchLedger
    artifacts: ArtifactStore
    transcripts: TranscriptStore

    @classmethod
    def open(cls, root: str | Path, *, clock: Callable[[], datetime] | None = None) -> "Workspace":
        """``clock`` stamps recorded events; tests pass their own."""
        root = Path(root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        store = SQLiteEventStore(root / "events.sqlite")
        ledger = ResearchLedger(store, clock=clock) if clock is not None else ResearchLedger(store)
        return cls(root, store, ledger, ArtifactStore(root / "artifacts"), TranscriptStore(root / "transcripts"))

    def notebook_dir(self, investigation_id: str) -> Path:
        """Where the investigation's research notebook (generated Markdown) lives."""
        return self.root / "notebooks" / investigation_id

    def workspaces(self, investigation_id: str) -> Path:
        """Where the investigation's agents have their workspaces, one directory per hypothesis."""
        return self.root / "workspaces" / investigation_id

    def agent_workspace(self, investigation_id: str, hypothesis_id: str) -> Path:
        """The directory of a hypothesis's agent, the only place it writes; created on first use."""
        path = self.workspaces(investigation_id) / hypothesis_id
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        return path


@dataclass
class Services:
    """Engine services for one approved compute profile."""

    profile: ComputeProfile
    registry: ProtocolRegistry
    engine: ProtocolEngine
    manager: JobManager
    monitor: JobMonitor
    validator: ResultValidator
    analysis: AnalysisRunner
    deny: list[Path]
    local_backend: LocalBackend
    """Runs local calculations on the interactive node CEDAR runs on."""
    local_manager: JobManager
    local_monitor: JobMonitor

    def manager_for(self, calc: CalculationRecord) -> JobManager:
        return self.local_manager if calc.task.execution == "local" else self.manager


def registry_for(profile: ComputeProfile) -> ProtocolRegistry:
    protocols = ([FhiAimsRunReviewed(profile), FhiAimsRun(profile), FhiAimsRunSlurmOnly(profile)]
                 if "fhi-aims" in profile.codes else [])
    return ProtocolRegistry(protocols)


def services_for(workspace: Workspace, profile: ComputeProfile, backend: SchedulerBackend | None = None,
                 *, deny: list[Path] = (), analysis_slots: threading.Semaphore | None = None,
                 local_backend: LocalBackend | None = None) -> Services:
    """``analysis_slots``, shared across investigations, limits analysis scripts running at once; the dispatcher
    passes its own ``local_backend``, which keeps track of the local runs it started."""
    backend = backend or SlurmBackend()
    local = local_backend or LocalBackend(profile.runs_root, host=current_host)
    registry = registry_for(profile)
    return Services(
        profile=profile,
        registry=registry,
        engine=ProtocolEngine(workspace.ledger, registry, workspace.artifacts, profile.runs_root),
        manager=JobManager(workspace.ledger, backend, profile.runs_root),
        monitor=JobMonitor(workspace.ledger, backend),
        validator=ResultValidator(workspace.ledger, registry, workspace.artifacts, backend, profile.runs_root,
                                  local_backend=local),
        analysis=AnalysisRunner(workspace.ledger, workspace.artifacts, profile.runs_root, workspace.root / "analyses",
                                slots=analysis_slots),
        deny=[Path(d) for d in deny],
        local_backend=local,
        local_manager=JobManager(workspace.ledger, local, profile.runs_root, execution="local"),
        local_monitor=JobMonitor(workspace.ledger, local, execution="local"),
    )


def investigation_registrar(workspace: Workspace, profiles: ProfileStore, investigation_id: str, *,
                            deny: list[Path] = ()) -> RunRegistrar:
    """Registers shared runs for the researcher and the Chief Scientist, under the investigation's compute profile if
    it has a usable one."""
    try:
        profile = profile_for(workspace.ledger.load(investigation_id), profiles)
    except (LookupError, ProfileNotApproved, FileNotFoundError):
        profile = None
    access = agent_access(deny)
    return RunRegistrar(workspace.ledger, workspace.artifacts, profile=profile,
                        readable=lambda path: access.resolve(str(path)))


def research_context(workspace: Workspace, investigation_id: str, agent: AgentRecord, services: Services | None, *,
                     deny: list[Path] = (), analysis_slots: threading.Semaphore | None = None,
                     workbench: Workbench | None = None,
                     should_stop: Callable[[], bool] | None = None) -> ResearchContext:
    """What one hypothesis agent's research tools work with. Without ``services`` (no usable compute profile) it
    requests no calculation, but still registers existing runs, reads and analyses them, and runs commands in the
    ``workbench``, which ``should_stop`` interrupts."""
    directory = workspace.agent_workspace(investigation_id, agent.hypothesis_id)
    profile = services.profile if services is not None else None
    files = agent_access(deny)
    if services is not None:
        registry, engine, validator, analysis = services.registry, services.engine, services.validator, \
            services.analysis
    else:
        registry, engine = ProtocolRegistry([]), None
        validator = ResultValidator(workspace.ledger, registry, workspace.artifacts)
        analysis = AnalysisRunner(workspace.ledger, workspace.artifacts, None, workspace.root / "analyses",
                                  slots=analysis_slots)
    registrar = RunRegistrar(workspace.ledger, workspace.artifacts, profile=profile,
                             readable=lambda path: files.resolve(str(path)))
    bench = None
    if workbench is not None:
        local = profile.local_execution if profile is not None else LocalExecution()
        binaries = [Path(code.binary) for code in profile.codes.values()] if profile is not None else []

        def view() -> SandboxView:
            # The sandbox makes its own /proc and /dev; the credentials, DFT codes and launchers are hidden.
            hidden = [p for p in files.deny if str(p) not in PSEUDO_FILESYSTEMS]
            hidden += [*binaries, *program_paths(), *map(Path, local.hidden_paths)]
            return SandboxView(workspace=directory, hidden=tuple(dict.fromkeys(hidden)), shown=())

        bench = AgentBench(workbench, investigation_id, agent.hypothesis_id, agent.agent_id, view, local,
                           code_paths=[Path(p) / "bin" for p in local.python_environments], should_stop=should_stop)
    return ResearchContext(ledger=workspace.ledger, investigation_id=investigation_id,
                           hypothesis_id=agent.hypothesis_id,
                           actor=Actor(kind=ActorKind.HYPOTHESIS_AGENT, id=agent.agent_id), registry=registry,
                           artifacts=workspace.artifacts, engine=engine, validator=validator, files=files,
                           analysis=analysis, workspace=directory, registrar=registrar, bench=bench,
                           runs_root=Path(profile.runs_root) if profile is not None else None,
                           workspaces=workspace.workspaces(investigation_id),
                           external=ExternalEvidenceRecorder(workspace.ledger, workspace.artifacts,
                                                             readable=lambda path: files.resolve(str(path))))


def profile_for(state: InvestigationState, profiles: ProfileStore) -> ComputeProfile:
    """The approved profile the investigation names; raises if it names none or it is not approved."""
    name = state.spec.constraints.compute_profile
    if name is None:
        raise LookupError(f"investigation {state.investigation_id} names no compute profile")
    return profiles.load_approved(name)


@dataclass
class ModelClients:
    """One client per provider, created on first use; settings per agent role."""

    config: ModelConfig
    credentials: CredentialStore
    _clients: dict[str, ModelClient] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def settings(self, role: str) -> ModelSettings:
        return self.config.for_role(role)

    def client(self, role: str) -> ModelClient:
        """One client per provider, shared by agents running at the same time."""
        settings = self.settings(role)
        with self._lock:
            if settings.provider not in self._clients:
                self._clients[settings.provider] = create_client(settings, self.credentials)
            return self._clients[settings.provider]

    def secrets(self) -> list[str]:
        return self.credentials.secrets()
