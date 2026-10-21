"""Wiring: the research data directory, and the services an investigation's profile implies."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from cedar.agents.toolkits.research import agent_file_access
from cedar.agents.toolkits.files import FileAccess
from cedar.agents.transcripts import TranscriptStore
from cedar.engine import JobManager, JobMonitor, ProtocolEngine, ResultValidator
from cedar.engine.analysis import AnalysisRunner
from cedar.hpc.profile import ComputeProfile, ProfileStore
from cedar.hpc.slurm import SchedulerBackend, SlurmBackend
from cedar.models import CredentialStore, ModelClient, ModelConfig, ModelSettings, create_client
from cedar.protocols import FhiAimsRun, ProtocolRegistry
from cedar.state.ledger import ResearchLedger
from cedar.state.projection import InvestigationState
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
    shared_roots: list[Path]
    """Directories every agent may read, such as the codes' species defaults."""
    deny: list[Path]

    def agent_files(self, workspace: Workspace, investigation_id: str, hypothesis_id: str) -> FileAccess:
        """What one hypothesis's agent may read: its own calculations' run directories and the shared roots."""
        return agent_file_access(workspace.ledger, investigation_id, hypothesis_id, self.profile.runs_root,
                                 self.shared_roots, self.deny)


def registry_for(profile: ComputeProfile) -> ProtocolRegistry:
    protocols = [FhiAimsRun(profile)] if "fhi-aims" in profile.codes else []
    return ProtocolRegistry(protocols)


def services_for(workspace: Workspace, profile: ComputeProfile, backend: SchedulerBackend | None = None,
                 *, deny: list[Path] = (), analysis_slots: threading.Semaphore | None = None) -> Services:
    """``analysis_slots``, shared across investigations, limits analysis scripts running at once."""
    backend = backend or SlurmBackend()
    registry = registry_for(profile)
    return Services(
        profile=profile,
        registry=registry,
        engine=ProtocolEngine(workspace.ledger, registry, workspace.artifacts, profile.runs_root),
        manager=JobManager(workspace.ledger, backend, profile.runs_root),
        monitor=JobMonitor(workspace.ledger, backend),
        validator=ResultValidator(workspace.ledger, registry, workspace.artifacts, backend, profile.runs_root),
        analysis=AnalysisRunner(workspace.ledger, workspace.artifacts, profile.runs_root, workspace.root / "analyses",
                                slots=analysis_slots),
        shared_roots=[Path(p) for code in profile.codes.values() for p in code.data_paths.values()],
        deny=[Path(d) for d in deny],
    )


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
