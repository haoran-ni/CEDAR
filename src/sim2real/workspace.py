"""Wiring: the research data directory, and the services an investigation's profile implies."""

from dataclasses import dataclass, field
from pathlib import Path

from sim2real.agents.toolkits.files import FileAccess
from sim2real.agents.transcripts import TranscriptStore
from sim2real.engine import JobManager, JobMonitor, ProtocolEngine, ResultValidator
from sim2real.hpc.profile import ComputeProfile, ProfileStore
from sim2real.hpc.slurm import SchedulerBackend, SlurmBackend
from sim2real.models import CredentialStore, ModelClient, ModelConfig, ModelSettings, create_client
from sim2real.protocols import FhiAimsTotalEnergy, ProtocolRegistry
from sim2real.state.ledger import ResearchLedger
from sim2real.state.projection import InvestigationState
from sim2real.store import ArtifactStore, SQLiteEventStore


@dataclass
class Workspace:
    """Everything stored under the research data directory."""

    root: Path
    store: SQLiteEventStore
    ledger: ResearchLedger
    artifacts: ArtifactStore
    transcripts: TranscriptStore

    @classmethod
    def open(cls, root: str | Path) -> "Workspace":
        root = Path(root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        store = SQLiteEventStore(root / "events.sqlite")
        return cls(root, store, ResearchLedger(store), ArtifactStore(root / "artifacts"),
                   TranscriptStore(root / "transcripts"))

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
    files: FileAccess


def registry_for(profile: ComputeProfile) -> ProtocolRegistry:
    protocols = [FhiAimsTotalEnergy(profile)] if "fhi-aims" in profile.codes else []
    return ProtocolRegistry(protocols)


def services_for(workspace: Workspace, profile: ComputeProfile, backend: SchedulerBackend | None = None,
                 *, deny: list[Path] = ()) -> Services:
    backend = backend or SlurmBackend()
    registry = registry_for(profile)
    readable = [profile.runs_root, *(p for code in profile.codes.values() for p in code.data_paths.values())]
    return Services(
        profile=profile,
        registry=registry,
        engine=ProtocolEngine(workspace.ledger, registry, workspace.artifacts, profile.runs_root),
        manager=JobManager(workspace.ledger, backend, profile.runs_root),
        monitor=JobMonitor(workspace.ledger, backend),
        validator=ResultValidator(workspace.ledger, registry, workspace.artifacts, backend, profile.runs_root),
        files=FileAccess(roots=readable, deny=list(deny)),
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

    def settings(self, role: str) -> ModelSettings:
        return self.config.for_role(role)

    def client(self, role: str) -> ModelClient:
        settings = self.settings(role)
        if settings.provider not in self._clients:
            self._clients[settings.provider] = create_client(settings, self.credentials)
        return self._clients[settings.provider]

    def secrets(self) -> list[str]:
        return self.credentials.secrets()
