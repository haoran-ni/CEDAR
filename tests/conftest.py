import pytest

from sim2real.state import ResearchLedger
from sim2real.store import ArtifactStore, SQLiteEventStore


@pytest.fixture
def store(tmp_path):
    return SQLiteEventStore(tmp_path / "events.sqlite")


@pytest.fixture
def ledger(store):
    return ResearchLedger(store)


@pytest.fixture
def artifacts(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")
