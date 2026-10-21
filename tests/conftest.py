import pytest
from sites import build_site

import sim2real.chat

from sim2real.state import ResearchLedger
from sim2real.store import ArtifactStore, SQLiteEventStore


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No test sees the researcher's configuration, data or API keys, or starts a process that outlives it.

    A background dispatcher would run with the real configuration and keys of whoever runs the tests, so
    starting one fails here; a test that needs one says so and gives it an isolated configuration.
    """
    monkeypatch.setenv("SIM2REAL_CONFIG_DIR", str(tmp_path / "isolated-config"))
    monkeypatch.setenv("SIM2REAL_DATA_DIR", str(tmp_path / "isolated-data"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def refuse(workspace):
        raise AssertionError(f"a test tried to start a background dispatcher on {workspace.root}")

    monkeypatch.setattr(sim2real.chat, "start_background_dispatcher", refuse)


@pytest.fixture
def store(tmp_path):
    return SQLiteEventStore(tmp_path / "events.sqlite")


@pytest.fixture
def ledger(store):
    return ResearchLedger(store)


@pytest.fixture
def artifacts(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def site(tmp_path):
    return build_site(tmp_path / "site")
