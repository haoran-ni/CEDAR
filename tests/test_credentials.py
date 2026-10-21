import os
import stat
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from sim2real.hpc import SlurmBackend
from sim2real.models import CredentialStore, InsecureCredentials, MissingCredential
from sim2real.secrets import redact, scrubbed_environment

KEY = "sk-ant-test-0123456789abcdefghij"


@pytest.fixture
def store(tmp_path):
    return CredentialStore(tmp_path / "config" / "credentials.toml", environ={})


def test_keys_are_stored_privately_and_read_back(store):
    store.set("anthropic", KEY)
    assert store.get("anthropic") == KEY
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert store.status() == {"anthropic": "file"}
    assert not list(store.path.parent.glob(".*.tmp"))


def test_status_never_shows_the_key(store):
    store.set("anthropic", KEY)
    assert KEY not in repr(store.status())


def test_environment_variable_overrides_the_file(tmp_path):
    store = CredentialStore(tmp_path / "credentials.toml", environ={"ANTHROPIC_API_KEY": "from-env"})
    store.set("anthropic", KEY)
    assert store.get("anthropic") == "from-env"
    assert store.status() == {"anthropic": "environment"}


def test_missing_key_says_how_to_add_one(store):
    with pytest.raises(MissingCredential, match="sim2real auth add anthropic"):
        store.get("anthropic")


def test_files_readable_by_others_are_refused(store):
    store.set("anthropic", KEY)
    os.chmod(store.path, 0o644)
    with pytest.raises(InsecureCredentials, match="chmod 600"):
        store.get("anthropic")


def test_remove(store):
    store.set("anthropic", KEY)
    store.set("openai", "sk-" + "x" * 30)
    assert store.remove("anthropic") and not store.remove("anthropic")
    assert store.status() == {"openai": "file"}


@pytest.mark.parametrize("provider, key", [("Anthropic!", KEY), ("../x", KEY), ("anthropic", "two words"), ("anthropic", " ")])
def test_bad_names_and_keys_are_rejected(store, provider, key):
    with pytest.raises(ValueError):
        store.set(provider, key)


def test_quotes_in_keys_survive_the_file(store):
    store.set("local", 'abc"def\\ghi')
    assert store.get("local") == 'abc"def\\ghi'


def test_redaction_removes_known_and_key_shaped_secrets():
    text = f"failed with key {KEY} and sk-proj-{'a' * 30} and mine=s3cr3t-value"
    cleaned = redact(text, ["s3cr3t-value"])
    assert KEY not in cleaned and "sk-proj" not in cleaned and "s3cr3t-value" not in cleaned
    assert cleaned.count("[REDACTED]") == 3


def test_scrubbed_environment_drops_credentials_only():
    environ = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": KEY, "GITHUB_TOKEN_X": "kept", "MY_SERVICE_API_KEY": "k",
               "AWS_SECRET_ACCESS_KEY": "s", "VAULT_AUTH_TOKEN": "t", "SLURM_CONF": "/etc/slurm.conf"}
    assert scrubbed_environment(environ) == {"PATH": "/usr/bin", "GITHUB_TOKEN_X": "kept",
                                            "SLURM_CONF": "/etc/slurm.conf"}


def test_slurm_commands_run_without_credentials():
    seen = {}

    def runner(argv, **kwargs):
        seen.update(kwargs["env"])
        return CompletedProcess(argv, 0, stdout="12345\n", stderr="")

    backend = SlurmBackend(user="someone", runner=runner, environ={"PATH": "/bin", "ANTHROPIC_API_KEY": KEY})
    assert backend.submit(Path("/tmp/job.sh"), Path("/tmp"), "s2r-x") == "12345"
    assert seen == {"PATH": "/bin"}
