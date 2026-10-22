import stat

import pytest

from cedar import __version__, cli
from cedar.models.scripted import ScriptedModel, reply
from cedar.models.types import TextPart

KEY = "sk-ant-test-0123456789abcdefghij"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CEDAR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path / "config"


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"cedar {__version__}"


def test_auth_add_status_remove(monkeypatch, capsys, isolated_config):
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: KEY)
    assert cli.main(["auth", "add", "anthropic"]) == 0
    path = isolated_config / "credentials.toml"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert cli.main(["auth", "status"]) == 0
    out = capsys.readouterr().out
    assert "anthropic: key from file" in out and KEY not in out
    assert cli.main(["auth", "remove", "anthropic"]) == 0
    cli.main(["auth", "status"])
    assert "No API keys configured" in capsys.readouterr().out


def test_models_init_show(capsys, isolated_config):
    assert cli.main(["models", "init"]) == 0
    assert cli.main(["models", "init"]) == 0
    cli.main(["models", "show"])
    out = capsys.readouterr().out
    assert "already exists" in out and "default: anthropic claude-opus-5-5, effort high" in out


def test_models_check_reports_the_serving_model(monkeypatch, capsys):
    cli.main(["models", "init"])
    model = ScriptedModel([reply(TextPart(text="ready"), model="claude-opus-5-5")])
    monkeypatch.setattr("cedar.models.create_client", lambda settings, credentials: model)
    assert cli.main(["models", "check"]) == 0
    assert "answered with claude-opus-5-5" in capsys.readouterr().out
    assert model.calls[0].settings.max_output_tokens == 1024


def test_setup_before_models_init_names_the_missing_step(capsys, isolated_config):
    assert cli.main(["setup", "--name", "mycluster"]) == 1
    err = capsys.readouterr().err
    assert "cedar models init" in err and str(isolated_config / "models.toml") in err


def test_models_check_before_auth_add_names_the_missing_step(capsys):
    cli.main(["models", "init"])
    assert cli.main(["models", "check"]) == 1
    assert "cedar auth add anthropic" in capsys.readouterr().err


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "auth" in capsys.readouterr().out
