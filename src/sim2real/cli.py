"""Command-line entry point (``sim2real``).

Implemented: ``auth`` (API keys) and ``models`` (which model each agent role uses).
Investigation commands such as ``start`` are added when the features behind them work.
"""

import argparse
import getpass
import sys
from pathlib import Path

from sim2real import __version__
from sim2real.paths import config_dir


def _credentials():
    from sim2real.models import CredentialStore

    return CredentialStore(config_dir() / "credentials.toml")


def _auth_add(args) -> int:
    key = getpass.getpass(f"API key for {args.provider} (input hidden): ")
    store = _credentials()
    store.set(args.provider, key)
    print(f"Stored the {args.provider} key in {store.path} (readable only by you).")
    return 0


def _auth_status(args) -> int:
    sources = _credentials().status()
    if not sources:
        print("No API keys configured. Add one with: sim2real auth add <provider>")
    for provider, source in sources.items():
        print(f"{provider}: key from {source}")
    return 0


def _auth_remove(args) -> int:
    removed = _credentials().remove(args.provider)
    print(f"Removed the {args.provider} key." if removed else f"No stored key for {args.provider}.")
    return 0


def _models_path() -> Path:
    return config_dir() / "models.toml"


def _models_init(args) -> int:
    from sim2real.models import write_starter

    path = _models_path()
    print(f"Wrote {path}; edit it to choose models." if write_starter(path) else f"{path} already exists.")
    return 0


def _models_show(args) -> int:
    from sim2real.models import load_model_config

    config = load_model_config(_models_path())
    for role, settings in sorted(config.roles.items()):
        effort = f", effort {settings.effort}" if settings.effort else ""
        print(f"{role}: {settings.provider} {settings.model}{effort}")
    return 0


def _models_check(args) -> int:
    """One tiny real request, to confirm the key and model work. Uses a few tokens."""
    from sim2real.models import Message, ModelError, TextPart, create_client, load_model_config

    settings = load_model_config(_models_path()).for_role(args.role)
    client = create_client(settings, _credentials())
    try:
        reply = client.complete(
            system="You are checking connectivity for sim2real. Reply with the single word: ready",
            messages=[Message(role="user", parts=(TextPart(text="Are you reachable?"),))],
            tools=[], settings=settings.model_copy(update={"max_output_tokens": 1024}),
        )
    except ModelError as exc:
        print(f"Model check failed: {exc}", file=sys.stderr)
        return 1
    print(f"{settings.provider} answered with {reply.served_model} (stop: {reply.stop}; "
          f"{reply.usage.input_tokens} input / {reply.usage.output_tokens} output tokens): {reply.message.text!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sim2real", description="Sim2Real Scientist (pre-alpha).")
    parser.add_argument("--version", action="version", version=f"sim2real {__version__}")
    commands = parser.add_subparsers(dest="command")

    auth = commands.add_parser("auth", help="manage API keys").add_subparsers(dest="action", required=True)
    add = auth.add_parser("add", help="store an API key (asks for it without echoing)")
    add.add_argument("provider", help="e.g. anthropic")
    add.set_defaults(handler=_auth_add)
    auth.add_parser("status", help="show which providers have keys, and where from").set_defaults(handler=_auth_status)
    remove = auth.add_parser("remove", help="delete a stored API key")
    remove.add_argument("provider")
    remove.set_defaults(handler=_auth_remove)

    models = commands.add_parser("models", help="choose models for agent roles").add_subparsers(
        dest="action", required=True)
    models.add_parser("init", help="write a starter models.toml").set_defaults(handler=_models_init)
    models.add_parser("show", help="list the configured roles").set_defaults(handler=_models_show)
    check = models.add_parser("check", help="send one tiny request to confirm a role's model works")
    check.add_argument("--role", default="default")
    check.set_defaults(handler=_models_check)

    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    return args.handler(args)
