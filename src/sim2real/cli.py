"""Command-line entry point (``sim2real``).

Implemented: ``auth`` (API keys), ``models`` (which model each agent role uses), ``setup``
(the environment-setup agent) and ``profile`` (review and approve compute profiles).
Investigation commands such as ``start`` are added when the features behind them work.
"""

import argparse
import getpass
import json
import sys
from datetime import datetime, timezone
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


def _profiles():
    from sim2real.hpc import ProfileStore

    return ProfileStore(config_dir() / "profiles")


def _ask_in_terminal(question: str) -> str:
    print(f"\n? {question}")
    if not sys.stdin.isatty():
        print("  (no terminal attached; answering nothing)")
        return ""
    return input("> ")


def _show_progress(call, outcome) -> None:
    detail = next((str(v) for k, v in call.arguments.items() if k in ("path", "root", "query", "name")), "")
    marker = "!" if outcome.is_error else "-"
    print(f"  {marker} {call.name} {detail}".rstrip())
    if outcome.is_error:
        print("    " + outcome.content.splitlines()[0][:200])


def _setup(args) -> int:
    from sim2real.agents import AgentRunner, RunLimits, TranscriptStore
    from sim2real.agents.prompts import ENVIRONMENT_SETUP
    from sim2real.agents.toolkits import FileAccess, file_tools
    from sim2real.agents.toolkits.environment import SetupSession, environment_tools
    from sim2real.hpc.queries import SchedulerQueries
    from sim2real.models import create_client, load_model_config

    roots = [Path.home(), *[Path(r).resolve() for r in args.root]]
    deny = [config_dir()]
    settings = load_model_config(_models_path()).for_role("environment_setup")
    credentials = _credentials()
    transcripts = config_dir() / "transcripts"
    started = datetime.now(timezone.utc)
    if args.resume:
        session_id = _last_setup_session(args.name)
        if session_id is None:
            print(f"No earlier setup session for {args.name}; start one without --resume.", file=sys.stderr)
            return 1
    else:
        session_id = f"setup-{args.name}-{started:%Y%m%d%H%M%S}"
    session = SetupSession.load(transcripts / f"{session_id}.sources.json")
    access = FileAccess(roots=roots, deny=deny)
    tools = file_tools(access, on_read=session.file_read) + environment_tools(
        _profiles(), SchedulerQueries(), session, _ask_in_terminal, deny=deny)
    if args.resume:
        instruction = ("The researcher is back at the terminal and can answer your questions now. Continue where "
                       "you stopped and propose the profile.")
    else:
        instruction = (
            f"Propose a compute profile named {args.name!r} for this system. You can read files under: "
            + ", ".join(str(r) for r in roots)
            + ". You can inspect any other path without reading it. The researcher is at the terminal to answer "
              "questions."
        )
    runner = AgentRunner(create_client(settings, credentials), settings, TranscriptStore(transcripts),
                         secrets=credentials.secrets())
    print(f"Environment-setup agent ({settings.provider} {settings.model}) is working; its steps are listed below.")
    result = runner.run(session_id, system=ENVIRONMENT_SETUP, tools=tools, instruction=instruction,
                        limits=RunLimits(max_turns=args.max_turns, max_tokens=args.max_tokens), observer=_show_progress)
    session.save(transcripts / f"{session_id}.sources.json")
    print("\n" + (result.final_text or result.error or ""))
    usage = result.usage
    print(f"\n[{result.outcome}; {result.turns} turns; served by {', '.join(result.served_models) or '-'}; tokens: "
          f"{usage.input_tokens} input, {usage.cache_read_tokens} cached input read, {usage.cache_write_tokens} "
          f"written to cache, {usage.output_tokens} output]")
    config_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
    with (config_dir() / "setup_runs.jsonl").open("a") as log:
        log.write(json.dumps({"session": session_id, "profile": args.name, "started": started.isoformat(),
                              "model": settings.model, "served_models": result.served_models,
                              "outcome": result.outcome, "turns": result.turns,
                              "usage": result.usage.model_dump()}) + "\n")
    if args.name in _profiles().names():
        print(f"Review it with `sim2real profile show {args.name}`; approve it with "
              f"`sim2real profile approve {args.name}`.")
        return 0
    return 1


def _last_setup_session(name: str) -> str | None:
    log = config_dir() / "setup_runs.jsonl"
    if not log.exists():
        return None
    sessions = [json.loads(line)["session"] for line in log.read_text().splitlines()
                if line.strip() and json.loads(line).get("profile") == name]
    return sessions[-1] if sessions else None


def _profile_list(args) -> int:
    store = _profiles()
    names = store.names()
    if not names:
        print("No compute profiles yet. Propose one with: sim2real setup --name <name>")
    for name in names:
        print(f"{name}: {'approved' if store.is_approved(name) else 'not approved'}")
    return 0


def _profile_show(args) -> int:
    from sim2real.hpc import check_profile

    store = _profiles()
    profile = store.load(args.name)
    print(json.dumps(profile.model_dump(mode="json"), indent=2))
    findings = check_profile(profile)
    print("\nChecks:" + ("".join(f"\n  [{f.severity.value}] {f.message}" for f in findings) or " none failed"))
    print(f"Status: {'approved' if store.is_approved(args.name) else 'not approved'}")
    return 0


def _profile_approve(args) -> int:
    from sim2real.contracts import FindingSeverity
    from sim2real.hpc import check_profile

    store = _profiles()
    profile = store.load(args.name)
    errors = [f for f in check_profile(profile) if f.severity is FindingSeverity.ERROR]
    if errors:
        print("The profile fails its checks and cannot be approved:")
        for f in errors:
            print(f"  {f.message}")
        return 1
    print(profile.job_template)
    print(f"Partitions: {', '.join(profile.limits.partitions)}; runs under {profile.runs_root}")
    for name, code in profile.codes.items():
        print(f"{name}: {code.binary} (version {code.version})")
    answer = input(f"Approve profile {args.name}? Jobs will be submitted as shown. [y/N] ")
    if answer.strip().lower() not in ("y", "yes"):
        print("Not approved.")
        return 1
    store.approve(args.name, getpass.getuser(), datetime.now(timezone.utc))
    print(f"Approved {args.name}.")
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

    setup = commands.add_parser("setup", help="let the environment-setup agent propose a compute profile")
    setup.add_argument("--name", required=True, help="profile name, e.g. mycluster-cpu")
    setup.add_argument("--root", action="append", default=[], metavar="PATH",
                       help="another directory the agent may read (your home directory is always included)")
    setup.add_argument("--max-turns", type=int, default=60)
    setup.add_argument("--max-tokens", type=int, default=2_000_000,
                       help="stop after processing this many tokens, cached input included")
    setup.add_argument("--resume", action="store_true", help="continue the last setup session for this name")
    setup.set_defaults(handler=_setup)

    profile = commands.add_parser("profile", help="review and approve compute profiles").add_subparsers(
        dest="action", required=True)
    profile.add_parser("list", help="list profiles and whether each is approved").set_defaults(handler=_profile_list)
    show = profile.add_parser("show", help="print a profile, its checks and its status")
    show.add_argument("name")
    show.set_defaults(handler=_profile_show)
    approve = profile.add_parser("approve", help="approve a profile after reviewing it")
    approve.add_argument("name")
    approve.set_defaults(handler=_profile_approve)

    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    return args.handler(args)
