"""Command-line entry point (``cedar``).

Implemented: ``auth`` (API keys), ``models`` (which model each agent role uses), ``setup``
(the environment-setup agent), ``profile`` (review and approve compute profiles), ``start``
(talk with the Chief Scientist), ``investigations`` and ``dispatch`` (the process that
submits authorized work, follows jobs and runs agents between conversations).
"""

import argparse
import getpass
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from cedar import __version__
from cedar.paths import config_dir, data_dir


def _credentials():
    from cedar.models import CredentialStore

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
        print("No API keys configured. Add one with: cedar auth add <provider>")
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
    from cedar.models import write_starter

    path = _models_path()
    print(f"Wrote {path}; edit it to choose models." if write_starter(path) else f"{path} already exists.")
    return 0


def _models_show(args) -> int:
    from cedar.models import load_model_config

    from cedar.models.prices import prices_for

    config = load_model_config(_models_path())
    for role, settings in sorted(config.roles.items()):
        effort = f", effort {settings.effort}" if settings.effort else ""
        prices = prices_for(settings)
        priced = (f"; ${prices.input:g} input, ${prices.cache_read:g} cached input, ${prices.output:g} output per "
                  f"million tokens ({'as configured' if settings.prices else 'as published'})" if prices else
                  "; prices unknown: set [roles." + role + ".prices], or no budget can be kept")
        print(f"{role}: {settings.provider} {settings.model}{effort}{priced}")
    return 0


def _models_check(args) -> int:
    """One tiny real request, to confirm the key and model work. Uses a few tokens."""
    from cedar.models import Message, ModelError, TextPart, create_client, load_model_config

    settings = load_model_config(_models_path()).for_role(args.role)
    client = create_client(settings, _credentials())
    try:
        reply = client.complete(
            system="You are checking connectivity for CEDAR. Reply with the single word: ready",
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
    from cedar.hpc import ProfileStore

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


def _dollars_or_off(text: str) -> float | None:
    if text.lower() == "off":
        return None
    try:
        amount = float(text.lstrip("$"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an amount of dollars: {text}") from None
    if amount <= 0:
        raise argparse.ArgumentTypeError("give a positive amount, or 'off'")
    return amount


def _setup(args) -> int:
    from cedar.agents import AgentRunner, RunLimits, TranscriptStore
    from cedar.agents.prompts import ENVIRONMENT_SETUP
    from cedar.agents.toolkits import file_tools
    from cedar.agents.toolkits.files import agent_access
    from cedar.agents.toolkits.environment import SetupSession, environment_tools
    from cedar.hpc.queries import SchedulerQueries
    from cedar.models import create_client, load_model_config

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
    access = agent_access(deny)
    tools = file_tools(access, on_read=session.file_read) + environment_tools(
        _profiles(), SchedulerQueries(), session, _ask_in_terminal, deny=deny)
    if args.resume:
        instruction = ("The researcher is back at the terminal and can answer your questions now. Continue where "
                       "you stopped and propose the profile.")
    else:
        instruction = (
            f"Propose a compute profile named {args.name!r} for this system. You can read any file the researcher "
            f"can, except credentials; the researcher's home directory is {Path.home()}. The researcher is at the "
            "terminal to answer questions."
        )
    runner = AgentRunner(create_client(settings, credentials), settings, TranscriptStore(transcripts),
                         secrets=credentials.secrets())
    print(f"Environment-setup agent ({settings.provider} {settings.model}) is working; its steps are listed below.")
    result = runner.run(session_id, system=ENVIRONMENT_SETUP, tools=tools, instruction=instruction,
                        limits=RunLimits(max_cost_usd=args.max_cost), observer=_show_progress)
    session.save(transcripts / f"{session_id}.sources.json")
    print("\n" + (result.final_text or result.error or ""))
    usage = result.usage
    cost = f"${result.cost_usd:.2f}" if result.cost_usd is not None else "cost unknown"
    print(f"\n[{result.outcome}; {result.turns} turns; {cost}; served by {', '.join(result.served_models) or '-'}; "
          f"tokens: "
          f"{usage.input_tokens} input, {usage.cache_read_tokens} cached input read, {usage.cache_write_tokens} "
          f"written to cache, {usage.output_tokens} output]")
    config_dir().mkdir(mode=0o700, parents=True, exist_ok=True)
    with (config_dir() / "setup_runs.jsonl").open("a") as log:
        log.write(json.dumps({"session": session_id, "profile": args.name, "started": started.isoformat(),
                              "model": settings.model, "served_models": result.served_models,
                              "outcome": result.outcome, "turns": result.turns,
                              "usage": result.usage.model_dump()}) + "\n")
    if args.name in _profiles().names():
        print(f"Review it with `cedar profile show {args.name}`; approve it with "
              f"`cedar profile approve {args.name}`.")
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
        print("No compute profiles yet. Propose one with: cedar setup --name <name>")
    for name in names:
        print(f"{name}: {'approved' if store.is_approved(name) else 'not approved'}")
    return 0


def _profile_show(args) -> int:
    from cedar.hpc import check_profile

    store = _profiles()
    profile = store.load(args.name)
    print(json.dumps(profile.model_dump(mode="json"), indent=2))
    findings = check_profile(profile)
    print("\nChecks:" + ("".join(f"\n  [{f.severity.value}] {f.message}" for f in findings) or " none failed"))
    print(f"Status: {'approved' if store.is_approved(args.name) else 'not approved'}")
    return 0


def _profile_approve(args) -> int:
    from cedar.contracts import FindingSeverity
    from cedar.hpc import check_profile

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


def _model_clients():
    from cedar.models import load_model_config
    from cedar.workspace import ModelClients

    return ModelClients(load_model_config(_models_path()), _credentials())


def _workspace():
    from cedar.workspace import Workspace

    return Workspace.open(data_dir())


def _full_screen(args) -> bool:
    """Whether to use the full-screen interface: asked for, and a terminal that can show it."""
    return (not args.plain and sys.stdin.isatty() and sys.stdout.isatty()
            and os.environ.get("TERM", "") not in ("", "dumb"))


def _start(args) -> int:
    from cedar.chat import ChatSession
    from cedar.dispatch import heartbeat_age

    workspace = _workspace()
    if args.investigation and args.investigation not in workspace.store.investigation_ids():
        print(f"No investigation {args.investigation}. See `cedar investigations`.", file=sys.stderr)
        return 1
    session = ChatSession(workspace, _profiles(), _model_clients(), investigation_id=args.investigation,
                          auto_dispatch=not args.no_dispatch)
    started = session.ensure_dispatcher()
    if _full_screen(args):
        from cedar.tui import run_tui

        run_tui(session, started)
        if session.investigation_id:
            print(f"Resume with: cedar start {session.investigation_id}")
        return 0
    if started:
        print(started)
    age = heartbeat_age(workspace)
    if args.no_dispatch and (age is None or age > 600):
        print("Note: no dispatcher has run recently, so queued agent work and jobs will not progress. Start one in "
              "another terminal (or tmux/screen): cedar dispatch")
    if args.investigation:
        print(f"Investigation {args.investigation}. Type /help for commands.")
    else:
        print("New conversation with the Chief Scientist. Describe the simulation-experiment discrepancy you want to "
              "investigate. Type /help for commands.")
    while True:
        for notice in session.notices():
            print(f"\nChief Scientist (review of the agents' reports):\n{notice.text}" if notice.review else notice.text)
        try:
            line = input("\nyou> ")
        except EOFError:
            break
        if line.strip() in ("/quit", "/exit"):
            break
        reply = session.handle(line)
        if reply:
            print(f"\n{reply}")
    if session.investigation_id:
        print(f"Resume with: cedar start {session.investigation_id}")
    return 0


def _investigations(args) -> int:
    workspace = _workspace()
    ids = workspace.store.investigation_ids()
    if not ids:
        print("No investigations yet. Start one with: cedar start")
    for investigation_id in ids:
        state = workspace.ledger.load(investigation_id)
        print(f"{investigation_id}: {state.status.value}; {len(state.hypotheses)} hypotheses, "
              f"{len(state.calculations)} calculations; {state.spec.title}")
    return 0


def _report(args) -> int:
    from cedar.notebook import update_notebook

    workspace = _workspace()
    ids = workspace.store.investigation_ids()
    if args.investigation and args.investigation not in ids:
        print(f"No investigation {args.investigation}. See `cedar investigations`.", file=sys.stderr)
        return 1
    for investigation_id in [args.investigation] if args.investigation else ids:
        update = update_notebook(workspace, investigation_id)
        print(f"{investigation_id}: {update.index} ({update.summary})")
    return 0


def _make_dispatcher(max_agents: int = 4):
    from cedar.dispatch import Dispatcher

    return Dispatcher(_workspace(), _profiles(), _model_clients(), deny=[config_dir()], max_agents=max_agents)


def _dispatch(args) -> int:
    if args.stop:
        from cedar.dispatch import stop_dispatcher

        print(stop_dispatcher(_workspace()))
        return 0
    if args.agents < 1:
        print("--agents must be at least 1", file=sys.stderr)
        return 2
    dispatcher = _make_dispatcher(args.agents)
    if args.once:
        for note in dispatcher.tick():
            print(note)
        return 0
    idle = f"; stops after {args.until_idle:g} min with nothing to do" if args.until_idle is not None else ""
    print(f"Dispatcher running every {args.interval:g} s on {data_dir()}, up to {args.agents} agents at once{idle}; "
          "stop with Ctrl-C or `cedar dispatch --stop`.", flush=True)
    try:
        dispatcher.run_forever(args.interval, log=lambda note: print(note, flush=True),
                               idle_seconds=args.until_idle * 60 if args.until_idle is not None else None)
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cedar", description="CEDAR: Autonomous Scientific Agent Swarm for Computation-Experiment Discrepancies (pre-alpha).")
    parser.add_argument("--version", action="version", version=f"cedar {__version__}")
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
    setup.add_argument("--max-cost", type=_dollars_or_off, default=15.0, metavar="DOLLARS",
                       help="stop once the session has spent this much on the model; 'off' for no limit "
                            "(default: 15)")
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

    start = commands.add_parser("start", help="talk with the Chief Scientist (new or existing investigation)")
    start.add_argument("investigation", nargs="?", help="an existing investigation to continue")
    start.add_argument("--no-dispatch", action="store_true",
                       help="never start a dispatcher in the background; run `cedar dispatch` yourself")
    start.add_argument("--plain", action="store_true",
                       help="line-by-line chat instead of the full-screen interface (used anyway when the terminal "
                            "cannot show it)")
    start.set_defaults(handler=_start)
    commands.add_parser("investigations", help="list investigations").set_defaults(handler=_investigations)
    report = commands.add_parser("report", help="write the research notebook (Markdown) from the event log")
    report.add_argument("investigation", nargs="?", help="one investigation (default: all)")
    report.set_defaults(handler=_report)
    dispatch = commands.add_parser("dispatch", help="submit authorized work, follow jobs and run queued agents")
    dispatch.add_argument("--once", action="store_true", help="do one pass over all investigations and exit")
    dispatch.add_argument("--interval", type=float, default=60.0, help="seconds between passes (default 60)")
    dispatch.add_argument("--agents", type=int, default=4, metavar="N",
                          help="agents that may run at the same time (default 4)")
    dispatch.add_argument("--until-idle", type=float, metavar="MINUTES",
                          help="stop after this long with nothing to do (as when the chat starts it)")
    dispatch.add_argument("--stop", action="store_true",
                          help="stop the dispatcher running on this data directory, after its agent runs finish")
    dispatch.set_defaults(handler=_dispatch)

    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    return args.handler(args)
