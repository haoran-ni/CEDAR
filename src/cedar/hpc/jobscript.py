"""Job scripts: rendering from the approved profile's template, and examination of any script.

The examiner knows Slurm's directive syntax and nothing about any site. A script's
commands are accepted only if each matches a command line of the approved profile's job
template (after shell variables are expanded on both sides), with the template's
placeholders standing for numbers or the configured values. Exactly one command must be
the template's launch of the configured binary, or of one of the code's other builds. Anything
else needs user review. A code with its own template in the profile is examined against that
one.

A local script, for a run on the interactive node CEDAR runs on, is the same template
without its ``#SBATCH`` lines, and one ``#CEDAR --ntasks=N --time=T`` line that states what
it may use; its commands are examined the same way.
"""

import re
import shlex
from dataclasses import dataclass

from cedar.contracts.examination import Finding, FindingSeverity
from cedar.hpc.profile import CodeInstallation, ComputeProfile, fill_template, template_fields

_E, _R, _W = FindingSeverity.ERROR, FindingSeverity.REVIEW, FindingSeverity.WARNING

_SLURM_SHORT = {"p": "partition", "N": "nodes", "n": "ntasks", "t": "time", "J": "job-name",
                "o": "output", "e": "error", "c": "cpus-per-task"}
_SLURM_RESOURCE_OPTIONS = {"partition", "nodes", "ntasks", "ntasks-per-node", "time", "job-name", "output", "error"}
_PLAIN_FILE = re.compile(r"^[A-Za-z0-9_.%-]+$")
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_ASSIGNMENT = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=("?)((?:[A-Za-z0-9_./:+@,-]+|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?)*)\2$')
_PLACEHOLDER_PATTERNS = {"ntasks": r"(?P<ntasks>\d+)", "nodes": r"\d+", "walltime": r"\S+", "partition": r"\S+",
                         "job_name": r"\S+"}


@dataclass(frozen=True)
class JobResources:
    partition: str
    nodes: int
    ntasks: int
    walltime_minutes: int

    @property
    def core_hours(self) -> float:
        return self.ntasks * self.walltime_minutes / 60


def format_walltime(minutes: int) -> str:
    days, rest = divmod(minutes, 24 * 60)
    hours, mins = divmod(rest, 60)
    return f"{days}-{hours:02d}:{mins:02d}:00" if days else f"{hours:02d}:{mins:02d}:00"


def parse_walltime(text: str) -> int | None:
    """Slurm time formats, rounded up to whole minutes."""
    match = re.fullmatch(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?", text)
    if not match:
        return None
    days, first, second, third = match.groups()
    if days is not None:  # D-HH[:MM[:SS]]
        total, secs = int(days) * 1440 + int(first) * 60 + int(second or 0), int(third or 0)
    elif third is not None:  # HH:MM:SS
        total, secs = int(first) * 60 + int(second), int(third)
    elif second is not None:  # MM:SS
        total, secs = int(first), int(second)
    else:  # MM
        total, secs = int(first), 0
    return total + (1 if secs else 0)


LOCAL_DIRECTIVE = "#CEDAR"
LOCAL_PARTITION = "local"
"""The partition name a local run's resources carry."""


def render_local_script(profile: ComputeProfile, code: CodeInstallation, *, ntasks: int, walltime_minutes: int,
                        output_file: str, binary: str | None = None) -> str:
    """The approved job template without its Slurm directives, for a run on this interactive node; ``binary`` is
    one of the code's builds, ``code.binary`` by default."""
    text = fill_template(profile.template_for(code), {
        "job_name": "cedar", "partition": LOCAL_PARTITION, "nodes": 1, "ntasks": ntasks,
        "walltime": format_walltime(walltime_minutes), "binary": binary or code.binary, "output": output_file,
    })
    lines = [line for line in text.splitlines() if not line.strip().startswith("#SBATCH")]
    lines.insert(1, f"{LOCAL_DIRECTIVE} --ntasks={ntasks} --time={format_walltime(walltime_minutes)}")
    return "\n".join(lines) + "\n"


def render_job_script(profile: ComputeProfile, code: CodeInstallation, resources: JobResources,
                      *, job_name: str, output_file: str, binary: str | None = None) -> str:
    text = fill_template(profile.template_for(code), {
        "job_name": job_name, "partition": resources.partition, "nodes": resources.nodes,
        "ntasks": resources.ntasks, "walltime": format_walltime(resources.walltime_minutes),
        "binary": binary or code.binary, "output": output_file,
    })
    return text if text.endswith("\n") else text + "\n"


@dataclass(frozen=True)
class ScriptExamination:
    findings: tuple[Finding, ...]
    resources: JobResources | None
    binary: str | None = None
    """The build the script launches, if it launches one as the template does."""


def _expand(value: str, variables: dict[str, str]) -> str | None:
    missing = False

    def substitute(match: re.Match) -> str:
        nonlocal missing
        name = match.group(1) or match.group(2)
        if name not in variables:
            missing = True
            return ""
        return variables[name]

    expanded = _VARIABLE.sub(substitute, value)
    return None if missing else expanded


@dataclass(frozen=True)
class _Commands:
    commands: list[tuple[int, str]]
    """(line, command with variables expanded)"""
    directives: list[tuple[int, str]]
    unresolved: list[tuple[int, str]]


def _split(text: str) -> _Commands:
    variables: dict[str, str] = {}
    commands, directives, unresolved = [], [], []
    seen_command = False
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if number == 1 and line.startswith("#!"):
            continue
        if line.startswith("#SBATCH"):
            directives.append((number, line if not seen_command else ""))
            continue
        if not line or line.startswith("#"):
            continue
        seen_command = True
        if m := _ASSIGNMENT.match(line):
            value = _expand(m.group(3), variables)
            if value is None:
                unresolved.append((number, line))
            else:
                variables[m.group(1)] = value
            continue
        expanded = _expand(line, variables)
        if expanded is None:
            unresolved.append((number, line))
        else:
            commands.append((number, " ".join(expanded.split())))
    return _Commands(commands, directives, unresolved)


def _template_patterns(profile: ComputeProfile, code: CodeInstallation, output_file: str) -> list[tuple[re.Pattern, bool]]:
    """Each template command as a regex; the flag marks the launch line, whose ``binary`` group is the build."""
    template = profile.template_for(code)
    marked = fill_template(template, {name: f"\x00{name}\x00" for name in template_fields(template)})
    patterns = []
    for _, command in _split(marked).commands:
        pieces = re.split(r"\x00(\w+)\x00", command)
        regex = ""
        for i, piece in enumerate(pieces):
            if i % 2 == 0:
                regex += re.escape(piece)
            elif piece == "binary":
                regex += "(?P<binary>" + "|".join(map(re.escape, code.binaries)) + ")"
            elif piece == "output":
                regex += re.escape(output_file)
            else:
                regex += _PLACEHOLDER_PATTERNS.get(piece, r"\S+")
        patterns.append((re.compile(regex), "\x00binary\x00" in command))
    return patterns


def _directive_options(text: str) -> list[tuple[str, str | None]] | None:
    try:
        tokens = shlex.split(text)
    except ValueError:
        return None
    options: list[tuple[str, str | None]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if token.startswith("--"):
            name, sep, value = token[2:].partition("=")
            if not sep and nxt is not None and not nxt.startswith("-"):
                value, sep, i = nxt, "=", i + 1
            options.append((name, value if sep else None))
        elif token.startswith("-") and len(token) >= 2:
            name = _SLURM_SHORT.get(token[1], token[1:])
            value = token[2:] or None
            if value is None and nxt is not None and not nxt.startswith("-"):
                value, i = nxt, i + 1
            options.append((name, value))
        else:
            options.append((token, None))
        i += 1
    return options


def _template_directive_names(template: str) -> set[str]:
    names = set()
    for _, line in _split(template).directives:
        for name, _ in _directive_options(line[len("#SBATCH"):]) or []:
            names.add(name)
    return names


def examine_job_script(text: str, profile: ComputeProfile, code: CodeInstallation, *,
                       file: str = "job.sh", output_file: str) -> ScriptExamination:
    findings: list[Finding] = []

    def add(severity: FindingSeverity, code_: str, message: str, line: int | None = None) -> None:
        findings.append(Finding(severity=severity, code=code_, message=message, file=file, line=line))

    if profile.scheduler != "slurm":
        add(_R, "no_examiner", f"no job-script examiner for scheduler {profile.scheduler}")
        return ScriptExamination(tuple(findings), None)
    lines = text.splitlines()
    _examine_shell(lines, profile.template_for(code), add)

    script = _split(text)
    for number, line in script.unresolved:
        add(_R, "unresolved_line", f"cannot resolve shell variables in: {line}", number)

    allowed_names = _SLURM_RESOURCE_OPTIONS | _template_directive_names(profile.template_for(code))
    directives: dict[str, str | None] = {}
    for number, line in script.directives:
        if not line:
            add(_W, "late_directive", "Slurm ignores #SBATCH lines after the first command", number)
            continue
        options = _directive_options(line[len("#SBATCH"):])
        if options is None:
            add(_R, "unparsable_directive", f"cannot parse: {line}", number)
            continue
        for name, value in options:
            if name not in allowed_names:
                add(_R, "unapproved_directive", f"--{name} is not used by the approved job template", number)
            directives[name] = value

    launches = _examine_commands(script, profile, code, output_file, add)
    resources = _resources(directives, profile, add)
    if resources is not None:
        for number, ntasks, _ in launches:
            if ntasks is not None and ntasks > resources.ntasks:
                add(_E, "oversubscribed", f"launches {ntasks} tasks but requests {resources.ntasks}", number)
    for name in ("output", "error"):
        value = directives.get(name)
        if value is not None and not _PLAIN_FILE.match(value):
            add(_R, "output_path", f"--{name}={value} writes outside the run directory")
    return ScriptExamination(tuple(findings), resources, launches[0][2] if launches else None)


def _examine_shell(lines: list[str], template: str, add) -> None:
    if not lines or not lines[0].startswith("#!"):
        add(_E, "missing_shebang", "the first line must be a shebang such as #!/bin/bash", 1)
    if lines and template.splitlines()[0].strip() != lines[0].strip():
        add(_R, "different_shell", f"uses {lines[0].strip()}, not the profile's {template.splitlines()[0]}", 1)


def _examine_commands(script: _Commands, profile: ComputeProfile, code: CodeInstallation, output_file: str,
                      add) -> list[tuple[int, int | None, str]]:
    """Every command must be one of the template's; returns the launches, with the task count each starts and the
    build it launches."""
    patterns = _template_patterns(profile, code, output_file)
    launches: list[tuple[int, int | None, str]] = []
    for number, command in script.commands:
        for pattern, is_launch in patterns:
            if m := pattern.fullmatch(command):
                if is_launch:
                    ntasks = m.groupdict().get("ntasks")
                    launches.append((number, int(ntasks) if ntasks else None, m.group("binary")))
                break
        else:
            add(_R, "unapproved_command", f"not a command of the approved job template: {command}", number)
    if not launches:
        add(_E, "no_launch", f"the script never launches {code.binary} writing {output_file} as the template does")
    elif len(launches) > 1:
        add(_R, "several_launches", "the script launches the code more than once", launches[1][0])
    return launches


def local_resources(text: str) -> JobResources | None:
    """What a local script's ``#CEDAR`` line says it may use; None if it has no readable one."""
    for line in text.splitlines():
        if not line.strip().startswith(LOCAL_DIRECTIVE):
            continue
        options = dict(_directive_options(line.strip()[len(LOCAL_DIRECTIVE):]) or [])
        ntasks, time = options.get("ntasks"), options.get("time")
        minutes = parse_walltime(time) if time else None
        if ntasks is None or not ntasks.isdigit() or int(ntasks) < 1 or minutes is None or minutes < 1:
            return None
        return JobResources(LOCAL_PARTITION, 1, int(ntasks), minutes)
    return None


def examine_local_script(text: str, profile: ComputeProfile, code: CodeInstallation, *, file: str = "local.sh",
                         output_file: str) -> ScriptExamination:
    """A script for a run on this interactive node: the template's commands, no Slurm directives, one ``#CEDAR``
    line stating its task count and time limit."""
    findings: list[Finding] = []

    def add(severity: FindingSeverity, code_: str, message: str, line: int | None = None) -> None:
        findings.append(Finding(severity=severity, code=code_, message=message, file=file, line=line))

    lines = text.splitlines()
    _examine_shell(lines, profile.template_for(code), add)
    script = _split(text)
    for number, line in script.unresolved:
        add(_R, "unresolved_line", f"cannot resolve shell variables in: {line}", number)
    for number, _ in script.directives:
        add(_E, "slurm_directive", "a local run is not a Slurm job; it takes no #SBATCH lines", number)
    if sum(1 for line in lines if line.strip().startswith(LOCAL_DIRECTIVE)) > 1:
        add(_E, "several_directives", f"give one {LOCAL_DIRECTIVE} line")
    resources = local_resources(text)
    if resources is None:
        add(_E, "no_local_resources", f"the script must state its task count and time limit in a line such as "
                                      f"'{LOCAL_DIRECTIVE} --ntasks=8 --time=01:00:00'")
    launches = _examine_commands(script, profile, code, output_file, add)
    if resources is not None:
        for number, ntasks, _ in launches:
            if ntasks is not None and ntasks > resources.ntasks:
                add(_E, "oversubscribed", f"launches {ntasks} tasks but states {resources.ntasks}", number)
    return ScriptExamination(tuple(findings), resources, launches[0][2] if launches else None)


def _resources(directives: dict[str, str | None], profile: ComputeProfile, add) -> JobResources | None:
    limits = profile.limits
    ok = True
    partition = directives.get("partition")
    if partition is None:
        add(_E, "no_partition", "the script must name a partition")
        ok = False
    elif partition not in limits.partitions:
        add(_E, "partition_not_allowed", f"partition {partition} is not one of {list(limits.partitions)}")
        ok = False

    def integer(name: str) -> int | None:
        value = directives.get(name)
        if value is None:
            return None
        if not value.isdigit() or int(value) < 1:
            add(_E, "bad_count", f"--{name}={value} is not a positive integer")
            return None
        return int(value)

    nodes = integer("nodes") or 1
    ntasks = integer("ntasks")
    per_node = integer("ntasks-per-node")
    if ntasks is None and per_node is not None:
        ntasks = per_node * nodes
    if ntasks is None:
        add(_E, "no_task_count", "the script must request --ntasks or --ntasks-per-node")
        ok = False
    elif ntasks > nodes * limits.cores_per_node:
        add(_E, "too_many_tasks", f"{ntasks} tasks exceed {nodes} node(s) x {limits.cores_per_node} cores")
        ok = False
    if nodes > limits.max_nodes:
        add(_E, "too_many_nodes", f"{nodes} nodes exceed the profile limit of {limits.max_nodes}")
        ok = False
    time = directives.get("time")
    minutes = parse_walltime(time) if time else None
    if time is None:
        add(_E, "no_walltime", "the script must set --time")
        ok = False
    elif minutes is None:
        add(_E, "bad_walltime", f"cannot interpret --time={time}")
        ok = False
    elif minutes > limits.max_walltime_minutes:
        add(_E, "walltime_too_long", f"{minutes} minutes exceed the profile limit of {limits.max_walltime_minutes}")
        ok = False
    return JobResources(partition, nodes, ntasks, minutes) if ok else None
