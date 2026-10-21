import pytest

from sim2real.contracts import FindingSeverity
from sim2real.hpc import JobResources, examine_job_script, render_job_script
from sim2real.hpc.jobscript import format_walltime, parse_walltime

RESOURCES = JobResources(partition="compute", nodes=1, ntasks=32, walltime_minutes=90)


def codes(examination, *severities):
    return {f.code for f in examination.findings if not severities or f.severity in severities}


def blocking(examination):
    return codes(examination, FindingSeverity.ERROR, FindingSeverity.REVIEW)


@pytest.fixture
def aims(site):
    return site.codes["fhi-aims"]


@pytest.fixture
def rendered(site, aims):
    return render_job_script(site, aims, RESOURCES, job_name="test", output_file="aims.out")


def examine(text, site, aims):
    return examine_job_script(text, site, aims, output_file="aims.out")


def test_rendered_script_passes_and_reports_its_resources(site, aims, rendered):
    result = examine(rendered, site, aims)
    assert blocking(result) == set()
    assert result.resources == RESOURCES


def test_script_written_with_shell_variables_is_recognized(site, aims, rendered):
    """An agent may restructure the template with variables; commands expand to the same thing."""
    directory, binary = aims.binary.rsplit("/", 1)
    env_line = next(l for l in rendered.splitlines() if l.startswith("source "))
    text = rendered.replace(f"mpirun -np 32 {aims.binary} > aims.out",
                            f'AIMS_DIR="{directory}"\nAIMS_BIN="$AIMS_DIR/{binary}"\nmpirun -np 32 $AIMS_BIN > aims.out')
    text = text.replace(env_line, f'ENV_FILE="{env_line.split()[1]}"\nsource ${{ENV_FILE}}')
    assert blocking(examine(text, site, aims)) == set()


@pytest.mark.parametrize("extra", [
    "rm -rf $HOME/old_runs",
    "cp aims.out /tmp/elsewhere",
    "echo $(whoami) > who.txt",
    "mpirun -np 32 /usr/bin/other-code > other.out",
])
def test_commands_outside_the_template_need_review(site, aims, rendered, extra):
    result = examine(rendered + extra + "\n", site, aims)
    assert "unapproved_command" in codes(result, FindingSeverity.REVIEW) | codes(result, FindingSeverity.ERROR) \
        or "unresolved_line" in codes(result, FindingSeverity.REVIEW)


def test_launching_another_binary_is_not_a_launch(site, aims, rendered):
    result = examine(rendered.replace(aims.binary, "/opt/other/aims.x"), site, aims)
    assert "no_launch" in codes(result, FindingSeverity.ERROR)
    assert "unapproved_command" in codes(result, FindingSeverity.REVIEW)


def test_output_must_go_to_the_protocol_file(site, aims, rendered):
    result = examine(rendered.replace("> aims.out", "> other.out"), site, aims)
    assert "no_launch" in codes(result, FindingSeverity.ERROR)


@pytest.mark.parametrize("change, code", [
    ("--partition=compute", "partition_not_allowed"),
    ("--time=01:30:00", "walltime_too_long"),
    ("--ntasks=32", "too_many_tasks"),
])
def test_resource_limits_come_from_the_profile(site, aims, rendered, change, code):
    replacement = {"--partition=compute": "--partition=gpu", "--time=01:30:00": "--time=3-00:00:00",
                   "--ntasks=32": "--ntasks=500"}[change]
    assert code in codes(examine(rendered.replace(change, replacement), site, aims), FindingSeverity.ERROR)


def test_launching_more_tasks_than_requested_is_an_error(site, aims, rendered):
    text = rendered.replace("mpirun -np 32", "mpirun -np 64")
    assert "oversubscribed" in codes(examine(text, site, aims), FindingSeverity.ERROR)


def test_directives_outside_the_template_need_review(site, aims, rendered):
    text = rendered.replace("#SBATCH --exclusive", "#SBATCH --exclusive\n#SBATCH --account=someone-else")
    assert "unapproved_directive" in codes(examine(text, site, aims), FindingSeverity.REVIEW)


def test_short_options_are_understood(site, aims, rendered):
    text = (rendered.replace("--partition=compute", "-p compute").replace("--ntasks=32", "-n 32")
            .replace("--time=01:30:00", "-t 90"))
    result = examine(text, site, aims)
    assert blocking(result) == set() and result.resources == RESOURCES


def test_log_files_must_stay_in_the_run_directory(site, aims, rendered):
    text = rendered.replace("#SBATCH --exclusive", "#SBATCH --exclusive\n#SBATCH --output=/home/x/log.txt")
    assert "output_path" in codes(examine(text, site, aims), FindingSeverity.REVIEW)


def test_missing_resources_are_errors(site, aims, rendered):
    text = "\n".join(l for l in rendered.splitlines() if "--time" not in l and "--partition" not in l)
    assert {"no_walltime", "no_partition"} <= codes(examine(text, site, aims), FindingSeverity.ERROR)


@pytest.mark.parametrize("text, minutes", [
    ("90", 90), ("01:30:00", 90), ("1:30", 2), ("2-00:00:00", 2880), ("1-12", 2160), ("00:00:30", 1), ("x", None),
])
def test_walltime_formats(text, minutes):
    assert parse_walltime(text) == minutes


def test_walltime_round_trip():
    assert parse_walltime(format_walltime(2 * 1440 + 75)) == 2 * 1440 + 75
