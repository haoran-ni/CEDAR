"""What the protocols of one run of a code share: the job script and where the run goes.

A run is one launch of the code, through Slurm (``job.sh``) or on the interactive node CEDAR
runs on (``execution: local``; ``local.sh``, the approved template without its Slurm
directives), never on a login node. When the agent does not write the job script, it is
rendered from the approved template for the code; either way it is examined against that
template, its resources against the profile's limits and the task's budget and, for a local
run, against the allocation.
"""

import math
from collections.abc import Callable, Mapping

from cedar.contracts.examination import Finding, FindingSeverity
from cedar.contracts.task import ScientificTaskSpec
from cedar.hpc.jobscript import (
    JobResources,
    ScriptExamination,
    examine_job_script,
    examine_local_script,
    render_job_script,
    render_local_script,
)
from cedar.hpc.node import WORKBENCH_CORES, ExecutionHost, HostWatch
from cedar.hpc.profile import ComputeProfile
from cedar.protocols.base import Protocol

_E, _I = FindingSeverity.ERROR, FindingSeverity.INFO

LOCAL_MARGIN_MINUTES = 5
"""A local run must end this long before the allocation does."""
LOGIN_NODE = ("CEDAR is running on a login node ({host}), and DFT cannot run locally on login nodes. Start CEDAR "
              "on an interactive node (inside a Slurm allocation) to run it locally, or request it through Slurm.")
current_host = HostWatch()
"""Where this process runs, asked again at most every minute."""


class CodeRun(Protocol):
    """A protocol whose run is one launch of the code ``code`` names in the compute profile, writing ``run_output``."""

    code: str
    """Key of the code's installation in a compute profile's ``codes``."""
    run_output: str
    """The file the launch writes the code's standard output to."""
    runs_locally = True

    def __init__(self, profile: ComputeProfile, host: Callable[[], ExecutionHost] | None = None) -> None:
        """``host`` says where CEDAR runs, which bounds a local run."""
        if self.code not in profile.codes:
            raise ValueError(f"compute profile {profile.name} has no {self.code!r} installation")
        self._profile = profile
        self._code = profile.codes[self.code]
        self._host = host or current_host

    @staticmethod
    def _script(task: ScientificTaskSpec) -> str:
        return "local.sh" if task.execution == "local" else "job.sh"

    @property
    def code_version(self) -> str:
        return self._code.version

    def _where(self) -> list[str]:
        if not self.runs_locally:
            return [f"Runs through Slurm only (job.sh); use version {self.superseded_by} for a local run."]
        host = self._host()
        local = (f"possible here: CEDAR runs on the {host.describe()}; a local run may use up to {host.cores} "
                 "tasks and must end before the allocation does" if host.interactive else
                 "not possible here: " + LOGIN_NODE.format(host=host.hostname))
        return ["Where it runs: execution 'slurm' (default) submits job.sh to the scheduler; execution 'local' runs "
                f"local.sh on the interactive node CEDAR runs on, inside the researcher's allocation. Local runs are "
                f"{local}. Both need the researcher's approval to start."]

    # -- generation -------------------------------------------------------------------

    def _generate_script(self, task: ScientificTaskSpec, present: Mapping[str, bytes], binary: str | None = None
                         ) -> tuple[dict[str, bytes], list[Finding]]:
        """The job script the task's execution takes, rendered from the approved template launching ``binary`` (the
        code's own by default), unless the agent wrote it."""
        script = self._script(task)
        other = "job.sh" if script == "local.sh" else "local.sh"
        if other in present:
            return {}, [Finding(severity=_E, code="wrong_script", file=other,
                                message=f"a {task.execution} run takes {script}, not {other}")]
        if script in present:
            return {}, []
        if script == "local.sh":
            if not (self.runs_locally and self._host().interactive):
                return {}, []
            resources = self._default_local_resources(task)
            host = self._host()
            if resources is None and host.dft_cores == 0:
                return {}, [Finding(severity=_E, code="no_cores_for_dft",
                                    message=f"the allocation has {host.cores} cores here, and {WORKBENCH_CORES} are "
                                            "kept for the agents' commands, which leaves none for a local run; run it "
                                            "through Slurm")]
            if resources is None:
                return {}, [Finding(severity=_E, code="budget_too_small",
                                    message="the task budget or the allocation's time left does not allow a local "
                                            "run of one minute")]
            return {"local.sh": render_local_script(
                self._profile, self._code, ntasks=resources.ntasks, walltime_minutes=resources.walltime_minutes,
                output_file=self.run_output, binary=binary).encode()}, []
        resources = self._default_resources(task)
        if resources is None:
            return {}, [Finding(severity=_E, code="budget_too_small",
                                message="the task budget does not allow one minute at the profile's default task "
                                        "count")]
        return {"job.sh": render_job_script(self._profile, self._code, resources, job_name="cedar",
                                            output_file=self.run_output, binary=binary).encode()}, []

    def _generated(self, generated: Mapping[str, bytes]) -> list[Finding]:
        return [Finding(severity=_I, code="generated", file=name,
                        message=f"{name} was generated by {self.ref.name} {self.ref.version}")
                for name in sorted(generated)]

    def _default_resources(self, task: ScientificTaskSpec) -> JobResources | None:
        limits = self._profile.limits
        ntasks = task.budget.cores or limits.default_ntasks
        minutes = limits.default_walltime_minutes
        if task.budget.max_walltime_hours is not None:
            minutes = min(minutes, math.floor(task.budget.max_walltime_hours * 60))
        if task.budget.max_core_hours is not None:
            minutes = min(minutes, math.floor(task.budget.max_core_hours * 60 / ntasks))
        if minutes < 1:
            return None
        nodes = math.ceil(ntasks / limits.cores_per_node)
        return JobResources(limits.default_partition, nodes, ntasks, minutes)

    def _default_local_resources(self, task: ScientificTaskSpec) -> JobResources | None:
        """The allocation's cores here (or the task's, if fewer), for the profile's default walltime, ending well
        before the allocation does."""
        host = self._host()
        if host.dft_cores == 0:
            return None
        ntasks = min(task.budget.cores or host.dft_cores, host.dft_cores)
        minutes = self._profile.limits.default_walltime_minutes
        if task.budget.max_walltime_hours is not None:
            minutes = min(minutes, math.floor(task.budget.max_walltime_hours * 60))
        if task.budget.max_core_hours is not None:
            minutes = min(minutes, math.floor(task.budget.max_core_hours * 60 / ntasks))
        left = host.seconds_left()
        if left is not None:
            minutes = min(minutes, math.floor(left / 60) - LOCAL_MARGIN_MINUTES)
        return JobResources("local", 1, ntasks, minutes) if minutes >= 1 else None

    # -- examination ------------------------------------------------------------------

    def _examined_script(self, files: Mapping[str, bytes]) -> tuple[str, ScriptExamination] | None:
        """The job script among ``files``, examined against the approved template: (its name, what was found)."""
        if "local.sh" in files:
            return "local.sh", examine_local_script(files["local.sh"].decode(errors="replace"), self._profile,
                                                    self._code, output_file=self.run_output)
        if "job.sh" in files:
            return "job.sh", examine_job_script(files["job.sh"].decode(errors="replace"), self._profile, self._code,
                                                output_file=self.run_output)
        return None

    def _examine_scripts(self, task: ScientificTaskSpec, files: Mapping[str, bytes]) -> list[Finding]:
        """The job script against the approved template, its resources against the budget and, for a local run,
        the allocation."""
        findings: list[Finding] = []
        if "job.sh" in files:
            script = examine_job_script(files["job.sh"].decode(errors="replace"), self._profile, self._code,
                                        output_file=self.run_output)
            findings += script.findings
            if script.resources is not None:
                findings += self._examine_budget(task, script.resources, "job.sh")
        if "local.sh" in files:
            script = examine_local_script(files["local.sh"].decode(errors="replace"), self._profile, self._code,
                                          output_file=self.run_output)
            findings += script.findings
            if script.resources is not None:
                findings += self._examine_budget(task, script.resources, "local.sh")
                findings += self._examine_fit(script.resources)
        return findings

    def _examine_where(self) -> list[Finding]:
        if not self.runs_locally:
            return [Finding(severity=_E, code="no_local_runs",
                            message=f"{self.ref.name} {self.ref.version} runs through Slurm only; request "
                                    f"{self.ref.name} {self.superseded_by} for a local run")]
        host = self._host()
        if not host.interactive:
            return [Finding(severity=_E, code="local_on_login_node", message=LOGIN_NODE.format(host=host.hostname))]
        return [Finding(severity=_I, code="local_host", message=f"runs on the {host.describe()}")]

    def _examine_fit(self, resources: JobResources) -> list[Finding]:
        """A local run must fit the allocation CEDAR runs in: its cores here, and its time left."""
        host = self._host()
        if not host.interactive:
            return []
        findings = []
        if resources.ntasks > host.dft_cores:
            findings.append(Finding(severity=_E, code="too_many_tasks", file="local.sh",
                                    message=f"{resources.ntasks} tasks exceed the {host.dft_cores} cores local DFT may "
                                            f"use on {host.hostname}: the allocation has {host.cores}, and "
                                            f"{WORKBENCH_CORES} are kept for the agents' commands"))
        left = host.seconds_left()
        if left is not None and resources.walltime_minutes > left / 60 - LOCAL_MARGIN_MINUTES:
            findings.append(Finding(severity=_E, code="past_the_allocation", file="local.sh",
                                    message=f"{resources.walltime_minutes} minutes would run past the end of "
                                            f"allocation {host.allocation}, {left / 60:.0f} minutes from now"))
        return findings

    def requested_resources(self, files: Mapping[str, bytes]):
        examined = self._examined_script(files)
        return examined[1].resources if examined is not None else None

    @staticmethod
    def _examine_budget(task: ScientificTaskSpec, resources: JobResources, script: str) -> list[Finding]:
        findings = []
        budget = task.budget
        if budget.max_core_hours is not None and resources.core_hours > budget.max_core_hours:
            findings.append(Finding(severity=_E, code="over_budget", file=script,
                                    message=f"{resources.core_hours:g} core-hours requested; the task allows "
                                            f"{budget.max_core_hours:g}"))
        if budget.max_walltime_hours is not None and resources.walltime_minutes > budget.max_walltime_hours * 60:
            findings.append(Finding(severity=_E, code="over_budget", file=script,
                                    message=f"{resources.walltime_minutes} minutes requested; the task allows "
                                            f"{budget.max_walltime_hours:g} h"))
        return findings

    def _other_files(self, files: Mapping[str, bytes]) -> list[Finding]:
        """Files besides the protocol's own inputs, as facts for the input reviewer, which checks them; CEDAR does
        not read them. Only a file named like one of the run's outputs is refused: the run writes it."""
        findings = []
        for name in sorted(set(files) - set(self.input_files)):
            if name in self.output_files:
                findings.append(Finding(severity=_E, code="output_name", file=name,
                                        message=f"{name} is the run's output, which the run writes; an input file "
                                                "cannot have its name"))
            else:
                findings.append(Finding(severity=_I, code="other_file", file=name,
                                        message=f"{name} ({len(files[name])} bytes) is not read by CEDAR; the input "
                                                "reviewer checks it"))
        return findings
