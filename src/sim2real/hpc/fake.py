"""TEST DOUBLE. An in-memory stand-in for Slurm that runs nothing.

Tests and demonstrations that use it must say so; job ids it returns are not real.
"""

import itertools
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from sim2real.contracts.status import SchedulerState
from sim2real.hpc.slurm import SchedulerObservation


class FakeScheduler:
    def __init__(self, *, first_job_id: int = 900000) -> None:
        self._ids = itertools.count(first_job_id)
        self.jobs: dict[str, dict] = {}
        self.submissions: list[tuple[Path, Path, str]] = []
        self.on_submit: Callable[[str, Path], None] | None = None
        """Called with (job_id, workdir) after each submission, e.g. to drop fixture outputs."""

    def submit(self, script: Path, workdir: Path, job_name: str) -> str:
        job_id = str(next(self._ids))
        self.jobs[job_id] = {"name": job_name, "state": SchedulerState.PENDING, "workdir": workdir}
        self.submissions.append((script, workdir, job_name))
        if self.on_submit is not None:
            self.on_submit(job_id, workdir)
        return job_id

    def set_state(self, job_id: str, state: SchedulerState) -> None:
        self.jobs[job_id]["state"] = state

    def forget(self, job_id: str) -> None:
        """Simulate a job the scheduler no longer reports."""
        del self.jobs[job_id]

    def query(self, job_ids: Sequence[str]) -> dict[str, SchedulerObservation]:
        return {
            job_id: SchedulerObservation(job_id, self.jobs[job_id]["state"], self.jobs[job_id]["state"].value)
            for job_id in job_ids
            if job_id in self.jobs
        }

    def find_by_name(self, job_name: str, since: datetime) -> list[SchedulerObservation]:
        return [
            SchedulerObservation(job_id, job["state"], job["state"].value)
            for job_id, job in self.jobs.items()
            if job["name"] == job_name
        ]

    def cancel(self, job_id: str) -> None:
        self.jobs[job_id]["state"] = SchedulerState.CANCELLED

    def log_file_names(self, job_id: str) -> list[str]:
        return [f"fake-{job_id}.log"]
