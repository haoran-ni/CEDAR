"""``cedar reset`` and ``cedar purge``.

The data and configuration directories are the isolated ones of each test; the Slurm scheduler is the FakeScheduler
TEST DOUBLE, and local runs use a TEST DOUBLE host (an allocation that does not exist), so no job runs.
"""

import io
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from factories import acting_for, add_hypothesis, start_investigation, submit, task

from cedar import cli
from cedar.contracts import SchedulerState
from cedar.dispatch import LOCK
from cedar.hpc import ProfileStore
from cedar.hpc.fake import FakeScheduler
from cedar.hpc.local import RECORDS, LocalBackend
from cedar.hpc.node import ExecutionHost
from cedar.hpc.slurm import SchedulerError
from cedar.paths import config_dir, data_dir
from cedar.workspace import Workspace


def interactive() -> ExecutionHost:
    return ExecutionHost(socket.gethostname(), allocation="777001", cores=8,
                         ends_at=datetime.now(timezone.utc) + timedelta(hours=2),
                         reason="TEST DOUBLE: an allocation that does not exist")


class Terminal(io.StringIO):
    """TEST DOUBLE for a terminal the researcher types in."""

    def isatty(self) -> bool:
        return True


@pytest.fixture
def stored(site, monkeypatch):
    """An investigation with a Slurm job and a local run that have not ended, a run directory, an agent workspace
    with a read-only directory in it, and the setup."""
    slurm = FakeScheduler()
    monkeypatch.setattr("cedar.hpc.slurm.SlurmBackend", lambda: slurm)
    monkeypatch.setattr("cedar.hpc.local.LocalBackend", lambda root, host: LocalBackend(root, host=interactive))
    profiles = ProfileStore(config_dir() / "profiles")
    profiles.save(site)
    (config_dir() / "models.toml").write_text("# test\n")
    (config_dir() / "credentials.toml").write_text("")
    os.chmod(config_dir() / "credentials.toml", 0o600)

    ledger = Workspace.open(data_dir()).ledger
    inv = start_investigation(ledger, compute_profile=site.name)
    hyp = add_hypothesis(ledger, inv)
    actor = acting_for(ledger, inv, hyp)
    on_slurm = ledger.request_calculation(inv, actor, hyp, task(), "test").calculation_id
    here = ledger.request_calculation(inv, actor, hyp, task(xc="PBE0", execution="local"), "test").calculation_id
    runs = Path(site.runs_root)
    slurm_job = slurm.submit(runs / "job.sh", runs / inv / on_slurm, f"cedar-{on_slurm}")
    slurm.set_state(slurm_job, SchedulerState.RUNNING)
    submit(ledger, inv, on_slurm, job_id=slurm_job)
    (runs / inv / here).mkdir(parents=True)
    script = runs / inv / here / "local.sh"
    script.write_text("#!/bin/bash\n#CEDAR --ntasks=2 --time=10\ntrue\n")
    local_job = LocalBackend(runs, host=interactive).submit(script, script.parent, f"cedar-{here}")
    submit(ledger, inv, here, job_id=local_job)
    (runs / inv / on_slurm).mkdir(parents=True)
    (runs / inv / on_slurm / "aims.out").write_text("synthetic output\n")
    (runs / "not-cedars.txt").write_text("the researcher's own file\n")
    locked = data_dir() / "workspaces" / inv / hyp / "locked"
    locked.mkdir(parents=True)
    (locked / "inside.txt").write_text("x")
    locked.chmod(0o555)
    return {"slurm": slurm, "slurm_job": slurm_job, "local_job": local_job, "runs": runs, "inv": inv,
            "run_dir": runs / inv / on_slurm}


def test_reset_waits_for_calculations_that_have_not_ended(stored, capsys):
    assert cli.main(["reset", "--yes"]) == 1
    err = capsys.readouterr().err
    assert f"Slurm job {stored['slurm_job']}, RUNNING" in err
    assert f"local run {stored['local_job']}, PENDING (waiting for 2 free cores)" in err
    assert "`cedar reset --cancel-jobs`" in err
    assert (data_dir() / "events.sqlite").exists() and (stored["runs"] / RECORDS).is_dir()


def test_reset_asks_the_scheduler_not_the_record(stored, capsys):
    """A closed investigation's record stops following its jobs: what the scheduler says counts."""
    stored["slurm"].set_state(stored["slurm_job"], SchedulerState.COMPLETED)
    LocalBackend(stored["runs"], host=interactive).cancel(stored["local_job"])
    assert cli.main(["reset", "--yes"]) == 0
    assert "Deleted." in capsys.readouterr().out


def test_reset_cancels_calculations_first_when_asked(stored, capsys):
    assert cli.main(["reset", "--cancel-jobs", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "First it cancels:" in out and "Cancelled 2 calculations." in out
    assert stored["slurm"].jobs[stored["slurm_job"]]["state"] is SchedulerState.CANCELLED
    assert not data_dir().exists()


def test_reset_deletes_the_investigations_and_keeps_the_setup_and_run_directories(stored, capsys):
    (data_dir() / "notes-of-mine.txt").write_text("not CEDAR's\n")
    assert cli.main(["reset", "--cancel-jobs", "--yes"]) == 0
    out = capsys.readouterr().out
    assert f"1 investigation ({stored['inv']})" in out and "CEDAR's records of 1 local run" in out
    assert "your setup in" in out and "2 run directories under" in out and "--runs deletes them too" in out
    assert sorted(p.name for p in data_dir().iterdir()) == ["notes-of-mine.txt"]
    assert f"Kept {data_dir()}, which also holds notes-of-mine.txt" in out
    assert not (stored["runs"] / RECORDS).exists()
    assert (stored["run_dir"] / "aims.out").exists()
    assert sorted(p.name for p in config_dir().iterdir()) == ["credentials.toml", "models.toml", "profiles"]


def test_purge_deletes_the_setup_and_with_runs_the_run_directories(stored, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0123456789")
    assert cli.main(["purge", "--runs", "--cancel-jobs", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "the compute profile synthetic-site and approvals, models.toml and the stored API keys" in out
    assert "ANTHROPIC_API_KEY is still set in your environment" in out and "pip uninstall cedar" in out
    assert not data_dir().exists() and not config_dir().exists()
    assert sorted(p.name for p in stored["runs"].iterdir()) == ["not-cedars.txt"]


def test_nothing_is_deleted_without_confirmation(stored, monkeypatch, capsys):
    stored["slurm"].set_state(stored["slurm_job"], SchedulerState.COMPLETED)
    LocalBackend(stored["runs"], host=interactive).cancel(stored["local_job"])
    assert cli.main(["reset"]) == 1  # no terminal
    assert "Run `cedar reset --yes` to go ahead" in capsys.readouterr().err
    monkeypatch.setattr("sys.stdin", Terminal())
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert cli.main(["reset"]) == 1
    assert "Nothing deleted." in capsys.readouterr().out and (data_dir() / "events.sqlite").exists()
    monkeypatch.setattr("builtins.input", lambda prompt: "reset")
    assert cli.main(["reset"]) == 0 and not (data_dir() / "events.sqlite").exists()


def test_a_scheduler_that_cannot_be_asked_needs_leave_jobs(stored, monkeypatch, capsys):
    def unreachable(job_ids):
        raise SchedulerError("squeue failed (1): slurm_load_jobs error: Unable to contact slurm controller")

    monkeypatch.setattr(stored["slurm"], "query", unreachable)
    LocalBackend(stored["runs"], host=interactive).cancel(stored["local_job"])
    assert cli.main(["reset", "--cancel-jobs", "--yes"]) == 1
    err = capsys.readouterr().err
    assert "cannot tell whether these calculations still run" in err and "Unable to contact" in err
    assert cli.main(["reset", "--leave-jobs", "--yes"]) == 0
    assert "It leaves these to run on, with nothing recording them:" in capsys.readouterr().out
    assert not (data_dir() / "events.sqlite").exists()


def test_nothing_is_deleted_while_a_dispatcher_works(stored, capsys):
    holder = subprocess.Popen(
        [sys.executable, "-c", "import fcntl, sys, time; f = open(sys.argv[1], 'a'); fcntl.lockf(f, fcntl.LOCK_EX); "
         "print('held', flush=True); time.sleep(60)", str(data_dir() / LOCK)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert cli.main(["reset", "--cancel-jobs", "--yes"]) == 1
        assert "a dispatcher is working on" in capsys.readouterr().err
        assert stored["slurm"].jobs[stored["slurm_job"]]["state"] is SchedulerState.RUNNING
        assert (data_dir() / "events.sqlite").exists()
    finally:
        holder.kill()
        holder.wait()


def test_nothing_stored_is_nothing_to_delete(capsys):
    assert cli.main(["purge", "--yes"]) == 0
    assert "Nothing to delete" in capsys.readouterr().out and not data_dir().exists()
