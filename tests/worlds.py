"""A ledger, a SYNTHETIC site and every engine service wired together, for tests.

Uses the FakeScheduler TEST DOUBLE: no job runs.
"""

from pathlib import Path

from factories import AGENT, add_hypothesis, authorize, start_investigation
from sites import co_on_cu111, declared, geometry_text

from sim2real.contracts import PermissionAction, ProtocolRef, SchedulerState, ScientificTaskSpec
from sim2real.engine import JobManager, JobMonitor, ProtocolEngine, ResultValidator
from sim2real.hpc.fake import FakeScheduler
from sim2real.protocols import TOTAL_ENERGY, FhiAimsTotalEnergy, ProtocolRegistry


class World:
    """A ledger, a synthetic site and every engine service wired together."""

    def __init__(self, ledger, store, artifacts, site, **constraints):
        self.ledger, self.store, self.artifacts, self.site = ledger, store, artifacts, site
        self.scheduler = FakeScheduler()
        self.registry = ProtocolRegistry([FhiAimsTotalEnergy(site)])
        self.runs = Path(site.runs_root)
        self.engine = ProtocolEngine(ledger, self.registry, artifacts, self.runs)
        self.manager = JobManager(ledger, self.scheduler, self.runs)
        self.monitor = JobMonitor(ledger, self.scheduler)
        self.validator = ResultValidator(ledger, self.registry, artifacts, self.scheduler, self.runs)
        self.inv = start_investigation(ledger, **constraints)
        self.hyp = add_hypothesis(ledger, self.inv)
        slab, bottom = co_on_cu111()
        self.structure = artifacts.put_bytes(geometry_text(slab, fixed=bottom).encode(), label="geometry.in")

    def task(self, authored=None, **overrides) -> ScientificTaskSpec:
        return ScientificTaskSpec(
            protocol=ProtocolRef(name="fhiaims.total_energy", version="0.1"), observable=TOTAL_ENERGY,
            estimand="total energy of CO/Cu(111) atop", system_label="CO/Cu(111) atop (synthetic)",
            structure=self.structure, authored_files=authored or {}, **declared(**overrides))

    def request(self, **kwargs) -> str:
        return self.ledger.request_calculation(self.inv, AGENT, self.hyp, self.task(**kwargs), "test").calculation_id

    def state(self):
        return self.ledger.load(self.inv)

    def calc(self, cid):
        return self.state().calculations[cid]

    def approve_submission(self, cid):
        authorize(self.ledger, self.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")

    def run(self, cid, *, output: str | None = None, final: SchedulerState = SchedulerState.COMPLETED) -> str:
        """Examine, authorize, submit and drive the fake job to its end, writing a synthetic aims.out."""
        self.engine.examine(self.inv, cid)
        self.approve_submission(cid)
        job = self.manager.submit(self.inv, cid)
        self.scheduler.set_state(job, SchedulerState.RUNNING)
        self.monitor.poll(self.inv)
        if output is not None:
            (self.runs / self.calc(cid).workdir / "aims.out").write_text(output)
        self.scheduler.set_state(job, final)
        self.monitor.poll(self.inv)
        return job
