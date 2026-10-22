"""A ledger, a SYNTHETIC site and every engine service wired together, for tests.

Uses the FakeScheduler TEST DOUBLE: no job runs.
"""

from pathlib import Path

from factories import AGENT, add_hypothesis, authorize, start_investigation
from sites import co_on_cu111, declared, geometry_text

from cedar.contracts import PermissionAction, ProtocolRef, SchedulerState, ScientificTaskSpec
from cedar.engine import JobManager, JobMonitor, ProtocolEngine, ResultValidator
from cedar.engine.analysis import AnalysisRunner
from cedar.hpc.fake import FakeScheduler
from cedar.protocols import (
    TOTAL_ENERGY,
    FhiAimsRun,
    FhiAimsRunJobChecked,
    FhiAimsRunReviewed,
    FhiAimsRunSlurmOnly,
    ProtocolRegistry,
)


class World:
    """A ledger, a synthetic site and every engine service wired together."""

    def __init__(self, ledger, store, artifacts, site, **constraints):
        self.ledger, self.store, self.artifacts, self.site = ledger, store, artifacts, site
        self.scheduler = FakeScheduler()
        self.registry = ProtocolRegistry([FhiAimsRunJobChecked(site), FhiAimsRunReviewed(site), FhiAimsRun(site),
                                          FhiAimsRunSlurmOnly(site)])
        self.runs = Path(site.runs_root)
        self.engine = ProtocolEngine(ledger, self.registry, artifacts, self.runs)
        self.manager = JobManager(ledger, self.scheduler, self.runs)
        self.monitor = JobMonitor(ledger, self.scheduler)
        self.validator = ResultValidator(ledger, self.registry, artifacts, self.scheduler, self.runs)
        self.analysis = AnalysisRunner(ledger, artifacts, self.runs, self.runs.parent / "analyses")
        self.inv = start_investigation(ledger, **constraints)
        self.hyp = add_hypothesis(ledger, self.inv)
        slab, bottom = co_on_cu111()
        self.structure = artifacts.put_bytes(geometry_text(slab, fixed=bottom).encode(), label="geometry.in")

    def task(self, authored=None, version="0.1", **overrides) -> ScientificTaskSpec:
        """A task under ``version``; 0.1 by default, so the examination alone decides, as it did under 0.1 and 0.2,
        and 0.3 for inputs the input reviewer judges."""
        return ScientificTaskSpec(
            protocol=ProtocolRef(name="fhiaims.run", version=version), observable=TOTAL_ENERGY,
            estimand="total energy of CO/Cu(111) atop", system_label="CO/Cu(111) atop (synthetic)",
            structure=self.structure, authored_files=authored or {}, **declared(**overrides))

    def request(self, **kwargs) -> str:
        return self.ledger.request_calculation(self.inv, AGENT, self.hyp, self.task(**kwargs), "test").calculation_id

    def checked(self, cid, **kwargs) -> str:
        """Run a calculation to its end and record the run check."""
        job = self.run(cid, **kwargs)
        self.validator.check_run(self.inv, cid)
        return job

    def state(self):
        return self.ledger.load(self.inv)

    def review(self, cid, decision="input_agent_approved", findings=(), summary="The inputs fit the test.",
               settings=None, deny=()):
        """The input reviewer's verdict on ``cid``, recorded through its own tool in a task the dispatcher would
        queue; returns the tool's outcome."""
        from cedar.agents.toolkits.files import agent_access
        from cedar.agents.toolkits.reviewer import ReviewerContext, reviewer_tools
        from cedar.contracts import Actor, ActorKind, AgentRunRecorded, AgentTaskQueued, EventDraft
        from cedar.contracts.base import new_id

        task_id = new_id("task")
        self.ledger.record(self.inv, [EventDraft(actor=Actor(kind=ActorKind.SYSTEM, id="dispatcher"),
                                                 payload=AgentTaskQueued(
            task_id=task_id, role="input_reviewer", hypothesis_id=self.calc(cid).hypothesis_ids[0],
            calculation_ids=(cid,), instruction="Review."))])
        ctx = ReviewerContext(ledger=self.ledger, investigation_id=self.inv,
                              actor=Actor(kind=ActorKind.INPUT_REVIEWER, id="input-reviewer"),
                              artifacts=self.artifacts, task=self.state().tasks[task_id], files=agent_access(deny),
                              profile=self.site)
        tools = {t.name: t for t in reviewer_tools(ctx)}
        if settings is None:
            settings = {name: "physical_model" for name in self.calc(cid).examination.settings}
        outcome = tools["record_input_review"].invoke({"calculation_id": cid, "decision": decision,
                                                        "summary": summary, "findings": list(findings),
                                                        "settings": settings})
        if outcome.is_error:
            return outcome  # a reviewer corrects itself within the same task
        self.ledger.record(self.inv, [EventDraft(actor=ctx.actor, payload=AgentRunRecorded(
            session_id="review", role="input_reviewer", task_id=task_id, provider="p", requested_model="m", turns=1,
            input_tokens=1, output_tokens=1, cache_read_tokens=0, cache_write_tokens=0, outcome="finished",
            report=outcome.content))])
        return outcome

    def reviewer_material(self, cid) -> str:
        """What the input reviewer is shown for ``cid``."""
        from cedar.agents.toolkits.files import agent_access
        from cedar.agents.toolkits.reviewer import ReviewerContext, review_material
        from cedar.contracts import Actor, ActorKind

        task = type("Task", (), {"calculation_ids": (cid,), "task_id": "t"})()
        ctx = ReviewerContext(ledger=self.ledger, investigation_id=self.inv,
                              actor=Actor(kind=ActorKind.INPUT_REVIEWER, id="input-reviewer"),
                              artifacts=self.artifacts, task=task, files=agent_access(), profile=self.site)
        return review_material(ctx, self.state(), self.calc(cid))

    def calc(self, cid):
        return self.state().calculations[cid]

    def approve_submission(self, cid):
        authorize(self.ledger, self.inv, PermissionAction.SUBMIT, cid, intervention_id=f"submit-{cid}")

    def run(self, cid, *, output: str | None = None, final: SchedulerState = SchedulerState.COMPLETED,
            extra: dict[str, str] | None = None) -> str:
        """Examine, authorize, submit and drive the fake job to its end, writing a synthetic aims.out."""
        self.engine.examine(self.inv, cid)
        self.approve_submission(cid)
        job = self.manager.submit(self.inv, cid)
        self.scheduler.set_state(job, SchedulerState.RUNNING)
        self.monitor.poll(self.inv)
        if output is not None:
            (self.runs / self.calc(cid).workdir / "aims.out").write_text(output)
        for name, text in (extra or {}).items():
            (self.runs / self.calc(cid).workdir / name).write_text(text)
        self.scheduler.set_state(job, final)
        self.monitor.poll(self.inv)
        return job
