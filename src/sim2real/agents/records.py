"""Recording agent runs in an investigation's research record."""

from sim2real.agents.runtime import RunResult
from sim2real.contracts.base import Actor
from sim2real.contracts.events import AgentRunRecorded, EventDraft
from sim2real.models.types import ModelSettings
from sim2real.state.ledger import RecordResult, ResearchLedger

REPORT_LIMIT = 8000


def record_agent_run(ledger: ResearchLedger, investigation_id: str, actor: Actor, *, session_id: str, role: str,
                     settings: ModelSettings, result: RunResult) -> RecordResult:
    report = result.final_text or result.error
    if report and len(report) > REPORT_LIMIT:
        report = report[:REPORT_LIMIT] + " …[truncated]"
    usage = result.usage
    return ledger.record(investigation_id, [EventDraft(actor=actor, payload=AgentRunRecorded(
        session_id=session_id, role=role, provider=settings.provider, requested_model=settings.model,
        served_models=tuple(result.served_models), turns=result.turns, input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens, cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens, outcome=result.outcome, report=report or None,
    ))])
