"""Recording agent runs in an investigation's research record."""

from cedar.agents.runtime import RunResult
from cedar.contracts.agents import AgentReport
from cedar.contracts.base import Actor
from cedar.contracts.events import AgentRunRecorded, EventDraft
from cedar.models.types import ModelSettings
from cedar.state.ledger import RecordResult, ResearchLedger

REPORT_LIMIT = 8000


def record_agent_run(ledger: ResearchLedger, investigation_id: str, actor: Actor, *, session_id: str, role: str,
                     settings: ModelSettings, result: RunResult, task_id: str | None = None,
                     started_version: int | None = None, account: str | None = None,
                     structured: AgentReport | None = None) -> RecordResult:
    """``account`` is CEDAR's account of a run that ended without the agent's own report; ``structured`` the report
    a hypothesis agent ended its task with."""
    if structured is not None:
        report = structured.text() + (f"\n\n{result.final_text.strip()}" if result.final_text.strip() else "")
    else:
        report = account or result.final_text or result.error
    if report and len(report) > REPORT_LIMIT:
        report = report[:REPORT_LIMIT] + " …[truncated]"
    usage = result.usage
    return ledger.record(investigation_id, [EventDraft(actor=actor, payload=AgentRunRecorded(
        session_id=session_id, role=role, task_id=task_id, provider=settings.provider, requested_model=settings.model,
        served_models=tuple(result.served_models), turns=result.turns, input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens, cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens, outcome=result.outcome, report=report or None,
        dropped_reasoning_blocks=result.dropped_reasoning_blocks, web_searches=usage.web_searches,
        web_fetches=usage.web_fetches, cost_usd=result.cost_usd, started_version=started_version,
        report_by_cedar=account is not None and structured is None, structured=structured,
    ))])
