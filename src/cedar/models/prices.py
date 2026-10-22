"""What model calls cost, in US dollars, so that agents stay within the researcher's budgets.

Budgets are in dollars because the kinds of tokens cost very different amounts: for Claude
Opus 5.5, input read from the prompt cache costs a twentieth of fresh input, and output five
times as much. A role's prices come from ``models.toml`` (``[roles.<role>.prices]``) or, for
the models listed here, from the prices the provider published.
"""

from cedar.models.types import ModelPrices, ModelSettings

PUBLISHED: dict[tuple[str, str], ModelPrices] = {
    ("anthropic", "claude-opus-5-5"): ModelPrices(input=4.0, output=20.0, cache_read=0.20, cache_write=5.0,
                                                  web_search=0.01),
    ("anthropic", "claude-sonnet-5-5"): ModelPrices(input=2.0, output=10.0, cache_read=0.20, cache_write=2.5,
                                                    web_search=0.01),
    ("anthropic", "claude-fable-5-1"): ModelPrices(input=10.0, output=50.0, cache_read=0.25, cache_write=12.5,
                                                   web_search=0.01),
}
"""As Anthropic published them in October 2026: cache writes are five-minute writes, at 1.25 times fresh input
(CEDAR asks for no longer cache); web search is $10 per 1,000 searches. Set prices in models.toml for any other
model, or when these change."""


def prices_for(settings: ModelSettings) -> ModelPrices | None:
    """The prices of a role's model: as configured, else as published; None if neither is known."""
    return settings.prices or PUBLISHED.get((settings.provider, settings.model))


def dollars(amount: float | None) -> str:
    """An amount of money as the researcher reads it: $0.042, $3.17, $15.00."""
    if amount is None:
        return "unknown"
    return f"${amount:.3f}" if 0 < abs(amount) < 0.1 else f"${amount:,.2f}"
