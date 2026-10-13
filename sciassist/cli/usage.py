"""Token and cost accounting for the terminal.

Counts come from ``usage_metadata`` on every model response, the main agent's
and its subagents'. Prices come from ``genai-prices``; a model it doesn't know
is counted in tokens only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost: float = 0.0
    priced: bool = True
    """False once any response came from a model without a known price."""

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cost += other.cost
        self.priced = self.priced and other.priced

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class UsageTracker:
    """Session totals, plus the main agent's latest context size."""

    session: Usage = field(default_factory=Usage)
    context_tokens: int = 0

    def record(self, message: AIMessage, model: str, *, main: bool) -> Usage:
        usage = usage_of(message, model)
        self.session.add(usage)
        if main and usage.input_tokens:
            self.context_tokens = usage.input_tokens + usage.output_tokens
        return usage


def usage_of(message: AIMessage, model: str) -> Usage:
    meta = message.usage_metadata or {}
    details = meta.get("input_token_details") or {}
    usage = Usage(
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
        cache_read_tokens=details.get("cache_read", 0) or 0,
    )
    price = _price(model, usage.input_tokens, usage.cache_read_tokens, usage.output_tokens)
    if price is None:
        usage.priced = usage.total_tokens == 0
    else:
        usage.cost = price
    return usage


def _price(model: str, input_tokens: int, cache_read: int, output_tokens: int) -> float | None:
    from genai_prices import Usage as PriceUsage
    from genai_prices import calc_price

    provider, _, name = model.partition(":")
    if not name:
        provider, name = "", provider
    try:
        result = calc_price(
            PriceUsage(
                input_tokens=input_tokens,
                cache_read_tokens=cache_read,
                output_tokens=output_tokens,
            ),
            model_ref=name,
            provider_id=provider or None,
        )
    except LookupError:
        return None
    return float(result.total_price)


def format_tokens(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


def format_cost(usage: Usage) -> str:
    if usage.cost == 0 and not usage.priced:
        return "cost unknown"
    text = f"${usage.cost:.2f}" if usage.cost >= 0.01 or usage.cost == 0 else "<$0.01"
    return text if usage.priced else text + "+"
