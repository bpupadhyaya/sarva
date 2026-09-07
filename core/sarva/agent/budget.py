"""sarva.agent.budget — resource budgets for an agent run.

Exceeding a budget is a normal terminal state (BUDGET_EXCEEDED), not an
exception — the run stops cleanly with a spend summary.
"""

from __future__ import annotations

from pydantic import BaseModel


class Budget(BaseModel):
    model_config = {"frozen": True}
    max_model_calls: int = 50
    max_total_tokens: int = 2_000_000  # input+output across the run
    max_wall_seconds: float = 3600.0
    # A real, documented-but-easy-to-miss gap found by a fresh-eyes
    # sweep across all three real provider adapters: only
    # AnthropicProvider computes a real, non-zero `cost_usd` (from a
    # verified-current models.yaml pricing entry); OpenaiProvider and
    # GoogleProvider both deliberately report `cost_usd=0.0`
    # unconditionally, since this project has no verified-current
    # pricing for either (see each module's own docstring -- an honest
    # "unknown" rather than a guessed/fabricated number). The
    # consequence one layer up, easy to miss from either provider file
    # alone: `Spend.exceeded()`'s cost check below can NEVER trip for an
    # OpenAI- or Google-routed run, no matter how many real, billed
    # calls it makes -- this dimension only actually protects spend on
    # Anthropic (and is trivially irrelevant, not silently broken, for
    # genuinely free local/mock models). A caller relying on
    # `max_cost_usd` specifically to cap real dollar spend on OpenAI or
    # Google gets no such protection; `max_model_calls`/`max_wall_seconds`
    # are the dimensions that actually bound a runaway run against
    # those two providers today.
    max_cost_usd: float = 10.0


class Spend(BaseModel):
    model_calls: int = 0
    total_tokens: int = 0
    wall_seconds: float = 0.0
    cost_usd: float = 0.0

    def exceeded(self, b: Budget) -> str | None:
        if self.model_calls >= b.max_model_calls:
            return "model_calls"
        if self.total_tokens >= b.max_total_tokens:
            return "tokens"
        if self.wall_seconds >= b.max_wall_seconds:
            return "wall_time"
        if self.cost_usd >= b.max_cost_usd:
            return "cost"
        return None
