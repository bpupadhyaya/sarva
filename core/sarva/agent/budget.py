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
    max_cost_usd: float = 10.0  # irrelevant when only local/mock models run


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
