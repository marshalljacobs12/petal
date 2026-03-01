"""
Cost scorer — checks if the agent stayed under a cost threshold.

Deterministic, fast, free (no LLM call needed).
"""

from __future__ import annotations

from petal.eval.scorer import scorer


@scorer(name="cost")
def cost_scorer(cost: float, thresholds: dict) -> float:
    """Score based on whether the agent stayed under a cost threshold.

    Looks for thresholds["cost_usd"]. Returns 1.0 if under, 0.0 if over.
    """
    max_cost = thresholds.get("cost_usd")
    if max_cost is None:
        return 1.0

    return 1.0 if cost <= max_cost else 0.0
