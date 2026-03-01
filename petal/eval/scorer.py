"""
Scorer system — functions that grade an agent's output.

A scorer takes the agent's input/output/expected values and returns
a float from 0.0 (bad) to 1.0 (perfect).

Usage:
    @scorer(name="contains_city")
    def contains_city(output: str, expected: dict) -> float:
        return 1.0 if expected["city"].lower() in output.lower() else 0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class PetalScorer:
    """A scorer that can grade agent output."""
    name: str
    score_fn: Callable[..., float | Any]
    judge_model: str | None = None


def scorer(
    name: str,
    judge_model: str | None = None,
) -> Callable:
    """Decorator that turns a function into a PetalScorer.

    Example (rule-based):
        @scorer(name="contains_city")
        def contains_city(output: str, expected: dict) -> float:
            return 1.0 if expected["city"].lower() in output.lower() else 0.0

    Example (LLM-as-judge):
        @scorer(name="answer_relevance", judge_model="gpt-4o-mini")
        async def answer_relevance(input: str, output: str) -> float:
            ...
    """

    def decorator(fn: Callable) -> PetalScorer:
        return PetalScorer(
            name=name,
            score_fn=fn,
            judge_model=judge_model,
        )

    return decorator
