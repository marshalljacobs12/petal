"""
Eval runner — loads YAML test cases, runs them against an agent, scores results.

Usage:
    results = run_eval("evals/weather-agent.yaml", agent=my_agent, scorers=[...])
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import yaml

from petal.agent import Agent, AgentResult
from petal.eval.scorer import PetalScorer


# --- Result types ---

@dataclass
class CaseResult:
    """Result of running one test case."""
    name: str
    input: str
    output: str
    scores: dict[str, float]        # scorer_name -> score
    thresholds: dict[str, float]    # scorer_name -> minimum score required
    passed: bool                    # True if all scores meet their thresholds
    duration_ms: float
    cost: float


@dataclass
class EvalResult:
    """Result of running an entire eval file."""
    eval_file: str
    cases: list[CaseResult]
    passed: int
    failed: int
    duration_ms: float


# --- YAML loading ---

@dataclass
class EvalCase:
    """One test case parsed from YAML."""
    name: str
    input: str
    expected: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalFile:
    """A parsed YAML eval file."""
    agent: str
    scorer_names: list[str]
    cases: list[EvalCase]


def load_eval_file(path: str) -> EvalFile:
    """Parse a YAML eval file into an EvalFile object.

    Expected YAML format:
        agent: weather-assistant
        scorers: [tool_accuracy, cost]
        cases:
          - name: basic query
            input: "What's the weather in Tokyo?"
            expected:
              tools_called: ["get_weather"]
            thresholds:
              tool_accuracy: 0.8
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    cases = []
    for case_data in data.get("cases", []):
        cases.append(EvalCase(
            name=case_data["name"],
            input=case_data["input"],
            expected=case_data.get("expected", {}),
            thresholds=case_data.get("thresholds", {}),
        ))

    return EvalFile(
        agent=data.get("agent", "unknown"),
        scorer_names=data.get("scorers", []),
        cases=cases,
    )


# --- Runner ---

async def _run_case(
    agent: Agent,
    case: EvalCase,
    scorers: list[PetalScorer],
) -> CaseResult:
    """Run one test case: call the agent, then score the result."""
    start = time.time()

    # Run the agent
    result: AgentResult = await agent.arun(case.input)

    # Score the result with each scorer
    scores: dict[str, float] = {}
    for s in scorers:
        score = _call_scorer(s, result, case)
        if inspect.isawaitable(score):
            score = await score
        scores[s.name] = float(score)

    duration_ms = (time.time() - start) * 1000

    # Check if all thresholds are met
    passed = True
    for scorer_name, threshold in case.thresholds.items():
        if scorer_name in scores and scores[scorer_name] < threshold:
            passed = False
            break

    return CaseResult(
        name=case.name,
        input=case.input,
        output=result.text,
        scores=scores,
        thresholds=case.thresholds,
        passed=passed,
        duration_ms=duration_ms,
        cost=result.cost,
    )


def _call_scorer(
    s: PetalScorer,
    result: AgentResult,
    case: EvalCase,
) -> Any:
    """Call a scorer function, passing whichever keyword arguments it accepts.

    Different scorers need different data:
    - tool_accuracy needs: steps, expected
    - cost_scorer needs: cost, thresholds
    - answer_relevance needs: input, output
    - custom scorers might need any combination

    We inspect the function's parameter names and pass only what it asks for.
    """
    # Build a dict of all available data the scorer might want
    available = {
        "input": case.input,
        "output": result.text,
        "expected": case.expected,
        "thresholds": case.thresholds,
        "steps": result.steps,
        "cost": result.cost,
        "usage": result.usage,
        "result": result,
        "judge_model": s.judge_model,
    }

    # Inspect which parameters the scorer function actually accepts
    sig = inspect.signature(s.score_fn)
    kwargs = {}
    for param_name in sig.parameters:
        if param_name in available:
            kwargs[param_name] = available[param_name]

    return s.score_fn(**kwargs)


def run_eval(
    eval_path: str,
    agent: Agent,
    scorers: list[PetalScorer],
) -> EvalResult:
    """Run an eval file synchronously. Wraps the async version."""
    return asyncio.run(arun_eval(eval_path, agent, scorers))


async def arun_eval(
    eval_path: str,
    agent: Agent,
    scorers: list[PetalScorer],
) -> EvalResult:
    """Run all test cases in an eval file against an agent."""
    start = time.time()
    eval_file = load_eval_file(eval_path)

    # Filter scorers to only those requested in the YAML file
    requested = set(eval_file.scorer_names)
    active_scorers = [s for s in scorers if s.name in requested]

    # Run each test case
    case_results: list[CaseResult] = []
    for case in eval_file.cases:
        case_result = await _run_case(agent, case, active_scorers)
        case_results.append(case_result)

    duration_ms = (time.time() - start) * 1000
    passed = sum(1 for c in case_results if c.passed)
    failed = len(case_results) - passed

    eval_result = EvalResult(
        eval_file=eval_path,
        cases=case_results,
        passed=passed,
        failed=failed,
        duration_ms=duration_ms,
    )

    # Save to store
    from petal.store import get_default_store
    store = get_default_store()
    if store:
        store.save_eval_run({
            "id": str(uuid4()),
            "eval_file": eval_path,
            "start_time": start * 1000,
            "end_time": time.time() * 1000,
            "passed": passed,
            "failed": failed,
            "results": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "scores": c.scores,
                    "cost": c.cost,
                }
                for c in case_results
            ],
        })

    return eval_result
