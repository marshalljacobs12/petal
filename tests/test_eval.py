"""Tests for the eval system: YAML loading, scorers, and runner integration."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from petal.agent import AgentResult, StepInfo, TokenUsage
from petal.eval.runner import (
    CaseResult,
    EvalCase,
    EvalResult,
    _call_scorer,
    _run_case,
    load_eval_file,
)
from petal.eval.scorer import PetalScorer, scorer
from petal.eval.scorers import cost_scorer, tool_accuracy


# ── YAML loading ─────────────────────────────────────────────────────

SAMPLE_YAML = """\
agent: weather-assistant
scorers: [tool_accuracy, cost]
cases:
  - name: basic query
    input: "What's the weather in Tokyo?"
    expected:
      tools_called: ["get_weather"]
    thresholds:
      tool_accuracy: 0.8
  - name: multi-city
    input: "Compare Tokyo and London weather"
    expected:
      tools_called: ["get_weather", "get_weather"]
    thresholds:
      tool_accuracy: 1.0
      cost: 0.5
"""


class TestLoadEvalFile:
    """load_eval_file() should parse YAML into EvalFile objects."""

    @pytest.fixture
    def yaml_path(self, tmp_path):
        path = tmp_path / "test.yaml"
        path.write_text(SAMPLE_YAML)
        return str(path)

    def test_agent_name(self, yaml_path):
        ef = load_eval_file(yaml_path)
        assert ef.agent == "weather-assistant"

    def test_scorer_names(self, yaml_path):
        ef = load_eval_file(yaml_path)
        assert ef.scorer_names == ["tool_accuracy", "cost"]

    def test_case_count(self, yaml_path):
        ef = load_eval_file(yaml_path)
        assert len(ef.cases) == 2

    def test_case_fields(self, yaml_path):
        ef = load_eval_file(yaml_path)
        case = ef.cases[0]
        assert case.name == "basic query"
        assert case.input == "What's the weather in Tokyo?"
        assert case.expected["tools_called"] == ["get_weather"]
        assert case.thresholds["tool_accuracy"] == 0.8

    def test_default_expected(self, tmp_path):
        """Cases without 'expected' should get an empty dict."""
        yaml = "agent: test\nscorers: []\ncases:\n  - name: bare\n    input: hi\n"
        path = tmp_path / "bare.yaml"
        path.write_text(yaml)
        ef = load_eval_file(str(path))
        assert ef.cases[0].expected == {}
        assert ef.cases[0].thresholds == {}


# ── Scorer decorator ────────────────────────────────────────────────

class TestScorerDecorator:
    """@scorer should produce a PetalScorer with correct fields."""

    def test_creates_scorer(self):
        @scorer(name="test_scorer")
        def my_scorer(output: str) -> float:
            return 1.0

        assert isinstance(my_scorer, PetalScorer)
        assert my_scorer.name == "test_scorer"

    def test_judge_model(self):
        @scorer(name="judge", judge_model="gpt-4o-mini")
        async def my_scorer(input: str, output: str) -> float:
            return 0.9

        assert my_scorer.judge_model == "gpt-4o-mini"


# ── Built-in scorers ────────────────────────────────────────────────

class TestToolAccuracyScorer:
    """tool_accuracy should check tool names and arguments."""

    def test_correct_tools(self):
        steps = [
            StepInfo(type="tool_call", name="get_weather"),
            StepInfo(type="llm_call", name="gpt-4o"),
        ]
        expected = {"tools_called": ["get_weather"]}
        score = tool_accuracy.score_fn(steps=steps, expected=expected)
        assert score == 1.0

    def test_wrong_tools(self):
        steps = [StepInfo(type="tool_call", name="search")]
        expected = {"tools_called": ["get_weather"]}
        score = tool_accuracy.score_fn(steps=steps, expected=expected)
        assert score == 0.0

    def test_no_expected(self):
        """No expected = nothing to check = perfect score."""
        steps = [StepInfo(type="tool_call", name="anything")]
        score = tool_accuracy.score_fn(steps=steps, expected={})
        assert score == 1.0

    def test_tool_args_match(self):
        steps = [
            StepInfo(type="tool_call", name="get_weather", input={"city": "Tokyo"}),
        ]
        expected = {
            "tools_called": ["get_weather"],
            "tool_args": {"get_weather": {"city": "Tokyo"}},
        }
        score = tool_accuracy.score_fn(steps=steps, expected=expected)
        assert score == 1.0  # 2 checks, both pass

    def test_tool_args_mismatch(self):
        steps = [
            StepInfo(type="tool_call", name="get_weather", input={"city": "NYC"}),
        ]
        expected = {
            "tools_called": ["get_weather"],
            "tool_args": {"get_weather": {"city": "Tokyo"}},
        }
        score = tool_accuracy.score_fn(steps=steps, expected=expected)
        assert score == 0.5  # 2 checks, only tools_called passes


class TestCostScorer:
    """cost_scorer should check if cost is under the threshold."""

    def test_under_threshold(self):
        score = cost_scorer.score_fn(cost=0.01, thresholds={"cost_usd": 0.05})
        assert score == 1.0

    def test_over_threshold(self):
        score = cost_scorer.score_fn(cost=0.10, thresholds={"cost_usd": 0.05})
        assert score == 0.0

    def test_no_threshold(self):
        """No cost_usd threshold = always pass."""
        score = cost_scorer.score_fn(cost=999.0, thresholds={})
        assert score == 1.0


# ── _call_scorer introspection ───────────────────────────────────────

class TestCallScorer:
    """_call_scorer() should pass only the kwargs each scorer function accepts."""

    def test_passes_matching_kwargs(self):
        @scorer(name="only_output")
        def only_output(output: str) -> float:
            return 1.0 if "good" in output else 0.0

        result = AgentResult(text="good answer", steps=[], usage=TokenUsage(), cost=0.0)
        case = EvalCase(name="t", input="q")
        score = _call_scorer(only_output, result, case)
        assert score == 1.0

    def test_ignores_extra_params(self):
        """Scorer that only needs 'cost' shouldn't receive 'steps' etc."""
        @scorer(name="cheap")
        def cheap(cost: float) -> float:
            return 1.0 if cost < 1.0 else 0.0

        result = AgentResult(text="x", steps=[], usage=TokenUsage(), cost=0.5)
        case = EvalCase(name="t", input="q")
        score = _call_scorer(cheap, result, case)
        assert score == 1.0


# ── Runner integration ───────────────────────────────────────────────

class TestRunCase:
    """_run_case() should call the agent, score the result, and check thresholds."""

    @pytest.mark.asyncio
    async def test_passing_case(self):
        from petal.agent import Agent

        agent = Agent(name="test", tools=[])

        # Mock the agent's arun to return a known result
        fake_result = AgentResult(
            text="It's 72°F in Tokyo.",
            steps=[StepInfo(type="tool_call", name="get_weather")],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost=0.001,
        )
        with patch.object(agent, "arun", new=AsyncMock(return_value=fake_result)):
            case = EvalCase(
                name="basic",
                input="Weather in Tokyo?",
                expected={"tools_called": ["get_weather"]},
                thresholds={"tool_accuracy": 0.8},
            )
            case_result = await _run_case(agent, case, [tool_accuracy])

        assert isinstance(case_result, CaseResult)
        assert case_result.passed is True
        assert case_result.scores["tool_accuracy"] == 1.0

    @pytest.mark.asyncio
    async def test_failing_case(self):
        from petal.agent import Agent

        agent = Agent(name="test", tools=[])

        fake_result = AgentResult(
            text="I don't know.",
            steps=[],  # No tools called
            usage=TokenUsage(),
            cost=0.0,
        )
        with patch.object(agent, "arun", new=AsyncMock(return_value=fake_result)):
            case = EvalCase(
                name="fail",
                input="Weather?",
                expected={"tools_called": ["get_weather"]},
                thresholds={"tool_accuracy": 0.8},
            )
            case_result = await _run_case(agent, case, [tool_accuracy])

        assert case_result.passed is False
        assert case_result.scores["tool_accuracy"] == 0.0
