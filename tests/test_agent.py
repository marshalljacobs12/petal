"""Tests for the Agent runtime with mocked LLM responses.

All tests mock litellm.acompletion so no API keys are needed.
The mock returns fake ModelResponse objects that mimic LiteLLM's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from petal.agent import Agent, AgentResult, StepInfo, TokenUsage
from petal.tool import tool


# ── Fake LiteLLM response objects ────────────────────────────────────
# These mimic the shape of litellm's ModelResponse just enough
# for Agent.arun() to work.


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15


@dataclass
class FakeFunctionCall:
    name: str = ""
    arguments: str = "{}"


@dataclass
class FakeToolCall:
    id: str = "tc_1"
    type: str = "function"
    function: FakeFunctionCall | None = None


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None
    role: str = "assistant"

    def model_dump(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return result


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    usage: FakeUsage | None = None


def text_response(content: str) -> FakeResponse:
    """A simple LLM response with just text (no tool calls)."""
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(content=content))],
        usage=FakeUsage(),
    )


def tool_call_response(tool_name: str, args: str, tool_call_id: str = "tc_1") -> FakeResponse:
    """An LLM response that requests a tool call."""
    return FakeResponse(
        choices=[
            FakeChoice(
                message=FakeMessage(
                    content=None,
                    tool_calls=[
                        FakeToolCall(
                            id=tool_call_id,
                            function=FakeFunctionCall(name=tool_name, arguments=args),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=FakeUsage(),
    )


# ── Sample tools ─────────────────────────────────────────────────────

class CityParams(BaseModel):
    city: str = Field(description="City name")


@tool(name="get_weather", description="Get weather for a city")
async def get_weather(params: CityParams) -> dict:
    return {"temp": 72, "city": params.city}


# ── Test fixtures ────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return Agent(
        name="test-agent",
        model="gpt-4o",
        instructions="You are a helpful assistant.",
        tools=[get_weather],
    )




# ── Tests ────────────────────────────────────────────────────────────

class TestAgentInit:
    """Agent constructor should store configuration."""

    def test_defaults(self):
        a = Agent()
        assert a.name == "agent"
        assert a.model == "gpt-4o"
        assert a.tools == []
        assert a.max_steps == 10

    def test_custom_config(self):
        a = Agent(name="my-bot", model="anthropic/claude-sonnet-4-20250514", max_steps=5)
        assert a.name == "my-bot"
        assert a.model == "anthropic/claude-sonnet-4-20250514"
        assert a.max_steps == 5


class TestSimpleTextResponse:
    """Agent should return text when the LLM responds without tool calls."""

    @pytest.mark.asyncio
    async def test_returns_text(self, agent):
        mock = AsyncMock(return_value=text_response("Hello!"))
        with patch("litellm.acompletion", mock):
            result = await agent.arun("Hi")

        assert isinstance(result, AgentResult)
        assert result.text == "Hello!"

    @pytest.mark.asyncio
    async def test_tracks_usage(self, agent):
        mock = AsyncMock(return_value=text_response("Hi"))
        with patch("litellm.acompletion", mock):
            result = await agent.arun("Hi")

        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_records_llm_step(self, agent):
        mock = AsyncMock(return_value=text_response("Hi"))
        with patch("litellm.acompletion", mock):
            result = await agent.arun("Hi")

        assert len(result.steps) == 1
        assert result.steps[0].type == "llm_call"
        assert result.steps[0].name == "gpt-4o"


class TestMessageAssembly:
    """Agent should build the messages array correctly."""

    @pytest.mark.asyncio
    async def test_system_and_user_messages(self, agent):
        mock = AsyncMock(return_value=text_response("Hi"))
        with patch("litellm.acompletion", mock):
            await agent.arun("Hello")

        call_args = mock.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert messages[1] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_no_system_message_when_empty(self):
        agent = Agent(name="test", instructions="")
        mock = AsyncMock(return_value=text_response("Hi"))
        with patch("litellm.acompletion", mock):
            await agent.arun("Hello")

        messages = mock.call_args.kwargs["messages"]
        assert messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_tools_passed_to_llm(self, agent):
        mock = AsyncMock(return_value=text_response("Hi"))
        with patch("litellm.acompletion", mock):
            await agent.arun("Hello")

        tool_defs = mock.call_args.kwargs["tools"]
        assert len(tool_defs) == 1
        assert tool_defs[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_no_tools_passed_when_empty(self):
        agent = Agent(name="test", tools=[])
        mock = AsyncMock(return_value=text_response("Hi"))
        with patch("litellm.acompletion", mock):
            await agent.arun("Hello")

        assert mock.call_args.kwargs["tools"] is None


class TestToolCalling:
    """Agent should execute tools and feed results back to the LLM."""

    @pytest.mark.asyncio
    async def test_single_tool_call(self, agent):
        """LLM calls get_weather → agent executes it → LLM gives final text."""
        mock = AsyncMock(
            side_effect=[
                tool_call_response("get_weather", '{"city": "Tokyo"}'),
                text_response("It's 72°F in Tokyo."),
            ]
        )
        with patch("litellm.acompletion", mock):
            result = await agent.arun("Weather in Tokyo?")

        assert result.text == "It's 72°F in Tokyo."
        assert len(result.steps) == 3  # llm_call, tool_call, llm_call
        tool_steps = [s for s in result.steps if s.type == "tool_call"]
        assert len(tool_steps) == 1
        assert tool_steps[0].name == "get_weather"
        assert tool_steps[0].output == {"temp": 72, "city": "Tokyo"}

    @pytest.mark.asyncio
    async def test_tool_result_sent_back(self, agent):
        """After tool execution, the result should be in the messages for the next LLM call."""
        mock = AsyncMock(
            side_effect=[
                tool_call_response("get_weather", '{"city": "NYC"}'),
                text_response("72°F"),
            ]
        )
        with patch("litellm.acompletion", mock):
            await agent.arun("Weather in NYC?")

        # Second call should have the tool result in messages
        second_call_messages = mock.call_args_list[1].kwargs["messages"]
        tool_msg = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msg) == 1
        assert '"temp": 72' in tool_msg[0]["content"]

    @pytest.mark.asyncio
    async def test_unknown_tool_handled(self, agent):
        """If LLM calls a tool that doesn't exist, agent sends an error back."""
        mock = AsyncMock(
            side_effect=[
                tool_call_response("nonexistent_tool", "{}"),
                text_response("Sorry, I couldn't do that."),
            ]
        )
        with patch("litellm.acompletion", mock):
            result = await agent.arun("Do something impossible")

        assert result.text == "Sorry, I couldn't do that."
        tool_steps = [s for s in result.steps if s.type == "tool_call"]
        assert "Unknown tool" in str(tool_steps[0].output)

    @pytest.mark.asyncio
    async def test_usage_accumulated_across_steps(self, agent):
        """Token usage should sum across all LLM calls."""
        mock = AsyncMock(
            side_effect=[
                tool_call_response("get_weather", '{"city": "LA"}'),
                text_response("Sunny"),
            ]
        )
        with patch("litellm.acompletion", mock):
            result = await agent.arun("Weather?")

        # Two LLM calls, each with 10 prompt + 5 completion tokens
        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 10
        assert result.usage.total_tokens == 30


class TestMaxSteps:
    """Agent should stop after max_steps to prevent infinite loops."""

    @pytest.mark.asyncio
    async def test_stops_at_max_steps(self):
        agent = Agent(name="test", tools=[get_weather], max_steps=2)
        # LLM always requests a tool call — never produces a final answer
        mock = AsyncMock(
            return_value=tool_call_response("get_weather", '{"city": "X"}')
        )
        with patch("litellm.acompletion", mock):
            result = await agent.arun("Loop forever")

        assert "stopped after 2 steps" in result.text


class TestTraceAttached:
    """The result should include a TraceData object."""

    @pytest.mark.asyncio
    async def test_trace_on_result(self, agent):
        mock = AsyncMock(return_value=text_response("Hi"))
        with patch("litellm.acompletion", mock):
            result = await agent.arun("Hello")

        assert result.trace is not None
        assert result.trace.name == "test-agent"
        assert result.trace.input == "Hello"
        assert result.trace.output == "Hi"
