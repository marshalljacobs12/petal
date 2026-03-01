"""
Agent runtime — the core orchestration loop.

Takes a user message, calls the LLM, handles tool calls,
and returns a final response.

Usage:
    from petal import Agent
    agent = Agent(name="assistant", model="gpt-4o", tools=[get_weather])
    result = agent.run("What's the weather in Tokyo?")
    print(result.text)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import litellm

from petal.tool import PetalTool


# --- Result types ---

@dataclass
class TokenUsage:
    """Token counts from the LLM calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class StepInfo:
    """One step in the agent loop (either an LLM call or a tool call)."""
    type: str            # "llm_call" or "tool_call"
    name: str            # model name or tool name
    input: Any = None
    output: Any = None
    tokens: TokenUsage | None = None


@dataclass
class AgentResult:
    """The return value of agent.run()."""
    text: str                              # The final text response
    steps: list[StepInfo] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost: float = 0.0
    # TODO (Phase 2): trace: TraceData = None


# --- Agent class ---

class Agent:
    """An AI agent that can use tools to accomplish tasks.

    Args:
        name: Human-readable name for this agent (used in traces).
        model: LiteLLM model string, e.g. "gpt-4o", "anthropic/claude-sonnet-4-20250514".
        instructions: System prompt — tells the LLM how to behave.
        tools: List of PetalTool objects the agent can call.
        max_steps: Safety limit on loop iterations to prevent infinite loops.
    """

    def __init__(
        self,
        *,                          # Forces all args to be keyword-only
        name: str = "agent",
        model: str = "gpt-4o",
        instructions: str = "",
        tools: list[PetalTool] | None = None,
        max_steps: int = 10,
    ):
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = tools or []
        self.max_steps = max_steps

    def run(self, message: str) -> AgentResult:
        """Run the agent synchronously. Internally calls the async version."""
        return asyncio.run(self.arun(message))

    async def arun(self, message: str) -> AgentResult:
        """Run the agent asynchronously. This is where the real work happens."""

        # --- Build the initial messages list ---
        messages: list[dict[str, Any]] = []

        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})

        messages.append({"role": "user", "content": message})

        # --- Build tool definitions for the LLM ---
        tool_defs = [t.to_litellm_tool() for t in self.tools] if self.tools else None

        # Map tool names to PetalTool objects for quick lookup when executing
        tool_map = {t.name: t for t in self.tools}

        # --- Agent loop ---
        steps: list[StepInfo] = []
        total_usage = TokenUsage()
        total_cost = 0.0

        for step_num in range(self.max_steps):

            # --- Call the LLM ---
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                tools=tool_defs,
            )

            # Extract the assistant's message from the response
            choice = response.choices[0]
            assistant_message = choice.message

            # Track token usage
            if response.usage:
                step_tokens = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    total_tokens=response.usage.total_tokens or 0,
                )
                total_usage.prompt_tokens += step_tokens.prompt_tokens
                total_usage.completion_tokens += step_tokens.completion_tokens
                total_usage.total_tokens += step_tokens.total_tokens
            else:
                step_tokens = None

            # Track cost — litellm provides this via completion_cost()
            try:
                step_cost = litellm.completion_cost(completion_response=response)
                total_cost += step_cost
            except Exception:
                step_cost = 0.0

            steps.append(StepInfo(
                type="llm_call",
                name=self.model,
                input=messages[-1],
                output=assistant_message,
                tokens=step_tokens,
            ))

            # --- Check if the LLM wants to call tools ---
            tool_calls = assistant_message.tool_calls

            if not tool_calls:
                # No tool calls — the LLM is done, it produced a text response.
                return AgentResult(
                    text=assistant_message.content or "",
                    steps=steps,
                    usage=total_usage,
                    cost=total_cost,
                )

            # --- Execute tool calls ---
            # Append the assistant's message (with tool calls) to the conversation
            messages.append(assistant_message.model_dump())

            for tc in tool_calls:
                tool_name = tc.function.name
                tool_args_str = tc.function.arguments  # JSON string of arguments
                tool_call_id = tc.id

                # Look up the tool
                petal_tool = tool_map.get(tool_name)

                if not petal_tool:
                    # LLM called a tool that doesn't exist — send error back
                    error_msg = f"Unknown tool: {tool_name}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"error": error_msg}),
                    })
                    steps.append(StepInfo(type="tool_call", name=tool_name, output=error_msg))
                    continue

                # Parse arguments JSON into the Pydantic model
                try:
                    args_dict = json.loads(tool_args_str)
                    params = petal_tool.parameters_model(**args_dict)
                except Exception as e:
                    error_msg = f"Invalid arguments for {tool_name}: {e}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"error": error_msg}),
                    })
                    steps.append(StepInfo(type="tool_call", name=tool_name, output=error_msg))
                    continue

                # Execute the tool
                try:
                    result = await petal_tool.execute(params)
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                except Exception as e:
                    result = {"error": str(e)}
                    result_str = json.dumps(result)

                # Append tool result to messages so the LLM can see it
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_str,
                })

                steps.append(StepInfo(
                    type="tool_call",
                    name=tool_name,
                    input=args_dict,
                    output=result,
                ))

            # Loop continues — LLM will see the tool results and decide next action.

        # Hit max_steps without the LLM finishing
        return AgentResult(
            text=f"Agent stopped after {self.max_steps} steps without completing.",
            steps=steps,
            usage=total_usage,
            cost=total_cost,
        )
