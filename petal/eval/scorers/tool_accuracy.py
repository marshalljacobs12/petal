"""
Tool accuracy scorer — checks if the agent called the right tools
with the right arguments.

Deterministic, fast, free (no LLM call needed).
"""

from __future__ import annotations

from petal.eval.scorer import scorer


@scorer(name="tool_accuracy")
def tool_accuracy(steps: list, expected: dict) -> float:
    """Score how accurately the agent used its tools.

    Checks two things from the expected dict:
    - tools_called: list of tool names that should have been called
    - tool_args: dict mapping tool name -> expected arguments

    Returns 1.0 if all checks pass, partial credit for partial matches.
    """
    if not expected:
        return 1.0

    score = 0.0
    checks = 0

    # Check which tools were called
    actual_tool_names = [s.name for s in steps if s.type == "tool_call"]
    expected_tools = expected.get("tools_called")

    if expected_tools is not None:
        checks += 1
        if sorted(actual_tool_names) == sorted(expected_tools):
            score += 1.0

    # Check tool arguments
    expected_args = expected.get("tool_args", {})
    for tool_name, expected_tool_args in expected_args.items():
        checks += 1
        # Find the actual call to this tool
        actual_call = next(
            (s for s in steps if s.type == "tool_call" and s.name == tool_name),
            None,
        )
        if actual_call and actual_call.input == expected_tool_args:
            score += 1.0

    return score / checks if checks > 0 else 1.0
