"""
Weather Agent — example showing how to build an agent with Petal.

This agent has one tool (get_weather) and answers weather questions.
Run it with a real API key to see it in action:

    export OPENAI_API_KEY=sk-...
    python examples/weather_agent/agent.py

Or run its evals:

    python examples/weather_agent/agent.py --eval
"""

from __future__ import annotations

import argparse
import sys

from pydantic import BaseModel, Field

from petal import Agent, tool


# ── Tool ─────────────────────────────────────────────────────────────

class WeatherParams(BaseModel):
    city: str = Field(description="City name to get weather for")
    units: str = Field(
        default="fahrenheit",
        description="Temperature units: 'fahrenheit' or 'celsius'",
    )


@tool(name="get_weather", description="Get the current weather for a city")
async def get_weather(params: WeatherParams) -> dict:
    """Fake weather API — returns hardcoded data for demo purposes.

    In a real agent, this would call a weather API like OpenWeatherMap.
    """
    weather_data = {
        "tokyo": {"temp": 72, "condition": "Partly cloudy", "humidity": 65},
        "london": {"temp": 55, "condition": "Rainy", "humidity": 80},
        "new york": {"temp": 68, "condition": "Sunny", "humidity": 45},
        "paris": {"temp": 61, "condition": "Overcast", "humidity": 70},
    }

    city_lower = params.city.lower()
    data = weather_data.get(city_lower, {"temp": 70, "condition": "Clear", "humidity": 50})

    if params.units == "celsius":
        data["temp"] = round((data["temp"] - 32) * 5 / 9)

    return {
        "city": params.city,
        "units": params.units,
        **data,
    }


# ── Agent ────────────────────────────────────────────────────────────

agent = Agent(
    name="weather-assistant",
    model="gpt-4o-mini",
    instructions=(
        "You are a helpful weather assistant. Use the get_weather tool to "
        "answer weather questions. Always include the temperature and "
        "conditions in your response. If the user asks about multiple cities, "
        "call get_weather for each one."
    ),
    tools=[get_weather],
)


# ── CLI ──────────────────────────────────────────────────────────────

def run_agent():
    """Run the agent with a sample query."""
    result = agent.run("What's the weather like in Tokyo?")
    print(f"Response: {result.text}")
    print(f"Steps: {len(result.steps)}")
    print(f"Tokens: {result.usage.total_tokens}")
    print(f"Cost: ${result.cost:.4f}")


def run_evals():
    """Run the eval suite for this agent."""
    import os

    from petal.eval.runner import run_eval
    from petal.eval.scorers import cost_scorer, tool_accuracy

    eval_path = os.path.join(os.path.dirname(__file__), "evals.yaml")
    result = run_eval(eval_path, agent=agent, scorers=[tool_accuracy, cost_scorer])

    print(f"Results: {result.passed}/{len(result.cases)} passed\n")
    for case in result.cases:
        status = "PASS" if case.passed else "FAIL"
        print(f"  [{status}] {case.name}")
        for scorer_name, score in case.scores.items():
            threshold = case.thresholds.get(scorer_name)
            if threshold is not None:
                print(f"         {scorer_name}: {score:.2f} (threshold: {threshold})")
            else:
                print(f"         {scorer_name}: {score:.2f}")

    sys.exit(0 if result.failed == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Agent Example")
    parser.add_argument("--eval", action="store_true", help="Run evals instead of the agent")
    args = parser.parse_args()

    if args.eval:
        run_evals()
    else:
        run_agent()
