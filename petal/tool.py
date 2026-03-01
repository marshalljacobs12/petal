"""
Tool system for giving agents abilities.

Tools are functions that an agent can call — search the web, query a DB, etc.
The LLM decides when to call a tool and what arguments to pass.

Usage:
    from petal import tool
    from pydantic import BaseModel, Field

    class WeatherParams(BaseModel):
        city: str = Field(description="City name")

    @tool(name="get_weather", description="Get weather for a city")
    async def get_weather(params: WeatherParams) -> dict:
        return {"temp": 72, "city": params.city}
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel


@dataclass
class PetalTool:
    """A tool that an agent can call.

    Holds everything the framework needs:
    - name/description: sent to the LLM so it knows what tools are available
    - parameters_schema: JSON Schema dict describing the arguments
    - parameters_model: the Pydantic class, used to validate args before calling
    - execute: the actual function to run
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    parameters_model: type[BaseModel]
    execute: Callable[..., Any]
    timeout_seconds: float = 30.0

    def to_litellm_tool(self) -> dict[str, Any]:
        """Convert to the format LiteLLM/OpenAI expects for tool definitions.

        LLM APIs expect tools in this shape:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": { ...json schema... }
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


def tool(
    name: str,
    description: str,
    timeout_seconds: float = 30.0,
) -> Callable:
    """Decorator that turns an async function into a PetalTool.

    The decorated function's first parameter must be typed as a Pydantic BaseModel.
    We inspect that type hint to auto-generate the JSON Schema.

    Example:
        @tool(name="get_weather", description="Get weather for a city")
        async def get_weather(params: WeatherParams) -> dict:
            return {"temp": 72}
    """

    def decorator(fn: Callable) -> PetalTool:

        # --- Step 1: Find the Pydantic model from the function's type hints ---
        hints = get_type_hints(fn)
        param_names = list(inspect.signature(fn).parameters.keys())
        if not param_names:
            raise ValueError(
                f"Tool function '{name}' must have at least one parameter (a Pydantic model)"
            )

        first_param_type = hints.get(param_names[0])
        if first_param_type is None or not (
            isinstance(first_param_type, type) and issubclass(first_param_type, BaseModel)
        ):
            raise ValueError(
                f"Tool function '{name}': first parameter must be typed as a Pydantic BaseModel, "
                f"got {first_param_type}"
            )

        # --- Step 2: Generate JSON Schema from the Pydantic model ---
        schema = first_param_type.model_json_schema()
        schema.pop("title", None)

        # --- Step 3: Wrap the function with timeout + error handling ---
        async def wrapped_execute(params: BaseModel) -> Any:
            try:
                result = fn(params)
                if inspect.isawaitable(result):
                    return await asyncio.wait_for(result, timeout=timeout_seconds)
                return result
            except asyncio.TimeoutError:
                raise TimeoutError(f'Tool "{name}" timed out after {timeout_seconds}s')
            except Exception as e:
                raise RuntimeError(f'Tool "{name}" failed: {e}') from e

        # --- Step 4: Build and return the PetalTool ---
        return PetalTool(
            name=name,
            description=description,
            parameters_schema=schema,
            parameters_model=first_param_type,
            execute=wrapped_execute,
            timeout_seconds=timeout_seconds,
        )

    return decorator
