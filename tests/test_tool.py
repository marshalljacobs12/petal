"""Tests for the @tool decorator, PetalTool, and JSON Schema generation."""

import asyncio

import pytest
from pydantic import BaseModel, Field

from petal.tool import PetalTool, tool


# --- Fixtures: sample tools ---

class WeatherParams(BaseModel):
    city: str = Field(description="City name")
    units: str = Field(default="fahrenheit", description="Temperature units")


@tool(name="get_weather", description="Get current weather")
async def get_weather(params: WeatherParams) -> dict:
    return {"temp": 72, "city": params.city, "units": params.units}


class AddParams(BaseModel):
    a: int
    b: int


@tool(name="add", description="Add two numbers")
def add_sync(params: AddParams) -> dict:
    """A synchronous tool — the decorator handles both sync and async."""
    return {"result": params.a + params.b}


# --- Tests ---

class TestToolDecorator:
    """The @tool decorator should produce a valid PetalTool."""

    def test_returns_petal_tool(self):
        assert isinstance(get_weather, PetalTool)

    def test_name_and_description(self):
        assert get_weather.name == "get_weather"
        assert get_weather.description == "Get current weather"

    def test_parameters_model_stored(self):
        assert get_weather.parameters_model is WeatherParams

    def test_default_timeout(self):
        assert get_weather.timeout_seconds == 30.0

    def test_custom_timeout(self):
        @tool(name="slow", description="Slow tool", timeout_seconds=5.0)
        async def slow_tool(params: WeatherParams) -> dict:
            return {}

        assert slow_tool.timeout_seconds == 5.0


class TestJsonSchema:
    """The generated JSON Schema should match the Pydantic model."""

    def test_schema_has_properties(self):
        schema = get_weather.parameters_schema
        assert "properties" in schema
        assert "city" in schema["properties"]
        assert "units" in schema["properties"]

    def test_required_fields(self):
        schema = get_weather.parameters_schema
        assert "city" in schema["required"]

    def test_field_descriptions(self):
        schema = get_weather.parameters_schema
        assert schema["properties"]["city"]["description"] == "City name"

    def test_title_stripped(self):
        """The 'title' key is stripped so it doesn't confuse the LLM."""
        assert "title" not in get_weather.parameters_schema


class TestLitellmFormat:
    """to_litellm_tool() should return the exact shape LLM APIs expect."""

    def test_top_level_type(self):
        fmt = get_weather.to_litellm_tool()
        assert fmt["type"] == "function"

    def test_function_block(self):
        fmt = get_weather.to_litellm_tool()
        fn = fmt["function"]
        assert fn["name"] == "get_weather"
        assert fn["description"] == "Get current weather"
        assert "properties" in fn["parameters"]


class TestToolExecution:
    """execute() should validate params and handle async/sync/errors."""

    @pytest.mark.asyncio
    async def test_async_tool_returns_result(self):
        params = WeatherParams(city="Tokyo")
        result = await get_weather.execute(params)
        assert result == {"temp": 72, "city": "Tokyo", "units": "fahrenheit"}

    @pytest.mark.asyncio
    async def test_sync_tool_returns_result(self):
        params = AddParams(a=2, b=3)
        result = await add_sync.execute(params)
        assert result == {"result": 5}

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        @tool(name="sleepy", description="Too slow", timeout_seconds=0.01)
        async def sleepy(params: WeatherParams) -> dict:
            await asyncio.sleep(5)
            return {}

        with pytest.raises(TimeoutError, match="timed out"):
            await sleepy.execute(WeatherParams(city="Slow"))

    @pytest.mark.asyncio
    async def test_execution_error_wrapped(self):
        @tool(name="broken", description="Always fails")
        async def broken(params: WeatherParams) -> dict:
            raise ValueError("bad input")

        with pytest.raises(RuntimeError, match="failed"):
            await broken.execute(WeatherParams(city="X"))


class TestToolValidation:
    """The decorator should reject invalid function signatures."""

    def test_no_params_raises(self):
        with pytest.raises(ValueError, match="at least one parameter"):

            @tool(name="bad", description="No params")
            async def bad_tool() -> dict:
                return {}

    def test_non_pydantic_param_raises(self):
        with pytest.raises(ValueError, match="Pydantic BaseModel"):

            @tool(name="bad", description="Wrong type")
            async def bad_tool(x: str) -> dict:
                return {}
