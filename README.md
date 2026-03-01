# Petal

A Python agent framework with built-in evals, tracing, and guardrails.

Petal gives you a tool-calling agent runtime, automatic tracing of every LLM call, and a YAML-based eval system — all in a single self-hosted package. No external services, no dashboard subscriptions, no vendor lock-in.

## Quickstart

```bash
pip install -e .
export OPENAI_API_KEY=sk-...
```

```python
from pydantic import BaseModel, Field
from petal import Agent, tool

class WeatherParams(BaseModel):
    city: str = Field(description="City name")

@tool(name="get_weather", description="Get weather for a city")
async def get_weather(params: WeatherParams) -> dict:
    return {"temp": 72, "city": params.city}

agent = Agent(
    name="weather-assistant",
    model="gpt-4o-mini",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)

result = agent.run("What's the weather in Tokyo?")
print(result.text)    # "It's 72°F in Tokyo."
print(result.cost)    # 0.0003
```

## Features

### Tools

The `@tool` decorator turns any function into something an LLM can call. Define parameters with a Pydantic model and Petal handles JSON Schema generation, argument validation, timeout, and error wrapping.

```python
from pydantic import BaseModel, Field
from petal import tool

class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=5, description="Max results")

@tool(name="search", description="Search the knowledge base", timeout_seconds=10.0)
async def search(params: SearchParams) -> dict:
    results = await my_search_api(params.query, params.limit)
    return {"results": results}
```

Both sync and async functions work. The decorator produces a `PetalTool` with:
- `name` / `description` — sent to the LLM
- `parameters_schema` — auto-generated JSON Schema
- `execute()` — the wrapped function with timeout and error handling

### Agent

The `Agent` class runs an LLM tool-calling loop. It sends your message to the model, executes any tool calls, feeds results back, and repeats until the LLM produces a final text response.

```python
agent = Agent(
    name="my-agent",
    model="gpt-4o",           # Any model LiteLLM supports (100+ providers)
    instructions="Be helpful.",
    tools=[search, get_weather],
    max_steps=10,              # Safety limit on loop iterations
)

result = agent.run("Find articles about Python")
print(result.text)             # Final response
print(result.steps)            # List of StepInfo (llm_call / tool_call)
print(result.usage)            # TokenUsage (prompt, completion, total)
print(result.cost)             # Total cost in USD
```

### Tracing

Every `agent.run()` automatically records a trace with spans for each LLM call and tool execution. Traces are persisted to SQLite (`./petal.db`) for inspection.

```python
from petal import get_traces, get_trace

# List recent traces
traces = get_traces(limit=10)
for t in traces:
    print(f"{t.name}: {t.total_tokens} tokens, ${t.total_cost:.4f}")

# Inspect a specific trace
trace = get_trace(trace_id)
for span in trace.spans:
    print(f"  [{span.type}] {span.name} — {span.status}")
```

Tool call spans are children of the LLM call span that triggered them, creating a tree structure for debugging.

### Evals

Write test cases in YAML to validate your agent's behavior:

```yaml
# evals/weather.yaml
agent: weather-assistant
scorers: [tool_accuracy, cost]

cases:
  - name: single city query
    input: "What's the weather in Tokyo?"
    expected:
      tools_called: ["get_weather"]
    thresholds:
      tool_accuracy: 1.0
      cost_usd: 0.01
```

Run evals in Python:

```python
from petal.eval.runner import run_eval
from petal.eval.scorers import tool_accuracy, cost_scorer

result = run_eval("evals/weather.yaml", agent=agent, scorers=[tool_accuracy, cost_scorer])
print(f"{result.passed}/{len(result.cases)} passed")
```

**Built-in scorers:**

| Scorer | What it checks | Cost |
|---|---|---|
| `tool_accuracy` | Correct tools called with correct arguments | Free |
| `cost` | Agent stayed under a USD cost threshold | Free |
| `answer_relevance` | LLM-as-judge relevance rating (0.0–1.0) | ~$0.001/case |

**Custom scorers:**

```python
from petal.eval.scorer import scorer

@scorer(name="mentions_temperature")
def mentions_temperature(output: str) -> float:
    return 1.0 if any(w in output.lower() for w in ["°f", "°c", "degrees"]) else 0.0
```

Scorers receive only the keyword arguments they ask for. Available kwargs: `input`, `output`, `expected`, `thresholds`, `steps`, `cost`, `usage`, `result`, `judge_model`.

## Project Structure

```
petal/
├── __init__.py          # Public API exports
├── agent.py             # Agent class — LLM tool-calling loop
├── tool.py              # @tool decorator, PetalTool dataclass
├── tracer.py            # Tracer, Span, TraceData — records agent runs
├── store.py             # SQLite persistence for traces and eval results
├── cli.py               # `petal eval` CLI (click-based)
└── eval/
    ├── runner.py         # YAML eval loader + test executor
    ├── scorer.py         # @scorer decorator, PetalScorer dataclass
    └── scorers/
        ├── tool_accuracy.py   # Checks correct tool usage
        ├── cost.py            # Checks cost thresholds
        └── relevance.py       # LLM-as-judge relevance scorer

examples/
└── weather_agent/
    ├── agent.py          # Working demo agent
    └── evals.yaml        # Eval test cases

tests/                    # pytest suite (86 tests, mocked LLM)
```

## Architecture

```
User Code  →  Agent Runtime  →  Tracer  →  SQLite Store
               (LiteLLM loop)                    ↑
                                         Eval Runner reads from here
```

- **LiteLLM** — Model-agnostic LLM calls (100+ providers via `acompletion()`)
- **Pydantic** — Tool parameter schemas and validation
- **SQLite** — Built-in storage for traces and eval results (`./petal.db`)

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
