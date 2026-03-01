"""Tests for SQLite storage: trace round-trip, eval run persistence, filtering."""

import pytest

from petal.store import Store
from petal.tracer import SpanData, TraceData


# --- Helpers ---

def make_trace(
    trace_id: str = "trace-1",
    name: str = "test-agent",
    status: str = "ok",
    total_tokens: int = 100,
    total_cost: float = 0.01,
    spans: list[SpanData] | None = None,
) -> TraceData:
    """Build a TraceData with sensible defaults for testing."""
    return TraceData(
        id=trace_id,
        name=name,
        input="hello",
        output="world",
        start_time=1000.0,
        end_time=2000.0,
        total_tokens=total_tokens,
        total_cost=total_cost,
        metadata={"env": "test"},
        spans=spans or [],
        status=status,
    )


def make_span(
    span_id: str = "span-1",
    trace_id: str = "trace-1",
    parent_id: str | None = None,
    span_type: str = "llm_call",
    name: str = "gpt-4o",
    status: str = "ok",
) -> SpanData:
    """Build a SpanData with sensible defaults."""
    return SpanData(
        id=span_id,
        trace_id=trace_id,
        parent_id=parent_id,
        type=span_type,
        name=name,
        input={"messages": [{"role": "user", "content": "hi"}]},
        output={"content": "hello"},
        start_time=1000.0,
        end_time=1500.0,
        tokens={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        cost=0.001,
        status=status,
    )


@pytest.fixture
def store():
    """In-memory SQLite store — fast, no disk cleanup needed."""
    s = Store(db_path=":memory:")
    yield s
    s.close()


# --- Trace round-trip ---

class TestTraceRoundTrip:
    """save_trace() → get_trace() should preserve all fields."""

    def test_save_and_load(self, store: Store):
        trace = make_trace()
        store.save_trace(trace)
        loaded = store.get_trace("trace-1")

        assert loaded is not None
        assert loaded.id == "trace-1"
        assert loaded.name == "test-agent"
        assert loaded.input == "hello"
        assert loaded.output == "world"
        assert loaded.total_tokens == 100
        assert loaded.total_cost == 0.01
        assert loaded.metadata == {"env": "test"}
        assert loaded.status == "ok"

    def test_missing_trace_returns_none(self, store: Store):
        assert store.get_trace("nonexistent") is None

    def test_spans_round_trip(self, store: Store):
        spans = [
            make_span(span_id="s1", trace_id="trace-1"),
            make_span(
                span_id="s2",
                trace_id="trace-1",
                parent_id="s1",
                span_type="tool_call",
                name="get_weather",
            ),
        ]
        trace = make_trace(spans=spans)
        store.save_trace(trace)
        loaded = store.get_trace("trace-1")

        assert loaded is not None
        assert len(loaded.spans) == 2
        assert loaded.spans[0].id == "s1"
        assert loaded.spans[0].type == "llm_call"
        assert loaded.spans[1].parent_id == "s1"
        assert loaded.spans[1].type == "tool_call"

    def test_span_input_output_deserialized(self, store: Store):
        """Span input/output are stored as JSON strings, should be dicts on load."""
        span = make_span()
        trace = make_trace(spans=[span])
        store.save_trace(trace)
        loaded = store.get_trace("trace-1")

        assert loaded is not None
        assert loaded.spans[0].input == {"messages": [{"role": "user", "content": "hi"}]}
        assert loaded.spans[0].output == {"content": "hello"}

    def test_span_tokens_deserialized(self, store: Store):
        span = make_span()
        trace = make_trace(spans=[span])
        store.save_trace(trace)
        loaded = store.get_trace("trace-1")

        assert loaded is not None
        assert loaded.spans[0].tokens == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_span_with_error(self, store: Store):
        span = make_span(status="error")
        span = SpanData(
            id="err-span",
            trace_id="trace-1",
            parent_id=None,
            type="tool_call",
            name="broken",
            input="args",
            output=None,
            start_time=1000.0,
            end_time=1500.0,
            status="error",
            error="tool crashed",
        )
        trace = make_trace(spans=[span])
        store.save_trace(trace)
        loaded = store.get_trace("trace-1")

        assert loaded is not None
        assert loaded.spans[0].status == "error"
        assert loaded.spans[0].error == "tool crashed"


# --- Trace listing and filtering ---

class TestGetTraces:
    """get_traces() should list, filter, and limit results."""

    def test_list_traces(self, store: Store):
        store.save_trace(make_trace(trace_id="t1", name="agent-a"))
        store.save_trace(make_trace(trace_id="t2", name="agent-b"))
        traces = store.get_traces()
        assert len(traces) == 2

    def test_limit(self, store: Store):
        for i in range(5):
            store.save_trace(make_trace(trace_id=f"t{i}"))
        traces = store.get_traces(limit=3)
        assert len(traces) == 3

    def test_filter_by_status(self, store: Store):
        store.save_trace(make_trace(trace_id="t1", status="ok"))
        store.save_trace(make_trace(trace_id="t2", status="error"))
        store.save_trace(make_trace(trace_id="t3", status="ok"))

        ok = store.get_traces(status="ok")
        assert len(ok) == 2
        err = store.get_traces(status="error")
        assert len(err) == 1

    def test_filter_by_name(self, store: Store):
        store.save_trace(make_trace(trace_id="t1", name="agent-a"))
        store.save_trace(make_trace(trace_id="t2", name="agent-b"))
        traces = store.get_traces(name="agent-a")
        assert len(traces) == 1
        assert traces[0].name == "agent-a"

    def test_list_traces_has_empty_spans(self, store: Store):
        """get_traces() doesn't load spans (for efficiency)."""
        span = make_span()
        store.save_trace(make_trace(spans=[span]))
        traces = store.get_traces()
        assert traces[0].spans == []


# --- Eval run persistence ---

class TestEvalRuns:
    """save_eval_run() / get_eval_runs() should round-trip eval results."""

    def test_save_and_load_eval_run(self, store: Store):
        run = {
            "id": "eval-1",
            "eval_file": "evals/test.yaml",
            "start_time": 1000.0,
            "end_time": 2000.0,
            "passed": 3,
            "failed": 1,
            "results": [
                {"name": "case-1", "passed": True, "scores": {"accuracy": 1.0}},
                {"name": "case-2", "passed": False, "scores": {"accuracy": 0.5}},
            ],
        }
        store.save_eval_run(run)
        runs = store.get_eval_runs()

        assert len(runs) == 1
        assert runs[0]["id"] == "eval-1"
        assert runs[0]["passed"] == 3
        assert runs[0]["failed"] == 1
        assert len(runs[0]["results"]) == 2

    def test_eval_run_limit(self, store: Store):
        for i in range(5):
            store.save_eval_run({
                "id": f"eval-{i}",
                "eval_file": "test.yaml",
                "start_time": float(i),
                "end_time": float(i + 1),
                "passed": 1,
                "failed": 0,
                "results": [],
            })
        runs = store.get_eval_runs(limit=2)
        assert len(runs) == 2
