"""Tests for the tracing system: Span, Tracer, SpanData, TraceData."""

import time

from petal.tracer import Span, SpanData, TraceData, Tracer


class TestSpan:
    """A live Span records one operation (LLM call, tool call, etc.)."""

    def test_span_gets_unique_id(self):
        s1 = Span("trace-1", None, "llm_call", "gpt-4o")
        s2 = Span("trace-1", None, "llm_call", "gpt-4o")
        assert s1.id != s2.id

    def test_span_records_trace_id(self):
        s = Span("trace-abc", None, "llm_call", "gpt-4o")
        assert s.trace_id == "trace-abc"

    def test_span_records_type_and_name(self):
        s = Span("t", None, "tool_call", "get_weather")
        assert s.type == "tool_call"
        assert s.name == "get_weather"

    def test_span_start_time_set(self):
        before = time.time() * 1000
        s = Span("t", None, "llm_call", "gpt-4o")
        after = time.time() * 1000
        assert before <= s.start_time <= after

    def test_end_sets_end_time(self):
        s = Span("t", None, "llm_call", "gpt-4o")
        s.end(output="hello")
        assert s.end_time > 0
        assert s.output == "hello"
        assert s.status == "ok"

    def test_end_without_output(self):
        s = Span("t", None, "llm_call", "gpt-4o")
        s.output = "pre-set"
        s.end()
        assert s.output == "pre-set"
        assert s.end_time > 0

    def test_fail_sets_error_status(self):
        s = Span("t", None, "tool_call", "broken")
        s.fail("something broke")
        assert s.status == "error"
        assert s.error == "something broke"
        assert s.end_time > 0

    def test_parent_id(self):
        s = Span("t", "parent-123", "tool_call", "child")
        assert s.parent_id == "parent-123"


class TestSpanToData:
    """to_data() converts a live Span to a frozen SpanData."""

    def test_returns_span_data(self):
        s = Span("t", None, "llm_call", "gpt-4o")
        s.input = {"messages": []}
        s.tokens = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        s.cost = 0.001
        s.end(output="response")
        data = s.to_data()

        assert isinstance(data, SpanData)
        assert data.id == s.id
        assert data.trace_id == "t"
        assert data.type == "llm_call"
        assert data.input == {"messages": []}
        assert data.output == "response"
        assert data.tokens == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert data.cost == 0.001
        assert data.status == "ok"

    def test_unfinished_span_gets_end_time(self):
        """If end() was never called, to_data() still sets an end_time."""
        s = Span("t", None, "llm_call", "gpt-4o")
        data = s.to_data()
        assert data.end_time > 0


class TestTracer:
    """Tracer creates and collects Spans for one agent run."""

    def test_tracer_has_unique_trace_id(self):
        t1 = Tracer("agent-1")
        t2 = Tracer("agent-2")
        assert t1.trace_id != t2.trace_id

    def test_start_span_returns_span(self):
        tracer = Tracer("agent")
        span = tracer.start_span("llm_call", "gpt-4o")
        assert isinstance(span, Span)
        assert span.trace_id == tracer.trace_id

    def test_spans_collected(self):
        tracer = Tracer("agent")
        tracer.start_span("llm_call", "step-0")
        tracer.start_span("tool_call", "get_weather")
        assert len(tracer._spans) == 2

    def test_parent_child_hierarchy(self):
        tracer = Tracer("agent")
        llm_span = tracer.start_span("llm_call", "step-0")
        tool_span = tracer.start_span("tool_call", "get_weather", parent_id=llm_span.id)

        assert tool_span.parent_id == llm_span.id
        assert llm_span.parent_id is None


class TestTracerToTraceData:
    """to_trace_data() aggregates all spans into a TraceData snapshot."""

    def _build_tracer_with_spans(self) -> Tracer:
        tracer = Tracer("test-agent")

        s1 = tracer.start_span("llm_call", "step-0")
        s1.tokens = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        s1.cost = 0.005
        s1.end()

        s2 = tracer.start_span("tool_call", "get_weather", parent_id=s1.id)
        s2.end()

        s3 = tracer.start_span("llm_call", "step-1")
        s3.tokens = {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}
        s3.cost = 0.008
        s3.end()

        return tracer

    def test_returns_trace_data(self):
        tracer = self._build_tracer_with_spans()
        td = tracer.to_trace_data(input_text="hello", output_text="world")
        assert isinstance(td, TraceData)

    def test_aggregates_tokens(self):
        tracer = self._build_tracer_with_spans()
        td = tracer.to_trace_data(input_text="hi", output_text="bye")
        assert td.total_tokens == 430  # 150 + 280

    def test_aggregates_cost(self):
        tracer = self._build_tracer_with_spans()
        td = tracer.to_trace_data(input_text="hi", output_text="bye")
        assert abs(td.total_cost - 0.013) < 1e-9  # 0.005 + 0.008

    def test_all_spans_included(self):
        tracer = self._build_tracer_with_spans()
        td = tracer.to_trace_data(input_text="hi", output_text="bye")
        assert len(td.spans) == 3

    def test_status_ok_when_no_errors(self):
        tracer = self._build_tracer_with_spans()
        td = tracer.to_trace_data(input_text="hi", output_text="bye")
        assert td.status == "ok"

    def test_status_error_when_span_fails(self):
        tracer = Tracer("agent")
        s = tracer.start_span("tool_call", "broken")
        s.fail("crash")
        td = tracer.to_trace_data(input_text="hi", output_text="error")
        assert td.status == "error"

    def test_metadata_passed_through(self):
        tracer = Tracer("agent")
        td = tracer.to_trace_data(
            input_text="hi", output_text="bye", metadata={"env": "test"}
        )
        assert td.metadata == {"env": "test"}

    def test_input_output_text(self):
        tracer = Tracer("agent")
        td = tracer.to_trace_data(input_text="question", output_text="answer")
        assert td.input == "question"
        assert td.output == "answer"
