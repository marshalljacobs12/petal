"""
Tracing system — records what happens during an agent run.

Every agent.run() creates a Trace containing Spans for each LLM call,
tool execution, etc. This data is stored in SQLite for later inspection.

A Trace is like a receipt for one agent run.
A Span is one line item on that receipt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


# --- Data classes (frozen snapshots for storage) ---

@dataclass
class SpanData:
    """One recorded operation (LLM call, tool call, etc.)."""
    id: str
    trace_id: str
    parent_id: str | None
    type: str           # "llm_call", "tool_call"
    name: str
    input: Any
    output: Any
    start_time: float   # milliseconds since epoch
    end_time: float
    tokens: dict[str, int] | None = None
    cost: float | None = None
    status: str = "ok"  # "ok" or "error"
    error: str | None = None


@dataclass
class TraceData:
    """One complete agent.run() call."""
    id: str
    name: str
    input: str
    output: str
    start_time: float
    end_time: float
    total_tokens: int
    total_cost: float
    metadata: dict[str, Any]
    spans: list[SpanData]
    status: str = "ok"


# --- Live span (mutable, used during recording) ---

class Span:
    """A span that's currently being recorded.

    Created by Tracer.start_span(), filled in during execution,
    then finalized with .end() or .fail().
    """

    def __init__(self, trace_id: str, parent_id: str | None, span_type: str, name: str):
        self.id = str(uuid4())
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.type = span_type
        self.name = name
        self.start_time = time.time() * 1000
        self.input: Any = None
        self.output: Any = None
        self.end_time: float = 0
        self.tokens: dict[str, int] | None = None
        self.cost: float | None = None
        self.status: str = "ok"
        self.error: str | None = None

    def end(self, output: Any = None) -> None:
        """Mark this span as successfully completed."""
        if output is not None:
            self.output = output
        self.end_time = time.time() * 1000

    def fail(self, error: str) -> None:
        """Mark this span as failed."""
        self.status = "error"
        self.error = error
        self.end_time = time.time() * 1000

    def to_data(self) -> SpanData:
        """Convert to a frozen SpanData for storage."""
        return SpanData(
            id=self.id,
            trace_id=self.trace_id,
            parent_id=self.parent_id,
            type=self.type,
            name=self.name,
            input=self.input,
            output=self.output,
            start_time=self.start_time,
            end_time=self.end_time or time.time() * 1000,
            tokens=self.tokens,
            cost=self.cost,
            status=self.status,
            error=self.error,
        )


# --- Tracer (creates and collects spans for one agent run) ---

class Tracer:
    """Records spans during one agent run.

    Usage:
        tracer = Tracer("my-agent")
        span = tracer.start_span("llm_call", "gpt-4o")
        span.input = messages
        # ... do the LLM call ...
        span.output = response
        span.end()

        trace_data = tracer.to_trace_data(input_text="hello", output_text="hi")
    """

    def __init__(self, name: str):
        self.trace_id = str(uuid4())
        self.name = name
        self.start_time = time.time() * 1000
        self._spans: list[Span] = []

    def start_span(self, span_type: str, name: str, parent_id: str | None = None) -> Span:
        """Create and register a new span."""
        span = Span(
            trace_id=self.trace_id,
            parent_id=parent_id,
            span_type=span_type,
            name=name,
        )
        self._spans.append(span)
        return span

    def to_trace_data(
        self,
        input_text: str,
        output_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> TraceData:
        """Finalize the trace into a frozen TraceData for storage."""
        total_tokens = 0
        total_cost = 0.0
        has_error = False

        for span in self._spans:
            if span.tokens:
                total_tokens += span.tokens.get("total_tokens", 0)
            if span.cost:
                total_cost += span.cost
            if span.status == "error":
                has_error = True

        return TraceData(
            id=self.trace_id,
            name=self.name,
            input=input_text,
            output=output_text,
            start_time=self.start_time,
            end_time=time.time() * 1000,
            total_tokens=total_tokens,
            total_cost=total_cost,
            metadata=metadata or {},
            spans=[s.to_data() for s in self._spans],
            status="error" if has_error else "ok",
        )
