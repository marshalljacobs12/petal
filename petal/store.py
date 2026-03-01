"""
SQLite storage layer for traces and eval results.

Uses Python's built-in sqlite3 module — no external dependency needed.
The database is just a file on disk (default: ./petal.db).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from petal.tracer import SpanData, TraceData


# --- SQL schema ---

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'ok'
);

CREATE TABLE IF NOT EXISTS spans (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_id TEXT,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    input TEXT NOT NULL DEFAULT 'null',
    output TEXT NOT NULL DEFAULT 'null',
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    tokens TEXT,
    cost REAL,
    status TEXT NOT NULL DEFAULT 'ok',
    error TEXT,
    FOREIGN KEY (trace_id) REFERENCES traces(id)
);

CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);

CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    eval_file TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    passed INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    results TEXT NOT NULL DEFAULT '[]'
);
"""


class Store:
    """SQLite storage for traces and eval results."""

    def __init__(self, db_path: str = "./petal.db"):
        # check_same_thread=False allows the connection to be used across
        # async contexts (safe here because we don't do concurrent writes)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Rows act like dicts (row["name"])
        self.conn.executescript(_SCHEMA)

    # --- Traces ---

    def save_trace(self, trace: TraceData) -> None:
        """Save a trace and all its spans to the database."""
        self.conn.execute(
            """INSERT INTO traces (id, name, input, output, start_time, end_time,
               total_tokens, total_cost, metadata, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trace.id,
                trace.name,
                trace.input,
                trace.output,
                trace.start_time,
                trace.end_time,
                trace.total_tokens,
                trace.total_cost,
                json.dumps(trace.metadata),
                trace.status,
            ),
        )

        # executemany runs the same INSERT for each span in the list
        self.conn.executemany(
            """INSERT INTO spans (id, trace_id, parent_id, type, name, input, output,
               start_time, end_time, tokens, cost, status, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    span.id,
                    span.trace_id,
                    span.parent_id,
                    span.type,
                    span.name,
                    json.dumps(span.input),
                    json.dumps(span.output),
                    span.start_time,
                    span.end_time,
                    json.dumps(span.tokens) if span.tokens else None,
                    span.cost,
                    span.status,
                    span.error,
                )
                for span in trace.spans
            ],
        )

        self.conn.commit()

    def get_trace(self, trace_id: str) -> TraceData | None:
        """Load a trace and its spans by ID."""
        row = self.conn.execute(
            "SELECT * FROM traces WHERE id = ?", (trace_id,)
        ).fetchone()
        if not row:
            return None

        span_rows = self.conn.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        ).fetchall()

        return TraceData(
            id=row["id"],
            name=row["name"],
            input=row["input"],
            output=row["output"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            total_tokens=row["total_tokens"],
            total_cost=row["total_cost"],
            metadata=json.loads(row["metadata"]),
            spans=[_row_to_span(r) for r in span_rows],
            status=row["status"],
        )

    def get_traces(
        self,
        limit: int = 50,
        status: str | None = None,
        name: str | None = None,
    ) -> list[TraceData]:
        """List recent traces (without spans, for efficiency)."""
        conditions = []
        params: list[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if name:
            conditions.append("name = ?")
            params.append(name)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = self.conn.execute(
            f"SELECT * FROM traces {where} ORDER BY start_time DESC LIMIT ?",
            params,
        ).fetchall()

        return [
            TraceData(
                id=row["id"],
                name=row["name"],
                input=row["input"],
                output=row["output"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                total_tokens=row["total_tokens"],
                total_cost=row["total_cost"],
                metadata=json.loads(row["metadata"]),
                spans=[],  # Not loaded for list queries
                status=row["status"],
            )
            for row in rows
        ]

    # --- Eval runs ---

    def save_eval_run(self, run: dict[str, Any]) -> None:
        """Save an eval run result."""
        self.conn.execute(
            """INSERT INTO eval_runs (id, eval_file, start_time, end_time,
               passed, failed, results)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                run["id"],
                run["eval_file"],
                run["start_time"],
                run["end_time"],
                run["passed"],
                run["failed"],
                json.dumps(run["results"]),
            ),
        )
        self.conn.commit()

    def get_eval_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent eval runs."""
        rows = self.conn.execute(
            "SELECT * FROM eval_runs ORDER BY start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "eval_file": row["eval_file"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "passed": row["passed"],
                "failed": row["failed"],
                "results": json.loads(row["results"]),
            }
            for row in rows
        ]

    def close(self) -> None:
        self.conn.close()


def _row_to_span(row: sqlite3.Row) -> SpanData:
    """Convert a database row to a SpanData object."""
    return SpanData(
        id=row["id"],
        trace_id=row["trace_id"],
        parent_id=row["parent_id"],
        type=row["type"],
        name=row["name"],
        input=json.loads(row["input"]),
        output=json.loads(row["output"]),
        start_time=row["start_time"],
        end_time=row["end_time"],
        tokens=json.loads(row["tokens"]) if row["tokens"] else None,
        cost=row["cost"],
        status=row["status"],
        error=row["error"],
    )


# --- Default store singleton ---
# The agent auto-saves traces here. Users can also call get_default_store()
# to query traces programmatically.

_default_store: Store | None = None


def get_default_store() -> Store | None:
    """Get the global store instance, creating it if needed."""
    global _default_store
    if _default_store is None:
        _default_store = Store()
    return _default_store


def get_traces(limit: int = 50, status: str | None = None) -> list[TraceData]:
    """Convenience function: query traces from the default store."""
    store = get_default_store()
    if not store:
        return []
    return store.get_traces(limit=limit, status=status)


def get_trace(trace_id: str) -> TraceData | None:
    """Convenience function: get a single trace from the default store."""
    store = get_default_store()
    if not store:
        return None
    return store.get_trace(trace_id)
