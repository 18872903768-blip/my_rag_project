"""Offline regressions for query tracing and JSONL persistence."""

from __future__ import annotations

import json
from pathlib import Path

from rag_modules.observability import QueryTrace, append_trace, load_traces


def test_query_trace_records_events_and_finish(tmp_path: Path) -> None:
    trace = QueryTrace("红烧肉怎么做", role="user", pipeline="classic")
    trace.event("route", route="detail")
    trace.event("retrieve", hits=3)

    record = trace.finish(answer="先焯水…", trace_path=tmp_path / "t.jsonl")

    assert record["question"] == "红烧肉怎么做"
    assert record["role"] == "user"
    assert record["pipeline"] == "classic"
    assert record["answer"] == "先焯水…"
    assert record["error"] is None
    assert [e["event"] for e in record["events"]] == ["route", "retrieve"]
    assert record["elapsed_ms"] >= 0
    assert len(record["query_id"]) == 12


def test_append_and_load_traces_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    append_trace({"query_id": "a", "answer": "1"}, trace_path=path)
    append_trace({"query_id": "b", "error": "boom"}, trace_path=path)

    records = load_traces(trace_path=path)

    assert [r["query_id"] for r in records] == ["a", "b"]
    assert records[1]["error"] == "boom"


def test_load_traces_skips_corrupt_lines(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    payload = (
        json.dumps({"query_id": "ok"})
        + "\n{corrupt line\n"
        + json.dumps({"query_id": "ok2"})
        + "\n"
    )
    path.write_text(payload, encoding="utf-8")

    records = load_traces(trace_path=path)

    assert [r["query_id"] for r in records] == ["ok", "ok2"]


def test_load_traces_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_traces(trace_path=tmp_path / "none.jsonl") == []
