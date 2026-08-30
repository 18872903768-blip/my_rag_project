"""Query-level tracing: structured JSONL logs plus optional LangSmith wiring.

Every user query gets a ``query_id`` that follows it through routing, retrieval,
and generation.  Node outcomes (including every degraded fallback) are appended
to a JSONL file so any bad case can be replayed and audited later::

    {"query_id": "...", "question": "...", "role": "user", "events": [...],
     "answer": "...", "error": null, "elapsed_ms": 8123}

LangSmith is enabled purely through environment variables
(``LANGSMITH_TRACING=true`` + ``LANGSMITH_API_KEY``); all LangChain/LangGraph
calls are then traced automatically, and ``tracing_status()`` reports it.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TRACE_DIR = Path(__file__).resolve().parents[1] / "logs"
TRACE_FILENAME = "query_trace.jsonl"
_write_lock = threading.Lock()


def tracing_status() -> dict[str, Any]:
    """Report whether LangSmith tracing is active in this process."""
    import os

    enabled = os.getenv("LANGSMITH_TRACING", "").casefold() in {"1", "true", "yes", "on"}
    has_key = bool(os.getenv("LANGSMITH_API_KEY"))
    return {
        "langsmith_tracing": enabled and has_key,
        "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "default"),
    }


class QueryTrace:
    """Accumulates events for one query and persists them as one JSONL line."""

    def __init__(
        self,
        question: str,
        *,
        role: str = "user",
        pipeline: str = "classic",
        query_id: str | None = None,
    ):
        self.query_id = query_id or uuid.uuid4().hex[:12]
        self.question = question
        self.role = role
        self.pipeline = pipeline
        self.started_at = datetime.now(UTC)
        self.events: list[dict[str, Any]] = []
        self.answer: str | None = None
        self.error: str | None = None

    def event(self, name: str, **details: Any) -> None:
        self.events.append({"ts": datetime.now(UTC).isoformat(), "event": name, **details})

    def finish(
        self,
        answer: str | None = None,
        error: str | Exception | None = None,
        *,
        trace_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if answer is not None:
            self.answer = answer
        if error is not None:
            self.error = str(error)
        record = self.to_dict()
        try:
            append_trace(record, trace_path)
        except OSError as exc:
            logger.warning("写入查询追踪日志失败: %s", exc)
        return record

    def to_dict(self) -> dict[str, Any]:
        finished = datetime.now(UTC)
        return {
            "query_id": self.query_id,
            "question": self.question,
            "role": self.role,
            "pipeline": self.pipeline,
            "started_at": self.started_at.isoformat(),
            "elapsed_ms": int((finished - self.started_at).total_seconds() * 1000),
            "events": self.events,
            "answer": self.answer,
            "error": self.error,
        }


def append_trace(record: dict[str, Any], trace_path: str | Path | None = None) -> None:
    path = Path(trace_path) if trace_path else DEFAULT_TRACE_DIR / TRACE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _write_lock:
        with path.open("a", encoding="utf-8") as target:
            target.write(line + "\n")


def load_traces(
    trace_path: str | Path | None = None, *, limit: int | None = None
) -> list[dict[str, Any]]:
    """Read back trace records (newest last); tolerant of a missing/partial file."""
    path = Path(trace_path) if trace_path else DEFAULT_TRACE_DIR / TRACE_FILENAME
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("跳过损坏的追踪行: %s", line[:80])
    if limit is not None:
        records = records[-limit:]
    return records
