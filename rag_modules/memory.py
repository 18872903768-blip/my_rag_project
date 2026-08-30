"""会话与长期记忆管理（饮食偏好领域化）。

分层：
- ``SessionStore``：多轮会话历史持久化，替换 API 层的内存 dict（重启即失、
  多实例不可用）；
- ``MemoryStore``：结构化长期记忆，类型为饮食领域（allergen 过敏原 /
  diet_preference 饮食方式 / taste_preference 口味偏好 / constraint 阶段性
  约束），带写入策略（write policy）、同 key 冲突 supersede、过期与
  recency 衰减；
- ``MemoryExtractor``：LLM 从单轮对话抽取候选记忆，经写入策略过滤后落库。

设计取舍：
- SQLite（标准库，WAL 模式）作为默认实现；``SessionStore``/``MemoryStore``
  为 Protocol，迁移到英语平台项目时新写 ``MySQLStore`` 即可，策略层零改动；
- 记忆条目量级小（每用户个位数到几十条），检索用条件查询 + recency 加权
  排序而非向量库——诚实且够用；
- 所有 LLM 抽取失败静默跳过（fallback 原则：记忆故障不影响主回答）。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

logger = logging.getLogger(__name__)

MEMORY_TYPES = ("allergen", "diet_preference", "taste_preference", "constraint")
ACTIVE = "active"
SUPERSEDED = "superseded"

DEFAULT_IMPORTANCE_THRESHOLD = 0.6
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_HALF_LIFE_DAYS = 30.0
SESSION_MAX_MESSAGES = 12  # 最近 6 轮（user+assistant 各一条）


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@dataclass
class MemoryRecord:
    memory_id: str
    user_id: str
    type: str
    key: str
    value: str
    content: str
    importance: float
    confidence: float
    created_at: str = field(default_factory=lambda: _iso(_utcnow()))
    expires_at: str | None = None
    status: str = ACTIVE
    supersedes: str | None = None


# ------------------------------------------------------------------ Store 抽象


class SessionStore(Protocol):
    """会话历史存储（多轮对话的持久化层）。"""

    def load_history(self, session_id: str) -> list[dict[str, str]]: ...

    def save_turn(self, session_id: str, question: str, answer: str) -> None: ...


class MemoryStore(Protocol):
    """结构化长期记忆存储。"""

    def add_memory(self, record: MemoryRecord) -> str: ...

    def active_memories(self, user_id: str, now: datetime | None = None) -> list[MemoryRecord]: ...

    def list_memories(self, user_id: str) -> list[MemoryRecord]: ...

    def delete_memory(self, user_id: str, memory_id: str) -> bool: ...

    def delete_all(self, user_id: str) -> int: ...


class SQLiteMemoryStore(SessionStore, MemoryStore):
    """SQLite 默认实现（WAL + 单锁；多线程 FastAPI 下安全）。

    迁移说明：英语平台项目为 Gin + MySQL 栈，届时实现 ``MySQLMemoryStore``
    即可无缝替换本类，上层策略代码零改动。
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                messages   TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id   TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                type        TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                content     TEXT NOT NULL,
                importance  REAL NOT NULL,
                confidence  REAL NOT NULL,
                created_at  TEXT NOT NULL,
                expires_at  TEXT,
                status      TEXT NOT NULL,
                supersedes  TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, status)"
        )
        self._conn.commit()

    # ---------------------------------------------------------------- session

    def load_history(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT messages FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return json.loads(row[0]) if row else []

    def save_turn(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT messages FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            messages = json.loads(row[0]) if row else []
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer[:400]})
            del messages[:-SESSION_MAX_MESSAGES]
            self._conn.execute(
                """
                INSERT INTO sessions (session_id, messages, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    messages = excluded.messages, updated_at = excluded.updated_at
                """,
                (session_id, json.dumps(messages, ensure_ascii=False), _iso(_utcnow())),
            )
            self._conn.commit()

    # ----------------------------------------------------------------- memory

    def add_memory(self, record: MemoryRecord) -> str:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT memory_id FROM memories "
                "WHERE user_id = ? AND type = ? AND key = ? AND status = ?",
                (record.user_id, record.type, record.key, ACTIVE),
            )
            old = cursor.fetchone()
            if old:
                self._conn.execute(
                    "UPDATE memories SET status = ? WHERE memory_id = ?",
                    (SUPERSEDED, old[0]),
                )
                record.supersedes = old[0]
            if not record.memory_id:
                record.memory_id = uuid.uuid4().hex[:16]
            self._conn.execute(
                """
                INSERT INTO memories (
                    memory_id, user_id, type, key, value, content,
                    importance, confidence, created_at, expires_at, status, supersedes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id, record.user_id, record.type, record.key,
                    record.value, record.content, record.importance, record.confidence,
                    record.created_at, record.expires_at, record.status, record.supersedes,
                ),
            )
            self._conn.commit()
        return record.memory_id

    def active_memories(self, user_id: str, now: datetime | None = None) -> list[MemoryRecord]:
        now_iso = _iso(now or _utcnow())
        with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id, user_id, type, key, value, content, importance, "
                "confidence, created_at, expires_at, status, supersedes "
                "FROM memories WHERE user_id = ? AND status = ? "
                "AND (expires_at IS NULL OR expires_at > ?)",
                (user_id, ACTIVE, now_iso),
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def list_memories(self, user_id: str) -> list[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id, user_id, type, key, value, content, importance, "
                "confidence, created_at, expires_at, status, supersedes "
                "FROM memories WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE user_id = ? AND memory_id = ?",
                (user_id, memory_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def delete_all(self, user_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE user_id = ?", (user_id,)
            )
            self._conn.commit()
        return cursor.rowcount

    @staticmethod
    def _record_from_row(row: tuple) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row[0], user_id=row[1], type=row[2], key=row[3], value=row[4],
            content=row[5], importance=row[6], confidence=row[7], created_at=row[8],
            expires_at=row[9], status=row[10], supersedes=row[11],
        )


# ------------------------------------------------------------------ 写入策略


def apply_write_policy(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """过滤不应写入的记忆：临时性表述、低重要度/低置信度、非法类型、字段缺失。

    例："我今晚想吃面" → temporary=True 不写；"我对花生过敏" → 写。
    """
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("temporary"):
            continue
        if str(candidate.get("type", "")).strip() not in MEMORY_TYPES:
            continue
        key = str(candidate.get("key", "")).strip()
        content = str(candidate.get("content", "")).strip()
        if not key or not content:
            continue
        try:
            importance = float(candidate.get("importance", 0.0))
            confidence = float(candidate.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if importance < DEFAULT_IMPORTANCE_THRESHOLD or confidence < DEFAULT_CONFIDENCE_THRESHOLD:
            continue
        kept.append(
            {
                **candidate,
                "type": str(candidate["type"]).strip(),
                "key": key,
                "value": str(candidate.get("value", key)).strip(),
                "content": content,
                "importance": importance,
                "confidence": confidence,
            }
        )
    return kept


# ------------------------------------------------------------------ 衰减与排序


def recency_score(created_at: str, now: datetime | None = None) -> float:
    """指数衰减：recency = exp(-ln2 * age / half_life)，30 天半衰期。"""
    reference = now or _utcnow()
    try:
        created = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return 0.5
    age_days = max((reference - created).total_seconds() / 86400, 0.0)
    return 2.0 ** (-age_days / DEFAULT_HALF_LIFE_DAYS)


def rank_memories(
    records: list[MemoryRecord],
    question: str,
    *,
    now: datetime | None = None,
) -> list[MemoryRecord]:
    """按 importance x recency（命中当前问题时加权）降序排列。"""

    def score(record: MemoryRecord) -> float:
        base = record.importance * recency_score(record.created_at, now)
        hit = any(
            token and token in question
            for token in (record.key, record.value, record.content[:20])
        )
        return base * (1.8 if hit else 1.0)

    return sorted(records, key=score, reverse=True)


# ------------------------------------------------------------------ LLM 抽取


EXTRACT_PROMPT = """分析下面这轮对话，判断用户是否表达了值得长期记住的饮食相关偏好。

用户: {user_message}
助手: {assistant_answer}

可用的记忆类型与 key 规范（key 必须严格按此填写，用于后续记忆更新时的冲突匹配）：
- allergen: 过敏原。key=过敏原名称（如"花生"、"海鲜"）
- taste_preference: 口味偏好。key 固定为"口味"，value=具体偏好（如"不吃辣"、"能吃微辣"、"喜欢清淡"）
- diet_preference: 饮食方式。key 固定为"饮食方式"，value=具体方式（如"素食"、"不吃猪肉"）
- constraint: 阶段性约束。key 固定为"当前约束"，value=约束内容（如"减脂"、"控糖"）

判断标准：
- 稳定、可复用的偏好必须记录，importance 与 confidence 给 0.8 以上（如"我对花生过敏"、"我不吃辣"）
- 阶段性约束（"我最近在减脂"）也要记录：temporary=false，并给 expires_days（如 30）
- 一次性需求（"我今晚想吃面"）必须标记 temporary=true，会被过滤
- 同一类偏好的变化（如从"不吃辣"变成"能吃微辣"）也要记录，系统会自动替换旧记忆
- 没有值得记的内容时返回空列表

只输出一行 JSON：
{{"memories": [{{"type": "...", "key": "...", "value": "...", "content": "...", "importance": 0.0-1.0, "confidence": 0.0-1.0, "temporary": false, "expires_days": null}}]}}"""


class MemoryExtractor:
    """LLM 从单轮对话抽取候选记忆（结构化 JSON，失败返回空列表）。"""

    def __init__(self, llm: Any):
        self.llm = llm

    def extract(self, user_message: str, assistant_answer: str) -> list[dict[str, Any]]:
        prompt = EXTRACT_PROMPT.format(
            user_message=user_message[:600], assistant_answer=assistant_answer[:800]
        )
        try:
            raw = str(self.llm.invoke(prompt).content)
        except Exception as error:  # noqa: BLE001 - 抽取失败静默跳过
            logger.warning("记忆抽取 LLM 调用失败: %s", error)
            return []
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
        memories = payload.get("memories")
        if not isinstance(memories, list):
            return []
        return [item for item in memories if isinstance(item, dict)]


def remember_turn(
    store: MemoryStore,
    extractor: MemoryExtractor,
    *,
    user_id: str,
    question: str,
    answer: str,
    now: datetime | None = None,
) -> list[MemoryRecord]:
    """单轮对话的记忆写入入口：抽取 → 写入策略 → 落库（含 supersede/过期）。

    任何失败都不抛出（fallback 原则），返回实际写入的记录。
    """
    reference = now or _utcnow()
    try:
        candidates = apply_write_policy(extractor.extract(question, answer))
    except Exception as error:  # noqa: BLE001
        logger.warning("记忆抽取失败（跳过写入）: %s", error)
        return []

    written: list[MemoryRecord] = []
    for candidate in candidates:
        expires_days = candidate.get("expires_days")
        expires_at = (
            _iso(reference + timedelta(days=float(expires_days)))
            if expires_days
            else None
        )
        record = MemoryRecord(
            memory_id=uuid.uuid4().hex[:16],
            user_id=user_id,
            type=candidate["type"],
            key=candidate["key"],
            value=candidate["value"],
            content=candidate["content"],
            importance=candidate["importance"],
            confidence=candidate["confidence"],
            created_at=_iso(reference),
            expires_at=expires_at,
        )
        try:
            store.add_memory(record)
            written.append(record)
        except Exception as error:  # noqa: BLE001
            logger.warning("记忆写入失败: %s", error)
    return written
