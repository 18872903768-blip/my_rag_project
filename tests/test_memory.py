"""Offline regressions for session/long-term memory (store, policy, decay)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rag_modules.memory import (
    MemoryExtractor,
    MemoryRecord,
    SQLiteMemoryStore,
    apply_write_policy,
    rank_memories,
    recency_score,
    remember_turn,
)


def _store(tmp_path) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(str(tmp_path / "memory_test.sqlite3"))


def _record(user_id: str = "u1", key: str = "花生", **overrides) -> MemoryRecord:
    defaults = {
        "memory_id": "",
        "user_id": user_id,
        "type": "allergen",
        "key": key,
        "value": key,
        "content": f"用户对{key}过敏",
        "importance": 0.9,
        "confidence": 0.9,
    }
    defaults.update(overrides)
    return MemoryRecord(**defaults)


# ------------------------------------------------------------------ session


def test_session_persists_across_store_reopen(tmp_path) -> None:
    path = str(tmp_path / "session.sqlite3")
    store = SQLiteMemoryStore(path)
    store.save_turn("s1", "红烧肉怎么做", "做法如下……")
    store.save_turn("s1", "那需要什么食材", "需要五花肉……")

    reopened = SQLiteMemoryStore(path)
    history = reopened.load_history("s1")

    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[-1]["content"] == "需要五花肉……"
    assert reopened.load_history("missing") == []


def test_session_trims_to_recent_six_messages(tmp_path) -> None:
    store = SQLiteMemoryStore(str(tmp_path / "trim.sqlite3"))
    for i in range(7):
        store.save_turn("s1", f"问题{i}", f"回答{i}")

    history = store.load_history("s1")

    assert len(history) == 12
    assert history[0]["content"] == "问题1"
    assert history[-2]["content"] == "问题6"


# ------------------------------------------------------------------ 写入策略


def test_write_policy_filters_temporary_and_low_quality() -> None:
    candidates = [
        {"type": "taste_preference", "key": "口味", "value": "清淡", "content": "喜欢清淡",
         "importance": 0.8, "confidence": 0.9},  # 保留
        {"type": "taste_preference", "key": "今晚", "value": "吃面", "content": "今晚想吃面",
         "importance": 0.8, "confidence": 0.9, "temporary": True},  # 临时 → 过滤
        {"type": "allergen", "key": "花生", "value": "花生", "content": "花生过敏",
         "importance": 0.4, "confidence": 0.9},  # 低重要度 → 过滤
        {"type": "invalid_type", "key": "x", "value": "x", "content": "x",
         "importance": 0.9, "confidence": 0.9},  # 非法类型 → 过滤
    ]

    kept = apply_write_policy(candidates)

    assert [row["key"] for row in kept] == ["口味"]


# ------------------------------------------------------------------ 冲突与衰减


def test_same_key_new_memory_supersedes_old(tmp_path) -> None:
    store = _store(tmp_path)
    old_id = store.add_memory(
        _record(type="taste_preference", key="吃辣", value="不吃辣", content="用户不吃辣")
    )
    new_id = store.add_memory(
        _record(type="taste_preference", key="吃辣", value="能吃微辣", content="用户现在能吃微辣")
    )

    records = {r.memory_id: r for r in store.list_memories("u1")}

    assert records[old_id].status == "superseded"
    assert records[old_id].supersedes is None
    assert records[new_id].status == "active"
    assert records[new_id].supersedes == old_id
    assert [r for r in store.active_memories("u1") if r.key == "吃辣"][0].value == "能吃微辣"


def test_active_memories_respect_expiry(tmp_path) -> None:
    store = _store(tmp_path)
    expired = _record(key="减脂", content="最近在减脂")
    expired.expires_at = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    store.add_memory(expired)
    store.add_memory(_record(key="花生", content="用户对花生过敏"))

    active = store.active_memories("u1")

    assert [r.key for r in active] == ["花生"]


def test_recency_decay_and_question_boost_ranking(tmp_path) -> None:
    store = _store(tmp_path)
    now = datetime.now(UTC)
    old_record = _record(key="花生", content="用户对花生过敏",
                         created_at=(now - timedelta(days=90)).isoformat())
    fresh_record = _record(key="花生", content="用户对花生过敏", user_id="u1")
    fresh_record.created_at = now.isoformat()
    unrelated = _record(key="清淡", content="用户喜欢清淡")
    unrelated.created_at = now.isoformat()

    ranked = rank_memories([old_record, fresh_record, unrelated], "花生能吃什么", now=now)

    # 相同重要度：新记忆衰减少排前；命中问题的记录获得加权
    assert ranked[0] is fresh_record
    assert ranked.index(unrelated) > ranked.index(fresh_record)


# ------------------------------------------------------------------ 隐私端点支撑


def test_delete_memory_and_delete_all(tmp_path) -> None:
    store = _store(tmp_path)
    id1 = store.add_memory(_record(key="花生"))
    store.add_memory(_record(key="海鲜", content="用户对海鲜过敏"))

    assert store.delete_memory("u1", id1) is True
    assert store.delete_memory("u1", "missing") is False
    assert store.delete_all("u1") == 1
    assert store.list_memories("u1") == []


# ------------------------------------------------------------------ LLM 抽取


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[str] = []

    def invoke(self, prompt: str):
        self.calls.append(prompt)

        class _Resp:
            pass

        resp = _Resp()
        resp.content = self.content
        return resp


def test_remember_turn_writes_policy_passing_memories(tmp_path) -> None:
    store = _store(tmp_path)
    extractor = MemoryExtractor(
        _FakeLLM('{"memories": [{"type": "allergen", "key": "花生", "value": "花生", '
                 '"content": "用户对花生过敏", "importance": 0.9, "confidence": 0.9, '
                 '"temporary": false, "expires_days": null}, '
                 '{"type": "taste_preference", "key": "今晚", "value": "吃面", '
                 '"content": "今晚想吃面", "importance": 0.9, "confidence": 0.9, '
                 '"temporary": true}]}')
    )

    written = remember_turn(store, extractor, user_id="u1", question="推荐几个菜", answer="好的")

    assert len(written) == 1
    assert written[0].key == "花生"
    assert [r.key for r in store.active_memories("u1")] == ["花生"]


def test_remember_turn_llm_failure_returns_empty(tmp_path) -> None:
    class _BrokenLLM:
        def invoke(self, prompt: str):
            raise RuntimeError("llm down")

    store = _store(tmp_path)

    written = remember_turn(store, MemoryExtractor(_BrokenLLM()), user_id="u1",
                            question="q", answer="a")

    assert written == []
    assert store.list_memories("u1") == []


def test_extractor_parses_memory_record_expiry(tmp_path) -> None:
    store = _store(tmp_path)
    extractor = MemoryExtractor(
        _FakeLLM('{"memories": [{"type": "constraint", "key": "减脂", "value": "减脂", '
                 '"content": "最近在减脂", "importance": 0.7, "confidence": 0.8, '
                 '"temporary": false, "expires_days": 7}]}')
    )

    written = remember_turn(store, extractor, user_id="u1", question="q", answer="a")

    assert len(written) == 1
    assert written[0].expires_at is not None  # 7 天后过期
