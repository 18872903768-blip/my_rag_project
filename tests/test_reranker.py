"""Offline regressions for cross-encoder reranking integration."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.documents import Document

from rag_modules.reranker import RerankerModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule


class FakeCrossEncoder:
    """Score = number of shared characters with the query (deterministic)."""

    def __init__(self) -> None:
        self.seen_pairs: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.seen_pairs.append(list(pairs))
        return [float(len(set(q) & set(d))) for q, d in pairs]


class FakeVectorStore:
    def __init__(self, results: list[Document]) -> None:
        self.results = results
        self.requested_k: list[int] = []

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        self.requested_k.append(k)
        return self.results[:k]

    def hybrid_search(self, query: str, k: int = 5, **kwargs: Any) -> list[Document]:
        self.requested_k.append(k)
        return self.results[:k]


def _chunk(chunk_id: str, dish: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={"chunk_id": chunk_id, "dish_name": dish, "visibility": "public"},
    )


def test_reranker_reorders_pool_by_cross_encoder_scores() -> None:
    reranker = RerankerModule(model_name="fake", pool_size=20, model=FakeCrossEncoder())
    pool = [
        _chunk("a", "拍黄瓜", "拍黄瓜需要黄瓜"),
        _chunk("b", "红烧肉", "红烧肉需要五花肉和酱油冰糖"),
        _chunk("c", "蛋炒饭", "蛋炒饭需要米饭鸡蛋"),
    ]

    ranked = reranker.rerank("红烧肉 冰糖 老抽", pool)

    # "红烧肉"与查询共享字符最多，应排第一
    assert ranked[0].metadata["dish_name"] == "红烧肉"
    assert all("rerank_score" in doc.metadata for doc in ranked)
    scores = [doc.metadata["rerank_score"] for doc in ranked]
    assert scores == sorted(scores, reverse=True)


def test_disabled_reranker_is_identity() -> None:
    reranker = RerankerModule(model_name="fake", enabled=False, model=FakeCrossEncoder())
    pool = [_chunk("a", "拍黄瓜", "拍黄瓜"), _chunk("b", "红烧肉", "红烧肉")]

    assert reranker.rerank("红烧肉", pool) is pool


def test_hybrid_search_expands_pool_and_reranks_to_top_k() -> None:
    chunks = [
        _chunk("a", "拍黄瓜", "拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜"),
        _chunk("b", "红烧肉", "红烧肉需要五花肉和酱油冰糖红烧肉红烧肉红烧肉红烧肉红烧肉"),
    ]
    vectorstore = FakeVectorStore(chunks)
    reranker = RerankerModule(model_name="fake", pool_size=20, model=FakeCrossEncoder())
    retrieval = RetrievalOptimizationModule(
        cast(Any, vectorstore),
        chunks,
        candidate_k=5,
        reranker=reranker,
    )

    results = retrieval.hybrid_search("红烧肉 冰糖 老抽", top_k=1)

    # 召回池按 pool_size=20 请求，再由重排截断到 top_k=1
    assert vectorstore.requested_k[-1] == 20
    assert len(results) == 1
    assert results[0].metadata["dish_name"] == "红烧肉"
    assert "rerank_score" in results[0].metadata


def test_no_reranker_keeps_original_behavior() -> None:
    chunks = [_chunk("a", "拍黄瓜", "拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜拍黄瓜")]
    vectorstore = FakeVectorStore(chunks)
    retrieval = RetrievalOptimizationModule(cast(Any, vectorstore), chunks, candidate_k=5)

    results = retrieval.hybrid_search("拍黄瓜", top_k=1)

    # 未启用重排时召回池由 candidate_k 决定（保持原有行为），不扩到 pool_size
    assert vectorstore.requested_k[-1] == 5
    assert "rerank_score" not in results[0].metadata


# ------------------------------------------------------------------ dashscope


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | str) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else str(payload)

    def json(self) -> dict[str, Any]:
        assert isinstance(self._payload, dict)
        return self._payload


def _dashscope_module(**overrides: Any) -> RerankerModule:
    defaults: dict[str, Any] = {
        "model_name": "qwen3-vl-rerank",
        "provider": "dashscope",
        "api_key": "sk-test",
        "model": None,
    }
    defaults.update(overrides)
    return RerankerModule(**defaults)


def _install_fake_post(monkeypatch: Any, responder: Any) -> list[dict[str, Any]]:
    """Replace requests.post (imported lazily by the dashscope path); record calls."""
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return responder(calls[-1])

    monkeypatch.setattr("requests.post", fake_post)
    return calls


def test_dashscope_provider_uses_api_scores(monkeypatch: Any) -> None:
    pool = [_chunk("a", "拍黄瓜", "拍黄瓜"), _chunk("b", "红烧肉", "红烧肉")]
    response = FakeResponse(
        200,
        {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
            "usage": {"total_tokens": 30},
        },
    )
    calls = _install_fake_post(monkeypatch, lambda _: response)

    ranked = _dashscope_module().rerank("红烧肉", pool)

    assert len(calls) == 1
    assert "text-rerank" in calls[0]["url"]
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["json"]["model"] == "qwen3-vl-rerank"
    assert calls[0]["json"]["input"]["documents"] == ["拍黄瓜", "红烧肉"]
    # index=1（红烧肉）分数更高，应排第一，分数写入 metadata
    assert ranked[0].metadata["dish_name"] == "红烧肉"
    assert ranked[0].metadata["rerank_score"] == 0.9
    assert ranked[1].metadata["rerank_score"] == 0.1


def test_dashscope_failure_falls_back_to_local_model(monkeypatch: Any) -> None:
    pool = [_chunk("a", "拍黄瓜", "拍黄瓜"), _chunk("b", "红烧肉", "红烧肉")]

    def always_500(_: dict[str, Any]) -> FakeResponse:
        return FakeResponse(500, {"code": "InternalError", "message": "boom"})

    _install_fake_post(monkeypatch, always_500)

    ranked = _dashscope_module(model=FakeCrossEncoder()).rerank("红烧肉", pool)

    # API 不可用 → 本地 cross-encoder 兜底，仍完成重排
    assert ranked[0].metadata["dish_name"] == "红烧肉"
    assert "rerank_score" in ranked[0].metadata


def test_dashscope_malformed_results_fall_back(monkeypatch: Any) -> None:
    pool = [_chunk("a", "拍黄瓜", "拍黄瓜"), _chunk("b", "红烧肉", "红烧肉")]
    response = FakeResponse(
        200,
        {"output": {"results": [{"index": 0, "relevance_score": 0.5}]}},  # 缺 index=1
    )
    _install_fake_post(monkeypatch, lambda _: response)

    ranked = _dashscope_module(model=FakeCrossEncoder()).rerank("红烧肉", pool)

    assert ranked[0].metadata["dish_name"] == "红烧肉"


def test_dashscope_missing_key_falls_back_without_request(monkeypatch: Any) -> None:
    pool = [_chunk("a", "拍黄瓜", "拍黄瓜"), _chunk("b", "红烧肉", "红烧肉")]

    def unexpected(_: dict[str, Any]) -> FakeResponse:
        raise AssertionError("无 key 时不应发起 HTTP 请求")

    _install_fake_post(monkeypatch, unexpected)

    ranked = _dashscope_module(api_key="", model=FakeCrossEncoder()).rerank("红烧肉", pool)

    assert ranked[0].metadata["dish_name"] == "红烧肉"


def test_parse_dashscope_results_accepts_flat_and_nested_bodies() -> None:
    nested = {"output": {"results": [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.2}]}}
    flat = {"results": [{"index": 0, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.8}]}

    assert RerankerModule._parse_dashscope_results(nested, 2) == [0.2, 0.8]
    assert RerankerModule._parse_dashscope_results(flat, 2) == [0.2, 0.8]
    assert RerankerModule._parse_dashscope_results({"output": {"results": []}}, 2) is None
    assert RerankerModule._parse_dashscope_results({"oops": 1}, 2) is None


def test_unknown_provider_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="provider"):
        RerankerModule(model_name="fake", provider="openai")


# ------------------------------------------------------- title 注入（方案一）


def test_context_flag_prepends_dish_title_to_scoring_text() -> None:
    pool = [
        _chunk("a", "拍黄瓜", "黄瓜拍碎加蒜末"),
        _chunk("b", "红烧肉", "五花肉加冰糖慢炖"),
    ]
    fake = FakeCrossEncoder()
    RerankerModule(
        model_name="fake", model=fake, include_context=True, pool_size=20
    ).rerank("红烧肉", pool)

    # 打分文本变成"菜名（分类）｜正文"；_chunk 未设置 category → 只有菜名
    assert fake.seen_pairs[0][1][1] == "红烧肉｜五花肉加冰糖慢炖"
    assert fake.seen_pairs[0][0][1] == "拍黄瓜｜黄瓜拍碎加蒜末"


def test_context_flag_includes_category_in_prefix() -> None:
    doc = Document(
        page_content="切段焯水",
        metadata={"chunk_id": "a", "dish_name": "凉拌黄瓜", "category": "凉菜"},
    )
    fake = FakeCrossEncoder()
    RerankerModule(model_name="fake", model=fake, include_context=True).rerank("黄瓜", [doc])

    assert fake.seen_pairs[0][0][1] == "凉拌黄瓜（凉菜）｜切段焯水"


def test_dashscope_payload_contains_prefixed_documents(monkeypatch: Any) -> None:
    pool = [_chunk("a", "拍黄瓜", "拍黄瓜"), _chunk("b", "红烧肉", "红烧肉")]
    response = FakeResponse(
        200,
        {"output": {"results": [{"index": 0, "relevance_score": 0.5}, {"index": 1, "relevance_score": 0.4}]}},
    )
    calls = _install_fake_post(monkeypatch, lambda _: response)

    _dashscope_module(include_context=True).rerank("凉菜", pool)

    assert calls[0]["json"]["input"]["documents"] == ["拍黄瓜｜拍黄瓜", "红烧肉｜红烧肉"]


def test_context_flag_skips_missing_dish_name(monkeypatch: Any) -> None:
    doc = Document(page_content="无元数据片段", metadata={})
    fake = FakeCrossEncoder()
    RerankerModule(model_name="fake", model=fake, include_context=True).rerank("查询", [doc])

    assert fake.seen_pairs[0][0][1] == "无元数据片段"
