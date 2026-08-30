"""Offline regressions for Chinese sparse retrieval and RRF fusion."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.documents import Document

from rag_modules.retrieval_optimization import (
    RetrievalOptimizationModule,
    tokenize_chinese,
)


class FakeVectorStore:
    """Return a fixed dense ranking without loading an embedding model."""

    def __init__(self, results: list[Document]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        self.calls.append((query, k))
        return self.results[:k]


def _document(
    chunk_id: str,
    dish_name: str,
    content: str,
    *,
    category: str = "其他",
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "dish_name": dish_name,
            "category": category,
        },
    )


def test_tokenize_chinese_produces_characters_bigrams_and_exact_phrase() -> None:
    tokens = tokenize_chinese("红烧鲤鱼 BGE-v1")

    assert {"红", "烧", "鲤", "鱼"}.issubset(tokens)
    assert {"红烧", "烧鲤", "鲤鱼"}.issubset(tokens)
    assert "红烧鲤鱼" in tokens
    assert "bge" in tokens
    assert "v1" in tokens


def test_chinese_bm25_recalls_the_exact_dish_name() -> None:
    red_carp = _document(
        "red-carp",
        "红烧鲤鱼",
        "# 红烧鲤鱼\n\n鲤鱼煎至两面金黄，再加入酱油烧制。",
        category="水产",
    )
    chunks = [
        _document(
            "steamed-fish",
            "清蒸鲈鱼",
            "# 清蒸鲈鱼\n\n鲈鱼加葱姜清蒸。",
            category="水产",
        ),
        _document(
            "tomato-eggs",
            "番茄炒蛋",
            "# 番茄炒蛋\n\n番茄和鸡蛋下锅翻炒。",
            category="荤菜",
        ),
        red_carp,
        _document(
            "cucumber",
            "拍黄瓜",
            "# 拍黄瓜\n\n黄瓜拍碎后凉拌。",
            category="素菜",
        ),
    ]
    retrieval = RetrievalOptimizationModule(
        cast(Any, FakeVectorStore([])), chunks, candidate_k=len(chunks)
    )

    results = retrieval.bm25_retriever.invoke("红烧鲤鱼怎么做")

    assert results
    assert results[0].metadata["chunk_id"] == red_carp.metadata["chunk_id"]


def test_rrf_fuses_and_deduplicates_by_chunk_id() -> None:
    first = _document("first", "红烧肉", "红烧肉操作")
    second = _document("second", "拍黄瓜", "拍黄瓜操作")
    third = _document("third", "蛋炒饭", "蛋炒饭操作")
    retrieval = RetrievalOptimizationModule(
        cast(Any, FakeVectorStore([first, second])), [first, second, third]
    )

    fused = retrieval._rrf_rerank(
        vector_docs=[first, second],
        bm25_docs=[first, third],
        k=60,
    )

    assert [item.metadata["chunk_id"] for item in fused] == [
        "first",
        "second",
        "third",
    ]
    assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]
    assert fused[1].metadata["rrf_score"] == fused[2].metadata["rrf_score"]
    assert "rrf_score" not in first.metadata


def test_rrf_keeps_identical_content_when_chunk_ids_are_different() -> None:
    first = _document("first-copy", "菜谱甲", "相同的操作正文")
    second = _document("second-copy", "菜谱乙", "相同的操作正文")
    retrieval = RetrievalOptimizationModule(
        cast(Any, FakeVectorStore([first, second])), [first, second]
    )

    fused = retrieval._rrf_rerank([first], [second], k=60)

    assert [item.metadata["chunk_id"] for item in fused] == [
        "first-copy",
        "second-copy",
    ]


def test_hybrid_search_uses_requested_candidate_pool_and_returns_top_k() -> None:
    first = _document("first", "红烧肉", "红烧肉需要五花肉")
    second = _document("second", "拍黄瓜", "拍黄瓜需要黄瓜")
    third = _document("third", "蛋炒饭", "蛋炒饭需要米饭")
    vectorstore = FakeVectorStore([third, second, first])
    retrieval = RetrievalOptimizationModule(
        cast(Any, vectorstore), [first, second, third], candidate_k=2
    )

    results = retrieval.hybrid_search("红烧肉", top_k=2, candidate_k=3)

    assert len(results) == 2
    assert vectorstore.calls == [("红烧肉", 3)]
    assert all("rrf_score" in result.metadata for result in results)


def test_hybrid_search_supports_top_k_greater_than_five() -> None:
    chunks = [
        _document(
            f"chunk-{index}",
            f"菜谱{index}",
            f"# 菜谱{index}\n\n第{index}份不同的食谱正文和操作步骤。",
        )
        for index in range(8)
    ]
    vectorstore = FakeVectorStore(chunks)
    retrieval = RetrievalOptimizationModule(cast(Any, vectorstore), chunks, candidate_k=3)

    results = retrieval.hybrid_search("食谱操作步骤", top_k=7)

    assert len(results) == 7
    assert vectorstore.calls == [("食谱操作步骤", 7)]
    assert len({result.metadata["chunk_id"] for result in results}) == 7
