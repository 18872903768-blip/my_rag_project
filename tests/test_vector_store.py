"""Offline regressions for the Milvus backend (no server required)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_modules.retrieval_optimization import document_matches_expr
from rag_modules.vector_store import (
    MilvusVectorStore,
    build_filter_expr,
    combine_expr,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


@dataclass
class FakeInsertResult:
    insert_count: int


class FakeMilvusClient:
    """Records calls without touching a real Milvus server."""

    def __init__(self, *, row_count: int = 0, analyzer_ok: bool = True) -> None:
        self.row_count = row_count
        self.analyzer_ok = analyzer_ok
        self.inserted_rows: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self._exists = row_count > 0

    def get_server_version(self) -> str:
        return "fake-2.5.14"

    def has_collection(self, name: str) -> bool:
        return self._exists

    def drop_collection(self, name: str) -> None:
        self.row_count = 0
        self._exists = False
        self.calls.append(f"drop:{name}")

    def create_collection(self, **kwargs: Any) -> None:
        self.calls.append("create")
        self._exists = True
        self.created = kwargs

    def prepare_index_params(self) -> Any:
        from pymilvus import MilvusClient

        return MilvusClient.prepare_index_params()

    def insert(self, collection_name: str, data: list[dict[str, Any]]) -> FakeInsertResult:
        self.inserted_rows.extend(data)
        return FakeInsertResult(insert_count=len(data))

    def flush(self, collection_name: str) -> None:
        self.calls.append("flush")

    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        return {"row_count": len(self.inserted_rows)}


def _chunk(chunk_id: str, content: str, **metadata: Any) -> Document:
    base = {
        "chunk_id": chunk_id,
        "parent_id": "parent-1",
        "revision_id": "revision-1",
        "doc_type": "child",
        "category": "荤菜",
        "dish_name": "红烧肉",
        "difficulty": "中等",
        "modality": "text",
        "visibility": "public",
        "source_path": "dishes/meat_dish/红烧肉/红烧肉.md",
        "source": "E:/data/红烧肉.md",
        "source_hash": "abc",
        "chunking_version": "markdown-headers-v1",
        "chunk_index": 0,
        "chunk_size": len(content),
    }
    base.update(metadata)
    return Document(page_content=content, metadata=base)


def _make_store(tmp_path: Path, client: FakeMilvusClient) -> MilvusVectorStore:
    store = MilvusVectorStore(
        collection_name="test_collection",
        uri="http://localhost:19530",
        embeddings=FakeEmbeddings(),
        manifest_dir=tmp_path,
    )
    store._client = client
    return store


def test_build_filter_expr_handles_values_lists_and_escaping() -> None:
    assert build_filter_expr(None) is None
    assert build_filter_expr({}) is None
    assert build_filter_expr({"category": "荤菜"}) == 'category == "荤菜"'
    assert build_filter_expr({"visibility": ["public"]}) == 'visibility in ["public"]'
    assert (
        build_filter_expr({"category": "荤菜", "difficulty": "简单"})
        == 'category == "荤菜" && difficulty == "简单"'
    )
    assert build_filter_expr({"dish_name": '带"引号"的菜'}) == 'dish_name == "带\\"引号\\"的菜"'
    assert build_filter_expr({"visibility": [None, ""]}) is None


def test_build_filter_expr_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="不支持"):
        build_filter_expr({"payload; drop table": "x"})


def test_combine_expr_joins_truthy_parts() -> None:
    assert combine_expr(None, None) is None
    assert combine_expr('a == "1"', None) == 'a == "1"'
    assert combine_expr('a == "1"', 'b == "2"') == 'a == "1" && b == "2"'


def test_build_index_creates_collection_and_skips_sparse_field(tmp_path: Path) -> None:
    client = FakeMilvusClient()
    store = _make_store(tmp_path, client)
    chunks = [_chunk("c1", "红烧肉需要五花肉"), _chunk("c2", "炖煮四十分钟")]

    rebuilt = store.build_index(chunks)

    assert rebuilt is True
    assert "create" in client.calls and "flush" in client.calls
    assert len(client.inserted_rows) == 2
    for row in client.inserted_rows:
        assert "sparse" not in row  # server-side BM25 function owns this field
        assert row["vector"] == [float(len(row["content"])), 1.0]
        assert row["modality"] == "text"
    assert (tmp_path / "milvus_manifest.json").is_file()
    assert client.created["collection_name"] == "test_collection"


def test_build_index_reuses_matching_collection(tmp_path: Path) -> None:
    client = FakeMilvusClient()
    store = _make_store(tmp_path, client)
    chunks = [_chunk("c1", "红烧肉需要五花肉")]
    store.build_index(chunks)

    rebuilt_again = store.build_index(chunks)

    assert rebuilt_again is False
    assert len(client.inserted_rows) == 1  # no duplicate inserts
    assert "drop:test_collection" not in client.calls


def test_build_index_rebuilds_when_manifest_fingerprint_changes(tmp_path: Path) -> None:
    client = FakeMilvusClient()
    store = _make_store(tmp_path, client)
    store.build_index([_chunk("c1", "旧内容")])

    rebuilt = store.build_index([_chunk("c1", "新内容更新了语料")])

    assert rebuilt is True
    assert "drop:test_collection" in client.calls


def test_count_reflects_inserted_rows(tmp_path: Path) -> None:
    client = FakeMilvusClient()
    store = _make_store(tmp_path, client)
    store.build_index([_chunk("c1", "内容"), _chunk("c2", "更多内容")])

    assert store.count() == 2


def test_document_matches_expr_supports_generated_subset() -> None:
    document = _chunk("c1", "内容", category="荤菜", visibility="public")

    assert document_matches_expr(document, 'category == "荤菜"')
    assert document_matches_expr(document, 'visibility in ["public", "internal"]')
    assert document_matches_expr(
        document, 'category == "荤菜" && visibility in ["public"]'
    )
    assert not document_matches_expr(document, 'category == "素菜"')
    assert not document_matches_expr(document, 'visibility in ["internal"]')
