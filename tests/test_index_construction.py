"""Offline tests for deterministic FAISS manifests and safe persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_modules.index_construction import IndexConstructionModule


class FakeEmbeddings(Embeddings):
    """Small deterministic embedding implementation that never loads a model."""

    def __init__(self) -> None:
        self.query_calls = 0
        self.document_calls = 0

    @staticmethod
    def _embed(text: str) -> list[float]:
        return [
            float(len(text) + 1),
            float(sum(ord(character) for character in text) % 997 + 1),
            float(text.count("菜") + 1),
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._embed(text)


@pytest.fixture()
def chunks() -> list[Document]:
    return [
        Document(
            page_content="# 红烧肉\n\n## 操作\n\n小火炒糖色。",
            metadata={
                "chunk_id": "chunk-red-cooking",
                "parent_id": "parent-red",
                "revision_id": "revision-red-1",
                "chunking_version": "markdown-headers-v1",
                "source_path": "dishes/meat_dish/红烧肉.md",
                "source": "C:/old-corpus/dishes/meat_dish/红烧肉.md",
                "chunk_index": 1,
                "dish_name": "红烧肉",
            },
        ),
        Document(
            page_content="# 拍黄瓜\n\n## 操作\n\n拍碎黄瓜后调味。",
            metadata={
                "chunk_id": "chunk-cucumber-cooking",
                "parent_id": "parent-cucumber",
                "revision_id": "revision-cucumber-1",
                "chunking_version": "markdown-headers-v1",
                "source_path": "dishes/vegetable_dish/拍黄瓜.md",
                "source": "C:/old-corpus/dishes/vegetable_dish/拍黄瓜.md",
                "chunk_index": 1,
                "dish_name": "拍黄瓜",
            },
        ),
    ]


def _module(index_path: Path, embeddings: FakeEmbeddings | None = None) -> IndexConstructionModule:
    return IndexConstructionModule(
        model_name="tests/fake-embedding",
        model_revision="fixture-v1",
        index_save_path=index_path,
        embeddings=embeddings or FakeEmbeddings(),
    )


def test_build_manifest_is_order_independent_and_records_configuration(
    tmp_path: Path, chunks: list[Document]
) -> None:
    embeddings = FakeEmbeddings()
    module = _module(tmp_path, embeddings)

    manifest = module.build_manifest(chunks)
    reordered_manifest = module.build_manifest(list(reversed(chunks)))

    assert manifest == reordered_manifest
    assert manifest["schema_version"] == module.MANIFEST_SCHEMA_VERSION
    assert manifest["storage_format"] == module.STORAGE_FORMAT
    assert manifest["chunk_count"] == 2
    assert manifest["document_count"] == 2
    assert manifest["chunking"] == {
        "versions": ["markdown-headers-v1"],
        "headers": ["#", "##", "###"],
        "strip_headers": False,
    }
    assert manifest["faiss"] == {"index_type": "IndexFlatL2", "metric": "L2"}
    assert len(manifest["corpus_fingerprint"]) == 64
    assert manifest["embedding"] == {
        "model_name": "tests/fake-embedding",
        "model_revision": "fixture-v1",
        "normalize_embeddings": True,
        "dimension": 3,
    }
    assert embeddings.query_calls == 1
    assert embeddings.document_calls == 0


def test_manifest_fingerprint_changes_when_chunk_content_changes(
    tmp_path: Path, chunks: list[Document]
) -> None:
    module = _module(tmp_path)
    original = module.build_manifest(chunks)
    changed_chunks = list(chunks)
    changed_chunks[0] = Document(
        page_content=chunks[0].page_content + "\n加入八角。",
        metadata=dict(chunks[0].metadata),
    )

    changed = module.build_manifest(changed_chunks)

    assert changed["corpus_fingerprint"] != original["corpus_fingerprint"]


@pytest.mark.parametrize(
    "invalid_chunks, message",
    [
        ([Document(page_content="缺少 ID", metadata={})], "chunk_id"),
        (
            [
                Document(page_content="第一块", metadata={"chunk_id": "duplicate"}),
                Document(page_content="第二块", metadata={"chunk_id": "duplicate"}),
            ],
            "唯一",
        ),
    ],
)
def test_build_vector_index_rejects_missing_or_duplicate_chunk_ids(
    tmp_path: Path,
    invalid_chunks: list[Document],
    message: str,
) -> None:
    module = _module(tmp_path)

    with pytest.raises(ValueError, match=message):
        module.build_vector_index(invalid_chunks)


def test_safe_index_round_trip_uses_manifest_and_json_document_map(
    tmp_path: Path, chunks: list[Document]
) -> None:
    index_path = tmp_path / "vector_index"
    builder = _module(index_path)

    vectorstore = builder.build_vector_index(chunks)
    builder.save_index()

    assert vectorstore.index.ntotal == len(chunks)
    assert {path.name for path in index_path.iterdir()} == {
        builder.INDEX_FILENAME,
        builder.DOCUMENTS_FILENAME,
        builder.MANIFEST_FILENAME,
    }
    assert not (index_path / "index.pkl").exists()

    manifest = json.loads((index_path / builder.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    document_map = json.loads((index_path / builder.DOCUMENTS_FILENAME).read_text(encoding="utf-8"))
    assert manifest["chunk_count"] == len(chunks)
    assert set(manifest["artifacts"]) == {
        builder.INDEX_FILENAME,
        builder.DOCUMENTS_FILENAME,
    }
    assert len(document_map["documents"]) == len(chunks)
    assert "page_content" not in document_map

    loader = _module(index_path)
    loaded = loader.load_index(chunks)

    assert loaded is not None
    assert loaded.index.ntotal == len(chunks)
    result = loader.similarity_search("黄瓜菜", k=1)
    assert len(result) == 1
    assert result[0].metadata["chunk_id"] in {chunk.metadata["chunk_id"] for chunk in chunks}


def test_load_index_hydrates_documents_from_current_corpus(
    tmp_path: Path, chunks: list[Document]
) -> None:
    index_path = tmp_path / "vector_index"
    builder = _module(index_path)
    builder.build_vector_index(chunks)
    builder.save_index()

    relocated_chunks = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata["source"] = "D:/relocated-corpus/" + metadata["source_path"]
        relocated_chunks.append(Document(page_content=chunk.page_content, metadata=metadata))

    loader = _module(index_path)
    loaded = loader.load_index(relocated_chunks)

    assert loaded is not None
    hydrated_sources: set[str] = set()
    for docstore_id in loaded.index_to_docstore_id.values():
        hydrated = loaded.docstore.search(docstore_id)
        assert isinstance(hydrated, Document)
        hydrated_sources.add(hydrated.metadata["source"])
    assert hydrated_sources == {chunk.metadata["source"] for chunk in relocated_chunks}
    assert all(source.startswith("D:/relocated-corpus/") for source in hydrated_sources)


def test_load_index_rejects_stale_corpus_without_deserializing_pickle(
    tmp_path: Path, chunks: list[Document]
) -> None:
    index_path = tmp_path / "vector_index"
    builder = _module(index_path)
    builder.build_vector_index(chunks)
    builder.save_index()

    changed_chunks = list(chunks)
    changed_chunks[0] = Document(
        page_content=chunks[0].page_content + "\n内容已经更新。",
        metadata=dict(chunks[0].metadata),
    )
    loader = _module(index_path)

    assert loader.load_index(changed_chunks) is None
    assert loader.vectorstore is None


def test_load_index_rejects_tampered_artifact(tmp_path: Path, chunks: list[Document]) -> None:
    index_path = tmp_path / "vector_index"
    builder = _module(index_path)
    builder.build_vector_index(chunks)
    builder.save_index()

    documents_path = index_path / builder.DOCUMENTS_FILENAME
    documents_path.write_text(documents_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    loader = _module(index_path)

    assert loader.load_index(chunks) is None
    assert loader.vectorstore is None
