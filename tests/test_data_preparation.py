"""Offline regression tests for the V0 data-preparation baseline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from shutil import copytree

import pytest
from langchain_core.documents import Document

from rag_modules.data_preparation import DataPreparationModule

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REAL_DATA_PATH = WORKSPACE_ROOT / "data" / "C8" / "cook"


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture()
def recipe_dir(tmp_path: Path) -> Path:
    """Create a small but representative recipe tree without external services."""
    data_dir = tmp_path / "cook"

    _write_markdown(
        data_dir / "dishes" / "meat_dish" / "红烧肉.md",
        """# 红烧肉

预估烹饪难度：★★

## 必备原料和工具

- 五花肉
- 冰糖

## 操作

1. 五花肉切块。
2. 小火炒糖色。
""",
    )
    _write_markdown(
        data_dir / "dishes" / "vegetable_dish" / "拍黄瓜.md",
        """# 拍黄瓜

预估烹饪难度：★

## 必备原料和工具

- 黄瓜
- 蒜

## 操作

拍碎黄瓜后调味。
""",
    )
    _write_markdown(
        data_dir / "dishes" / "semi-finished" / "速冻水饺.md",
        """# 速冻水饺

预估烹饪难度：★★★

## 操作

水开后下入速冻水饺，煮熟即可。
""",
    )
    _write_markdown(
        data_dir / "dishes" / "template" / "示例菜" / "示例菜.md",
        """# 示例菜

预估烹饪难度：★★★★★

## 操作

这只是编写菜谱时使用的模板，不应进入知识库。
""",
    )
    return data_dir


def _chunk_identity(chunk: Document) -> tuple[str, int, str]:
    """Return the logical identity of a chunk, excluding generated identifiers."""
    return (
        chunk.metadata["dish_name"],
        chunk.metadata["chunk_index"],
        chunk.page_content,
    )


def test_module_initializes_with_empty_collections(recipe_dir: Path) -> None:
    module = DataPreparationModule(str(recipe_dir))

    assert module.documents == []
    assert module.chunks == []
    assert module.parent_child_map == {}
    assert isinstance(module.parent_child_map, dict)


def test_load_documents_enriches_metadata_and_skips_template(recipe_dir: Path) -> None:
    module = DataPreparationModule(str(recipe_dir))

    documents = module.load_documents()

    assert module.documents is documents
    assert len(documents) == 3
    assert {doc.metadata["dish_name"] for doc in documents} == {
        "红烧肉",
        "拍黄瓜",
        "速冻水饺",
    }
    assert all(doc.metadata["doc_type"] == "parent" for doc in documents)
    assert all(doc.metadata["parent_id"] for doc in documents)
    assert len({doc.metadata["parent_id"] for doc in documents}) == len(documents)
    assert all("template" not in Path(doc.metadata["source"]).parts for doc in documents)

    by_name = {doc.metadata["dish_name"]: doc for doc in documents}
    assert by_name["红烧肉"].metadata["category"] == "荤菜"
    assert by_name["拍黄瓜"].metadata["category"] == "素菜"
    assert by_name["速冻水饺"].metadata["category"] == "半成品"
    assert by_name["红烧肉"].metadata["difficulty"] == "简单"
    assert by_name["拍黄瓜"].metadata["difficulty"] == "非常简单"
    assert by_name["速冻水饺"].metadata["difficulty"] == "中等"


def test_real_dataset_discovers_323_files_and_loads_322_recipes() -> None:
    """Protect the known V0 corpus while explicitly excluding its authoring template."""
    assert REAL_DATA_PATH.is_dir(), f"C8 数据目录不存在: {REAL_DATA_PATH}"
    assert len(list(REAL_DATA_PATH.rglob("*.md"))) == 323

    module = DataPreparationModule(str(REAL_DATA_PATH))
    documents = module.load_documents()

    assert len(documents) == 322
    assert not any(doc.metadata["dish_name"] == "示例菜" for doc in documents)
    assert all("template" not in Path(doc.metadata["source"]).parts for doc in documents)

    semi_finished = [
        doc for doc in documents if "semi-finished" in Path(doc.metadata["source"]).parts
    ]
    assert len(semi_finished) == 10
    assert {doc.metadata["category"] for doc in semi_finished} == {"半成品"}


def test_chunk_documents_returns_a_flat_document_list(recipe_dir: Path) -> None:
    module = DataPreparationModule(str(recipe_dir))
    module.load_documents()

    chunks = module.chunk_documents()

    assert module.chunks is chunks
    assert len(chunks) > len(module.documents)
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert not any(isinstance(chunk, list) for chunk in chunks)
    assert [chunk.metadata["batch_index"] for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)

    for chunk in chunks:
        assert chunk.metadata["doc_type"] == "child"
        assert isinstance(chunk.metadata["chunk_index"], int)
        assert chunk.metadata["chunk_size"] == len(chunk.page_content)
        assert module.parent_child_map[chunk.metadata["chunk_id"]] == chunk.metadata["parent_id"]


def test_chunk_ids_are_deterministic_across_reloads(recipe_dir: Path) -> None:
    first = DataPreparationModule(str(recipe_dir))
    first.load_documents()
    first_chunks = first.chunk_documents()

    second = DataPreparationModule(str(recipe_dir))
    second.load_documents()
    second_chunks = second.chunk_documents()

    first_ids = {_chunk_identity(chunk): chunk.metadata["chunk_id"] for chunk in first_chunks}
    second_ids = {_chunk_identity(chunk): chunk.metadata["chunk_id"] for chunk in second_chunks}

    assert first_ids == second_ids
    assert len(first_ids) == len(first_chunks)


def test_document_and_chunk_ids_do_not_depend_on_absolute_data_root(
    recipe_dir: Path, tmp_path: Path
) -> None:
    relocated_dir = tmp_path / "relocated" / "cook"
    copytree(recipe_dir, relocated_dir)

    original = DataPreparationModule(str(recipe_dir))
    original_documents = original.load_documents()
    original_chunks = original.chunk_documents()

    relocated = DataPreparationModule(str(relocated_dir))
    relocated_documents = relocated.load_documents()
    relocated_chunks = relocated.chunk_documents()

    original_parent_ids = {
        doc.metadata["dish_name"]: doc.metadata["parent_id"] for doc in original_documents
    }
    relocated_parent_ids = {
        doc.metadata["dish_name"]: doc.metadata["parent_id"] for doc in relocated_documents
    }
    original_chunk_ids = {
        _chunk_identity(chunk): chunk.metadata["chunk_id"] for chunk in original_chunks
    }
    relocated_chunk_ids = {
        _chunk_identity(chunk): chunk.metadata["chunk_id"] for chunk in relocated_chunks
    }

    assert original_parent_ids == relocated_parent_ids
    assert original_chunk_ids == relocated_chunk_ids


def test_statistics_and_filters_are_instance_methods(recipe_dir: Path) -> None:
    module = DataPreparationModule(str(recipe_dir))
    assert module.get_statistics() == {}

    module.load_documents()
    chunks = module.chunk_documents()
    statistics = module.get_statistics()

    assert statistics["total_documents"] == 3
    assert statistics["total_chunks"] == len(chunks)
    assert statistics["categories"] == {"荤菜": 1, "素菜": 1, "半成品": 1}
    assert statistics["difficulties"] == {"简单": 1, "非常简单": 1, "中等": 1}
    assert statistics["avg_chunk_size"] == pytest.approx(
        sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
    )
    assert sum(statistics["categories"].values()) == statistics["total_documents"]
    assert sum(statistics["difficulties"].values()) == statistics["total_documents"]

    meat_dishes = module.filter_documents_by_category("荤菜")
    assert [doc.metadata["dish_name"] for doc in meat_dishes] == ["红烧肉"]
    very_easy_dishes = module.filter_documents_by_difficulty("非常简单")
    assert [doc.metadata["dish_name"] for doc in very_easy_dishes] == ["拍黄瓜"]
    assert "半成品" in module.get_supported_categories()


def test_get_parent_documents_deduplicates_and_ranks_by_matching_chunks(
    recipe_dir: Path,
) -> None:
    module = DataPreparationModule(str(recipe_dir))
    module.load_documents()
    chunks = module.chunk_documents()

    meat_chunks = [chunk for chunk in chunks if chunk.metadata["dish_name"] == "红烧肉"]
    vegetable_chunk = next(chunk for chunk in chunks if chunk.metadata["dish_name"] == "拍黄瓜")
    orphan = Document(page_content="无父文档", metadata={"parent_id": "missing"})

    parents = module.get_parent_documents([meat_chunks[0], vegetable_chunk, meat_chunks[1], orphan])

    assert [doc.metadata["dish_name"] for doc in parents] == ["红烧肉", "拍黄瓜"]
    assert len({doc.metadata["parent_id"] for doc in parents}) == len(parents)
    assert all(doc.metadata["doc_type"] == "parent" for doc in parents)


def test_chunking_requires_loaded_documents(recipe_dir: Path) -> None:
    module = DataPreparationModule(str(recipe_dir))

    with pytest.raises(ValueError, match="先加载文档"):
        module.chunk_documents()


def test_all_chunks_have_a_parent_in_the_loaded_document_set(recipe_dir: Path) -> None:
    module = DataPreparationModule(str(recipe_dir))
    documents = module.load_documents()
    chunks = module.chunk_documents()

    parent_ids = {doc.metadata["parent_id"] for doc in documents}
    actual_counts = Counter(chunk.metadata["parent_id"] for chunk in chunks)

    assert set(actual_counts) == parent_ids
    assert all(count >= 1 for count in actual_counts.values())
