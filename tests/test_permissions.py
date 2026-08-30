"""Offline regressions for visibility metadata and role-based filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.documents import Document

from main import DEFAULT_ROLE, ROLE_ALLOWED_VISIBILITY, RecipeRAGSystem
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule


def _write_dish(root: Path, category_dir: str, dish: str) -> None:
    dish_dir = root / "dishes" / category_dir / dish
    dish_dir.mkdir(parents=True)
    (dish_dir / f"{dish}.md").write_text(
        f"# {dish}\n\n预估烹饪难度：★★\n\n## 必备原料\n\n鸡蛋两个。", encoding="utf-8"
    )


def test_internal_categories_get_internal_visibility(tmp_path: Path) -> None:
    _write_dish(tmp_path, "semi-finished", "速冻水饺")
    _write_dish(tmp_path, "soup", "番茄蛋汤")
    module = DataPreparationModule(tmp_path)
    documents = module.load_documents()

    by_name = {doc.metadata["dish_name"]: doc for doc in documents}
    assert by_name["速冻水饺"].metadata["visibility"] == "internal"
    assert by_name["番茄蛋汤"].metadata["visibility"] == "public"

    chunks = module.chunk_documents()
    for chunk in chunks:
        parent_visibility = by_name[chunk.metadata["dish_name"]].metadata["visibility"]
        assert chunk.metadata["visibility"] == parent_visibility


def test_internal_categories_are_configurable(tmp_path: Path) -> None:
    _write_dish(tmp_path, "dessert", "双皮奶")
    module = DataPreparationModule(tmp_path, internal_categories={"甜品"})
    documents = module.load_documents()
    assert documents[0].metadata["visibility"] == "internal"


class FakeVectorStore:
    def __init__(self, results: list[Document]) -> None:
        self.results = results

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        return self.results[:k]


def test_hybrid_search_client_path_post_filters_visibility() -> None:
    public_chunk = Document(
        page_content="公开菜谱",
        metadata={"chunk_id": "c-pub", "visibility": "public"},
    )
    internal_chunk = Document(
        page_content="内部工艺",
        metadata={"chunk_id": "c-int", "visibility": "internal"},
    )
    retrieval = RetrievalOptimizationModule(
        cast(Any, FakeVectorStore([public_chunk, internal_chunk])),
        [public_chunk, internal_chunk],
        candidate_k=5,
    )

    restricted = retrieval.hybrid_search("菜谱", top_k=2, expr='visibility == "public"')
    unrestricted = retrieval.hybrid_search("菜谱", top_k=2)

    assert [c.metadata["chunk_id"] for c in restricted] == ["c-pub"]
    assert len(unrestricted) == 2


class TestRoleVisibility:
    def test_guest_and_user_see_public_only(self) -> None:
        expr = RecipeRAGSystem._visibility_expr_for_role("guest")
        assert expr == 'visibility in ["public"]'
        assert RecipeRAGSystem._visibility_expr_for_role("user") == expr

    def test_staff_sees_everything_without_filter(self) -> None:
        assert RecipeRAGSystem._visibility_expr_for_role("staff") is None

    def test_unknown_role_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="未知角色"):
            RecipeRAGSystem._visibility_expr_for_role("admin")

    def test_default_role_is_registered(self) -> None:
        assert DEFAULT_ROLE in ROLE_ALLOWED_VISIBILITY
