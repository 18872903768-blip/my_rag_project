"""Offline regressions for citation formatting."""

from __future__ import annotations

from langchain_core.documents import Document

from rag_modules.generation_integration import format_citations


def _parent(dish: str, source: str) -> Document:
    return Document(page_content=f"# {dish}", metadata={"dish_name": dish, "source_path": source})


def test_format_citations_lists_unique_sources() -> None:
    docs = [
        _parent("简易红烧肉", "dishes/meat_dish/简易红烧肉/简易红烧肉.md"),
        _parent("简易红烧肉", "dishes/meat_dish/简易红烧肉/简易红烧肉.md"),  # 去重
        _parent("湖南家常红烧肉", "dishes/meat_dish/湖南家常红烧肉/湖南家常红烧肉.md"),
    ]

    text = format_citations(docs)

    assert "简易红烧肉（dishes/meat_dish/简易红烧肉/简易红烧肉.md）" in text
    assert "湖南家常红烧肉" in text
    assert text.count("简易红烧肉（") == 1  # 重复条目只出现一次


def test_format_citations_empty_returns_empty_string() -> None:
    assert format_citations([]) == ""
    assert format_citations([Document(page_content="x", metadata={})]) == ""
