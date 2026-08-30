"""Domain-swap regressions: a new DomainConfig must re-skin the whole engine.

These tests simulate the future "English-learning KB" migration without
touching any engine code — the whole point of the domain config layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.documents import Document

from rag_modules.agentic_rag import RecipeAgent, build_agent_tools
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.domain_config import (
    RECIPE_DOMAIN,
    DomainConfig,
    get_domain,
    set_domain,
)
from rag_modules.retrieval_optimization import RetrievalOptimizationModule

ENGLISH_DOMAIN = DomainConfig(
    name="english",
    assistant_name="英语学习助手",
    item_noun="知识点",
    item_classifier="个",
    unknown_item_name="未知知识点",
    category_mapping={
        "grammar": "语法",
        "vocabulary": "词汇",
        "phonetics": "音标",
        "exam": "真题",
    },
    difficulty_labels=["入门", "初级", "中级", "高级"],
    internal_categories=frozenset({"真题"}),
    agent_router_prompt="你是英语学习助手的任务规划器……",
    tool_descriptions={
        "search_recipes": "在英语学习知识库中检索语法、词汇讲解",
        "search_images": "检索配图示例",
        "list_by_metadata": "按类型/难度浏览知识点列表",
        "direct_answer": "与英语学习无关的问题直接回答",
        "query_param_example": "适合检索的知识点查询词，如'现在完成时'",
        "image_query_param": "图片内容描述",
        "category_param": "知识点类型（可选）",
        "difficulty_param": "难度级别（可选）",
        "category_param_browse": "知识点类型",
        "difficulty_param_browse": "难度级别",
        "direct_answer_param": "给用户的回答",
    },
    agent_no_results="抱歉，知识库中没有相关知识点。",
)


class FakeVectorStore:
    def __init__(self) -> None:
        self.results: list[Document] = []

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        return []

    def hybrid_search(self, query: str, k: int = 5, **kwargs: Any) -> list[Document]:
        return []


@pytest.fixture()
def english_domain():
    set_domain(ENGLISH_DOMAIN)
    yield ENGLISH_DOMAIN
    set_domain(RECIPE_DOMAIN)


def test_default_domain_is_recipe() -> None:
    assert get_domain().name == "recipe"
    assert "荤菜" in get_domain().category_labels


def test_custom_domain_swaps_metadata_vocabularies(
    tmp_path: Path, english_domain: DomainConfig
) -> None:
    grammar_dir = tmp_path / "kb" / "grammar" / "现在完成时"
    grammar_dir.mkdir(parents=True)
    (grammar_dir / "现在完成时.md").write_text("# 现在完成时", encoding="utf-8")

    module = DataPreparationModule(tmp_path / "kb")
    documents = module.load_documents()

    assert documents[0].metadata["category"] == "语法"
    assert documents[0].metadata["visibility"] == "public"
    assert module.get_supported_categories() == ["语法", "词汇", "音标", "真题"]
    assert module.get_supported_difficulties() == ["入门", "初级", "中级", "高级"]


def test_custom_domain_internal_categories_change_visibility(
    tmp_path: Path, english_domain: DomainConfig
) -> None:
    exam_dir = tmp_path / "kb" / "exam" / "2024年考研英语"
    exam_dir.mkdir(parents=True)
    (exam_dir / "2024年考研英语.md").write_text("# 考研真题", encoding="utf-8")

    module = DataPreparationModule(tmp_path / "kb")
    documents = module.load_documents()

    assert documents[0].metadata["visibility"] == "internal"


def test_custom_domain_swaps_agent_tool_enums_and_descriptions(
    english_domain: DomainConfig,
) -> None:
    tools = build_agent_tools()
    by_name = {tool["function"]["name"]: tool["function"] for tool in tools}

    assert by_name["search_recipes"]["description"] == "在英语学习知识库中检索语法、词汇讲解"
    category_enum = by_name["search_recipes"]["parameters"]["properties"]["category"]["enum"]
    assert "语法" in category_enum and "荤菜" not in category_enum
    difficulty_enum = by_name["list_by_metadata"]["parameters"]["properties"]["difficulty"]["enum"]
    assert difficulty_enum == ["入门", "初级", "中级", "高级"]


def test_custom_domain_swaps_refusal_answer(english_domain: DomainConfig) -> None:
    from tests.test_agentic_rag import (
        FakeAIMessage,
        FakeChatModel,
        FakeGeneration,
        _chunk,
        _make_data_module,
    )

    chunk = _chunk("c1", "现在完成时")
    llm = FakeChatModel(
        responses=[
            FakeAIMessage(
                tool_calls=[{"name": "search_recipes", "args": {"query": "虚拟语气"}, "id": "1"}]
            ),
            FakeAIMessage(content="虚拟语气 语法讲解"),
        ]
    )
    generation = FakeGeneration(llm)
    vectorstore = FakeVectorStore()
    vectorstore.supports_server_side_hybrid = True
    retrieval = RetrievalOptimizationModule(cast(Any, vectorstore), [chunk], candidate_k=5)
    agent = RecipeAgent(retrieval, _make_data_module([chunk]), cast(Any, generation), top_k=3)

    result = agent.invoke("什么是虚拟语气")

    assert result["answer"] == "抱歉，知识库中没有相关知识点。"
