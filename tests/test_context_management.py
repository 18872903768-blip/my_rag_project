"""Offline regressions for the agent-pipeline Context Manager."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from rag_modules.context_management import (
    ContextItem,
    ContextManager,
    build_agent_context_items,
    count_tokens,
)
from rag_modules.generation_integration import GenerationIntegrationModule


def _parent(dish: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={"dish_name": dish, "category": "荤菜", "difficulty": "简单"},
    )


def _item(
    content: str,
    source_type: str,
    priority: int,
    order: int = 0,
    truncatable: bool = True,
    item_id: str = "",
) -> ContextItem:
    return ContextItem(
        content=content,
        source_type=source_type,
        priority=priority,
        order=order,
        truncatable=truncatable,
        item_id=item_id,
    )


def test_count_tokens_positive_and_monotonic() -> None:
    short = count_tokens("红烧肉")
    long = count_tokens("红烧肉" * 100)

    assert short >= 1
    assert long > short


def test_build_within_budget_keeps_all_in_render_order() -> None:
    manager = ContextManager(budget_tokens=10_000)
    items = [
        _item("用户: 红烧肉怎么做", "conversation", 2, order=1, item_id="c1"),
        _item("助手: 红烧肉的做法如下", "conversation", 2, order=0, item_id="c2"),
        _item("【食谱 1】 红烧肉\n正文", "retrieval", 1, order=1, item_id="r1"),
    ]

    result = manager.build(items)

    assert result.stats["input_items"] == 3
    assert result.stats["dropped"] == []
    # 渲染顺序：对话块在最前，检索其次
    assert result.context.index("对话历史") < result.context.index("【食谱 1】")
    # 时间序渲染：用户在助手之前（传入顺序）
    assert result.context.index("用户: 红烧肉") < result.context.index("助手: 红烧肉")


def test_build_over_budget_drops_memory_keeps_retrieval() -> None:
    manager = ContextManager(budget_tokens=500)
    items = [
        _item("【食谱 1】 红烧肉\n" + "步骤 " * 100, "retrieval", 1, item_id="r1"),
        _item("用户: 之前问了什么", "conversation", 2, item_id="c1"),
        _item("- 对花生过敏（allergen）", "memory", 3, item_id="m1"),
    ]

    result = manager.build(items)

    assert "r1" not in result.stats["dropped"]
    assert "m1" in result.stats["dropped"]
    assert "allergen" not in result.context


def test_build_tight_budget_keeps_newest_conversation_turn() -> None:
    manager = ContextManager(budget_tokens=300)
    items = [
        _item("用户: 第一轮很老的问题 " + "旧 " * 60, "conversation", 2, order=2, item_id="c_old"),
        _item("助手: 老回答", "conversation", 2, order=1, item_id="c_mid"),
        _item("用户: 最新一轮的问题", "conversation", 2, order=0, item_id="c_new"),
    ]

    result = manager.build(items)

    # 最新轮次 order=0 优先保留；最老的先被丢弃/截断
    assert "c_new" not in result.stats["dropped"]
    kept_text = result.context
    assert "最新一轮的问题" in kept_text


def test_build_empty_fallback_truncates_first_item() -> None:
    manager = ContextManager(budget_tokens=60)
    items = [
        _item("【食谱 1】 红烧肉\n" + "很长的正文 " * 200, "retrieval", 1, truncatable=True, item_id="r1"),
        _item("【食谱 2】 短内容", "retrieval", 1, item_id="r2"),
    ]

    result = manager.build(items)

    # 预算极小：非首篇整条丢弃，首篇按"至少保留一部分"兜底截断
    assert result.context.startswith("【食谱 1】")
    assert "r2" in result.stats["dropped"]
    assert "r1" in result.stats["truncated"]


def test_exact_duplicate_content_deduplicated() -> None:
    manager = ContextManager(budget_tokens=10_000)
    items = [
        _item("【食谱 1】 红烧肉\n相同正文", "retrieval", 1, item_id="r1"),
        _item("【食谱 1】 红烧肉\n相同正文", "retrieval", 1, item_id="r2"),
    ]

    result = manager.build(items)

    assert result.context.count("相同正文") == 1


def test_build_agent_context_items_formats_recipe_headers() -> None:
    history = [
        {"role": "user", "content": "徽派红烧肉怎么做"},
        {"role": "assistant", "content": "做法是……"},
        {"role": "user", "content": "那第二种呢"},
    ]
    parents = [_parent("徽派红烧肉", "正文A"), _parent("南派红烧肉", "正文B")]

    items = build_agent_context_items(history, parents)

    conv = [i for i in items if i.source_type == "conversation"]
    retr = [i for i in items if i.source_type == "retrieval"]

    # conversation 分配顺序最新优先（standalone 用），且只取最近 6 轮
    assert [i.order for i in conv] == [0, 1, 2]
    assert conv[2].content.startswith("用户: 徽派红烧肉怎么做")
    # retrieval 保留【食谱 N】头，与 format_citations 引用序号对齐
    assert retr[0].content.startswith("【食谱 1】 徽派红烧肉")
    assert retr[1].content.startswith("【食谱 2】 南派红烧肉")
    assert retr[0].truncatable is True and retr[1].truncatable is False


def test_generate_basic_answer_context_text_bypasses_build_context() -> None:
    """classic 路径不传 context_text 行为不变；agent 路径传入则直接使用。"""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from pydantic import ConfigDict

    class RecordingChatModel(FakeListChatModel):
        model_config = ConfigDict(extra="allow")  # 允许挂 prompts 记录列表

        def __init__(self) -> None:
            super().__init__(responses=["ok", "ok"])
            self.prompts: list[str] = []

        def _call(self, messages: Any, stop: Any = None, **kwargs: Any) -> Any:
            self.prompts.append("\n".join(str(m.content) for m in messages))
            return super()._call(messages, stop, **kwargs)

    llm = RecordingChatModel()
    module = GenerationIntegrationModule(llm=llm, grounded_answer=False)
    docs = [_parent("徽派红烧肉", "正文A")]

    # classic 路径：仍走 _build_context 的【食谱 N】格式
    module.generate_basic_answer("徽派红烧肉", docs)
    assert "【食谱 1】 徽派红烧肉" in llm.prompts[-1]

    # agent 路径：context_text 原样进入 prompt，不再经过 _build_context
    module.generate_basic_answer("徽派红烧肉", docs, context_text="CTX_MANAGER_OUTPUT")
    assert "CTX_MANAGER_OUTPUT" in llm.prompts[-1]
    assert "【食谱 1】" not in llm.prompts[-1]
