"""Agent 管线的统一上下文管理：多源 ContextItem 组装（token 预算 + 优先级 + 去重）。

针对的现状：agent 生成侧把对话历史硬拼进 question（`_history_prefix` 固定
4 轮 x 120 字符）、检索 parents 走 `_build_context` 的字符顺序拼接，两类
来源（后续加入记忆后是三类）没有统一的预算与降级机制。本模块把所有上下文
来源抽象为 ContextItem，按优先级分配 token 预算，超限按来源类型降级：

- retrieval (P1)：整条保留或整条丢弃（保留 rerank 靠前的），预算耗尽时兜底
  截断第一条，对齐 `_build_context` 的"至少保留首篇一部分"行为；
- conversation (P2)：可截断；分配时最新轮次优先保留，渲染保持时间顺序；
- memory (P3)：超预算直接丢弃（偏好信息缺失不致命）。

仅服务 agent 管线（`RAG_CONTEXT_MANAGER=true` 启用）；classic 链路的
`_build_context` 保持原样，行为零变化。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

SOURCE_CONVERSATION = "conversation"
SOURCE_RETRIEVAL = "retrieval"
SOURCE_MEMORY = "memory"

PRIORITY_RETRIEVAL = 1
PRIORITY_CONVERSATION = 2
PRIORITY_MEMORY = 3

# token 计数：tiktoken（ragas 依赖已存在）优先；编码表不可用时按中文经验估算
try:
    import tiktoken as _tiktoken

    _ENCODING = _tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001 - tiktoken 不可用只影响计数精度
    _ENCODING = None


def count_tokens(text: str) -> int:
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text))
        except Exception:  # noqa: BLE001
            pass
    return max(1, int(len(text) * 0.85))


@dataclass
class ContextItem:
    """一条候选上下文：content 为最终进入 prompt 的文本。"""

    content: str
    source_type: str  # conversation | retrieval | memory
    priority: int  # 数字越小越优先保留（retrieval=1, conversation=2, memory=3）
    order: int = 0  # 同优先级内的分配顺序（conversation 建议最新轮次取更小值）
    truncatable: bool = True
    item_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ContextResult:
    context: str
    stats: dict


class ContextManager:
    """按 token 预算组装多源上下文；输出可直接作为生成 prompt 的 context 槽。"""

    def __init__(
        self,
        budget_tokens: int = 6000,
        *,
        dedup_embeddings: Any | None = None,
        dedup_threshold: float = 0.93,
    ):
        if budget_tokens < 1:
            raise ValueError("budget_tokens 必须为正整数")
        self.budget_tokens = budget_tokens
        self.dedup_embeddings = dedup_embeddings
        self.dedup_threshold = dedup_threshold

    # ------------------------------------------------------------------ build

    def build(
        self,
        items: list[ContextItem],
        *,
        budget_tokens: int | None = None,
        image_paths: list[str] | None = None,
    ) -> ContextResult:
        budget = budget_tokens or self.budget_tokens
        input_tokens = sum(count_tokens(item.content) for item in items)
        deduped = self._dedup(items)

        # 分配：优先级升序，同优先级按 order 升序（conversation 最新优先）
        kept_ids: set[int] = set()
        truncated_ids: list[str] = []
        dropped_ids: list[str] = []
        kept_contents: dict[int, str] = {}
        used = 0
        for item in sorted(deduped, key=lambda i: (i.priority, i.order)):
            tokens = count_tokens(item.content)
            remaining = budget - used
            key = item.item_id or f"{item.source_type}:{len(kept_ids)}"
            if tokens <= remaining:
                kept_ids.add(id(item))
                kept_contents[id(item)] = item.content
                used += tokens
                continue
            if not item.truncatable or remaining <= 50:
                dropped_ids.append(key)
                continue
            cut = self._truncate_to_tokens(item.content, remaining)
            if not cut:
                dropped_ids.append(key)
                continue
            kept_ids.add(id(item))
            kept_contents[id(item)] = cut
            used += count_tokens(cut)
            truncated_ids.append(key)

        # 兜底：一条都没保留时截断第一条可截断项，避免空上下文
        if not kept_ids:
            for item in deduped:
                cut = self._truncate_to_tokens(item.content, max(budget // 2, 200))
                if cut:
                    kept_ids.add(id(item))
                    kept_contents[id(item)] = cut
                    truncated_ids.append(item.item_id or item.source_type)
                    break

        # 渲染保持调用方传入顺序（时间序/引用序），不按分配序
        kept_in_order = [item for item in deduped if id(item) in kept_ids]
        context = self._render(kept_in_order, kept_contents, image_paths)
        stats = {
            "input_items": len(items),
            "input_tokens": input_tokens,
            "final_tokens": used,
            "dropped": dropped_ids,
            "truncated": truncated_ids,
        }
        return ContextResult(context=context, stats=stats)

    # --------------------------------------------------------------- internals

    def _dedup(self, items: list[ContextItem]) -> list[ContextItem]:
        """精确内容去重；提供 embeddings 时再做语义去重（保留先到项）。"""
        seen_texts: set[str] = set()
        unique: list[ContextItem] = []
        for item in items:
            normalized = item.content.strip()
            if not normalized or normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            unique.append(item)
        if self.dedup_embeddings is None or len(unique) < 2:
            return unique
        try:
            vectors = self.dedup_embeddings.embed_documents(
                [item.content[:2000] for item in unique]
            )
        except Exception as error:  # noqa: BLE001 - 去重失败不影响主流程
            logger.warning("语义去重嵌入失败，跳过: %s", error)
            return unique
        kept: list[ContextItem] = []
        kept_vectors: list[list[float]] = []
        for item, vector in zip(unique, vectors, strict=True):
            if any(_cosine(vector, other) > self.dedup_threshold for other in kept_vectors):
                continue
            kept.append(item)
            kept_vectors.append(vector)
        return kept

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if _ENCODING is not None:
            try:
                tokens = _ENCODING.encode(text)
                if len(tokens) <= max_tokens:
                    return text
                return _ENCODING.decode(tokens[:max_tokens]).strip()
            except Exception:  # noqa: BLE001
                pass
        char_limit = max(1, int(max_tokens / 0.85))
        return text[:char_limit].strip()

    @staticmethod
    def _render(
        kept: list[ContextItem],
        contents: dict[int, str],
        image_paths: list[str] | None,
    ) -> str:
        parts: list[str] = []
        divider = "\n" + "=" * 50 + "\n"

        conversation_lines = [
            contents[id(item)] for item in kept if item.source_type == SOURCE_CONVERSATION
        ]
        if conversation_lines:
            parts.append(
                "【对话历史（供理解指代用，回答只针对当前问题）】\n"
                + "\n".join(conversation_lines)
            )

        parts.extend(
            contents[id(item)] for item in kept if item.source_type == SOURCE_RETRIEVAL
        )

        memory_lines = [contents[id(item)] for item in kept if item.source_type == SOURCE_MEMORY]
        if memory_lines:
            parts.append("【用户偏好（回答时必须遵守）】\n" + "\n".join(memory_lines))

        if image_paths:
            listing = "\n".join(f"- {path}" for path in image_paths)
            parts.append(f"【相关图片（可在回答中提示用户查看）】\n{listing}")

        if not parts:
            return "暂无相关食谱信息。"
        return divider.join(parts)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_agent_context_items(
    history: list[dict[str, Any]] | None,
    parents: list[Any],
    memories: list[dict[str, Any]] | None = None,
) -> list[ContextItem]:
    """把 agent 的三类上下文来源转换为 ContextItem 列表。

    - conversation：每轮一条，分配顺序最新优先（order 递减），渲染保持时间序；
    - retrieval：单篇一条，内容带【食谱 N】头（与 `_build_context` 格式一致，
      保证引用序号与 format_citations 对齐）；
    - memories：结构化记忆 dict（content/type 必填），渲染为偏好条目。
    """
    items: list[ContextItem] = []

    turns = [
        turn
        for turn in (history or [])
        if turn.get("role") in {"user", "assistant"} and turn.get("content")
    ]
    recent = turns[-6:]
    for offset, turn in enumerate(reversed(recent)):
        role_label = "用户" if turn["role"] == "user" else "助手"
        items.append(
            ContextItem(
                content=f"{role_label}: {str(turn['content'])[:200]}",
                source_type=SOURCE_CONVERSATION,
                priority=PRIORITY_CONVERSATION,
                order=offset,  # 最新一轮 offset=0，预算紧张时优先保留
                item_id=f"conv:{len(recent) - offset}",
            )
        )

    for index, doc in enumerate(parents, 1):
        metadata_info = f"【食谱 {index}】"
        if "dish_name" in doc.metadata:
            metadata_info += f" {doc.metadata['dish_name']}"
        if "category" in doc.metadata:
            metadata_info += f" | 分类: {doc.metadata['category']}"
        if "difficulty" in doc.metadata:
            metadata_info += f" | 难度: {doc.metadata['difficulty']}"
        items.append(
            ContextItem(
                content=f"{metadata_info}\n{doc.page_content}\n",
                source_type=SOURCE_RETRIEVAL,
                priority=PRIORITY_RETRIEVAL,
                order=index,
                truncatable=index == 1,  # 首篇可部分保留（对齐 _build_context 行为）
                item_id=f"retrieval:{index}",
                metadata={"dish_name": str(doc.metadata.get("dish_name", ""))},
            )
        )

    for offset, memory in enumerate(memories or []):
        items.append(
            ContextItem(
                content=f"- {memory['content']}（{memory.get('type', 'preference')}）",
                source_type=SOURCE_MEMORY,
                priority=PRIORITY_MEMORY,
                order=offset,
                item_id=f"memory:{offset}",
            )
        )
    return items
