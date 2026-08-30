"""Domain configuration: the single place that knows this is a recipe KB.

Everything domain-specific lives here — metadata vocabularies, LLM prompts,
agent tool descriptions, and user-facing wording.  The retrieval engine
(Milvus store, RRF fusion, LangGraph skeleton, tracing, permissions) is
domain-agnostic and only reads these values.  Swapping to another knowledge
base (e.g. an English-learning corpus) means writing a new ``DomainConfig``
plus new document parsers — no engine changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainConfig:
    """All domain-coupled strings and vocabularies used by the engine."""

    name: str
    assistant_name: str
    item_noun: str = "菜品"
    item_classifier: str = "道"
    unknown_item_name: str = "未知菜品"

    # ---- 元数据词汇表 ----
    category_mapping: dict[str, str] = field(default_factory=dict)
    difficulty_labels: list[str] = field(default_factory=list)
    internal_categories: frozenset[str] = frozenset()

    # ---- 经典链路 prompt（{question}/{context}/{query} 为占位符）----
    router_prompt: str = ""
    rewrite_prompt: str = ""
    basic_answer_prompt: str = ""
    step_by_step_prompt: str = ""

    # ---- Agentic 链路 ----
    agent_router_prompt: str = ""
    tool_descriptions: dict[str, str] = field(default_factory=dict)
    rewrite_prompt_short: str = ""
    direct_answer_default: str = ""

    # ---- 用户可见话术 ----
    classic_no_results: str = ""
    agent_no_results: str = ""
    retrieval_unavailable_answer: str = ""
    offline_summary_header: str = ""
    list_recommend_single: str = ""
    list_recommend_intro: str = ""
    list_recommend_more: str = ""

    @property
    def category_labels(self) -> list[str]:
        return list(dict.fromkeys(self.category_mapping.values()))


RECIPE_DOMAIN = DomainConfig(
    name="recipe",
    assistant_name="菜谱客服助手",
    item_noun="菜品",
    item_classifier="道",
    unknown_item_name="未知菜品",
    category_mapping={
        "meat_dish": "荤菜",
        "vegetable_dish": "素菜",
        "soup": "汤品",
        "dessert": "甜品",
        "breakfast": "早餐",
        "staple": "主食",
        "aquatic": "水产",
        "condiment": "调料",
        "drink": "饮品",
        "semi-finished": "半成品",
        "semi_finished": "半成品",
    },
    difficulty_labels=["非常简单", "简单", "中等", "困难", "非常困难"],
    internal_categories=frozenset({"半成品"}),
    router_prompt="""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

2. 'detail' - 用户想要具体的制作方法或详细信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

3. 'general' - 其他一般性问题
   例如：什么是川菜、制作技巧、营养价值

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""",
    rewrite_prompt="""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体菜品名称：如"宫保鸡丁怎么做"、"红烧肉的制作方法"
   - 明确的制作询问：如"蛋炒饭需要什么食材"、"糖醋排骨的步骤"
   - 具体的烹饪技巧：如"如何炒菜不粘锅"、"怎样调制糖醋汁"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"做菜"、"有什么好吃的"、"推荐个菜"
   - 缺乏具体信息：如"川菜"、"素菜"、"简单的"
   - 口语化表达：如"想吃点什么"、"有饮品推荐吗"

重写原则：
- 保持原意不变
- 增加相关烹饪术语
- 优先推荐简单易做的
- 保持简洁性

示例：
- "做菜" → "简单易做的家常菜谱"
- "有饮品推荐吗" → "简单饮品制作方法"
- "推荐个菜" → "简单家常菜推荐"
- "川菜" → "经典川菜菜谱"
- "宫保鸡丁怎么做" → "宫保鸡丁怎么做"（保持原查询）
- "红烧肉需要什么食材" → "红烧肉需要什么食材"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""",
    basic_answer_prompt="""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。
引用某份食谱的内容时，请在对应句子后用【食谱 N】标注序号。

回答:""",
    step_by_step_prompt="""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的技巧要分享，可以省略制作技巧部分
- 引用某份食谱的内容时，在对应内容后用【食谱 N】标注序号

回答:""",
    agent_router_prompt="""你是一个菜谱客服助手的任务规划器。根据用户问题选择一个工具调用。

规则：
- 问做法/食材/技巧 → search_recipes（把用户话提炼成适合检索的查询词）
- 要看图/问菜品长什么样 → search_images
- 要推荐列表、按分类难度浏览 → list_by_metadata
- 问候、闲聊、问你能干什么 → direct_answer
- 用户查询模糊时也要选 search_recipes 并自行提炼关键词，不要反问用户""",
    tool_descriptions={
        "search_recipes": "在菜谱知识库中检索菜品的做法、食材、技巧等详细内容",
        "search_images": "根据文字描述检索菜品的成品图或步骤图",
        "list_by_metadata": "按分类/难度浏览菜品列表，适合'推荐几个简单的汤'这类请求",
        "direct_answer": "与菜谱知识库无关的问候、闲聊或关于本助手的问题，直接回答",
        "query_param_example": "适合向量检索的中文查询词，如'红烧肉怎么做'",
        "image_query_param": "图片内容描述，如'红烧肉成品图'",
        "category_param": "菜品分类（可选）",
        "difficulty_param": "烹饪难度（可选）",
        "category_param_browse": "菜品分类",
        "difficulty_param_browse": "烹饪难度",
        "direct_answer_param": "给用户的回答",
    },
    rewrite_prompt_short=(
        "用户的菜谱检索没有命中结果。请把查询改写得更具体、更适合菜谱检索"
        "（加入菜品名、食材或烹饪方式关键词），只输出改写后的查询。\n"
        "原查询: {question}"
    ),
    direct_answer_default="你好，我是菜谱助手。",
    classic_no_results="抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。",
    agent_no_results=(
        "抱歉，知识库中没有找到与您问题相关的菜谱内容。"
        "您可以换个菜名或食材关键词再试试，比如'红烧肉怎么做'。"
    ),
    retrieval_unavailable_answer=(
        "检索服务暂时不可用，请稍后重试。"
        "您也可以直接浏览菜谱分类：荤菜、素菜、汤品、甜品、主食等。"
    ),
    offline_summary_header="检索到以下相关菜谱（当前生成服务降级，仅展示条目）：",
    list_recommend_single="为您推荐：{name}",
    list_recommend_intro="为您推荐以下菜品：",
    list_recommend_more="还有其他 {count} {classifier}{noun}可供选择。",
)

_active_domain: DomainConfig | None = None


def get_domain() -> DomainConfig:
    """Return the active domain (cached; overridable via RAG_DOMAIN for now)."""
    global _active_domain
    if _active_domain is None:
        _active_domain = RECIPE_DOMAIN
    return _active_domain


def set_domain(domain: DomainConfig) -> None:
    """Override the active domain — used by tests and future domain packs."""
    global _active_domain
    _active_domain = domain


def default_internal_categories() -> frozenset[str]:
    """Internal categories from the domain, unless overridden by env."""
    override = os.getenv("RAG_INTERNAL_CATEGORIES", "")
    if override.strip():
        return frozenset(
            item.strip() for item in override.split(",") if item.strip()
        )
    return get_domain().internal_categories
