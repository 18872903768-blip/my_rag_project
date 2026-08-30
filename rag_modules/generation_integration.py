"""
生成集成模块
"""

import logging
import os
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag_modules.domain_config import get_domain

logger = logging.getLogger(__name__)

# 忠实性硬约束：追加到生成 prompt（RAG_GROUNDED_ANSWER=true 时启用）。
# 针对 RAGAS 评测定位的失分模式：编造时间/用量/尺寸等原文没有的细节、
# 添加常识性口感与场景评价。
GROUNDING_CLAUSE = """

【忠实性硬约束（最高优先级，覆盖其他所有指令）】
- 回答中的每一处事实性内容（食材、用量、时间、温度、尺寸、步骤）都必须来自"相关食谱信息"原文。
- 严禁自行估算或补充原文没有的数字与细节（如具体时长、尺寸、毫升数、厘米数）；原文未写明的直接省略，或明确写"原文未提及"。
- 菜品介绍只能复述或转写原文已有的描述，禁止添加原文没有的口感、场景、适用人群等评价性内容。
- 不要输出与"相关食谱信息"矛盾或其未涉及的常识性烹饪知识。"""


def apply_grounding(prompt: str, enabled: bool) -> str:
    """按开关把忠实性硬约束追加到 prompt 模板（保留原占位符不变）。"""
    return prompt + GROUNDING_CLAUSE if enabled else prompt


def format_citations(docs: list[Document]) -> str:
    """Deterministic source appendix: unique dish names with their file paths.

    Inline 【食谱 N】 markers depend on the LLM; this appendix is computed
    from the actual retrieved parents so every answer carries verifiable
    provenance even when generation degrades.
    """
    entries: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        name = str(doc.metadata.get("dish_name", "未知来源"))
        source = str(doc.metadata.get("source_path", "")).strip()
        key = f"{name}|{source}"
        if key in seen or name == "未知来源" and not source:
            continue
        seen.add(key)
        entries.append(f"- {name}" + (f"（{source}）" if source else ""))
    if not entries:
        return ""
    return "\n\n——\n📚 以上回答参考自：\n" + "\n".join(entries)


class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_context_chars: int = 6000,
        *,
        llm: Any | None = None,
        grounded_answer: bool = False,
    ):
        """
        初始化生成集成模块

        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_chars = max_context_chars
        self.grounded_answer = grounded_answer
        self.llm: Any = llm if llm is not None else self.setup_llm()

    def setup_llm(self) -> Any:
        """初始化大语言模型"""
        from langchain_deepseek import ChatDeepSeek

        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        # 显式将 key 写入系统的 OPENAI_API_KEY 环境变量，防止底层 SDK 报错
        # os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatDeepSeek(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        logger.info("LLM初始化完成")
        return llm

    def generate_basic_answer(
        self,
        query: str,
        context_docs: list[Document],
        *,
        image_paths: list[str] | None = None,
        context_text: str | None = None,
    ) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表
            image_paths: 检索命中的相关图片路径（可选）
            context_text: 外部组装好的上下文（agent 管线的 Context Manager
                产出）；传入时跳过 _build_context。classic 链路不传，行为不变

        Returns:
            生成的回答
        """
        if context_text is not None:
            context = context_text
        else:
            context = self._build_context(context_docs, self.max_context_chars, image_paths)

        prompt = ChatPromptTemplate.from_template(
            apply_grounding(get_domain().basic_answer_prompt, self.grounded_answer)
        )

        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response + format_citations(context_docs)

    def generate_step_by_step_answer(
        self, query: str, context_docs: list[Document], *, image_paths: list[str] | None = None
    ) -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表
            image_paths: 检索命中的相关图片路径（可选）

        Returns:
            分步骤的详细回答
        """
        context = self._build_context(context_docs, self.max_context_chars, image_paths)

        prompt = ChatPromptTemplate.from_template(
            apply_grounding(get_domain().step_by_step_prompt, self.grounded_answer)
        )

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response + format_citations(context_docs)

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate.from_template(get_domain().rewrite_prompt)

        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response

    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template(get_domain().router_prompt)

        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()

        result = chain.invoke(query).strip().lower()

        # 确保返回有效的路由类型
        if result in ["list", "detail", "general"]:
            return result
        else:
            return "general"  # 默认类型

    def generate_list_answer(self, query: str, context_docs: list[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """
        if not context_docs:
            return "抱歉，没有找到相关的信息。"

        domain = get_domain()
        # 提取条目名称
        names = []
        for doc in context_docs:
            name = doc.metadata.get("dish_name", domain.unknown_item_name)
            if name not in names:
                names.append(name)

        # 构建简洁的列表回答
        if len(names) == 1:
            return domain.list_recommend_single.format(name=names[0])
        elif len(names) <= 3:
            return domain.list_recommend_intro + "\n" + "\n".join(
                [f"{i + 1}. {name}" for i, name in enumerate(names)]
            )
        else:
            return (
                domain.list_recommend_intro
                + "\n"
                + "\n".join([f"{i + 1}. {name}" for i, name in enumerate(names[:3])])
                + "\n\n"
                + domain.list_recommend_more.format(
                    count=len(names) - 3,
                    noun=domain.item_noun,
                    classifier=domain.item_classifier,
                )
            )

    def generate_basic_answer_stream(self, query: str, context_docs: list[Document]):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs, self.max_context_chars)

        prompt = ChatPromptTemplate.from_template(
            apply_grounding(get_domain().basic_answer_prompt, self.grounded_answer)
        )

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        yield from chain.stream(query)
        citations = format_citations(context_docs)
        if citations:
            yield citations

    def generate_step_by_step_answer_stream(self, query: str, context_docs: list[Document]):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细步骤回答片段
        """
        context = self._build_context(context_docs, self.max_context_chars)

        prompt = ChatPromptTemplate.from_template(
            apply_grounding(get_domain().step_by_step_prompt, self.grounded_answer)
        )

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        yield from chain.stream(query)
        citations = format_citations(context_docs)
        if citations:
            yield citations

    def _build_context(
        self, docs: list[Document], max_length: int = 6000, image_paths: list[str] | None = None
    ) -> str:
        """
        构建上下文字符串

        Args:
            docs: 文档列表
            max_length: 最大长度
            image_paths: 命中的相关图片相对路径（可选）

        Returns:
            格式化的上下文
        """
        if not docs:
            return "暂无相关食谱信息。"

        if max_length < 1:
            raise ValueError("max_length 必须为正整数")

        context_parts: list[str] = []
        current_length = 0

        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【食谱 {i}】"
            if "dish_name" in doc.metadata:
                metadata_info += f" {doc.metadata['dish_name']}"
            if "category" in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['category']}"
            if "difficulty" in doc.metadata:
                metadata_info += f" | 难度: {doc.metadata['difficulty']}"

            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"

            remaining = max_length - current_length
            if remaining <= 0:
                break

            if len(doc_text) > remaining:
                # 至少保留首篇食谱的一部分，避免长文档导致空上下文。
                if not context_parts:
                    context_parts.append(doc_text[:remaining])
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)
        divider = "\n" + "=" * 50 + "\n"
        parts = list(context_parts)
        if parts and image_paths:
            listing = "\n".join(f"- {path}" for path in image_paths)
            parts.append(f"【相关图片（可在回答中提示用户查看）】\n{listing}")
        return divider.join(parts) if parts else "暂无相关食谱信息。"
