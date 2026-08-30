"""Agentic RAG on LangGraph: tool routing, reflection, and graded fallbacks.

The graph exposes the retrieval stack as four tools the LLM chooses between:
full-text recipe search, image search, structured metadata browsing, and direct
answers for chitchat.  A reflection step rewrites and retries once when nothing
was found; every stage appends audit events used by tracing (Day 5) and records
degraded outcomes so bad cases can be replayed later.

Graph::

    START -> route -> act -> reflect -> generate -> END
                        ^        |
                        |        +-> rewrite -> act (once)
                        +-> fallback -> END
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from rag_modules.data_preparation import DataPreparationModule
from rag_modules.domain_config import DomainConfig, get_domain
from rag_modules.retrieval_optimization import RetrievalOptimizationModule

logger = logging.getLogger(__name__)

MAX_REWRITES = 1


class AgentState(TypedDict, total=False):
    question: str
    role: str
    history: list[dict[str, Any]]
    messages: Annotated[list[Any], add_messages]
    route: str
    tool_args: dict[str, Any]
    chunks: list[Document]
    parents: list[Document]
    image_paths: list[str]
    answer: str
    rewrites_used: int
    retry_pending: bool
    error: str | None
    events: list[dict[str, Any]]
    context_stats: dict[str, Any]
    standalone_query: str


def _event(name: str, **details: Any) -> dict[str, Any]:
    return {"ts": datetime.now(UTC).isoformat(), "event": name, **details}


def build_agent_tools(domain: DomainConfig | None = None) -> list[dict[str, Any]]:
    """Tool schemas for the router LLM; vocabularies come from the domain config."""
    domain = domain or get_domain()
    descriptions = domain.tool_descriptions
    categories = domain.category_labels
    difficulties = list(domain.difficulty_labels)
    return [
        {
            "type": "function",
            "function": {
                "name": "search_recipes",
                "description": descriptions["search_recipes"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": descriptions["query_param_example"],
                        },
                        "category": {
                            "type": "string",
                            "enum": categories,
                            "description": descriptions["category_param"],
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": difficulties,
                            "description": descriptions["difficulty_param"],
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_images",
                "description": descriptions["search_images"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": descriptions["image_query_param"],
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_by_metadata",
                "description": descriptions["list_by_metadata"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": categories,
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": difficulties,
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "direct_answer",
                "description": descriptions["direct_answer"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": descriptions["direct_answer_param"],
                        }
                    },
                    "required": ["answer"],
                },
            },
        },
    ]


class RecipeAgent:
    """LangGraph agent wrapping the retrieval stack with graded fallbacks."""

    def __init__(
        self,
        retrieval_module: RetrievalOptimizationModule,
        data_module: DataPreparationModule,
        generation_module: Any,
        *,
        top_k: int = 3,
        visibility_expr_builder: Any | None = None,
        context_manager: Any | None = None,
    ):
        self.retrieval_module = retrieval_module
        self.data_module = data_module
        self.generation_module = generation_module
        self.top_k = top_k
        self.visibility_expr_builder = visibility_expr_builder
        self.context_manager = context_manager
        self.domain = get_domain()
        self.tools = build_agent_tools(self.domain)
        self.graph = self._build_graph()

    # ------------------------------------------------------------- LLM helpers

    def _llm(self) -> Any:
        return self.generation_module.llm

    @staticmethod
    def _history_messages(
        history: list[dict[str, Any]] | None, *, limit: int = 6
    ) -> list[dict[str, Any]]:
        """Recent turns as chat messages (used by routing for reference resolution)."""
        if not history:
            return []
        turns = [
            {"role": turn["role"], "content": str(turn["content"])}
            for turn in history[-limit:]
            if turn.get("role") in {"user", "assistant"} and turn.get("content")
        ]
        return turns

    @staticmethod
    def _history_prefix(history: list[dict[str, Any]] | None, *, limit: int = 4) -> str:
        """Compact transcript prepended to the generation question."""
        if not history:
            return ""
        lines = [
            f"{'用户' if turn['role'] == 'user' else '助手'}: {str(turn['content'])[:120]}"
            for turn in history[-limit:]
            if turn.get("role") in {"user", "assistant"} and turn.get("content")
        ]
        if not lines:
            return ""
        header = "（对话历史，供理解指代用，回答只针对当前问题）\n"
        return header + "\n".join(lines) + "\n\n当前问题: "

    def _route_llm_call(
        self, question: str, history: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": self.domain.agent_router_prompt},
            *self._history_messages(history),
            {"role": "user", "content": question},
        ]
        message = self._llm().bind_tools(self.tools).invoke(messages)
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            call = tool_calls[0]
            return {"route": call["name"], "tool_args": dict(call["args"])}
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            return {"route": "direct_answer", "tool_args": {"answer": content}}
        raise ValueError("路由 LLM 未返回工具调用")

    def _rewrite_query(self, question: str, history: list[dict[str, Any]] | None = None) -> str:
        history_note = ""
        if history:
            recent = "; ".join(
                f"{turn['role']}说:{str(turn['content'])[:60]}"
                for turn in history[-2:]
                if turn.get("content")
            )
            history_note = f"\n对话背景: {recent}"
        prompt = (
            "用户的菜谱检索没有命中结果。请把查询改写得更具体、更适合菜谱检索"
            "（结合对话背景解决指代，加入菜品名、食材或烹饪方式关键词），只输出改写后的查询。\n"
            f"原查询: {question}{history_note}"
        )
        return str(self._llm().invoke(prompt).content).strip()

    # ------------------------------------------------------------- graph nodes

    def _route_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        try:
            decision = self._route_llm_call(question, history=state.get("history"))
        except Exception as exc:  # noqa: BLE001 - routing failure degrades to default search
            logger.warning("路由节点失败，降级为默认检索: %s", exc)
            decision = {"route": "search_recipes", "tool_args": {"query": question}}
            state["error"] = f"route_fallback: {exc}"
        state["route"] = decision["route"]
        state["tool_args"] = decision["tool_args"]
        state.setdefault("events", []).append(
            _event("route", route=decision["route"], args=decision["tool_args"])
        )
        return state

    def _visibility_expr(self, role: str) -> str | None:
        if self.visibility_expr_builder is None:
            return None
        return self.visibility_expr_builder(role)

    def _search_recipes(self, query: str, role: str, **filters: Any) -> list[Document]:
        clean_filters = {k: v for k, v in filters.items() if v}
        return self.retrieval_module.metadata_filtered_search(
            query, clean_filters, top_k=self.top_k, extra_expr=self._visibility_expr(role)
        )

    def _act_node(self, state: AgentState) -> AgentState:
        # LangGraph 对缺失 key 保留旧值，必须显式覆盖才能结束重试循环
        state["retry_pending"] = False
        route = state.get("route", "search_recipes")
        args = state.get("tool_args", {})
        role = state.get("role", "user")
        question = state["question"]
        state["chunks"] = []
        state["parents"] = []
        state["image_paths"] = []

        if route == "direct_answer":
            fallback = self.domain.direct_answer_default
            state["answer"] = str(args.get("answer", "")).strip() or fallback
            state.setdefault("events", []).append(_event("direct_answer"))
            return state

        try:
            if route == "search_images":
                chunks = self.retrieval_module.metadata_filtered_search(
                    args.get("query", question),
                    {"modality": "image"},
                    top_k=self.top_k,
                    extra_expr=self._visibility_expr(role),
                )
            elif route == "list_by_metadata":
                filters = {k: v for k, v in args.items() if v}
                query = " ".join(str(v) for v in filters.values()) or "家常菜谱推荐"
                chunks = self.retrieval_module.metadata_filtered_search(
                    query, filters, top_k=max(self.top_k, 5),
                    extra_expr=self._visibility_expr(role),
                )
            else:
                chunks = self._search_recipes(args.get("query", question), role, **{
                    k: args.get(k) for k in ("category", "difficulty")
                })
        except Exception as exc:  # noqa: BLE001 - tool failure degrades to plain hybrid search
            logger.warning("工具 %s 执行失败，降级为默认检索: %s", route, exc)
            state["error"] = f"tool_fallback({route}): {exc}"
            state.setdefault("events", []).append(
                _event("tool_fallback", route=route, error=str(exc))
            )
            try:
                chunks = self.retrieval_module.hybrid_search(
                    question, top_k=self.top_k, expr=self._visibility_expr(role)
                )
            except Exception:  # noqa: BLE001 - retrieval itself is down
                state["answer"] = self._offline_answer(question)
                state["error"] = "retrieval_unavailable"
                state.setdefault("events", []).append(_event("retrieval_unavailable"))
                return state

        parents = self.data_module.get_parent_documents(chunks) if chunks else []
        state["chunks"] = chunks
        state["parents"] = parents
        image_paths = []
        for chunk in chunks:
            path = str(chunk.metadata.get("image_path", ""))
            if chunk.metadata.get("modality") == "image" and path and path not in image_paths:
                image_paths.append(path)
        state["image_paths"] = image_paths[:6]
        state.setdefault("events", []).append(
            _event("act", route=route, hits=len(chunks), parents=len(parents))
        )
        return state

    def _reflect_node(self, state: AgentState) -> AgentState:
        if state.get("answer") is not None:  # direct_answer / offline fallback path
            return state
        rewrites_used = state.get("rewrites_used", 0)
        if not state.get("chunks") and rewrites_used < MAX_REWRITES:
            try:
                rewritten = self._rewrite_query(state["question"], history=state.get("history"))
            except Exception as exc:  # noqa: BLE001 - rewrite failure keeps original query
                logger.warning("查询改写失败: %s", exc)
                state["error"] = f"rewrite_failed: {exc}"
                state["rewrites_used"] = MAX_REWRITES  # 消耗掉重试机会，避免死循环
                return state
            state["rewrites_used"] = rewrites_used + 1
            state["route"] = "search_recipes"
            state["tool_args"] = {"query": rewritten}
            state["retry_pending"] = True
            state.setdefault("events", []).append(_event("rewrite", query=rewritten))
            return state
        if not state.get("parents"):
            state.setdefault("events", []).append(_event("no_results"))
        return state

    def _generate_node(self, state: AgentState) -> AgentState:
        if state.get("answer") is not None:
            return state
        parents = state.get("parents", [])
        if not parents:
            state["answer"] = self.domain.agent_no_results
            return state
        history = state.get("history")
        try:
            if self.context_manager is not None:
                # 统一上下文管道：conversation/retrieval/memory 三类来源按
                # token 预算与优先级组装，替代 history 前缀硬拼 + 字符顺序拼接
                from rag_modules.context_management import build_agent_context_items

                items = build_agent_context_items(
                    history, parents, memories=state.get("memory_items")
                )
                built = self.context_manager.build(items, image_paths=state.get("image_paths"))
                state["context_stats"] = built.stats
                state.setdefault("events", []).append(_event("context_built", **built.stats))
                state["answer"] = self.generation_module.generate_basic_answer(
                    state["question"],
                    parents,
                    image_paths=state.get("image_paths"),
                    context_text=built.context,
                )
            else:
                # 多轮：历史以前缀并入生成问题，让模型理解"它/第二道"这类指代
                question = self._history_prefix(history) + state["question"]
                state["answer"] = self.generation_module.generate_basic_answer(
                    question, parents, image_paths=state.get("image_paths")
                )
        except Exception as exc:  # noqa: BLE001 - LLM failure falls back to retrieved summary
            logger.warning("生成节点失败，降级为检索摘要: %s", exc)
            state["answer"] = self._offline_answer(state["question"], parents)
            state["error"] = f"generate_fallback: {exc}"
            state.setdefault("events", []).append(
                _event("generate_fallback", error=str(exc))
            )
        return state

    def _offline_answer(self, question: str, parents: list[Document] | None = None) -> str:
        """No-LLM fallback: summarize whatever retrieval produced, rule-based."""
        if not parents:
            return self.domain.retrieval_unavailable_answer
        names = []
        for parent in parents:
            name = str(parent.metadata.get("dish_name", self.domain.unknown_item_name))
            if name not in names:
                names.append(name)
        return "检索到以下相关菜谱（当前生成服务降级，仅展示条目）：\n" + "\n".join(
            f"- {name}" for name in names
        )

    # ------------------------------------------------------------------ graph

    def _decide_after_reflect(self, state: AgentState) -> str:
        if state.get("answer") is not None:
            return END
        if state.get("retry_pending"):
            return "act"  # reflect 已生成改写查询，回到 act 重试一次
        if not state.get("chunks"):
            return "generate"  # 改写机会用尽仍无结果 → generate 节点拒答
        return "generate"

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route_node)
        graph.add_node("act", self._act_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("generate", self._generate_node)
        graph.add_edge(START, "route")
        graph.add_edge("route", "act")
        graph.add_edge("act", "reflect")
        graph.add_conditional_edges(
            "reflect",
            self._decide_after_reflect,
            {"act": "act", "generate": "generate", END: END},
        )
        graph.add_edge("generate", END)
        return graph.compile()

    # ------------------------------------------------------------------- API

    def invoke(
        self,
        question: str,
        *,
        role: str = "user",
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        initial: AgentState = {
            "question": question,
            "role": role,
            "history": history or [],
            "rewrites_used": 0,
            "events": [],
        }
        result = self.graph.invoke(initial, config={"recursion_limit": 8})
        logger.info(
            "Agent 完成: question=%r route=%s hits=%d rewrites=%d history=%d error=%s",
            question,
            result.get("route"),
            len(result.get("chunks", [])),
            result.get("rewrites_used", 0),
            len(history or []),
            result.get("error"),
        )
        return result
