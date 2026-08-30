"""Offline regressions for the LangGraph agent routing, reflection, fallbacks."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.documents import Document

from rag_modules.agentic_rag import RecipeAgent
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule


class FakeAIMessage:
    def __init__(self, content: str = "", tool_calls: list[dict[str, Any]] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChatModel:
    """Scripted LLM: pops one queued response per call, recording prompts."""

    def __init__(self, responses: list[FakeAIMessage]):
        self.responses = list(responses)
        self.prompts: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> FakeChatModel:
        return self

    def invoke(self, prompt: Any) -> FakeAIMessage:
        if not self.responses:
            raise AssertionError("FakeChatModel 队列为空")
        self.prompts.append(prompt)
        return self.responses.pop(0)


class FakeGeneration:
    def __init__(self, llm: FakeChatModel, answer: str = "生成的回答"):
        self.llm = llm
        self.answer = answer
        self.calls: list[tuple[str, list[Document]]] = []
        self.context_texts: list[str | None] = []

    def generate_basic_answer(
        self,
        query: str,
        docs: list[Document],
        *,
        image_paths: list[str] | None = None,
        context_text: str | None = None,
    ) -> str:
        self.calls.append((query, docs))
        self.context_texts.append(context_text)
        return self.answer


class FakeVectorStore:
    def __init__(self, results: list[Document], *, fail: bool = False, server_side: bool = False):
        self.results = results
        self.fail = fail
        self.hybrid_calls = 0
        if server_side:
            self.supports_server_side_hybrid = True

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        if self.fail:
            raise RuntimeError("vector backend down")
        return self.results[:k]

    def hybrid_search(self, query: str, k: int = 5, **kwargs: Any) -> list[Document]:
        self.hybrid_calls += 1
        if self.fail:
            raise RuntimeError("vector backend down")
        return self.results[:k]


def _chunk(chunk_id: str, dish: str, content: str = "内容", **metadata: Any) -> Document:
    base = {
        "chunk_id": chunk_id,
        "dish_name": dish,
        "visibility": "public",
        "modality": "text",
        "parent_id": f"parent-{chunk_id}",
    }
    base.update(metadata)
    return Document(page_content=content, metadata=base)


def _make_data_module(chunks: list[Document]) -> DataPreparationModule:
    data_module = DataPreparationModule.__new__(DataPreparationModule)
    data_module._parent_documents = {}
    data_module.documents = []
    for chunk in chunks:
        data_module._parent_documents[chunk.metadata["parent_id"]] = Document(
            page_content=f"# {chunk.metadata['dish_name']}",
            metadata={
                "parent_id": chunk.metadata["parent_id"],
                "dish_name": chunk.metadata["dish_name"],
            },
        )
    return data_module


def _agent(
    chunks: list[Document],
    responses: list[FakeAIMessage],
    results: list[Document] | None = None,
    *,
    fail_backend: bool = False,
    server_side: bool = False,
) -> tuple[RecipeAgent, FakeGeneration]:
    llm = FakeChatModel(responses)
    generation = FakeGeneration(llm)
    vectorstore = FakeVectorStore(
        results if results is not None else chunks, fail=fail_backend, server_side=server_side
    )
    retrieval = RetrievalOptimizationModule(cast(Any, vectorstore), chunks, candidate_k=5)
    agent = RecipeAgent(
        retrieval, _make_data_module(chunks), cast(Any, generation), top_k=3
    )
    return agent, generation


def test_agent_routes_search_recipes_and_generates() -> None:
    chunks = [_chunk("c1", "红烧肉"), _chunk("c2", "宫保鸡丁")]
    agent, generation = _agent(
        chunks,
        responses=[
            FakeAIMessage(
                tool_calls=[
                    {"name": "search_recipes", "args": {"query": "红烧肉 做法"}, "id": "1"}
                ]
            )
        ],
    )

    result = agent.invoke("红烧肉怎么做")

    assert result["answer"] == "生成的回答"
    assert len(generation.calls) == 1
    assert generation.calls[0][1][0].metadata["dish_name"] == "红烧肉"
    events = [event["event"] for event in result["events"]]
    assert "route" in events and "act" in events


def test_agent_direct_answer_skips_retrieval() -> None:
    chunks = [_chunk("c1", "红烧肉")]
    agent, generation = _agent(
        chunks,
        responses=[
            FakeAIMessage(
                tool_calls=[{"name": "direct_answer", "args": {"answer": "你好呀"}, "id": "1"}]
            )
        ],
    )

    result = agent.invoke("你好")

    assert result["answer"] == "你好呀"
    assert generation.calls == []


def test_agent_rewrites_once_then_refuses_when_nothing_found() -> None:
    chunks = [_chunk("c1", "红烧肉")]
    agent, _ = _agent(
        chunks,
        responses=[
            FakeAIMessage(
                tool_calls=[{"name": "search_recipes", "args": {"query": "法式鹅肝"}, "id": "1"}]
            ),
            FakeAIMessage(content="法式鹅肝 经典做法"),
        ],
        results=[],
        server_side=True,
    )

    result = agent.invoke("怎么做法式鹅肝")

    assert result["rewrites_used"] == 1
    assert "抱歉" in result["answer"]
    events = [event["event"] for event in result["events"]]
    assert "rewrite" in events and "no_results" in events


def test_agent_degrades_gracefully_when_backend_fails() -> None:
    chunks = [_chunk("c1", "红烧肉")]
    agent, _ = _agent(
        chunks,
        responses=[
            FakeAIMessage(
                tool_calls=[{"name": "search_recipes", "args": {"query": "红烧肉"}, "id": "1"}]
            )
        ],
        fail_backend=True,
    )

    result = agent.invoke("红烧肉怎么做")

    # 检索后端彻底不可用 → 无 LLM 的离线兜底回答
    assert "降级" in result["answer"] or "不可用" in result["answer"]
    assert result.get("error") == "retrieval_unavailable"


def test_agent_llm_generate_failure_returns_rule_based_summary() -> None:
    chunk = _chunk("c1", "红烧肉")

    class ExplodingGeneration(FakeGeneration):
        def generate_basic_answer(self, query, docs, *, image_paths=None):
            raise RuntimeError("llm quota exceeded")

    llm = FakeChatModel(
        [
            FakeAIMessage(
                tool_calls=[{"name": "search_recipes", "args": {"query": "红烧肉"}, "id": "1"}]
            )
        ]
    )
    generation = ExplodingGeneration(llm)
    retrieval = RetrievalOptimizationModule(
        cast(Any, FakeVectorStore([chunk])), [chunk], candidate_k=5
    )
    agent = RecipeAgent(
        retrieval, _make_data_module([chunk]), cast(Any, generation), top_k=3
    )

    result = agent.invoke("红烧肉怎么做")

    assert "红烧肉" in result["answer"]
    assert "generate_fallback" in str(result.get("error", ""))


def test_agent_history_reaches_router_and_generation() -> None:
    chunks = [_chunk("c1", "红烧肉")]
    history = [
        {"role": "user", "content": "推荐一道硬菜"},
        {"role": "assistant", "content": "为您推荐简易红烧肉"},
    ]
    agent, generation = _agent(
        chunks,
        responses=[
            FakeAIMessage(
                tool_calls=[
                    {"name": "search_recipes", "args": {"query": "简易红烧肉 怎么做"}, "id": "1"}
                ]
            )
        ],
    )

    agent.invoke("它怎么做", history=history)

    # 路由 LLM 收到对话历史（解决"它"的指代需要上下文）
    route_prompt = generation.llm.prompts[0]
    assert isinstance(route_prompt, list)
    roles = [message["role"] for message in route_prompt]
    assert roles == ["system", "user", "assistant", "user"]
    assert route_prompt[-1]["content"] == "它怎么做"

    # 生成问题带历史前缀
    generate_query = generation.calls[0][0]
    assert "对话历史" in generate_query
    assert "简易红烧肉" in generate_query
    assert generate_query.endswith("它怎么做")


def test_agent_without_history_behaves_as_before() -> None:
    chunks = [_chunk("c1", "红烧肉")]
    agent, generation = _agent(
        chunks,
        responses=[
            FakeAIMessage(
                tool_calls=[{"name": "search_recipes", "args": {"query": "红烧肉"}, "id": "1"}]
            )
        ],
    )

    agent.invoke("红烧肉怎么做")

    route_prompt = generation.llm.prompts[0]
    assert [message["role"] for message in route_prompt] == ["system", "user"]
    assert generation.calls[0][0] == "红烧肉怎么做"
