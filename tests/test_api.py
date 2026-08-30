"""Offline API tests: auth, role propagation, agent fields, SSE, traces.

Uses TestClient with a fake RAG system injected via ``create_app(rag=...)`` so
no model, Milvus, or LLM is needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import rag_modules.observability as observability
from api.app import create_app
from api.auth import mint_token


class FakeRAG:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.config = SimpleNamespace(backend="milvus")
        self.backend = None

    def ask_question(
        self,
        query: str,
        *,
        stream: bool = False,
        role: str = "user",
        query_id: str | None = None,
    ) -> Any:
        self.calls.append(("ask_question", query, role, stream, query_id))
        if stream:
            return iter(["答案片段1", "答案片段2"])
        return f"classic-answer[{role}]"

    def ask_agent(
        self,
        query: str,
        *,
        role: str = "user",
        query_id: str | None = None,
        history: list[dict[str, str]] | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("ask_agent", query, role, query_id, list(history or [])))
        chunks = [
            Document(
                page_content="红烧肉做法",
                metadata={
                    "dish_name": "简易红烧肉",
                    "category": "荤菜",
                    "visibility": "public",
                    "modality": "text",
                },
            ),
            Document(
                page_content="图片",
                metadata={
                    "dish_name": "简易红烧肉",
                    "modality": "image",
                    "image_path": "dishes/a/1.jpg",
                },
            ),
        ]
        parents = [
            Document(
                page_content="# 简易红烧肉",
                metadata={
                    "dish_name": "简易红烧肉",
                    "source_path": "dishes/meat_dish/简易红烧肉/简易红烧肉.md",
                },
            )
        ]
        return {
            "answer": f"agent-answer[{role}]",
            "route": "search_recipes",
            "chunks": chunks,
            "parents": parents,
            "error": None,
            "query_id": query_id,
        }

    def retrieve(
        self, query: str, *, top_k: int = 5, role: str = "user", **kwargs: Any
    ) -> list[Document]:
        self.calls.append(("retrieve", query, top_k, role))
        return [
            Document(
                page_content="简易红烧肉的做法……",
                metadata={
                    "dish_name": "简易红烧肉",
                    "category": "荤菜",
                    "difficulty": "中等",
                    "visibility": "public",
                    "modality": "text",
                    "rrf_score": 0.03,
                },
            )
        ]

    def retrieve_images(
        self, query: str, *, top_k: int = 5, role: str = "user", **kwargs: Any
    ) -> list[Document]:
        self.calls.append(("retrieve_images", query, top_k, role))
        return [
            Document(
                page_content="《简易红烧肉》图片",
                metadata={
                    "dish_name": "简易红烧肉",
                    "modality": "image",
                    "image_path": "dishes/meat/简易红烧肉/000.jpg",
                    "rrf_score": 0.02,
                },
            )
        ]


@pytest.fixture()
def client() -> TestClient:
    app = create_app(rag=FakeRAG())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def trace_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(observability, "DEFAULT_TRACE_DIR", tmp_path)
    return tmp_path


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["backend"] == "milvus"


def test_json_responses_declare_utf8_charset(client: TestClient) -> None:
    # PowerShell 5.1 等客户端在缺少 charset 时按 Latin-1 解码导致中文乱码
    response = client.post("/api/ask", json={"query": "红烧肉怎么做"})

    assert response.headers["content-type"] == "application/json; charset=utf-8"
    assert response.json()["answer"].startswith("classic-answer")


def test_ask_classic_returns_answer_and_query_id(client: TestClient, trace_dir: Any) -> None:
    response = client.post("/api/ask", json={"query": "红烧肉怎么做", "role": "staff"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["pipeline"] == "classic"
    assert payload["answer"] == "classic-answer[staff]"  # 开发模式允许 body 指定角色
    assert len(payload["query_id"]) == 12


def test_ask_agent_exposes_route_and_deduped_hits(
    client: TestClient, trace_dir: Any
) -> None:
    response = client.post(
        "/api/ask", json={"query": "红烧肉怎么做", "pipeline": "agent"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["pipeline"] == "agent"
    assert payload["route"] == "search_recipes"
    assert payload["hits"] == ["简易红烧肉"]  # 两个 chunk 去重成一个菜名
    assert payload["answer"].startswith("agent-answer")


def test_ask_stream_emits_sse_tokens_and_done(client: TestClient, trace_dir: Any) -> None:
    with client.stream(
        "POST", "/api/ask/stream", json={"query": "红烧肉怎么做"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert "答案片段1" in body and "答案片段2" in body
    assert '"type": "token"' in body
    assert '"type": "done"' in body
    assert '"query_id"' in body


def test_search_returns_metadata_items(client: TestClient, trace_dir: Any) -> None:
    response = client.post("/api/search", json={"query": "红烧肉", "top_k": 3})

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["dish_name"] == "简易红烧肉"
    assert item["category"] == "荤菜"
    assert item["visibility"] == "public"
    assert item["score"] == pytest.approx(0.03)


def test_search_images_only_uses_image_endpoint(client: TestClient, trace_dir: Any) -> None:
    response = client.post("/api/search", json={"query": "成品图", "images_only": True})

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert item["modality"] == "image"
    assert item["image_path"].endswith("000.jpg")


def test_trace_roundtrip_after_search(client: TestClient, trace_dir: Any) -> None:
    search = client.post("/api/search", json={"query": "红烧肉"}).json()
    query_id = search["query_id"]

    found = client.get(f"/api/traces/{query_id}")
    missing = client.get("/api/traces/nonexistent00")

    assert found.status_code == 200
    assert found.json()["found"] is True
    assert found.json()["record"]["pipeline"] == "search"
    assert missing.json()["found"] is False


def test_auth_disabled_by_default_in_dev(client: TestClient, trace_dir: Any) -> None:
    # 未设置 RAG_JWT_SECRET → 开发模式，无需 token
    response = client.post("/api/ask", json={"query": "你好"})

    assert response.status_code == 200


def test_auth_enabled_rejects_missing_or_bad_token(
    monkeypatch: pytest.MonkeyPatch, trace_dir: Any
) -> None:
    monkeypatch.setenv("RAG_JWT_SECRET", "test-secret")
    app = create_app(rag=FakeRAG())
    with TestClient(app) as protected:
        no_token = protected.post("/api/ask", json={"query": "红烧肉怎么做"})
        bad_token = protected.post(
            "/api/ask",
            json={"query": "红烧肉怎么做"},
            headers={"Authorization": "Bearer not-a-jwt"},
        )

        assert no_token.status_code == 401
        assert bad_token.status_code == 401


def test_auth_role_comes_from_token_not_body(
    monkeypatch: pytest.MonkeyPatch, trace_dir: Any
) -> None:
    monkeypatch.setenv("RAG_JWT_SECRET", "test-secret")
    fake = FakeRAG()
    app = create_app(rag=fake)
    token = mint_token("staff", "test-secret")
    with TestClient(app) as protected:
        # body 里试图以 guest 越权，但角色必须来自 token（staff）
        response = protected.post(
            "/api/ask",
            json={"query": "速冻水饺的生产工艺", "role": "guest"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "classic-answer[staff]"
    assert fake.calls[-1][2] == "staff"  # 系统实际收到的 role


def test_agent_ask_includes_structured_citations(client: TestClient, trace_dir: Any) -> None:
    response = client.post(
        "/api/ask", json={"query": "红烧肉怎么做", "pipeline": "agent"}
    )

    payload = response.json()
    assert payload["citations"] == ["简易红烧肉（dishes/meat_dish/简易红烧肉/简易红烧肉.md）"]


def test_session_memory_passes_history_to_agent(client: TestClient, trace_dir: Any) -> None:
    # 第一轮：无历史
    first = client.post(
        "/api/ask",
        json={"query": "推荐一道红烧菜", "pipeline": "agent", "session_id": "s1"},
    ).json()
    # 第二轮：同 session，应携带第一轮的历史
    second = client.post(
        "/api/ask",
        json={"query": "它怎么做", "pipeline": "agent", "session_id": "s1"},
    )

    assert first is not None
    assert second.status_code == 200


def test_session_history_content_passed_to_rag(
    monkeypatch: pytest.MonkeyPatch, trace_dir: Any
) -> None:
    fake = FakeRAG()
    app = create_app(rag=fake)
    with TestClient(app) as c:
        c.post(
            "/api/ask",
            json={"query": "推荐一道红烧菜", "pipeline": "agent", "session_id": "s9"},
        )
        c.post("/api/ask", json={"query": "它怎么做", "pipeline": "agent", "session_id": "s9"})
        c.post("/api/ask", json={"query": "换个素菜", "pipeline": "agent"})  # 无session，无历史

    agent_calls = [call for call in fake.calls if call[0] == "ask_agent"]
    assert agent_calls[0][4] == []  # 第一轮无历史
    history = agent_calls[1][4]
    assert history[0] == {"role": "user", "content": "推荐一道红烧菜"}
    assert history[1]["role"] == "assistant"
    assert "agent-answer" in history[1]["content"]
    assert agent_calls[2][4] == []  # 未带 session 的请求不携带历史
