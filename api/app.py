"""FastAPI application: HTTP/SSE shell over the RAG engine.

The RAG system (embeddings, Milvus connection, agent graph) is loaded once in
the lifespan and reused by every request — this is the service-mode answer to
the CLI's per-run cold start.  All RAG logic lives in ``rag_modules``; this
layer only handles HTTP, auth, and response shaping.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from api.auth import AuthError, Principal, authenticate
from api.schemas import (
    AskRequest,
    AskResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    TraceResponse,
)

logger = logging.getLogger(__name__)


def _new_query_id() -> str:
    return uuid.uuid4().hex[:12]


def create_app(rag: Any | None = None) -> FastAPI:
    """Build the app; tests inject a fake RAG system via ``rag``."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.rag is None:
            from main import RecipeRAGSystem

            logger.info("服务模式启动：正在加载 RAG 系统（模型/索引常驻内存）")
            system = RecipeRAGSystem()
            system.initialize_system(load_generation=True)
            system.build_knowledge_base()
            app.state.rag = system
            logger.info("RAG 系统就绪，开始对外服务")
        yield

    # 响应显式声明 charset=utf-8：PowerShell 5.1 / 老版 .NET 客户端在
    # Content-Type 缺少 charset 时会按 Latin-1 解码，中文变成 äçé 乱码。
    class UTF8JSONResponse(JSONResponse):
        media_type = "application/json; charset=utf-8"

    app = FastAPI(
        title="Recipe RAG Service",
        version="2.0.0",
        lifespan=lifespan,
        default_response_class=UTF8JSONResponse,
    )
    app.state.rag = rag
    # 多轮会话与长期记忆：SQLite 持久化（重启可恢复；多副本部署实现
    # MySQLStore 替换，见 rag_modules/memory.py 的 Store 抽象）。
    # fake rag（测试注入）没有 config 时用一次性临时库，保证用例隔离。
    from rag_modules.memory import SQLiteMemoryStore

    memory_db = getattr(getattr(rag, "config", None), "memory_db_path", None)
    if not memory_db:
        memory_db = os.path.join(
            tempfile.gettempdir(), f"rag_memory_{uuid.uuid4().hex[:8]}.sqlite3"
        )
    app.state.memory_store = SQLiteMemoryStore(memory_db)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_rag(request: Request) -> Any:
        system = request.app.state.rag
        if system is None:
            raise HTTPException(status_code=503, detail="RAG 系统尚未就绪")
        return system

    def get_principal(request: Request) -> Principal:
        try:
            return authenticate(request.headers.get("Authorization"))
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    def effective_role(principal: Principal, requested: str | None) -> str:
        # 鉴权开启时角色以 token 为准，请求体不可越权
        return requested if (principal.dev_mode and requested) else principal.role

    def load_history(request: Request, session_id: str | None) -> list[dict[str, str]]:
        if not session_id:
            return []
        try:
            return request.app.state.memory_store.load_history(session_id)
        except Exception as exc:  # noqa: BLE001 - 会话故障降级为无历史
            logger.warning("会话历史读取失败（按无历史处理）: %s", exc)
            return []

    def save_turn(request: Request, session_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        try:
            request.app.state.memory_store.save_turn(session_id, question, answer)
        except Exception as exc:  # noqa: BLE001 - 会话写入失败不影响本次回答
            logger.warning("会话历史写入失败（跳过）: %s", exc)

    @app.get("/healthz")
    def healthz(rag: Any = Depends(get_rag)) -> dict[str, Any]:
        backend = getattr(rag, "backend", None)
        rows: int | None = None
        try:
            if backend is not None and hasattr(backend, "count"):
                rows = backend.count()
        except Exception:  # noqa: BLE001 - healthz 不应因依赖抖动而 5xx
            rows = None
        return {"status": "ok", "backend": getattr(rag.config, "backend", "?"), "rows": rows}

    @app.post("/api/ask", response_model=AskResponse)
    def ask(
        request: Request,
        body: AskRequest,
        rag: Any = Depends(get_rag),
        principal: Principal = Depends(get_principal),
    ) -> AskResponse:
        role = effective_role(principal, body.role)
        query_id = _new_query_id()
        if body.pipeline == "agent":
            try:
                result = rag.ask_agent(
                    body.query,
                    role=role,
                    query_id=query_id,
                    history=load_history(request, body.session_id),
                    user_id=principal.subject or body.session_id,
                )
            except Exception as exc:  # noqa: BLE001 - agent 链路失败已落 trace，这里转 5xx
                logger.exception("agent 问答失败")
                raise HTTPException(status_code=500, detail=f"agent 问答失败: {exc}") from exc
            save_turn(request, body.session_id, body.query, str(result.get("answer", "")))
            hits = [
                str(chunk.metadata.get("dish_name", ""))
                for chunk in result.get("chunks", [])
            ]
            citations: list[str] = []
            for parent in result.get("parents", []):
                name = str(parent.metadata.get("dish_name", ""))
                source = str(parent.metadata.get("source_path", "")).strip()
                entry = f"{name}（{source}）" if source else name
                if entry not in citations:
                    citations.append(entry)
            return AskResponse(
                query_id=query_id,
                answer=str(result.get("answer", "")),
                pipeline="agent",
                route=result.get("route"),
                hits=list(dict.fromkeys(hits)),
                citations=citations,
                error=result.get("error"),
            )
        answer = rag.ask_question(body.query, stream=False, role=role, query_id=query_id)
        return AskResponse(query_id=query_id, answer=str(answer), pipeline="classic")

    @app.post("/api/ask/stream")
    async def ask_stream(
        request: Request,
        body: AskRequest,
        rag: Any = Depends(get_rag),
        principal: Principal = Depends(get_principal),
    ) -> StreamingResponse:
        role = effective_role(principal, body.role)
        query_id = _new_query_id()

        def sse(payload: dict[str, Any]) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        async def event_stream() -> AsyncIterator[str]:
            try:
                if body.pipeline == "agent":
                    result = rag.ask_agent(
                        body.query,
                        role=role,
                        query_id=query_id,
                        history=load_history(request, body.session_id),
                        user_id=principal.subject or body.session_id,
                    )
                    answer = str(result.get("answer", ""))
                    save_turn(request, body.session_id, body.query, answer)
                    yield sse({"type": "token", "content": answer})
                    yield sse(
                        {
                            "type": "done",
                            "query_id": query_id,
                            "route": result.get("route"),
                            "error": result.get("error"),
                        }
                    )
                    return
                for chunk in rag.ask_question(
                    body.query, stream=True, role=role, query_id=query_id
                ):
                    yield sse({"type": "token", "content": chunk})
                yield sse({"type": "done", "query_id": query_id})
            except Exception as exc:  # noqa: BLE001 - 流中途失败以 error 事件收尾
                logger.exception("流式问答失败")
                yield sse({"type": "error", "query_id": query_id, "message": str(exc)})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/search", response_model=SearchResponse)
    def search(
        body: SearchRequest,
        rag: Any = Depends(get_rag),
        principal: Principal = Depends(get_principal),
    ) -> SearchResponse:
        role = effective_role(principal, body.role)
        query_id = _new_query_id()
        from rag_modules.observability import QueryTrace

        trace = QueryTrace(body.query, role=role, pipeline="search", query_id=query_id)
        try:
            if body.images_only:
                documents = rag.retrieve_images(body.query, top_k=body.top_k, role=role)
            else:
                documents = rag.retrieve(body.query, top_k=body.top_k, role=role)
            trace.event("search", hits=len(documents), images_only=body.images_only)
        except Exception as exc:  # noqa: BLE001
            trace.event("search_error", error=str(exc))
            raise HTTPException(status_code=500, detail=f"检索失败: {exc}") from exc
        finally:
            trace.finish(answer=None)

        items = [
            SearchResultItem(
                dish_name=str(doc.metadata.get("dish_name", "")),
                category=doc.metadata.get("category"),
                difficulty=doc.metadata.get("difficulty"),
                visibility=doc.metadata.get("visibility"),
                modality=doc.metadata.get("modality"),
                image_path=doc.metadata.get("image_path") or None,
                score=doc.metadata.get("rrf_score"),
                content_preview=doc.page_content[:120],
            )
            for doc in documents
        ]
        return SearchResponse(query_id=query_id, results=items)

    @app.get("/api/traces/{query_id}", response_model=TraceResponse)
    def traces(query_id: str) -> TraceResponse:
        from rag_modules.observability import load_traces

        for record in reversed(load_traces()):
            if record.get("query_id") == query_id:
                return TraceResponse(query_id=query_id, found=True, record=record)
        return TraceResponse(query_id=query_id, found=False, record=None)

    # ------------------------------------------------ 长期记忆隐私端点（P2.8）

    def _memory_user(principal: Principal, session_id: str | None) -> str:
        user_id = principal.subject or session_id
        if not user_id:
            raise HTTPException(status_code=400, detail="需要认证 token 或提供 session_id")
        return user_id

    @app.get("/api/memory")
    def list_memory(
        request: Request,
        session_id: str | None = None,
        principal: Principal = Depends(get_principal),
    ) -> dict[str, Any]:
        user_id = _memory_user(principal, session_id)
        from dataclasses import asdict

        records = request.app.state.memory_store.list_memories(user_id)
        return {"user_id": user_id, "memories": [asdict(record) for record in records]}

    @app.delete("/api/memory/{memory_id}")
    def delete_memory(
        memory_id: str,
        request: Request,
        session_id: str | None = None,
        principal: Principal = Depends(get_principal),
    ) -> dict[str, Any]:
        user_id = _memory_user(principal, session_id)
        deleted = request.app.state.memory_store.delete_memory(user_id, memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="记忆不存在")
        return {"user_id": user_id, "deleted": memory_id}

    @app.delete("/api/memory")
    def delete_all_memory(
        request: Request,
        session_id: str | None = None,
        principal: Principal = Depends(get_principal),
    ) -> dict[str, Any]:
        user_id = _memory_user(principal, session_id)
        deleted = request.app.state.memory_store.delete_all(user_id)
        return {"user_id": user_id, "deleted_count": deleted}

    return app


app = create_app()
