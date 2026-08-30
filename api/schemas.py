"""Request/response models for the RAG service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="用户问题")
    role: str | None = Field(
        default=None,
        description="访问角色（仅当鉴权关闭的开发模式生效；鉴权开启时以 token 中的 role 为准）",
    )
    pipeline: Literal["classic", "agent"] = Field(default="classic", description="问答链路")
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="会话 ID：传入且 pipeline=agent 时启用多轮记忆（内存存储，最近3轮）",
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    role: str | None = Field(default=None, description="同 AskRequest.role")
    images_only: bool = Field(default=False, description="True 时只在图片 chunk 中检索")


class AskResponse(BaseModel):
    query_id: str
    answer: str
    pipeline: str
    route: str | None = None
    hits: list[str] = Field(default_factory=list, description="命中条目名称（agent 链路）")
    citations: list[str] = Field(
        default_factory=list,
        description="答案引用来源（agent 链路：菜名 + 语料路径；classic 链路见 answer 末尾附录）",
    )
    error: str | None = None


class SearchResultItem(BaseModel):
    dish_name: str
    category: str | None = None
    difficulty: str | None = None
    visibility: str | None = None
    modality: str | None = None
    image_path: str | None = None
    score: float | None = None
    content_preview: str


class SearchResponse(BaseModel):
    query_id: str
    results: list[SearchResultItem]


class TraceResponse(BaseModel):
    query_id: str
    found: bool
    record: dict[str, Any] | None = None
