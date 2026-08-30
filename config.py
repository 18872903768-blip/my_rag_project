"""Configuration for the CPU-friendly recipe RAG baseline."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parents[1]
DEFAULT_DATA_PATH = WORKSPACE_DIR / "data" / "C8" / "cook"
DEFAULT_INDEX_PATH = PROJECT_DIR / "vector_index"


def _path_from_env(name: str, default: Path) -> str:
    raw = os.getenv(name)
    if not raw:
        return str(default.resolve())
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return str(path.resolve())


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_MILVUS_COLLECTION = "recipe_rag_v1"

# 百炼 rerank HTTP 端点（国内站），适配 qwen3-vl-rerank / gte-rerank 系列。
DEFAULT_DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


@dataclass(slots=True)
class RAGConfig:
    """Runtime settings, with paths independent of the process working directory."""

    data_path: str = str(DEFAULT_DATA_PATH.resolve())
    index_save_path: str = str(DEFAULT_INDEX_PATH.resolve())

    backend: str = "milvus"
    milvus_uri: str = DEFAULT_MILVUS_URI
    milvus_collection: str = DEFAULT_MILVUS_COLLECTION
    milvus_token: str = ""

    enable_image_ingestion: bool = True
    vision_model: str = "deepseek-v4-flash-vision-exp"
    vision_base_url: str = "https://api.deepseek.com"
    vision_max_workers: int = 4

    internal_categories: str = "半成品"

    # 重排实验结论（73条 golden set，milvus/top_k=5）：
    #   2026-08-23: RRF基线 MRR 0.931，纯 CE 重排 0.879，CE+RRF 二次融合 0.902。
    #   2026-08-29: 基线 0.9315，本地 bge-reranker-base 融合 0.9018，
    #               dashscope qwen3-vl-rerank 融合 0.9315（keyword recall
    #               0.749 vs 基线 0.738，两条 query 互抵，噪声级）。
    #   2026-08-29 模糊集 24 条（同义/错字/间接/混淆，eval/golden_set_fuzzy.json）：
    #               基线 MRR 0.861，qwen3 融合 0.854，本地 CE 0.767；qwen3 赢
    #               同义组（麻辣豆腐→麻婆豆腐 MRR 0.33→1.0）但输错字组，
    #               Hit@5 三组均 0.917——净效应仍≈基线，CE+RRF 融合抵消了
    #               语义增益。
    #   2026-08-29 方案一 title 注入（打分文本前缀"菜名（分类）"，修复
    #               盲段）：qwen3+注入 精确集 keyword recall 0.768（历史最优）；
    #               模糊集 MRR 0.896 首次超过基线 0.861（同义 0.900、
    #               错字 1.000、混淆 0.667 全部回到或超过基线）；本地 CE
    #               也从 0.767 升到 0.854——盲段是精排失效的主因，而非模型。
    #               故 RAG_RERANK_CONTEXT 默认开启。
    # 本语料 BM25 精确菜名信号已足够强，重排无增益，故默认关闭；但云端
    # qwen3 精排不伤 MRR 且明显优于本地小模型（4 条精确名 query 本地 CE
    # 掉到 rank2+，qwen3 全部 rank1），换语料需精排时优先
    # RAG_RERANK_PROVIDER=dashscope（失败自动回退本地模型）。
    rerank_enabled: bool = False
    rerank_provider: str = "local"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_pool: int = 20
    rerank_api_key: str = ""
    rerank_base_url: str = DEFAULT_DASHSCOPE_RERANK_URL
    rerank_timeout_seconds: float = 10.0
    # 精排打分文本是否注入"菜名（分类）"前缀（方案一 title 注入，修复盲段）。
    # 重排默认关闭；开启重排时默认启用注入（2026-08-29 实测对两套集合均有增益）。
    rerank_context_enabled: bool = True

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_revision: str | None = "7999e1d3359715c523056ef9478215996d62a620"
    embedding_device: str = "cpu"
    normalize_embeddings: bool = True

    llm_model: str = "deepseek-chat"
    top_k: int = 3
    candidate_k: int = 10
    rrf_k: int = 60
    temperature: float = 0.1
    max_tokens: int = 2048
    max_context_chars: int = 6000
    # 生成 prompt 追加忠实性硬约束（禁编造时间/用量/尺寸与常识性补充）。
    # RAGAS A/B 实测（45 条 golden set，2026-08-30，judge=deepseek-chat）：
    #   faithfulness 0.650 → 0.950（最差样本 0.22 → 0.91），answer_correctness
    #   0.791 → 0.801，context 指标不变（检索层未动）；answer_relevancy
    #   0.582 → 0.535（答案变精炼后反向提问相似度略降，属指标噪声级）。
    # 忠实性收益显著且无实质代价，故默认开启；置 RAG_GROUNDED_ANSWER=false
    # 可回退旧行为。
    grounded_answer: bool = True

    # agent 管线统一上下文管理（ContextItem/token 预算/优先级/去重）。
    # false 时走原 _history_prefix 硬拼 + _build_context 字符拼接路径。
    context_manager_enabled: bool = False
    context_budget_tokens: int = 6000

    # 多轮指代消解：agent 检索前把"那第二种呢"改写为独立查询（LLM 一次调用）
    query_contextualization_enabled: bool = False

    # 长期记忆（饮食偏好领域化）：SQLite 存储 + LLM 抽取 + 写入策略。
    # 开启后 agent 回答结束会尝试抽取记忆，检索前注入相关偏好。
    memory_enabled: bool = False
    memory_db_path: str = str((PROJECT_DIR / "memory_store.sqlite3").resolve())

    @classmethod
    def from_env(cls) -> RAGConfig:
        revision = os.getenv(
            "RAG_EMBEDDING_REVISION",
            "7999e1d3359715c523056ef9478215996d62a620",
        )
        return cls(
            data_path=_path_from_env("RAG_DATA_PATH", DEFAULT_DATA_PATH),
            index_save_path=_path_from_env("RAG_INDEX_PATH", DEFAULT_INDEX_PATH),
            backend=os.getenv("RAG_BACKEND", "milvus").strip().lower(),
            milvus_uri=os.getenv("RAG_MILVUS_URI", DEFAULT_MILVUS_URI),
            milvus_collection=os.getenv("RAG_MILVUS_COLLECTION", DEFAULT_MILVUS_COLLECTION),
            milvus_token=os.getenv("RAG_MILVUS_TOKEN", ""),
            enable_image_ingestion=os.getenv("RAG_ENABLE_IMAGE_INGESTION", "true").casefold()
            not in {"0", "false", "no", "off"},
            vision_model=os.getenv("RAG_VISION_MODEL", "deepseek-v4-flash-vision-exp"),
            vision_base_url=os.getenv("RAG_VISION_BASE_URL", "https://api.deepseek.com"),
            vision_max_workers=_int_from_env("RAG_VISION_MAX_WORKERS", 4),
            internal_categories=os.getenv("RAG_INTERNAL_CATEGORIES", "半成品"),
            rerank_enabled=os.getenv("RAG_RERANK_ENABLED", "false").casefold()
            not in {"0", "false", "no", "off"},
            rerank_provider=os.getenv("RAG_RERANK_PROVIDER", "local").strip().lower(),
            rerank_model=os.getenv("RAG_RERANK_MODEL", "").strip()
            or (
                "qwen3-vl-rerank"
                if os.getenv("RAG_RERANK_PROVIDER", "local").strip().lower() == "dashscope"
                else "BAAI/bge-reranker-base"
            ),
            rerank_pool=_int_from_env("RAG_RERANK_POOL", 20),
            rerank_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            rerank_base_url=os.getenv("RAG_RERANK_BASE_URL", DEFAULT_DASHSCOPE_RERANK_URL),
            rerank_timeout_seconds=_float_from_env("RAG_RERANK_TIMEOUT", 10.0),
            rerank_context_enabled=os.getenv("RAG_RERANK_CONTEXT", "true").casefold()
            not in {"0", "false", "no", "off"},
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            embedding_revision=revision or None,
            embedding_device=os.getenv("RAG_EMBEDDING_DEVICE", "cpu"),
            normalize_embeddings=os.getenv("RAG_NORMALIZE_EMBEDDINGS", "true").casefold()
            not in {"0", "false", "no", "off"},
            llm_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            top_k=_int_from_env("RAG_TOP_K", 3),
            candidate_k=_int_from_env("RAG_CANDIDATE_K", 10),
            rrf_k=_int_from_env("RAG_RRF_K", 60),
            temperature=_float_from_env("RAG_TEMPERATURE", 0.1),
            max_tokens=_int_from_env("RAG_MAX_TOKENS", 2048),
            max_context_chars=_int_from_env("RAG_MAX_CONTEXT_CHARS", 6000),
            grounded_answer=os.getenv("RAG_GROUNDED_ANSWER", "true").casefold()
            not in {"0", "false", "no", "off"},
            context_manager_enabled=os.getenv("RAG_CONTEXT_MANAGER", "false").casefold()
            not in {"0", "false", "no", "off"},
            context_budget_tokens=_int_from_env("RAG_CONTEXT_BUDGET", 6000),
            query_contextualization_enabled=os.getenv(
                "RAG_QUERY_CONTEXTUALIZATION", "false"
            ).casefold()
            not in {"0", "false", "no", "off"},
            memory_enabled=os.getenv("RAG_MEMORY_ENABLED", "false").casefold()
            not in {"0", "false", "no", "off"},
            memory_db_path=_path_from_env(
                "RAG_MEMORY_DB", PROJECT_DIR / "memory_store.sqlite3"
            ),
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> RAGConfig:
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RAGConfig.from_env()
