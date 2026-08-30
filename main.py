"""Command-line entry point for the recipe RAG system."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from config import RAGConfig
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.domain_config import get_domain
from rag_modules.index_construction import IndexConstructionModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule

# Load configuration before importing config.DEFAULT_CONFIG. Existing shell values
# always win, and indexing remains usable when no LLM key is configured.
PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parents[1]
load_dotenv(WORKSPACE_DIR / ".env", override=False)
load_dotenv(PROJECT_DIR / ".env", override=False)
RUNTIME_CONFIG = RAGConfig.from_env()

logger = logging.getLogger(__name__)

# 访问角色 → 可见范围。客服场景：游客/注册用户只看公开菜谱，内部员工可看
# 半成品工艺等内部文档。权限以元数据过滤形式在检索层强制执行。
ROLE_ALLOWED_VISIBILITY: dict[str, list[str]] = {
    "guest": ["public"],
    "user": ["public"],
    "staff": ["public", "internal"],
}
DEFAULT_ROLE = "user"


class RecipeRAGSystem:
    """Orchestrate corpus preparation, indexing, retrieval, and optional generation."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RUNTIME_CONFIG
        data_path = Path(self.config.data_path)
        if not data_path.is_dir():
            raise FileNotFoundError(f"数据路径不存在或不是目录: {data_path}")

        self.data_module: DataPreparationModule | None = None
        self.index_module: IndexConstructionModule | None = None
        self.retrieval_module: RetrievalOptimizationModule | None = None
        self.generation_module: Any | None = None
        self.backend: Any | None = None
        self._agent: Any | None = None

    def initialize_system(self, *, load_generation: bool = True) -> None:
        """Initialize local modules and, when requested, the remote LLM client."""
        logger.info("正在初始化 RAG 系统")
        internal_categories = {
            item.strip()
            for item in self.config.internal_categories.split(",")
            if item.strip()
        }
        self.data_module = DataPreparationModule(
            self.config.data_path, internal_categories=internal_categories
        )
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            model_revision=self.config.embedding_revision,
            device=self.config.embedding_device,
            normalize_embeddings=self.config.normalize_embeddings,
            index_save_path=self.config.index_save_path,
        )

        if load_generation:
            from rag_modules.generation_integration import GenerationIntegrationModule

            self.generation_module = GenerationIntegrationModule(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                max_context_chars=self.config.max_context_chars,
                grounded_answer=self.config.grounded_answer,
            )
        logger.info("系统初始化完成")

    def _ingest_images(self, documents: list[Any]) -> list[Any]:
        """Caption dish photos; degrade to text-only when disabled or unreachable."""
        import os

        assert self.data_module is not None
        if not self.config.enable_image_ingestion:
            logger.info("图片摄取已关闭（RAG_ENABLE_IMAGE_INGESTION=false）")
            return []
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("未设置 DEEPSEEK_API_KEY，跳过图片摄取，仅使用文本语料")
            return []

        from rag_modules.image_ingestion import ImageIngestionModule

        image_module = ImageIngestionModule(
            self.config.data_path,
            Path(self.config.index_save_path) / "image_caption_cache.json",
            vision_model=self.config.vision_model,
            api_key=api_key,
            base_url=self.config.vision_base_url,
            max_workers=self.config.vision_max_workers,
        )
        try:
            image_chunks = image_module.ingest_images(documents)
        except Exception:
            logger.exception("图片摄取失败，降级为纯文本索引")
            return []
        self.data_module.chunks = list(self.data_module.chunks) + list(image_chunks)
        return image_chunks

    def _build_reranker(self) -> Any | None:
        """精排（本地 cross-encoder 或 dashscope API，懒加载；禁用时返回 None）。"""
        if not self.config.rerank_enabled:
            logger.info("重排已关闭（RAG_RERANK_ENABLED=false）")
            return None
        from rag_modules.reranker import RerankerModule

        logger.info(
            "启用重排: provider=%s, model=%s", self.config.rerank_provider, self.config.rerank_model
        )
        return RerankerModule(
            self.config.rerank_model,
            enabled=True,
            pool_size=self.config.rerank_pool,
            device="cpu",
            provider=self.config.rerank_provider,
            api_key=self.config.rerank_api_key,
            base_url=self.config.rerank_base_url,
            timeout_seconds=self.config.rerank_timeout_seconds,
            include_context=self.config.rerank_context_enabled,
        )

    def build_knowledge_base(self, *, force_rebuild: bool = False) -> dict[str, Any]:
        """Load the corpus and build or safely reuse a matching vector index."""
        if self.data_module is None or self.index_module is None:
            self.initialize_system(load_generation=False)
        assert self.data_module is not None
        assert self.index_module is not None

        documents = self.data_module.load_documents()
        chunks = self.data_module.chunk_documents()
        logger.info("已准备 %d 篇食谱、%d 个文本 chunk", len(documents), len(chunks))

        image_chunks = self._ingest_images(documents)
        all_chunks = chunks + image_chunks

        backend_name = self.config.backend
        if backend_name == "milvus":
            from rag_modules.vector_store import MilvusVectorStore

            milvus_store = MilvusVectorStore(
                collection_name=self.config.milvus_collection,
                uri=self.config.milvus_uri,
                token=self.config.milvus_token,
                embeddings=self.index_module.setup_embeddings(),
                manifest_dir=self.config.index_save_path,
            )
            rebuilt = milvus_store.build_index(all_chunks, force_rebuild=force_rebuild)
            self.backend = milvus_store
            self.retrieval_module = RetrievalOptimizationModule(
                milvus_store,
                all_chunks,
                candidate_k=self.config.candidate_k,
                rrf_k=self.config.rrf_k,
                reranker=self._build_reranker(),
            )
            logger.info(
                "知识库就绪（Milvus: %s, %s）: rows=%d",
                self.config.milvus_collection,
                "已重建" if rebuilt else "复用现有索引",
                milvus_store.count(),
            )
        elif backend_name == "faiss":
            from rag_modules.vector_store import FaissBackend

            vectorstore = None
            if not force_rebuild:
                vectorstore = self.index_module.load_index(all_chunks)
            if vectorstore is None:
                logger.info("正在从当前语料重建向量索引")
                vectorstore = self.index_module.build_vector_index(all_chunks)
                self.index_module.save_index()
            else:
                logger.info("已复用与当前语料及配置完全匹配的向量索引")
            self.backend = FaissBackend(vectorstore)
            self.retrieval_module = RetrievalOptimizationModule(
                self.backend,
                all_chunks,
                candidate_k=self.config.candidate_k,
                rrf_k=self.config.rrf_k,
                reranker=self._build_reranker(),
            )
        else:
            raise ValueError(f"未知的向量后端: {backend_name!r}（可选 milvus / faiss）")

        statistics = self.data_module.get_statistics()
        logger.info(
            "知识库就绪: documents=%d chunks=%d",
            statistics["total_documents"],
            statistics["total_chunks"],
        )
        return statistics

    def _extract_filters_from_query(self, query: str) -> dict[str, str]:
        filters: dict[str, str] = {}
        for category in DataPreparationModule.get_supported_categories():
            if category in query:
                filters["category"] = category
                break
        for difficulty in sorted(
            DataPreparationModule.get_supported_difficulties(), key=len, reverse=True
        ):
            if difficulty in query:
                filters["difficulty"] = difficulty
                break
        return filters

    @staticmethod
    def _visibility_expr_for_role(role: str) -> str | None:
        from rag_modules.vector_store import build_filter_expr

        allowed = ROLE_ALLOWED_VISIBILITY.get(role)
        if allowed is None:
            raise ValueError(f"未知角色: {role!r}（可选 {sorted(ROLE_ALLOWED_VISIBILITY)}）")
        if set(allowed) >= {"public", "internal"}:
            return None  # 可见全部内容，无需过滤
        return build_filter_expr({"visibility": allowed})

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        role: str = DEFAULT_ROLE,
    ) -> list[Any]:
        """Run the local hybrid retriever without requiring an LLM API key."""
        if self.retrieval_module is None:
            raise ValueError("请先构建知识库")
        limit = top_k or self.config.top_k
        visibility_expr = self._visibility_expr_for_role(role)
        active_filters = filters if filters is not None else self._extract_filters_from_query(query)
        if active_filters or visibility_expr:
            return self.retrieval_module.metadata_filtered_search(
                query,
                active_filters or {},
                top_k=limit,
                extra_expr=visibility_expr,
            )
        return self.retrieval_module.hybrid_search(query, top_k=limit)

    def retrieve_images(
        self, query: str, *, top_k: int | None = None, role: str = DEFAULT_ROLE
    ) -> list[Any]:
        """Text-to-image search: hybrid retrieval restricted to image chunks."""
        if self.retrieval_module is None:
            raise ValueError("请先构建知识库")
        limit = top_k or self.config.top_k
        visibility_expr = self._visibility_expr_for_role(role)
        return self.retrieval_module.metadata_filtered_search(
            query, {"modality": "image"}, top_k=limit, extra_expr=visibility_expr
        )

    def ask_question(
        self,
        question: str,
        *,
        stream: bool = False,
        role: str = DEFAULT_ROLE,
        query_id: str | None = None,
    ) -> Any:
        if self.retrieval_module is None:
            raise ValueError("请先构建知识库")
        if self.generation_module is None:
            raise ValueError("生成模块未初始化；离线模式请使用 retrieve()")
        assert self.data_module is not None

        from rag_modules.observability import QueryTrace

        trace = QueryTrace(question, role=role, pipeline="classic", query_id=query_id)
        try:
            answer = self._ask_question_inner(question, stream=stream, role=role, trace=trace)
            trace.finish(answer=str(answer) if isinstance(answer, str) else None)
            return answer
        except Exception as exc:
            logger.exception("经典链路处理失败")
            trace.event("pipeline_error", error=str(exc))
            trace.finish(error=exc)
            return f"处理问题时出错: {exc}"

    def _ask_question_inner(
        self, question: str, *, stream: bool, role: str, trace: Any
    ) -> Any:
        assert self.data_module is not None
        assert self.generation_module is not None

        route_type = self.generation_module.query_router(question)
        trace.event("route", route=route_type)
        rewritten_query = (
            question if route_type == "list" else self.generation_module.query_rewrite(question)
        )
        trace.event("rewrite", query=rewritten_query)
        relevant_chunks = self.retrieve(
            rewritten_query,
            filters=self._extract_filters_from_query(question),
            role=role,
        )
        trace.event("retrieve", hits=len(relevant_chunks))
        if not relevant_chunks:
            trace.event("no_results")
            return get_domain().classic_no_results

        relevant_documents = self.data_module.get_parent_documents(relevant_chunks)
        image_paths = self._collect_image_paths(relevant_chunks)
        trace.event("parents", count=len(relevant_documents), images=len(image_paths))
        if route_type == "list":
            return self.generation_module.generate_list_answer(question, relevant_documents)
        if route_type == "detail":
            if stream:
                return self.generation_module.generate_step_by_step_answer_stream(
                    question, relevant_documents
                )
            return self.generation_module.generate_step_by_step_answer(
                question, relevant_documents, image_paths=image_paths
            )
        if stream:
            return self.generation_module.generate_basic_answer_stream(
                question, relevant_documents
            )
        return self.generation_module.generate_basic_answer(
            question, relevant_documents, image_paths=image_paths
        )

    def _get_agent(self) -> Any:
        """Create the LangGraph agent lazily (requires LLM + built knowledge base)."""
        if self._agent is None:
            if self.retrieval_module is None or self.data_module is None:
                raise ValueError("请先构建知识库")
            if self.generation_module is None:
                self.initialize_system(load_generation=True)

            from rag_modules.agentic_rag import RecipeAgent

            context_manager = None
            if self.config.context_manager_enabled:
                from rag_modules.context_management import ContextManager

                context_manager = ContextManager(
                    budget_tokens=self.config.context_budget_tokens,
                    dedup_embeddings=self.index_module.setup_embeddings(),
                )

            self._agent = RecipeAgent(
                self.retrieval_module,
                self.data_module,
                self.generation_module,
                top_k=self.config.top_k,
                visibility_expr_builder=self._visibility_expr_for_role,
                context_manager=context_manager,
            )
        return self._agent

    def ask_agent(
        self,
        question: str,
        *,
        role: str = DEFAULT_ROLE,
        query_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run the agentic RAG graph; the result carries answer + audit events."""
        from rag_modules.observability import QueryTrace

        trace = QueryTrace(question, role=role, pipeline="agent", query_id=query_id)
        try:
            result = self._get_agent().invoke(question, role=role, history=history)
            trace.events.extend(result.get("events", []))
            trace.event(
                "summary",
                route=result.get("route"),
                hits=len(result.get("chunks", [])),
                rewrites=result.get("rewrites_used", 0),
            )
            trace.finish(
                answer=result.get("answer"),
                error=result.get("error"),
            )
            result["query_id"] = trace.query_id
            return result
        except Exception as exc:
            logger.exception("Agent 链路处理失败")
            trace.event("pipeline_error", error=str(exc))
            trace.finish(error=exc)
            raise

    @staticmethod
    def _collect_image_paths(chunks: list[Any], *, limit: int = 6) -> list[str]:
        paths: list[str] = []
        for chunk in chunks:
            if chunk.metadata.get("modality") == "image":
                path = str(chunk.metadata.get("image_path", ""))
                if path and path not in paths:
                    paths.append(path)
                    if len(paths) >= limit:
                        break
        return paths

    @staticmethod
    def _print_answer(answer: str | Iterator[str]) -> None:
        if isinstance(answer, str):
            print(answer)
            return
        for chunk in answer:
            print(chunk, end="", flush=True)
        print()

    def run_interactive(
        self, *, force_rebuild: bool = False, role: str = DEFAULT_ROLE, agent_mode: bool = False
    ) -> None:
        self.initialize_system(load_generation=True)
        statistics = self.build_knowledge_base(force_rebuild=force_rebuild)
        print(
            f"知识库就绪：{statistics['total_documents']} 篇食谱，"
            f"{statistics['total_chunks']} 个文本块。"
        )
        print(f"当前角色: {role}（可见范围: {'全部' if role == 'staff' else '公开菜谱'}）")
        pipeline = "LangGraph agent" if agent_mode else "固定管线"
        print(f"交互式问答已启动（{pipeline}，输入 exit、quit 或 退出结束）。")

        history: list[dict[str, Any]] = []
        while True:
            try:
                question = input("\n您的问题: ").strip()
                if question.casefold() in {"", "退出", "quit", "exit"}:
                    break
                if agent_mode:
                    result = self.ask_agent(question, role=role, history=history)
                    answer = str(result.get("answer", ""))
                    self._print_answer(answer)
                    history.append({"role": "user", "content": question})
                    history.append({"role": "assistant", "content": answer[:400]})
                    history = history[-6:]  # 只保留最近3轮
                    print(
                        f"\n[agent] route={result.get('route')} "
                        f"hits={len(result.get('chunks', []))} "
                        f"rewrites={result.get('rewrites_used', 0)} "
                        f"query_id={result.get('query_id', '-')}"
                    )
                else:
                    self._print_answer(self.ask_question(question, stream=True, role=role))
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.exception("处理问题失败")
                print(f"处理问题时出错: {exc}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-friendly recipe RAG")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-only",
        action="store_true",
        help="仅构建/校验本地索引，不需要 DEEPSEEK_API_KEY",
    )
    mode.add_argument(
        "--retrieve",
        metavar="QUERY",
        help="执行一次本地双路检索并打印命中，不调用 LLM",
    )
    mode.add_argument(
        "--retrieve-images",
        metavar="QUERY",
        help="文搜图：在图片 caption 索引中检索并打印命中的图片",
    )
    mode.add_argument(
        "--agent",
        metavar="QUERY",
        help="Agentic RAG：由 LangGraph agent 路由工具并生成回答（需 DEEPSEEK_API_KEY）",
    )
    parser.add_argument(
        "--chat-mode",
        choices=("classic", "agent"),
        default="classic",
        help="交互模式链路：classic=固定管线，agent=LangGraph 图",
    )
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量索引")
    parser.add_argument(
        "--backend",
        choices=("milvus", "faiss"),
        default=None,
        help="向量后端（默认取 RAG_BACKEND 环境变量，未设置时为 milvus）",
    )
    parser.add_argument("--top-k", type=int, default=None, help="检索结果数量")
    parser.add_argument(
        "--role",
        choices=("guest", "user", "staff"),
        default=None,
        help="访问角色（决定 visibility 权限过滤范围，默认 user）",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        system = RecipeRAGSystem()
        if args.backend:
            system.config.backend = args.backend
        role = args.role or DEFAULT_ROLE
        if args.agent is not None:
            system.initialize_system(load_generation=True)
            system.build_knowledge_base(force_rebuild=args.rebuild)
            result = system.ask_agent(args.agent, role=role)
            print(result.get("answer", ""))
            print(
                f"\n[trace] route={result.get('route')} hits={len(result.get('chunks', []))} "
                f"rewrites={result.get('rewrites_used', 0)} error={result.get('error') or '-'}"
            )
            return 0
        if args.build_only or args.retrieve is not None or args.retrieve_images is not None:
            system.initialize_system(load_generation=False)
            statistics = system.build_knowledge_base(force_rebuild=args.rebuild)
            print(
                f"知识库就绪：{statistics['total_documents']} 篇食谱，"
                f"{statistics['total_chunks']} 个文本块。"
            )
            if args.retrieve is not None:
                results = system.retrieve(args.retrieve, top_k=args.top_k, role=role)
                for rank, document in enumerate(results, start=1):
                    print(
                        f"{rank}. {document.metadata.get('dish_name', '未知菜品')} | "
                        f"category={document.metadata.get('category', '未知')} | "
                        f"visibility={document.metadata.get('visibility', '?')} | "
                        f"rrf={document.metadata.get('rrf_score', 0.0):.6f}"
                    )
                    print(document.page_content[:180].replace("\n", " "))
            if args.retrieve_images is not None:
                results = system.retrieve_images(args.retrieve_images, top_k=args.top_k, role=role)
                if not results:
                    print("未检索到相关图片。")
                for rank, document in enumerate(results, start=1):
                    print(
                        f"{rank}. {document.metadata.get('dish_name', '未知菜品')} | "
                        f"image={document.metadata.get('image_path', '?')} | "
                        f"rrf={document.metadata.get('rrf_score', 0.0):.6f}"
                    )
                    print(document.page_content[:180].replace("\n", " "))
            return 0

        system.run_interactive(
            force_rebuild=args.rebuild, role=role, agent_mode=args.chat_mode == "agent"
        )
        return 0
    except Exception as exc:
        logger.exception("系统运行出错")
        print(f"系统错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
