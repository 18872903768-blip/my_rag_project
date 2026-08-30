"""Hybrid dense/sparse retrieval and reciprocal-rank fusion."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag_modules.vector_store import VectorStoreBackend, build_filter_expr, combine_expr

logger = logging.getLogger(__name__)


def tokenize_chinese(text: str) -> list[str]:
    """Tokenize CJK text without a model or external dictionary.

    LangChain's BM25 default uses whitespace splitting, which treats an entire
    Chinese recipe as one token.  Character, bigram, and short-phrase tokens give
    the V0 sparse route useful exact-name and partial-name recall on any CPU.
    """
    normalized = text.casefold()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    for sequence in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", normalized):
        tokens.extend(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
        if len(sequence) <= 8:
            tokens.append(sequence)
    return tokens


class RetrievalOptimizationModule:
    """Combine dense and Chinese-aware BM25 retrieval using RRF.

    With a Milvus backend both routes and the RRF fusion run inside the server
    (``supports_server_side_hybrid``); metadata filters become Milvus ``expr``
    pushdowns.  With FAISS the fusion stays client side via ``BM25Retriever``
    and filters are applied after recall.
    """

    def __init__(
        self,
        vectorstore: VectorStoreBackend,
        chunks: list[Document],
        *,
        candidate_k: int = 10,
        rrf_k: int = 60,
        reranker: Any | None = None,
    ):
        if not chunks:
            raise ValueError("文档块列表不能为空")
        if candidate_k < 1 or rrf_k < 1:
            raise ValueError("candidate_k 和 rrf_k 必须为正整数")
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.server_side_hybrid = getattr(vectorstore, "supports_server_side_hybrid", False)
        self.bm25_retriever: BM25Retriever | None = None
        self.setup_retrievers()

    def setup_retrievers(self) -> None:
        if self.server_side_hybrid:
            logger.info("检测到服务端混合检索后端，跳过本地 BM25 索引")
            return
        logger.info("正在设置中文双路检索器...")
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=self.candidate_k,
            preprocess_func=tokenize_chinese,
        )
        logger.info("检索器设置完成")

    def _pool_size(self, top_k: int) -> int:
        """召回池大小：启用重排时扩大候选池给 cross-encoder 精选。"""
        if self.reranker is not None and getattr(self.reranker, "enabled", True):
            return max(top_k, getattr(self.reranker, "pool_size", 20))
        return top_k

    def _apply_rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        if self.reranker is None or not documents:
            return documents[:top_k]
        reranked = self.reranker.rerank(query, documents)
        # 单独让 cross-encoder 定序会在精确名查询上劣化（评测实测 MRR
        # 0.931→0.879）：把 CE 序与召回融合序再做一次 RRF，兼顾语义相关
        # 与 BM25 的精确菜名信号。
        recall_rank = {id(doc): rank for rank, doc in enumerate(documents, start=1)}
        def fused_score(document: Document, ce_rank: int) -> float:
            return 1.0 / (self.rrf_k + ce_rank) + 1.0 / (
                self.rrf_k + recall_rank[id(document)]
            )
        fused = sorted(
            enumerate(reranked, start=1),
            key=lambda pair: -fused_score(pair[1], pair[0]),
        )
        return [document for _, document in fused][:top_k]

    def hybrid_search(
        self, query: str, top_k: int = 3, *, candidate_k: int | None = None, expr: str | None = None
    ) -> list[Document]:
        if top_k < 1:
            return []
        pool_size = self._pool_size(top_k)
        candidate_count = max(pool_size, candidate_k or self.candidate_k)
        if self.server_side_hybrid:
            pool = self.vectorstore.hybrid_search(
                query, k=pool_size, expr=expr, candidate_k=candidate_count
            )
            return self._apply_rerank(query, pool, top_k)

        assert self.bm25_retriever is not None
        vector_docs = self.vectorstore.similarity_search(query, k=candidate_count)

        previous_k = self.bm25_retriever.k
        self.bm25_retriever.k = candidate_count
        try:
            bm25_docs = self.bm25_retriever.invoke(query)
        finally:
            self.bm25_retriever.k = previous_k

        fused = self._rrf_rerank(vector_docs, bm25_docs, self.rrf_k)
        if expr:
            # Client-side backends cannot push the expression down, so filter
            # the fused ranking here before truncating to top_k.
            fused = [document for document in fused if document_matches_expr(document, expr)]
        return self._apply_rerank(query, fused, top_k)

    @staticmethod
    def _document_id(document: Document) -> str:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        payload = "\0".join(
            (
                str(document.metadata.get("source_path", "")),
                str(document.metadata.get("chunk_index", "")),
                document.page_content,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _rrf_rerank(
        self,
        vector_docs: list[Document],
        bm25_docs: list[Document],
        k: int = 60,
    ) -> list[Document]:
        scores: dict[str, float] = {}
        documents: dict[str, Document] = {}

        for result_set in (vector_docs, bm25_docs):
            for rank, document in enumerate(result_set, start=1):
                document_id = self._document_id(document)
                documents.setdefault(document_id, document)
                scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)

        ranked = sorted(scores, key=lambda document_id: scores[document_id], reverse=True)
        result = []
        for document_id in ranked:
            original = documents[document_id]
            kwargs: dict[str, Any] = {
                "page_content": original.page_content,
                "metadata": {**original.metadata, "rrf_score": scores[document_id]},
            }
            if original.id is not None:
                kwargs["id"] = original.id
            result.append(Document(**kwargs))

        logger.info(
            "RRF 重排完成: dense=%d, sparse=%d, merged=%d",
            len(vector_docs),
            len(bm25_docs),
            len(result),
        )
        return result

    def metadata_filtered_search(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int = 5,
        *,
        extra_expr: str | None = None,
    ) -> list[Document]:
        """Search restricted to chunks matching ``filters``.

        Milvus backends translate the filters into one ``expr`` (plus any
        caller-provided expression, e.g. a visibility scope) so filtering and
        ranking happen server side.  FAISS keeps the recall-then-post-filter
        behaviour from V0.
        """
        if not filters and not extra_expr:
            return self.hybrid_search(query, top_k=top_k)

        if self.server_side_hybrid:
            expr = combine_expr(build_filter_expr(filters), extra_expr)
            if expr is None:
                return self.hybrid_search(query, top_k=top_k)
            pool_size = self._pool_size(top_k)
            pool = self.vectorstore.hybrid_search(
                query, k=pool_size, expr=expr, candidate_k=max(pool_size, self.candidate_k)
            )
            return self._apply_rerank(query, pool, top_k)

        candidate_count = min(len(self.chunks), max(self.candidate_k, top_k * 5))
        candidates = self.hybrid_search(query, top_k=candidate_count, candidate_k=candidate_count)
        filtered: list[Document] = []
        for document in candidates:
            matches = True
            for key, expected in filters.items():
                actual = document.metadata.get(key)
                if isinstance(expected, (list, tuple, set)):
                    matches = actual in expected
                else:
                    matches = actual == expected
                if not matches:
                    break
            if not matches:
                continue
            if extra_expr and not document_matches_expr(document, extra_expr):
                continue
            filtered.append(document)
            if len(filtered) >= top_k:
                break
        return filtered


_EXPR_CLAUSE = re.compile(r'^(\w+)\s*(==|in)\s*(.+)$')
_EXPR_LIST = re.compile(r'^\[(.*)\]$')
_EXPR_LITERAL = re.compile(r'^"((?:[^"\\]|\\.)*)"$')
_EXPR_LITERAL_SCAN = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _unescape(literal: str) -> str:
    return literal.replace('\\"', '"').replace("\\\\", "\\")


def document_matches_expr(document: Document, expr: str) -> bool:
    """Evaluate the restricted ``expr`` subset produced by ``build_filter_expr``.

    Only used by client-side (FAISS) backends where the expression cannot be
    pushed down; Milvus evaluates the same expression natively.
    """
    for clause in expr.split("&&"):
        clause = clause.strip()
        if not clause:
            continue
        match = _EXPR_CLAUSE.match(clause)
        if not match:
            logger.warning("无法解析过滤表达式子句，忽略: %r", clause)
            continue
        key, operator, raw_value = match.groups()
        actual = document.metadata.get(key)
        if operator == "==":
            literal = _EXPR_LITERAL.match(raw_value.strip())
            if not literal or actual != _unescape(literal.group(1)):
                return False
        else:
            list_match = _EXPR_LIST.match(raw_value.strip())
            if not list_match:
                return False
            values = [
                _unescape(item.group(1))
                for item in _EXPR_LITERAL_SCAN.finditer(list_match.group(1))
            ]
            if actual not in values:
                return False
    return True
