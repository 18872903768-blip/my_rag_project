"""Vector-store backend abstraction with a Milvus implementation.

The retrieval layer talks to a :class:`VectorStoreBackend` instead of a concrete
FAISS object.  The Milvus backend keeps every scalar field alongside the vector
so metadata filtering, permission checks, and image chunks are pushed down to
the server; hybrid dense/BM25 ranking also happens inside Milvus via RRF.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_modules.index_construction import IndexConstructionModule

logger = logging.getLogger(__name__)

# Fields stored in Milvus besides the primary key and the vectors.  ``content``
# carries the analyzer for server-side BM25.  Keys map 1:1 to chunk metadata.
SCALAR_FIELDS: dict[str, int] = {
    "content": 16384,
    "parent_id": 64,
    "revision_id": 64,
    "doc_type": 16,
    "category": 32,
    "dish_name": 128,
    "difficulty": 16,
    "modality": 16,
    "visibility": 16,
    "source_path": 512,
    "source": 1024,
    "source_hash": 64,
    "chunking_version": 64,
    "image_path": 512,
    "file_type": 16,
}
INT_FIELDS = ("chunk_index", "chunk_size")
# Filterable keys accepted by :func:`build_filter_expr`; anything else is
# rejected so untrusted input can never inject raw Milvus expressions.
FILTERABLE_FIELDS = frozenset(
    {"category", "difficulty", "dish_name", "modality", "visibility", "doc_type", "parent_id"}
)

MILVUS_MANIFEST_SCHEMA_VERSION = 1
MILVUS_STORAGE_FORMAT = "milvus-2.5-v1"
MILVUS_MANIFEST_FILENAME = "milvus_manifest.json"
INSERT_BATCH_SIZE = 64


def _escape_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_filter_expr(filters: dict[str, Any] | None) -> str | None:
    """Translate a metadata filter dict into a safe Milvus boolean expression.

    Only whitelisted keys are accepted; string values are escaped and list
    values become ``in`` predicates.  Returns ``None`` when nothing to filter.
    """
    if not filters:
        return None
    clauses: list[str] = []
    for key, expected in filters.items():
        if key not in FILTERABLE_FIELDS:
            raise ValueError(f"不支持作为过滤条件的字段: {key!r}")
        if expected is None or expected == "":
            continue
        if isinstance(expected, (list, tuple, set)):
            values = [item for item in expected if item is not None and item != ""]
            if not values:
                continue
            literals = ", ".join(_escape_literal(str(item)) for item in values)
            clauses.append(f"{key} in [{literals}]")
        else:
            clauses.append(f"{key} == {_escape_literal(str(expected))}")
    return " && ".join(clauses) if clauses else None


def combine_expr(*exprs: str | None) -> str | None:
    parts = [expr for expr in exprs if expr]
    return " && ".join(parts) if parts else None


class VectorStoreBackend(ABC):
    """Minimal contract shared by the Milvus and FAISS backends."""

    supports_server_side_hybrid: bool = False

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 5, *, expr: str | None = None
    ) -> list[Document]:
        """Dense-vector search; ``expr`` is a Milvus boolean expression."""

    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        *,
        expr: str | None = None,
        candidate_k: int | None = None,
    ) -> list[Document]:
        """Dense + sparse search fused with RRF."""

    def count(self) -> int:
        raise NotImplementedError

    def build_index(self, chunks: list[Document], *, force_rebuild: bool = False) -> bool:
        """Create the index for ``chunks``; returns True when a rebuild happened."""
        raise NotImplementedError

    def add_documents(self, chunks: list[Document]) -> None:
        raise NotImplementedError


class FaissBackend(VectorStoreBackend):
    """Thin adapter so the existing FAISS index satisfies the backend contract.

    ``expr`` cannot be pushed down; the retrieval module falls back to client
    side BM25 fusion and post-filtering for this backend.
    """

    supports_server_side_hybrid = False

    def __init__(self, vectorstore: Any):
        self.vectorstore = vectorstore

    def similarity_search(
        self, query: str, k: int = 5, *, expr: str | None = None
    ) -> list[Document]:
        if expr:
            logger.warning("FAISS 后端不支持表达式过滤，已忽略: %s", expr)
        return self.vectorstore.similarity_search(query, k=k)

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        *,
        expr: str | None = None,
        candidate_k: int | None = None,
    ) -> list[Document]:
        if expr:
            logger.warning("FAISS 后端不支持表达式过滤，已忽略: %s", expr)
        return self.vectorstore.similarity_search(query, k=candidate_k or k)


class MilvusVectorStore(VectorStoreBackend):
    """Milvus collection with scalar metadata, server-side BM25, and RRF fusion."""

    supports_server_side_hybrid = True
    BM25_ANALYZER_CANDIDATES: tuple[dict[str, Any], ...] = (
        {"tokenizer": "jieba"},
        {"type": "chinese"},
    )

    def __init__(
        self,
        collection_name: str,
        uri: str = "http://localhost:19530",
        token: str = "",
        *,
        embeddings: Embeddings,
        manifest_dir: str | Path,
        metric_type: str = "COSINE",
    ):
        self.collection_name = collection_name
        self.uri = uri
        self.token = token
        self.embeddings = embeddings
        self.manifest_dir = Path(manifest_dir).expanduser()
        self.metric_type = metric_type
        self._client: Any | None = None
        self._embedding_dimension: int | None = None
        self._analyzer_params: dict[str, Any] | None = None
        self._analyzer_decided = False

    # ------------------------------------------------------------------ client

    def connect(self) -> Any:
        if self._client is None:
            from pymilvus import MilvusClient

            logger.info("正在连接 Milvus: %s (collection=%s)", self.uri, self.collection_name)
            self._client = (
                MilvusClient(uri=self.uri, token=self.token)
                if self.token
                else MilvusClient(uri=self.uri)
            )
            version = None
            try:
                version = self._client.get_server_version()
            except Exception:  # version probe is best-effort only
                pass
            if version:
                logger.info("Milvus 服务端版本: %s", version)
        return self._client

    def _get_embedding_dimension(self) -> int:
        if self._embedding_dimension is None:
            probe = self.embeddings.embed_query("维度检查")
            if not probe:
                raise ValueError("嵌入模型返回了空向量")
            self._embedding_dimension = len(probe)
        return self._embedding_dimension

    # ------------------------------------------------------------------ schema

    def _build_schema(self, analyzer_params: dict[str, Any] | None) -> Any:
        from pymilvus import DataType, Function, FunctionType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._get_embedding_dimension())
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        content_kwargs: dict[str, Any] = {"max_length": SCALAR_FIELDS["content"]}
        if analyzer_params is not None:
            content_kwargs.update(enable_analyzer=True, analyzer_params=analyzer_params)
        schema.add_field("content", DataType.VARCHAR, **content_kwargs)
        for field_name, max_length in SCALAR_FIELDS.items():
            if field_name == "content":
                continue
            schema.add_field(
                field_name,
                DataType.VARCHAR,
                max_length=max_length,
                is_partition_key=(field_name == "category"),
            )
        for field_name in INT_FIELDS:
            schema.add_field(field_name, DataType.INT64)
        schema.add_function(
            Function(
                name="content_bm25",
                input_field_names=["content"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )
        return schema

    def _resolve_analyzer(self) -> dict[str, Any] | None:
        return self._analyzer_params

    # ------------------------------------------------------------------ build

    def _create_collection(self, client: Any) -> None:
        """Create the collection, preferring the first Chinese analyzer the server supports."""
        last_error: Exception | None = None
        for candidate in (*self.BM25_ANALYZER_CANDIDATES, None):
            schema = self._build_schema(candidate)
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type=self.metric_type,
                params={"M": 16, "efConstruction": 200},
            )
            index_params.add_index(
                field_name="sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
            for field_name in ("difficulty", "visibility", "modality"):
                index_params.add_index(field_name=field_name, index_type="INVERTED")
            try:
                client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=index_params,
                    num_partitions=16,
                )
            except Exception as exc:  # noqa: BLE001 - fall through to next analyzer candidate
                last_error = exc
                logger.warning("分词器 %s 建库失败: %s", candidate, exc)
                if client.has_collection(self.collection_name):
                    client.drop_collection(self.collection_name)
                continue
            self._analyzer_params = candidate
            self._analyzer_decided = True
            logger.info("BM25 分词器选用: %s", candidate)
            return
        raise RuntimeError(f"无法创建 Milvus collection（所有分词器候选均失败）: {last_error}")

    @staticmethod
    def _validate_chunks(chunks: list[Document]) -> None:
        ids = IndexConstructionModule._storage_ids(chunks)
        logger.info("待写入 chunk 数: %d（ID 唯一性已校验）", len(ids))
        oversized = [
            str(chunk.metadata.get("chunk_id"))
            for chunk in chunks
            if len(chunk.page_content) > SCALAR_FIELDS["content"]
        ]
        if oversized:
            raise ValueError(
                f"有 {len(oversized)} 个 chunk 超过 content 字段上限 "
                f"{SCALAR_FIELDS['content']} 字符，例如 {oversized[:3]}"
            )

    def _rows_for_insert(
        self, chunks: list[Document], vectors: list[list[float]]
    ) -> list[dict[str, Any]]:
        rows = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            metadata = chunk.metadata
            row: dict[str, Any] = {"chunk_id": str(metadata["chunk_id"]), "vector": vector}
            for field_name in SCALAR_FIELDS:
                if field_name == "content":
                    row["content"] = chunk.page_content
                else:
                    row[field_name] = str(metadata.get(field_name, ""))
            for field_name in INT_FIELDS:
                row[field_name] = int(metadata.get(field_name, 0) or 0)
            rows.append(row)
        return rows

    def build_index(self, chunks: list[Document], *, force_rebuild: bool = False) -> bool:
        """Create the collection and insert ``chunks``; returns True when rebuilt."""
        if not chunks:
            raise ValueError("文档块列表不能为空")
        self._validate_chunks(chunks)
        client = self.connect()

        expected_fingerprint = IndexConstructionModule._corpus_fingerprint(chunks)
        if not force_rebuild and self._collection_is_current(client, chunks, expected_fingerprint):
            logger.info("Milvus collection 与当前语料一致，复用: %s", self.collection_name)
            return False

        if client.has_collection(self.collection_name):
            logger.info("删除旧 collection: %s", self.collection_name)
            client.drop_collection(self.collection_name)

        self._create_collection(client)
        logger.info("collection %s 已创建，正在写入 %d 个 chunk", self.collection_name, len(chunks))

        total = 0
        for start in range(0, len(chunks), INSERT_BATCH_SIZE):
            batch = chunks[start : start + INSERT_BATCH_SIZE]
            vectors = self.embeddings.embed_documents([chunk.page_content for chunk in batch])
            result = client.insert(
                collection_name=self.collection_name,
                data=self._rows_for_insert(batch, vectors),
            )
            total += int(getattr(result, "insert_count", len(batch)))
        client.flush(self.collection_name)
        if total != len(chunks):
            raise ValueError(f"Milvus 写入数量不一致: 期望 {len(chunks)}, 实际 {total}")

        self._write_manifest(chunks, expected_fingerprint)
        logger.info("Milvus 索引构建完成: %d 个 chunk 已写入", total)
        return True

    def add_documents(self, chunks: list[Document]) -> None:
        if not chunks:
            return
        self._validate_chunks(chunks)
        client = self.connect()
        vectors = self.embeddings.embed_documents([chunk.page_content for chunk in chunks])
        client.insert(
            collection_name=self.collection_name,
            data=self._rows_for_insert(chunks, vectors),
        )
        client.flush(self.collection_name)

    def count(self) -> int:
        client = self.connect()
        stats = client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))

    # ---------------------------------------------------------------- manifest

    def _manifest_path(self) -> Path:
        return self.manifest_dir / MILVUS_MANIFEST_FILENAME

    def _write_manifest(self, chunks: list[Document], fingerprint: str) -> None:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": MILVUS_MANIFEST_SCHEMA_VERSION,
            "storage_format": MILVUS_STORAGE_FORMAT,
            "collection": self.collection_name,
            "chunk_count": len(chunks),
            "corpus_fingerprint": fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
        }
        path = self._manifest_path()
        path.with_name(path.name + ".tmp").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (path.with_name(path.name + ".tmp")).replace(path)

    def _collection_is_current(self, client: Any, chunks: list[Document], fingerprint: str) -> bool:
        if not client.has_collection(self.collection_name):
            return False
        path = self._manifest_path()
        if not path.is_file():
            return False
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logger.warning("Milvus manifest 读取失败，将重建: %s", exc)
            return False
        if (
            manifest.get("corpus_fingerprint") != fingerprint
            or int(manifest.get("chunk_count", -1)) != len(chunks)
            or manifest.get("collection") != self.collection_name
        ):
            return False
        row_count = self.count()
        if row_count != len(chunks):
            logger.warning(
                "Milvus 行数与 manifest 不一致 (%d != %d)，将重建", row_count, len(chunks)
            )
            return False
        return True

    # ----------------------------------------------------------------- search

    @staticmethod
    def _hit_field(hit: Any, name: str) -> Any:
        try:
            entity = hit.entity
            return entity.get(name)
        except Exception:  # pymilvus version differences between Hit/dict
            if isinstance(hit, dict):
                return hit.get(name)
            return getattr(hit, name, None)

    def _to_document(self, hit: Any) -> Document:
        metadata = {
            field_name: self._hit_field(hit, field_name)
            for field_name in SCALAR_FIELDS
            if field_name != "content"
        }
        for field_name in INT_FIELDS:
            value = self._hit_field(hit, field_name)
            if value is not None:
                metadata[field_name] = int(value)
        metadata["chunk_id"] = str(hit.id)
        metadata["distance"] = float(hit.distance)
        return Document(page_content=str(self._hit_field(hit, "content") or ""), metadata=metadata)

    def _output_fields(self) -> list[str]:
        return list(SCALAR_FIELDS) + list(INT_FIELDS)

    def similarity_search(
        self, query: str, k: int = 5, *, expr: str | None = None
    ) -> list[Document]:
        client = self.connect()
        vector = self.embeddings.embed_query(query)
        results = client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="vector",
            search_params={"metric_type": self.metric_type},
            limit=k,
            filter=expr,
            output_fields=self._output_fields(),
        )
        hits = results[0] if results else []
        return [self._to_document(hit) for hit in hits]

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        *,
        expr: str | None = None,
        candidate_k: int | None = None,
    ) -> list[Document]:
        from pymilvus import AnnSearchRequest, RRFRanker

        client = self.connect()
        limit = max(k, candidate_k or k)
        vector = self.embeddings.embed_query(query)
        # pymilvus 2.5 的 hybrid_search 会忽略 filter 关键字参数，
        # 表达式必须挂在每个 AnnSearchRequest 上才会在服务端生效。
        dense_request = AnnSearchRequest(
            data=[vector],
            anns_field="vector",
            param={"metric_type": self.metric_type},
            limit=limit,
            expr=expr,
        )
        sparse_request = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=limit,
            expr=expr,
        )
        results = client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_request, sparse_request],
            ranker=RRFRanker(60),
            limit=k,
            output_fields=self._output_fields(),
        )
        hits = results[0] if results else []
        documents = [self._to_document(hit) for hit in hits]
        for document in documents:
            # RRFRanker output arrives in ``distance``; expose it under the
            # name downstream consumers already print.
            document.metadata["rrf_score"] = document.metadata.pop("distance")
        logger.info(
            "Milvus hybrid 检索完成: query=%r, k=%d, expr=%s, 命中=%d",
            query,
            k,
            expr or "-",
            len(documents),
        )
        return documents
