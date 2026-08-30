"""FAISS index construction with reproducible manifests and safe persistence."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """Build, validate, save, and load the dense retrieval index.

    The document store is persisted as JSON instead of pickle.  A manifest binds
    the files to the exact corpus, chunking version, and embedding configuration,
    preventing a stale dense index from being paired with freshly built BM25 data.
    """

    MANIFEST_SCHEMA_VERSION = 1
    STORAGE_FORMAT = "faiss-json-v1"
    INDEX_FILENAME = "index.faiss"
    DOCUMENTS_FILENAME = "documents.json"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        index_save_path: str | Path = "./vector_index",
        *,
        device: str = "cpu",
        model_revision: str | None = None,
        normalize_embeddings: bool = True,
        embeddings: Embeddings | None = None,
    ):
        self.model_name = model_name
        self.index_save_path = Path(index_save_path).expanduser()
        self.device = device
        self.model_revision = model_revision
        self.normalize_embeddings = normalize_embeddings
        self.embeddings = embeddings
        self.vectorstore: FAISS | None = None
        self._indexed_chunks: list[Document] = []
        self._current_manifest: dict[str, Any] | None = None
        self._embedding_dimension: int | None = None

    def setup_embeddings(self) -> Embeddings:
        """Lazily initialize the CPU embedding model.

        huggingface_hub 在 import 时就把 ``HF_HUB_OFFLINE`` 固化进模块常量，
        之后改环境变量无效。因此必须在导入 langchain_huggingface 前先翻转
        常量，模型已缓存时完全不发网络请求（代理断开/被墙不会阻塞启动）；
        离线加载失败（本地无缓存）再恢复在线模式下载。
        """
        if self.embeddings is not None:
            return self.embeddings

        import huggingface_hub.constants as hf_constants

        forced_offline = False
        import os

        if os.getenv("RAG_HF_OFFLINE", "").casefold() in {"1", "true", "yes", "on"}:
            forced_offline = True
            logger.info("RAG_HF_OFFLINE 已启用，跳过 Hub 校验直接用本地缓存")

        previous_offline = hf_constants.HF_HUB_OFFLINE
        if not previous_offline:
            hf_constants.HF_HUB_OFFLINE = True
            os.environ["HF_HUB_OFFLINE"] = "1"

        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("正在初始化嵌入模型 %s（设备: %s）", self.model_name, self.device)
        model_kwargs: dict[str, Any] = {"device": self.device}
        if self.model_revision:
            model_kwargs["revision"] = self.model_revision
        encode_kwargs = {"normalize_embeddings": self.normalize_embeddings}

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        except Exception as offline_error:  # 本地没有缓存 → 恢复在线模式下载
            if not forced_offline:
                hf_constants.HF_HUB_OFFLINE = previous_offline
                os.environ.pop("HF_HUB_OFFLINE", None)
            logger.warning(
                "离线加载嵌入模型失败（%s），改为在线模式。首次运行需要下载约 100MB 模型。",
                offline_error,
            )
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        logger.info("嵌入模型初始化完成")
        return self.embeddings

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _corpus_fingerprint(chunks: list[Document]) -> str:
        records = []
        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
            chunk_id = (
                chunk.metadata.get("chunk_id")
                or hashlib.sha256(
                    (
                        str(chunk.metadata.get("source_path", ""))
                        + "\0"
                        + str(chunk.metadata.get("chunk_index", ""))
                        + "\0"
                        + content_hash
                    ).encode("utf-8")
                ).hexdigest()
            )
            records.append(
                {
                    "chunk_id": str(chunk_id),
                    "content_hash": content_hash,
                    "parent_id": str(chunk.metadata.get("parent_id", "")),
                    "revision_id": str(chunk.metadata.get("revision_id", "")),
                }
            )
        records.sort(key=lambda item: item["chunk_id"])
        payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _storage_ids(chunks: list[Document]) -> list[str]:
        ids = []
        for chunk in chunks:
            chunk_id = chunk.metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError("每个 chunk 都必须包含确定性的 chunk_id")
            ids.append(str(chunk_id))
        if len(ids) != len(set(ids)):
            raise ValueError("chunk_id 必须唯一")
        return ids

    def _get_embedding_dimension(self) -> int:
        if self._embedding_dimension is None:
            probe = self.setup_embeddings().embed_query("维度检查")
            if not probe:
                raise ValueError("嵌入模型返回了空向量")
            self._embedding_dimension = len(probe)
        return self._embedding_dimension

    def build_manifest(self, chunks: list[Document]) -> dict[str, Any]:
        versions = sorted(
            {str(chunk.metadata.get("chunking_version", "unknown")) for chunk in chunks}
        )
        return {
            "schema_version": self.MANIFEST_SCHEMA_VERSION,
            "storage_format": self.STORAGE_FORMAT,
            "embedding": {
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "normalize_embeddings": self.normalize_embeddings,
                "dimension": self._get_embedding_dimension(),
            },
            "chunking": {
                "versions": versions,
                "headers": ["#", "##", "###"],
                "strip_headers": False,
            },
            "document_count": len({str(chunk.metadata.get("parent_id", "")) for chunk in chunks}),
            "chunk_count": len(chunks),
            "corpus_fingerprint": self._corpus_fingerprint(chunks),
            "faiss": {"index_type": "IndexFlatL2", "metric": "L2"},
        }

    @staticmethod
    def _manifest_signature(manifest: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "schema_version",
            "storage_format",
            "embedding",
            "chunking",
            "document_count",
            "chunk_count",
            "corpus_fingerprint",
            "faiss",
        )
        return {key: manifest.get(key) for key in keys}

    def build_vector_index(self, chunks: list[Document]) -> FAISS:
        if not chunks:
            raise ValueError("文档列表不能为空")
        logger.info("正在构建 FAISS 向量索引（%d 个 chunk）", len(chunks))
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.setup_embeddings(),
            ids=self._storage_ids(chunks),
        )
        self._indexed_chunks = list(chunks)
        self._current_manifest = self.build_manifest(chunks)
        logger.info("向量索引构建完成")
        return self.vectorstore

    def add_documents(self, new_chunks: list[Document]) -> None:
        if self.vectorstore is None:
            raise ValueError("请先构建或加载向量索引")
        if not new_chunks:
            return
        self.vectorstore.add_documents(new_chunks, ids=self._storage_ids(new_chunks))
        self._indexed_chunks.extend(new_chunks)
        self._current_manifest = self.build_manifest(self._indexed_chunks)

    def _serialize_documents(self) -> dict[str, Any]:
        if self.vectorstore is None:
            raise ValueError("请先构建向量索引")

        documents = []
        for position, docstore_id in sorted(self.vectorstore.index_to_docstore_id.items()):
            document = self.vectorstore.docstore.search(docstore_id)
            if not isinstance(document, Document):
                raise ValueError(f"索引中的文档 {docstore_id!r} 无法读取")
            chunk_id = document.metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"索引中的文档 {docstore_id!r} 缺少 chunk_id")
            documents.append(
                {
                    "position": int(position),
                    "docstore_id": str(docstore_id),
                    "chunk_id": str(chunk_id),
                }
            )
        return {
            "schema_version": self.MANIFEST_SCHEMA_VERSION,
            "documents": documents,
        }

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    def save_index(self) -> None:
        if self.vectorstore is None:
            raise ValueError("请先构建向量索引")
        current_manifest = self._current_manifest
        if current_manifest is None:
            raise ValueError("缺少索引 manifest，无法安全保存")
        if int(self.vectorstore.index.ntotal) != int(current_manifest["chunk_count"]):
            raise ValueError("向量数量与 manifest 不一致，拒绝保存")

        directory = self.index_save_path.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        faiss = dependable_faiss_import()

        index_path = directory / self.INDEX_FILENAME
        index_temporary = directory / (self.INDEX_FILENAME + ".tmp")
        documents_path = directory / self.DOCUMENTS_FILENAME
        manifest_path = directory / self.MANIFEST_FILENAME

        faiss.write_index(self.vectorstore.index, str(index_temporary))
        index_temporary.replace(index_path)
        serialized = json.dumps(
            self._serialize_documents(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._atomic_write_text(documents_path, serialized)

        persisted_manifest = {
            **current_manifest,
            "created_at": datetime.now(UTC).isoformat(),
            "artifacts": {
                self.INDEX_FILENAME: self._sha256_file(index_path),
                self.DOCUMENTS_FILENAME: self._sha256_file(documents_path),
            },
        }
        self._atomic_write_text(
            manifest_path,
            json.dumps(
                persisted_manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
        )
        self._current_manifest = persisted_manifest
        logger.info("向量索引已安全保存到: %s", directory)

    def _read_verified_manifest(self, expected: dict[str, Any]) -> dict[str, Any] | None:
        directory = self.index_save_path.resolve()
        manifest_path = directory / self.MANIFEST_FILENAME
        index_path = directory / self.INDEX_FILENAME
        documents_path = directory / self.DOCUMENTS_FILENAME
        if not all(path.is_file() for path in (manifest_path, index_path, documents_path)):
            logger.info("索引文件不完整，将重新构建: %s", directory)
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logger.warning("索引 manifest 无法读取，将重新构建: %s", exc)
            return None

        if not isinstance(manifest, dict):
            logger.warning("索引 manifest 格式无效，将重新构建")
            return None

        if self._manifest_signature(manifest) != self._manifest_signature(expected):
            logger.info("索引 manifest 与当前语料或配置不匹配，将重新构建")
            return None

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            logger.warning("索引 manifest 缺少文件摘要，将重新构建")
            return None
        try:
            for path in (index_path, documents_path):
                expected_hash = artifacts.get(path.name)
                if not expected_hash or self._sha256_file(path) != expected_hash:
                    logger.warning("索引文件摘要校验失败: %s", path.name)
                    return None
        except OSError as exc:
            logger.warning("索引文件无法读取，将重新构建: %s", exc)
            return None
        return manifest

    def load_index(self, chunks: list[Document]) -> FAISS | None:
        """Load only an index that exactly matches ``chunks`` and this config."""
        if not chunks:
            raise ValueError("文档列表不能为空")
        self._storage_ids(chunks)
        expected = self.build_manifest(chunks)
        manifest = self._read_verified_manifest(expected)
        if manifest is None:
            return None

        directory = self.index_save_path.resolve()
        try:
            faiss = dependable_faiss_import()
            index = faiss.read_index(str(directory / self.INDEX_FILENAME))
            payload = json.loads((directory / self.DOCUMENTS_FILENAME).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("documents.json 格式无效")
            if payload.get("schema_version") != self.MANIFEST_SCHEMA_VERSION:
                raise ValueError("documents.json schema 不受支持")

            docstore: dict[str, Document] = {}
            position_map: dict[int, str] = {}
            current_chunks = {str(chunk.metadata["chunk_id"]): chunk for chunk in chunks}
            hydrated_chunk_ids: set[str] = set()
            for item in payload.get("documents", []):
                chunk_id = str(item["chunk_id"])
                if chunk_id not in current_chunks:
                    raise ValueError(f"当前语料缺少索引 chunk: {chunk_id}")
                docstore_id = str(item["docstore_id"])
                docstore[docstore_id] = current_chunks[chunk_id]
                position_map[int(item["position"])] = docstore_id
                hydrated_chunk_ids.add(chunk_id)

            expected_count = int(manifest["chunk_count"])
            if (
                len(docstore) != expected_count
                or len(position_map) != expected_count
                or int(index.ntotal) != expected_count
                or int(index.d) != int(manifest["embedding"]["dimension"])
                or set(position_map) != set(range(expected_count))
                or hydrated_chunk_ids != set(current_chunks)
                or type(index).__name__ != manifest["faiss"]["index_type"]
            ):
                raise ValueError("索引向量数与 manifest 不一致")

            self.vectorstore = FAISS(
                self.setup_embeddings(),
                index,
                InMemoryDocstore(docstore),
                position_map,
            )
            self._indexed_chunks = list(chunks)
            self._current_manifest = manifest
            logger.info("向量索引已从 %s 加载并通过完整性校验", directory)
            return self.vectorstore
        except (
            OSError,
            RuntimeError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            logger.warning("加载向量索引失败，将重新构建: %s", exc)
            self.vectorstore = None
            return None

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        if self.vectorstore is None:
            raise ValueError("请先构建或加载向量索引")
        return self.vectorstore.similarity_search(query, k=k)
