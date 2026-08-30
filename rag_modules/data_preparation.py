"""Markdown recipe loading, metadata enrichment, and deterministic chunking."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from rag_modules.domain_config import get_domain

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """Prepare the corpus for indexing.

    V0 intentionally supports Markdown only. Every parent and child identifier is
    derived from the portable relative path and content, so rebuilding the same
    corpus in another directory produces the same IDs. Category/difficulty
    vocabularies come from the active :data:`DomainConfig`.
    """
    #类常量
    #记录当前chunk的算法版本
    CHUNKING_VERSION = "markdown-headers-v1"
    #初始化函数
    def __init__(
        self,
        data_path: str | Path,
        #*后面的参数必须使用关键词参数传递
        *,
        include_templates: bool = False,
        strict: bool = True,
        internal_categories: set[str] | None = None,
    ):
        #获取路径并转为path对象，expanduser()是展开路径里面~
        self.data_path = Path(data_path).expanduser()
        #false代表跳过template目录，true代表也作为知识入库
        self.include_templates = include_templates
        #true代表一个文件失败，整个构建都失败
        self.strict = strict
        #调用方没有指定internal_categories参数，
        # 则用get_domain().internal_categories
        self.internal_categories = (
            set(get_domain().internal_categories)
            if internal_categories is None
            else set(internal_categories)
        )
        self.documents: list[Document] = []
        self.chunks: list[Document] = []
        self.parent_child_map: dict[str, str] = {}
        self._parent_documents: dict[str, Document] = {}
    #ID/路径辅助函数
    #@staticmethod说明这个方法不依赖self和cls
    #本质是放在类命名空间的普通工具函数
    @staticmethod
    #*parts 表示接受任意多个位置参数
    #将namespace和任意的参数*parts用\0拼接在一起，然后hash得到id
    def _stable_id(namespace: str, *parts: str) -> str:
        payload = "\0".join((namespace, *parts)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    #判断是否是template目录下的文件
    @staticmethod
    def _is_template(relative_path: Path) -> bool:
        return any(part.casefold() == "template" for part in relative_path.parts)
    #父文档加载阶段
    #将磁盘Markdown文件加载成Document
    def load_documents(self) -> list[Document]:
        """Load all Markdown recipes below ``data_path`` in stable path order."""
        #路径规范化
        data_root = self.data_path.resolve()
        if not data_root.exists():
            raise FileNotFoundError(f"数据路径不存在: {data_root}")
        if not data_root.is_dir():
            raise NotADirectoryError(f"数据路径不是目录: {data_root}")

        logger.info("正在从 %s 加载文档...", data_root)
        self.documents = []
        self.chunks = []
        self.parent_child_map = {}
        self._parent_documents = {}

        discovered = sorted(
            data_root.rglob("*.md"),
            key=lambda path: path.relative_to(data_root).as_posix().casefold(),
        )
        skipped_templates = 0
        failures: list[tuple[Path, Exception]] = []

        for md_file in discovered:
            relative = md_file.relative_to(data_root)
            if not self.include_templates and self._is_template(relative):
                skipped_templates += 1
                continue

            try:
                content = md_file.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                logger.warning("读取文档 %s 失败: %s", md_file, exc)
                failures.append((md_file, exc))
                continue

            source_path = relative.as_posix()
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            parent_id = self._stable_id("parent", source_path)
            revision_id = self._stable_id("revision", source_path, content_hash)
            document = Document(
                page_content=content,
                metadata={
                    "source": str(md_file.resolve()),
                    "source_path": source_path,
                    "source_hash": content_hash,
                    "parent_id": parent_id,
                    "revision_id": revision_id,
                    "doc_type": "parent",
                    "file_type": "markdown",
                    "modality": "text",
                    "visibility": "public",
                    "image_path": "",
                },
            )
            self._enhance_metadata(document)
            self.documents.append(document)
            self._parent_documents[parent_id] = document

        if failures and self.strict:
            examples = "; ".join(f"{path}: {error}" for path, error in failures[:3])
            raise RuntimeError(
                f"有 {len(failures)} 个 Markdown 文件读取失败，拒绝构建部分索引: {examples}"
            )
        if not self.documents:
            raise ValueError(f"数据目录中没有可索引的 Markdown 食谱: {data_root}")

        logger.info(
            "发现 %d 个 Markdown，跳过 %d 个模板，成功加载 %d 个食谱",
            len(discovered),
            skipped_templates,
            len(self.documents),
        )
        return self.documents

    def _enhance_metadata(self, document: Document) -> None:
        source_path = Path(document.metadata.get("source_path", ""))
        path_parts = {part.casefold() for part in source_path.parts}

        category = "其他"
        for key, value in get_domain().category_mapping.items():
            if key.casefold() in path_parts:
                category = value
                break

        document.metadata["category"] = category
        document.metadata["dish_name"] = source_path.stem
        # 权限演示：内部类别（如半成品工艺文档）默认不对公开访客可见。
        document.metadata["visibility"] = (
            "internal" if category in self.internal_categories else "public"
        )

        star_match = re.search(r"★+", document.page_content)
        if star_match:
            difficulty_map = {
                1: "非常简单",
                2: "简单",
                3: "中等",
                4: "困难",
                5: "非常困难",
            }
            document.metadata["difficulty"] = difficulty_map.get(len(star_match.group()), "未知")
        else:
            document.metadata["difficulty"] = "未知"
    #配置查询
    @classmethod
    def get_supported_categories(cls) -> list[str]:
        return get_domain().category_labels

    @classmethod
    #classmethod 和 staticmethod 区别
    #
    def get_supported_difficulties(cls) -> list[str]:
        return list(get_domain().difficulty_labels)
    #文档切块阶段
    def chunk_documents(self) -> list[Document]:
        """Split loaded documents by Markdown headings into a flat child list."""
        if not self.documents:
            raise ValueError("请先加载文档")

        logger.info("正在进行 Markdown 结构感知分块...")
        self.parent_child_map = {}
        chunks = self._markdown_header_split()

        for batch_index, chunk in enumerate(chunks):
            chunk.metadata["batch_index"] = batch_index
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = chunks
        logger.info("Markdown 分块完成，共生成 %d 个 chunk", len(chunks))
        return self.chunks

    def _markdown_header_split(self) -> list[Document]:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "主标题"),
                ("##", "二级标题"),
                ("###", "三级标题"),
            ],
            strip_headers=False,
        )
        all_chunks: list[Document] = []

        for parent in self.documents:
            try:
                split_chunks = splitter.split_text(parent.page_content)
            except Exception as exc:  # one malformed recipe must not stop the corpus
                logger.warning(
                    "文档 %s Markdown 分割失败，降级为整篇分块: %s",
                    parent.metadata.get("source", "未知"),
                    exc,
                )
                split_chunks = []

            if not split_chunks:
                split_chunks = [Document(page_content=parent.page_content, metadata={})]

            for chunk_index, split_chunk in enumerate(split_chunks):
                child_id = self._stable_id(
                    "chunk",
                    parent.metadata["parent_id"],
                    parent.metadata["revision_id"],
                    str(chunk_index),
                    split_chunk.page_content,
                )
                metadata = {
                    **parent.metadata,
                    **split_chunk.metadata,
                    "chunk_id": child_id,
                    "parent_id": parent.metadata["parent_id"],
                    "doc_type": "child",
                    "chunk_index": chunk_index,
                    "chunking_version": self.CHUNKING_VERSION,
                }
                child = Document(page_content=split_chunk.page_content, metadata=metadata)
                all_chunks.append(child)
                self.parent_child_map[child_id] = parent.metadata["parent_id"]

        return all_chunks
    #查询辅助
    def filter_documents_by_category(self, category: str) -> list[Document]:
        return [
            document for document in self.documents if document.metadata.get("category") == category
        ]

    def filter_documents_by_difficulty(self, difficulty: str) -> list[Document]:
        return [
            document
            for document in self.documents
            if document.metadata.get("difficulty") == difficulty
        ]
    #统计导出
    def get_statistics(self) -> dict[str, Any]:
        if not self.documents:
            return {}

        categories: dict[str, int] = {}
        difficulties: dict[str, int] = {}
        for document in self.documents:
            category = document.metadata.get("category", "未知")
            difficulty = document.metadata.get("difficulty", "未知")
            categories[category] = categories.get(category, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        average = (
            sum(len(chunk.page_content) for chunk in self.chunks) / len(self.chunks)
            if self.chunks
            else 0.0
        )
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "difficulties": difficulties,
            "avg_chunk_size": average,
        }

    def export_metadata(self, output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = [
            {
                "source": document.metadata.get("source"),
                "source_path": document.metadata.get("source_path"),
                "parent_id": document.metadata.get("parent_id"),
                "revision_id": document.metadata.get("revision_id"),
                "dish_name": document.metadata.get("dish_name"),
                "category": document.metadata.get("category"),
                "difficulty": document.metadata.get("difficulty"),
                "content_length": len(document.page_content),
            }
            for document in self.documents
        ]
        output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("元数据已导出到: %s", output)
    #父子回溯
    def get_parent_documents(self, child_chunks: list[Document]) -> list[Document]:
        """Return unique parents, ranked by how many matched chunks they own."""
        relevance: dict[str, int] = {}
        first_seen: dict[str, int] = {}

        for position, chunk in enumerate(child_chunks):
            parent_id = chunk.metadata.get("parent_id")
            if not parent_id or parent_id not in self._parent_documents:
                continue
            relevance[parent_id] = relevance.get(parent_id, 0) + 1
            first_seen.setdefault(parent_id, position)

        ranked_ids = sorted(
            relevance,
            key=lambda parent_id: (-relevance[parent_id], first_seen[parent_id]),
        )
        parents = [self._parent_documents[parent_id] for parent_id in ranked_ids]
        logger.info(
            "从 %d 个子块中找到 %d 个去重父文档",
            len(child_chunks),
            len(parents),
        )
        return parents
