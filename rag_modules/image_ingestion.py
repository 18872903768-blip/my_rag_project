"""Image ingestion: caption dish photos with a vision LLM and index the text.

Each image becomes a chunk whose ``page_content`` is the generated Chinese
caption; ``modality=image`` and ``image_path`` let retrieval return the picture
itself, while ``parent_id`` links the image back to its recipe so parent-level
ranking keeps working.  Captions are cached by (path, file hash, model, prompt
version) so rebuilding the index never re-bills the vision API.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"
MIN_REAL_IMAGE_BYTES = 2048
CAPTION_PROMPT_VERSION = "v2"
CAPTION_MAX_WORKERS = 4

CAPTION_PROMPT = (
    "这是菜品《{dish_name}》菜谱目录中的一张图片。"
    "请用中文描述图片内容：判断它是成品图还是制作步骤图，"
    "指出画面中可见的食材、炊具或烹饪动作。"
    "不超过80字，只描述实际可见的内容，不要推测，直接输出描述。"
)


class ImageIngestionModule:
    """Turn dish photos into captioned, parent-linked image chunks."""

    def __init__(
        self,
        data_path: str | Path,
        cache_path: str | Path,
        *,
        vision_model: str = "deepseek-v4-flash-vision-exp",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        max_workers: int = CAPTION_MAX_WORKERS,
        strict: bool = False,
        caption_fn: Callable[[Path, str, str], str] | None = None,
    ):
        self.data_path = Path(data_path).expanduser()
        self.cache_path = Path(cache_path).expanduser()
        self.vision_model = vision_model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_workers = max(1, max_workers)
        self.strict = strict
        self._caption_fn = caption_fn
        self._cache = self._load_cache()

    # ------------------------------------------------------------------ cache

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path.is_file():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(key): str(value) for key, value in payload.items()}
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            logger.warning("图片 caption 缓存读取失败，将重新生成: %s", exc)
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_name(self.cache_path.name + ".tmp")
        temporary.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        temporary.replace(self.cache_path)

    def _cache_key(self, image_path: Path, file_hash: str) -> str:
        payload = "\0".join(
            (
                str(image_path),
                file_hash,
                self.vision_model,
                CAPTION_PROMPT_VERSION,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------- caption

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _call_vision_model(self, image_path: Path, dish_name: str, prompt: str) -> str:
        if self._caption_fn is not None:
            return self._caption_fn(image_path, dish_name, prompt)
        if not self.api_key:
            raise ValueError("缺少视觉模型 API Key，无法生成图片 caption")

        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, max_retries=3, timeout=120
        )
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        response = client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
            # deepseek-v4-flash-vision-exp 是推理模型，reasoning 也要消耗
            # completion tokens，余量不足会导致 content 为空。
            max_tokens=2000,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError(f"视觉模型返回空 caption: {image_path.name}")
        return content.strip()

    def _caption_image(self, image_path: Path, dish_name: str, file_hash: str) -> str:
        cache_key = self._cache_key(image_path, file_hash)
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        prompt = CAPTION_PROMPT.format(dish_name=dish_name)
        caption = self._call_vision_model(image_path, dish_name, prompt)
        caption = f"《{dish_name}》图片：{caption}"
        self._cache[cache_key] = caption
        return caption

    # ---------------------------------------------------------------- ingest

    def _images_for_parent(self, parent: Document) -> list[Path]:
        md_path = Path(str(parent.metadata.get("source", "")))
        if not md_path.is_file():
            return []
        candidates: list[Path] = []
        for path in sorted(md_path.parent.iterdir()):
            if path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES or not path.is_file():
                continue
            # 语料中的 Git LFS 指针文件（约130字节的文本）不是真实图片，
            # 视觉模型会拒绝；跳过它们，真实图片就位后会被自动摄取。
            if path.stat().st_size < MIN_REAL_IMAGE_BYTES or path.read_bytes()[:40].startswith(
                LFS_POINTER_MAGIC
            ):
                logger.debug("跳过 LFS 指针/占位图片: %s", path.name)
                continue
            candidates.append(path)
        return candidates

    def _build_chunk(
        self,
        parent: Document,
        image_path: Path,
        caption: str,
        file_hash: str,
        chunk_index: int,
    ) -> Document:
        relative_image = image_path.relative_to(self.data_path).as_posix()
        chunk_id = hashlib.sha256(
            ("image\0" + relative_image + "\0" + file_hash).encode("utf-8")
        ).hexdigest()
        inherited_keys = (
            "parent_id",
            "category",
            "dish_name",
            "difficulty",
            "visibility",
            "source_path",
            "revision_id",
        )
        metadata = {
            **{key: parent.metadata[key] for key in inherited_keys if key in parent.metadata},
            "chunk_id": chunk_id,
            "doc_type": "child",
            "modality": "image",
            "file_type": "image",
            "image_path": relative_image,
            "chunk_index": chunk_index,
            "chunking_version": f"image-caption-{CAPTION_PROMPT_VERSION}",
            "chunk_size": len(caption),
            "source": str(image_path.resolve()),
            "source_hash": file_hash,
        }
        return Document(page_content=caption, metadata=metadata)

    def ingest_images(self, parent_documents: list[Document]) -> list[Document]:
        """Caption every supported image next to each recipe and return chunks."""
        jobs: list[tuple[Document, Path, str]] = []
        for parent in parent_documents:
            for image_path in self._images_for_parent(parent):
                jobs.append((parent, image_path, self._sha256_file(image_path)))
        if not jobs:
            logger.info("语料中没有发现可摄取的图片")
            return []
        logger.info("发现 %d 张图片，开始生成 caption（并发=%d）", len(jobs), self.max_workers)

        captions: dict[Path, str] = {}
        failures: list[tuple[Path, Exception]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._caption_image,
                    image_path,
                    str(parent.metadata.get("dish_name", "未知菜品")),
                    file_hash,
                ): (parent, image_path, file_hash)
                for parent, image_path, file_hash in jobs
            }
            for future in as_completed(futures):
                parent, image_path, file_hash = futures[future]
                try:
                    captions[image_path] = future.result()
                except Exception as exc:  # noqa: BLE001 - one bad image must not kill the run
                    failures.append((image_path, exc))
                    logger.warning("图片 caption 失败，跳过 %s: %s", image_path.name, exc)

        if failures and self.strict:
            examples = "; ".join(f"{path.name}: {error}" for path, error in failures[:3])
            raise RuntimeError(f"有 {len(failures)} 张图片 caption 失败（strict 模式）: {examples}")

        chunks: list[Document] = []
        per_parent_counter: dict[str, int] = {}
        for parent, image_path, file_hash in jobs:
            caption = captions.get(image_path)
            if caption is None:
                continue
            parent_id = str(parent.metadata.get("parent_id", ""))
            chunk_index = per_parent_counter.get(parent_id, 0)
            chunks.append(self._build_chunk(parent, image_path, caption, file_hash, chunk_index))
            per_parent_counter[parent_id] = chunk_index + 1

        self._save_cache()
        cache_hits = sum(
            1 for _, path, file_hash in jobs if self._cache_key(path, file_hash) in self._cache
        )
        logger.info(
            "图片摄取完成: 成功=%d, 失败=%d, 缓存命中=%d/%d",
            len(chunks),
            len(failures),
            cache_hits,
            len(jobs),
        )
        return chunks

    def get_statistics(self, chunks: list[Document]) -> dict[str, Any]:
        return {
            "image_chunks": len(chunks),
            "caption_cache_entries": len(self._cache),
        }
