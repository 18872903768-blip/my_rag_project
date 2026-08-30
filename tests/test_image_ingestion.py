"""Offline regressions for image caption ingestion (no vision API calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag_modules.image_ingestion import ImageIngestionModule


def _make_dish(root: Path, category: str = "soup", dish: str = "西红柿蛋汤") -> Path:
    dish_dir = root / "dishes" / category / dish
    dish_dir.mkdir(parents=True)
    (dish_dir / f"{dish}.md").write_text(f"# {dish}\n\n预估烹饪难度：★★", encoding="utf-8")
    # 超过 MIN_REAL_IMAGE_BYTES，避免被当作 LFS 指针跳过
    (dish_dir / "000.jpg").write_bytes(b"fake-jpeg-bytes" * 300)
    (dish_dir / "001.webp").write_bytes(b"fake-webp-bytes" * 300)
    (dish_dir / "notes.txt").write_text("不是图片", encoding="utf-8")
    return dish_dir


def _parent_document(dish_dir: Path, root: Path, dish: str = "西红柿蛋汤") -> Document:
    md_path = dish_dir / f"{dish}.md"
    return Document(
        page_content=f"# {dish}",
        metadata={
            "source": str(md_path.resolve()),
            "source_path": md_path.relative_to(root).as_posix(),
            "parent_id": "parent-abc",
            "revision_id": "revision-abc",
            "dish_name": dish,
            "category": "汤品",
            "difficulty": "简单",
            "visibility": "public",
            "modality": "text",
        },
    )


def test_ingest_images_builds_captioned_chunks(tmp_path: Path) -> None:
    root = tmp_path / "cook"
    dish_dir = _make_dish(root)
    parent = _parent_document(dish_dir, root)
    calls: list[str] = []

    def fake_caption(image_path: Path, dish_name: str, prompt: str) -> str:
        calls.append(image_path.name)
        return f"{dish_name}的成品图，色泽红亮"

    module = ImageIngestionModule(
        root, tmp_path / "cache" / "captions.json", caption_fn=fake_caption
    )

    chunks = module.ingest_images([parent])

    assert len(chunks) == 2  # 000.jpg + 001.webp；notes.txt 被忽略
    assert sorted(calls) == ["000.jpg", "001.webp"]
    first = chunks[0]
    assert first.metadata["modality"] == "image"
    assert first.metadata["parent_id"] == "parent-abc"
    assert first.metadata["dish_name"] == "西红柿蛋汤"
    assert first.metadata["image_path"].endswith(".jpg") or first.metadata["image_path"].endswith(
        ".webp"
    )
    assert first.page_content.startswith("《西红柿蛋汤》图片：")
    assert first.metadata["chunk_id"] != chunks[1].metadata["chunk_id"]


def test_caption_cache_prevents_repeat_api_calls(tmp_path: Path) -> None:
    root = tmp_path / "cook"
    dish_dir = _make_dish(root)
    parent = _parent_document(dish_dir, root)
    call_count = {"n": 0}

    def fake_caption(image_path: Path, dish_name: str, prompt: str) -> str:
        call_count["n"] += 1
        return "成品图"

    cache_path = tmp_path / "cache" / "captions.json"
    first_module = ImageIngestionModule(root, cache_path, caption_fn=fake_caption)
    first_chunks = first_module.ingest_images([parent])
    assert call_count["n"] == 2

    second_module = ImageIngestionModule(root, cache_path, caption_fn=fake_caption)
    second_chunks = second_module.ingest_images([parent])

    assert call_count["n"] == 2  # 全部命中缓存
    assert [c.page_content for c in second_chunks] == [c.page_content for c in first_chunks]


def test_failed_caption_is_skipped_but_others_ingest(tmp_path: Path) -> None:
    root = tmp_path / "cook"
    dish_dir = _make_dish(root)
    parent = _parent_document(dish_dir, root)

    def flaky_caption(image_path: Path, dish_name: str, prompt: str) -> str:
        if image_path.name == "000.jpg":
            raise RuntimeError("vision api down")
        return "步骤图"

    module = ImageIngestionModule(root, tmp_path / "cache" / "c.json", caption_fn=flaky_caption)
    chunks = module.ingest_images([parent])

    assert len(chunks) == 1
    assert chunks[0].metadata["image_path"].endswith("001.webp")

    strict = ImageIngestionModule(
        root, tmp_path / "cache" / "c2.json", caption_fn=flaky_caption, strict=True
    )
    with pytest.raises(RuntimeError, match="strict"):
        strict.ingest_images([parent])


def test_no_images_returns_empty(tmp_path: Path) -> None:
    root = tmp_path / "cook"
    dish_dir = root / "dishes" / "soup" / "白粥"
    dish_dir.mkdir(parents=True)
    (dish_dir / "白粥.md").write_text("# 白粥", encoding="utf-8")
    parent = _parent_document(dish_dir, root, dish="白粥")

    module = ImageIngestionModule(root, tmp_path / "cache" / "c.json", caption_fn=None)
    assert module.ingest_images([parent]) == []


def test_lfs_pointer_files_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "cook"
    dish_dir = _make_dish(root)
    # 模拟 Git LFS 指针文件：130 字节的文本，指向不存在的对象
    (dish_dir / "002.jpg").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123456\n"
    )
    parent = _parent_document(dish_dir, root)
    calls: list[str] = []

    def fake_caption(image_path: Path, dish_name: str, prompt: str) -> str:
        calls.append(image_path.name)
        return "成品图"

    module = ImageIngestionModule(
        root, tmp_path / "cache" / "c.json", caption_fn=fake_caption
    )

    chunks = module.ingest_images([parent])

    assert "002.jpg" not in calls  # 指针文件未触发视觉调用
    assert len(chunks) == 2
