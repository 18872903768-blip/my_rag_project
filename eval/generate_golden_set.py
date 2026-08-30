"""Generate a seeded golden set from the corpus metadata.

Producing the seed programmatically keeps dish names, categories, and
difficulties aligned with real metadata; the file is then hand-curated.
Rerunning overwrites the generated file, so curated edits belong in
``golden_set.json`` afterwards.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_modules.data_preparation import DataPreparationModule  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "golden_set.json"
SEED = 42
DISH_DETAIL_SAMPLES = 40
CATEGORY_SAMPLES_PER_CATEGORY = 2
KEYWORD_QUERIES = {
    "红烧": "红烧",
    "糖醋": "糖醋",
    "清蒸": "清蒸",
    "炒饭": "炒饭",
    "汤": "汤",
}


def build_items(data_module: DataPreparationModule) -> list[dict]:
    documents = data_module.load_documents()
    rng = random.Random(SEED)
    items: list[dict] = []

    public_docs = [d for d in documents if d.metadata["visibility"] == "public"]
    internal_docs = [d for d in documents if d.metadata["visibility"] == "internal"]

    # 1. 具体菜品做法/食材问题（期望命中该菜品，允许同名变体）
    for doc in rng.sample(public_docs, min(DISH_DETAIL_SAMPLES, len(public_docs))):
        dish = doc.metadata["dish_name"]
        items.append(
            {
                "query": f"{dish}怎么做",
                "expected_dishes": [dish],
                "role": "user",
                "intent": "detail",
            }
        )

    # 2. 关键词类查询（期望命中菜名含关键词的菜品）
    for keyword in KEYWORD_QUERIES:
        matching = sorted(
            {
                doc.metadata["dish_name"]
                for doc in public_docs
                if keyword in doc.metadata["dish_name"]
            }
        )
        if matching:
            items.append(
                {
                    "query": f"{keyword}菜品有哪些做法",
                    "expected_dishes": matching,
                    "role": "user",
                    "intent": "keyword",
                }
            )

    # 3. 分类/难度浏览类
    by_category: dict[str, list] = {}
    for doc in public_docs:
        by_category.setdefault(doc.metadata["category"], []).append(doc)
    for category, docs in sorted(by_category.items()):
        for doc in rng.sample(docs, min(CATEGORY_SAMPLES_PER_CATEGORY, len(docs))):
            difficulty = doc.metadata["difficulty"]
            if difficulty == "未知":
                continue
            items.append(
                {
                    "query": f"推荐几道{difficulty}的{category}",
                    "expected_dishes": [doc.metadata["dish_name"]],
                    "expected_filters": {"category": category, "difficulty": difficulty},
                    "role": "user",
                    "intent": "browse",
                }
            )

    # 4. 权限隔离：普通用户查内部（半成品）内容 → 期望零命中
    if internal_docs:
        internal_sample = rng.sample(internal_docs, min(5, len(internal_docs)))
        for doc in internal_sample:
            dish = doc.metadata["dish_name"]
            items.append(
                {
                    "query": f"{dish}的生产工艺",
                    "expected_dishes": [],
                    "expect_empty": True,
                    "role": "user",
                    "intent": "permission_denied",
                }
            )
            items.append(
                {
                    "query": f"{dish}的生产工艺",
                    "expected_dishes": [dish],
                    "role": "staff",
                    "intent": "permission_allowed",
                }
            )

    return items


def main() -> int:
    from config import RAGConfig

    config = RAGConfig.from_env()
    data_module = DataPreparationModule(config.data_path)
    items = build_items(data_module)
    OUTPUT_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    intents: dict[str, int] = {}
    for item in items:
        intents[item["intent"]] = intents.get(item["intent"], 0) + 1
    print(f"生成 {len(items)} 条 golden set → {OUTPUT_PATH}")
    print("意图分布:", intents)
    return 0


if __name__ == "__main__":
    sys.exit(main())
