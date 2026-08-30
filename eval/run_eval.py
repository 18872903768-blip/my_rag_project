"""Retrieval evaluation: Recall@k, MRR, and permission isolation.

Reads ``golden_set.json``, runs the configured pipeline per query with the
query's role, and writes a JSON + markdown report under ``.artifacts/eval``.
No LLM is involved, so the whole run stays offline and reproducible.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAGConfig  # noqa: E402

logger = logging.getLogger(__name__)


def _matches(expected: str, dish_name: str) -> bool:
    return expected in dish_name or dish_name in expected


def evaluate_item(system: Any, item: dict, top_k: int) -> dict:
    results = system.retrieve(item["query"], top_k=top_k, role=item.get("role", "user"))
    returned = [
        {
            "dish": str(doc.metadata.get("dish_name", "")),
            "visibility": str(doc.metadata.get("visibility", "unknown")),
            "category": str(doc.metadata.get("category", "")),
            "difficulty": str(doc.metadata.get("difficulty", "")),
        }
        for doc in results
    ]
    returned_dishes = [row["dish"] for row in returned]
    expected = item.get("expected_dishes", [])

    record: dict = {
        "query": item["query"],
        "role": item.get("role", "user"),
        "intent": item.get("intent", "unknown"),
        "returned": returned_dishes,
        "expected": expected,
    }

    if item.get("expect_empty"):
        # 权限隔离的标准：不是"零结果"，而是"没有任何 internal 内容泄漏"；
        # 返回相关公开菜谱属于预期内的优雅降级。
        record["leaks"] = [row["dish"] for row in returned if row["visibility"] == "internal"]
        record["hit"] = not record["leaks"]
        record["reciprocal_rank"] = 0.0
        record["recall"] = 0.0
        return record

    if item.get("expected_filters"):
        # 浏览类查询：验证过滤精确率（返回的都符合期望的分类/难度）且至少有一条
        filters = item["expected_filters"]
        violating = [
            row["dish"]
            for row in returned
            if (filters.get("category") and row["category"] != filters["category"])
            or (filters.get("difficulty") and row["difficulty"] != filters["difficulty"])
        ]
        record["filter_violations"] = violating
        record["hit"] = bool(returned) and not violating
        record["reciprocal_rank"] = 1.0 if returned else 0.0
        record["recall"] = 1.0 if returned else 0.0
        return record

    record["leaks"] = []
    ranks = []
    for dish in expected:
        for rank, returned_dish in enumerate(returned_dishes, start=1):
            if _matches(dish, returned_dish):
                ranks.append(rank)
                break
    record["hit"] = bool(ranks)
    record["reciprocal_rank"] = 1.0 / min(ranks) if ranks else 0.0
    record["recall"] = len(ranks) / len(expected) if expected else 0.0
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval evaluation on the golden set")
    parser.add_argument(
        "--golden-set", default=str(Path(__file__).resolve().parent / "golden_set.json")
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", choices=("milvus", "faiss"), default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / ".artifacts" / "eval"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT.parents[1] / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    from main import RecipeRAGSystem

    config = RAGConfig.from_env()
    if args.backend:
        config.backend = args.backend
    system = RecipeRAGSystem(config)
    system.initialize_system(load_generation=False)
    system.build_knowledge_base()

    items = json.loads(Path(args.golden_set).read_text(encoding="utf-8"))
    records = [evaluate_item(system, item, args.top_k) for item in items]

    by_intent: dict[str, list[dict]] = {}
    for record in records:
        by_intent.setdefault(record["intent"], []).append(record)

    def _metric(rows: list[dict], key: str) -> float:
        return statistics.mean(row[key] for row in rows) if rows else 0.0

    summary_rows = []
    for intent, rows in sorted(by_intent.items()):
        summary_rows.append(
            {
                "intent": intent,
                "count": len(rows),
                "hit_rate": round(_metric(rows, "hit"), 4),
                "mrr": round(_metric(rows, "reciprocal_rank"), 4),
                "recall": round(_metric(rows, "recall"), 4),
            }
        )
    overall = {
        "generated_at": datetime.now(UTC).isoformat(),
        "backend": config.backend,
        "top_k": args.top_k,
        "total": len(records),
        "overall_mrr": round(_metric(records, "reciprocal_rank"), 4),
        "overall_hit_rate": round(_metric(records, "hit"), 4),
        "by_intent": summary_rows,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    (output_dir / f"retrieval_eval_{stamp}.json").write_text(
        json.dumps({"summary": overall, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== 检索评测（backend={config.backend}, top_k={args.top_k}）===")
    print(
        f"总数 {overall['total']} | Hit@{args.top_k} {overall['overall_hit_rate']:.3f} "
        f"| MRR {overall['overall_mrr']:.3f}"
    )
    for row in summary_rows:
        print(
            f"  {row['intent']:<20} n={row['count']:<3} hit={row['hit_rate']:.3f} "
            f"mrr={row['mrr']:.3f} recall={row['recall']:.3f}"
        )
    leaks = [r for r in records if r.get("leaks")]
    if leaks:
        print(f"\n⚠ 权限泄漏 {len(leaks)} 条：")
        for record in leaks[:5]:
            print(f"  [{record['role']}] {record['query']} → {record['returned'][:3]}")
    else:
        print("权限隔离：无泄漏")
    print(f"报告已写入 {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
