"""一次性合成 golden set 的标准答案（reference answers），供 RAGAS 评测使用。

RAGAS 的 context_recall / answer_correctness 需要 reference（标准答案），
而 golden_set.json 只标注了期望菜名。本脚本按期望菜名直接取语料中的
parent 文档作为"金标准上下文"，用生成模块（DeepSeek）合成一条标准答案——
不经过检索链路，因此不依赖 Milvus，也不受被测系统排序质量影响。

产物 eval/reference_answers.json 结构：
    {"generated_at": ..., "model": ..., "answers": {query: {"reference": ..., "intent": ..., "expected_dishes": [...]}}}

合成结果应人工过目后再用于评测；默认合并写入（已存在的 query 跳过），
--force 全部重生成。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.run_ragas_eval import filter_golden_items, strip_citations  # noqa: E402

logger = logging.getLogger(__name__)


def gold_parents_by_dish(system: Any, expected_dishes: list[str]) -> list:
    """按期望菜名从语料中取 parent 文档（不经过检索）。"""
    docs_by_dish = {}
    for doc in system.data_module.documents:
        name = str(doc.metadata.get("dish_name", ""))
        if name and name not in docs_by_dish:
            docs_by_dish[name] = doc
    return [docs_by_dish[name] for name in expected_dishes if name in docs_by_dish]


def main() -> int:
    parser = argparse.ArgumentParser(description="合成 golden set 的标准答案")
    parser.add_argument(
        "--golden-set", default=str(Path(__file__).resolve().parent / "golden_set.json")
    )
    parser.add_argument(
        "--output", default=str(Path(__file__).resolve().parent / "reference_answers.json")
    )
    parser.add_argument("--force", action="store_true", help="忽略已有结果全部重生成")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条缺失项")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT.parents[1] / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    import os

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("错误：未设置 DEEPSEEK_API_KEY", file=sys.stderr)
        return 1

    from config import RAGConfig
    from main import RecipeRAGSystem

    system = RecipeRAGSystem(RAGConfig.from_env())
    system.initialize_system(load_generation=True)
    system.data_module.load_documents()

    items = filter_golden_items(json.loads(Path(args.golden_set).read_text(encoding="utf-8")))

    output_path = Path(args.output)
    existing: dict = {}
    if output_path.exists() and not args.force:
        existing = json.loads(output_path.read_text(encoding="utf-8")).get("answers", {})

    todo = [item for item in items if item["query"] not in existing]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print(f"全部 {len(items)} 条已有标准答案（--force 可重生成）")
        return 0

    print(f"待合成 {len(todo)} 条（已存在 {len(existing)} 条，共 {len(items)} 条）")
    answers = dict(existing)
    for index, item in enumerate(todo, start=1):
        question = item["query"]
        parents = gold_parents_by_dish(system, item.get("expected_dishes", []))
        if not parents:
            logger.warning("跳过（期望菜名均不在语料中）: %s", question)
            continue
        raw = str(system.generation_module.generate_basic_answer(question, parents))
        answers[question] = {
            "reference": strip_citations(raw).strip(),
            "intent": item.get("intent"),
            "expected_dishes": item.get("expected_dishes", []),
        }
        print(f"[{index}/{len(todo)}] 已合成: {question[:30]}")

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": system.config.llm_model,
        "count": len(answers),
        "answers": answers,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n共 {len(answers)} 条标准答案已写入 {output_path}")
    print("提示：合成结果基于期望菜谱的原文生成，请人工抽查后再用于评测；"
          "答案末尾的引用附录已剔除")
    return 0


if __name__ == "__main__":
    sys.exit(main())
