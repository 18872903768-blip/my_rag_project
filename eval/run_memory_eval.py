"""多轮记忆评测：驱动 agent 管线验证 Contextualization 与 Memory 的行为。

用例文件 ``multi_turn_eval.json`` 的断言类型（按 turn 逐条检查）：
- expect_recall_dishes: 检索命中的菜名（dish_name 子串匹配）
- expect_contextualized_query_contains: 指代消解后的 standalone query 需包含的关键词
- expect_memory_written / expect_memory_not_written: 记忆落库断言
- expect_supersede: 同 key 旧记忆被替换（status=superseded）
- expect_memory_recalled: 本轮发生记忆召回注入
- expect_memory_has_expiry: 写入的记忆带过期时间
- judge: LLM 判分（1-5，>=4 通过），用于答案层面的约束遵守检查

用法（需 Milvus 运行 + DEEPSEEK_API_KEY）：
    python eval/run_memory_eval.py                 # 全量用例
    python eval/run_memory_eval.py --case memory_supersede
产物: .artifacts/eval/memory_eval_{stamp}.json（envelope 格式）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _matches(expected: str, dish_name: str) -> bool:
    return expected in dish_name or dish_name in expected


def _events(result: dict) -> list[dict]:
    return result.get("events", [])


def _has_event(result: dict, name: str) -> bool:
    return any(e.get("event") == name for e in _events(result))


def check_turn_assertions(
    turn: dict,
    result: dict,
    memories_after: list,
    memories_before: list,
) -> list[str]:
    """返回未通过的断言描述；空列表 = 全部通过。"""
    failures: list[str] = []
    events = _events(result)
    hit_dishes = [
        str(chunk.metadata.get("dish_name", "")) for chunk in result.get("chunks", [])
    ]

    for dish in turn.get("expect_recall_dishes", []):
        if not any(_matches(dish, name) for name in hit_dishes):
            failures.append(f"expect_recall_dishes 缺 {dish}（命中: {hit_dishes}）")

    if "expect_contextualized_query_contains" in turn:
        ctx_events = [e for e in events if e.get("event") == "query_contextualized"]
        if not ctx_events:
            failures.append("未发生指代消解（缺 query_contextualized 事件）")
        else:
            query = str(ctx_events[-1].get("query", ""))
            for keyword in turn["expect_contextualized_query_contains"]:
                if keyword not in query:
                    failures.append(f"消解查询未包含 {keyword!r}（实际: {query!r}）")

    if "expect_memory_recalled" in turn:
        if not _has_event(result, "memory_recalled"):
            failures.append("未发生记忆召回（缺 memory_recalled 事件）")

    if "expect_memory_written" in turn:
        spec = turn["expect_memory_written"]
        hit = [
            m
            for m in memories_after
            if m.type == spec.get("type") and spec.get("keyword", "") in (m.key + m.content)
        ]
        if len(hit) <= len(
            [m for m in memories_before if m.type == spec.get("type")]
        ):
            failures.append(f"期望写入记忆 {spec}，实际未新增")
        elif turn.get("expect_memory_has_expiry") and not hit[0].expires_at:
            failures.append("写入的记忆缺少过期时间（constraint 应有 TTL）")

    if "expect_memory_not_written" in turn:
        grew = len(memories_after) > len(memories_before)
        if grew:
            failures.append(f"不应写入记忆，但新增了 {len(memories_after) - len(memories_before)} 条")

    if "expect_supersede" in turn:
        spec = turn["expect_supersede"]
        matched = [m for m in memories_after if m.type == spec.get("type")]
        superseded = [m for m in matched if m.status == "superseded"]
        active = [m for m in matched if m.status == "active"]
        if not superseded or not active:
            failures.append(
                f"期望旧记忆 superseded 且新记忆 active，实际 superseded={len(superseded)} active={len(active)}"
            )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="多轮记忆评测（agent 管线）")
    parser.add_argument(
        "--cases", default=str(Path(__file__).resolve().parent / "multi_turn_eval.json")
    )
    parser.add_argument("--case", default=None, help="只跑指定 case_id")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 判分")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / ".artifacts" / "eval"))
    parser.add_argument("--keep-db", action="store_true", help="保留临时记忆库以便检查")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT.parents[1] / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    # 本评测强制开启被测特性，并使用独立临时记忆库（可复现、不污染业务库）
    os.environ["RAG_MEMORY_ENABLED"] = "true"
    os.environ["RAG_QUERY_CONTEXTUALIZATION"] = "true"
    if not args.keep_db:
        os.environ["RAG_MEMORY_DB"] = os.path.join(tempfile.mkdtemp(), "memory_eval.sqlite3")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("错误：未设置 DEEPSEEK_API_KEY", file=sys.stderr)
        return 1

    from config import RAGConfig
    from main import RecipeRAGSystem

    config = RAGConfig.from_env()
    system = RecipeRAGSystem(config)
    system.initialize_system(load_generation=True)
    system.build_knowledge_base()
    extractor = system._get_memory_extractor()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if args.case:
        cases = [case for case in cases if case["case_id"] == args.case]

    store = system._get_memory_store()
    judge_llm = system.generation_module.llm
    records: list[dict] = []

    for case in cases:
        user_id = f"eval-{uuid.uuid4().hex[:8]}"
        case_record: dict = {"case_id": case["case_id"], "description": case["description"], "turns": []}
        case_failures: list[str] = []

        for session in case["sessions"]:
            history: list[dict[str, str]] = []
            for turn in session["turns"]:
                memories_before = store.list_memories(user_id)
                result = system.ask_agent(
                    turn["query"],
                    role="user",
                    history=history,
                    user_id=user_id,
                )
                answer = str(result.get("answer", ""))
                history.append({"role": "user", "content": turn["query"]})
                history.append({"role": "assistant", "content": answer[:400]})
                memories_after = store.list_memories(user_id)

                failures = check_turn_assertions(
                    turn, result, memories_after, memories_before
                )
                if "judge" in turn and not args.no_judge:
                    prompt = (
                        "以下是菜谱助手对用户的回答。请判断回答是否满足给定要求，1-5 分。\n"
                        f"【要求】{turn['judge']}\n【回答】{answer[:1000]}\n"
                        '只输出一行 JSON: {"score": <int>, "reason": "<20字>"}'
                    )
                    try:
                        raw = str(judge_llm.invoke(prompt).content)
                        start, end = raw.find("{"), raw.rfind("}")
                        payload = json.loads(raw[start : end + 1])
                        score = int(payload.get("score", 0))
                    except Exception as error:  # noqa: BLE001 - 判分失败按不通过计
                        failures.append(f"judge 解析失败: {error}")
                        score = 0
                    if score < 4:
                        failures.append(f"judge 判分 {score} < 4: {payload.get('reason', '')}")

                turn_record = {
                    "query": turn["query"],
                    "answer_head": answer[:120],
                    "events": [e.get("event") for e in _events(result)],
                    "hits": [
                        str(chunk.metadata.get("dish_name", ""))
                        for chunk in result.get("chunks", [])
                    ],
                    "memory_count_after": len(memories_after),
                    "failures": failures,
                    "passed": not failures,
                }
                case_record["turns"].append(turn_record)
                case_failures.extend(failures)

        case_record["passed"] = not case_failures
        records.append(case_record)
        print(f"[{case['case_id']}] {'PASS' if case_record['passed'] else 'FAIL'}")

    passed = [r for r in records if r["passed"]]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": len(records),
        "passed_cases": len(passed),
        "pass_rate": round(len(passed) / len(records), 4) if records else 0.0,
        "features": {
            "context_manager": config.context_manager_enabled,
            "query_contextualization": config.query_contextualization_enabled,
            "memory_enabled": config.memory_enabled,
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    (output_dir / f"memory_eval_{stamp}.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== 多轮记忆评测（{summary['passed_cases']}/{summary['cases']} 通过）===")
    for record in records:
        mark = "PASS" if record["passed"] else "FAIL"
        print(f"  [{mark}] {record['case_id']}")
        for turn in record["turns"]:
            for failure in turn["failures"]:
                print(f"      - {turn['query'][:20]}: {failure}")
    print(f"报告已写入 {output_dir}")
    return 0 if summary["passed_cases"] == len(records) else 2


if __name__ == "__main__":
    sys.exit(main())
