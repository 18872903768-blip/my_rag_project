"""Answer-quality evaluation: LLM-as-judge over both pipelines.

For a sample of golden-set queries, generate answers with the classic pipeline
and the LangGraph agent, then have DeepSeek score each answer for
faithfulness (grounded in retrieved context) and relevance (answers the
question) on a 1-5 scale.  Hand-rolled instead of RAGAS to avoid version
coupling; the judge prompt is explicit and auditable.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """你是一个严格的评测员，请对菜谱助手的回答打分。

【用户问题】{question}
【检索到的参考内容】
{context}
【助手回答】
{answer}

请按两个维度各打 1-5 分：
- faithfulness: 回答是否严格基于参考内容，有无编造或与参考矛盾（5=完全有据，1=大量编造）
- relevance: 回答是否切题、对用户有用（5=完全切题，1=答非所问）

只输出一行 JSON，格式：{{"faithfulness": <int>, "relevance": <int>, "reason": "<不超过30字>"}}"""


def _context_text(parents: list) -> str:
    parts = []
    for doc in parents[:3]:
        parts.append(
            f"【{doc.metadata.get('dish_name', '未知')}】"
            f"(分类:{doc.metadata.get('category', '?')}) "
            + doc.page_content[:800]
        )
    return "\n\n".join(parts) if parts else "（无检索结果）"


def parse_judge(raw: str) -> dict:
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            return {
                "faithfulness": max(1, min(5, int(payload.get("faithfulness", 0)))),
                "relevance": max(1, min(5, int(payload.get("relevance", 0)))),
                "reason": str(payload.get("reason", ""))[:60],
            }
        except (ValueError, TypeError):
            pass
    return {"faithfulness": 0, "relevance": 0, "reason": "解析失败"}


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-judge answer quality (classic vs agent)")
    parser.add_argument("--sample", type=int, default=10, help="抽样的查询数")
    parser.add_argument(
        "--golden-set", default=str(Path(__file__).resolve().parent / "golden_set.json")
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / ".artifacts" / "eval"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT.parents[1] / ".env", override=False)

    from config import RAGConfig
    from main import RecipeRAGSystem

    system = RecipeRAGSystem(RAGConfig.from_env())
    system.initialize_system(load_generation=True)
    system.build_knowledge_base()

    items = [
        item
        for item in json.loads(Path(args.golden_set).read_text(encoding="utf-8"))
        if item.get("intent") in ("detail", "keyword") and not item.get("expect_empty")
    ]
    import random

    rng = random.Random(args.seed)
    sample = rng.sample(items, min(args.sample, len(items)))

    judge_llm = system.generation_module.llm
    records = []
    for item in sample:
        question = item["query"]
        entry = {"question": question, "intent": item.get("intent")}

        classic_answer = str(system.ask_question(question, role="user"))
        classic_trace = system.retrieve(question, role="user")
        classic_parents = system.data_module.get_parent_documents(classic_trace)
        prompt = JUDGE_PROMPT.format(
            question=question, context=_context_text(classic_parents), answer=classic_answer[:1500]
        )
        classic_scores = parse_judge(str(judge_llm.invoke(prompt).content))
        entry["classic"] = {"answer_head": classic_answer[:100], **classic_scores}

        agent_result = system.ask_agent(question, role="user")
        agent_answer = str(agent_result.get("answer", ""))
        agent_parents = agent_result.get("parents", [])
        prompt = JUDGE_PROMPT.format(
            question=question, context=_context_text(agent_parents), answer=agent_answer[:1500]
        )
        agent_scores = parse_judge(str(judge_llm.invoke(prompt).content))
        entry["agent"] = {
            "answer_head": agent_answer[:100],
            "route": agent_result.get("route"),
            **agent_scores,
        }
        records.append(entry)
        logger.info(
            "judged: %s classic(f=%.0f,r=%.0f) agent(f=%.0f,r=%.0f)",
            question[:24],
            classic_scores["faithfulness"],
            classic_scores["relevance"],
            agent_scores["faithfulness"],
            agent_scores["relevance"],
        )

    def _avg(rows: list, key: str) -> float:
        values = [row[key] for row in rows if row[key] > 0]
        return round(statistics.mean(values), 3) if values else 0.0

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sample_size": len(records),
        "classic": {
            "faithfulness": _avg([r["classic"] for r in records], "faithfulness"),
            "relevance": _avg([r["classic"] for r in records], "relevance"),
        },
        "agent": {
            "faithfulness": _avg([r["agent"] for r in records], "faithfulness"),
            "relevance": _avg([r["agent"] for r in records], "relevance"),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    (output_dir / f"judge_eval_{stamp}.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== 答案质量评测（LLM judge, n={len(records)}）===")
    print(
        f"classic: faithfulness={summary['classic']['faithfulness']} "
        f"relevance={summary['classic']['relevance']}"
    )
    print(
        f"agent:  faithfulness={summary['agent']['faithfulness']} "
        f"relevance={summary['agent']['relevance']}"
    )
    print(f"报告已写入 {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
