"""RAGAS 生成质量评测：LLM-as-judge 自动打分（run_judge.py 的正式演进）。

run_eval.py 用确定性指标评测检索排序；本脚本补生成层：采集 classic 管线
真实输出 (question, answer, contexts, reference)，交给 RAGAS 指标打分——

- faithfulness（忠实度/反幻觉，无参考）、answer_relevancy（答案相关性，无参考）
- context_precision / context_recall / answer_correctness（需要标准答案，
  由 eval/generate_reference_answers.py 一次性合成，见 reference_answers.json）

judge LLM 复用 DeepSeek（OpenAI 兼容端点）；embeddings 复用项目 bge-small-zh。
产物沿用 {"summary", "records"} envelope 写 .artifacts/eval/ragas_eval_*.json。

兼容性：ragas==0.4.3 在导入期无条件引用 langchain_community 的 VertexAI 集成，
而 langchain-community>=0.4 已移除该模块；本脚本在导入 ragas 前注入占位模块
补齐（judge 走 OpenAI 兼容端点，不会触碰这些类）。ragas 仅在本脚本导入，
运行时代码（main/api）不依赖它。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import types
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

REFERENCE_FILE = Path(__file__).resolve().parent / "reference_answers.json"
CITATION_MARKER = "——\n📚 以上回答参考自："

# 指标名 -> ascore 需要的字段（v0.4 collections API 的关键字参数）
METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "answer_relevancy": ("user_input", "response"),
    "context_precision": ("user_input", "retrieved_contexts", "reference"),
    "context_recall": ("user_input", "retrieved_contexts", "reference"),
    "answer_correctness": ("user_input", "response", "reference"),
}
REFERENCE_FREE_METRICS = ("faithfulness", "answer_relevancy")
EMBEDDING_METRICS = ("answer_relevancy", "answer_correctness")


def install_ragas_compat_shims() -> None:
    """在导入 ragas 前补齐 langchain-community 0.4 已移除的 VertexAI 引用。"""
    import langchain_community.llms as community_llms

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        fake = types.ModuleType("langchain_community.chat_models.vertexai")
        fake.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = fake
    if not hasattr(community_llms, "VertexAI"):
        community_llms.VertexAI = type("VertexAI", (), {})


# ---------------------------------------------------------------- 纯函数（可离线单测）


def filter_golden_items(items: list[dict]) -> list[dict]:
    """与 run_judge.py 同规则：detail/keyword 且非 expect_empty。"""
    return [
        item
        for item in items
        if item.get("intent") in ("detail", "keyword") and not item.get("expect_empty")
    ]


def sample_items(items: list[dict], *, limit: int | None, sample: int | None, seed: int) -> list[dict]:
    """先 seed 抽样后截断；两者都不传则全量。"""
    picked = items
    if sample is not None and 0 < sample < len(picked):
        import random

        picked = random.Random(seed).sample(picked, sample)
    if limit is not None:
        picked = picked[:limit]
    return picked


def parent_texts(parents: list) -> list[str]:
    """RAGAS 的 retrieved_contexts：生成实际所见 parent 正文的文本列表。"""
    return [str(doc.page_content) for doc in parents]


def strip_citations(text: str) -> str:
    """去掉确定性引用附录（引用是 provenance，不是供 judge 校验的论断）。"""
    index = text.find(CITATION_MARKER)
    return text[:index].strip() if index >= 0 else text.strip()


def merge_reference(items: list[dict], reference_map: dict) -> list[dict]:
    """给每条样本挂 reference（无则 None），返回 (样本列表, 缺 reference 数)。"""
    merged, missing = [], 0
    for item in items:
        entry = dict(item)
        ref = reference_map.get(entry["query"], {}).get("reference")
        entry["reference"] = str(ref).strip() if ref else None
        if not entry["reference"]:
            missing += 1
        merged.append(entry)
    return merged, missing


def summarize_scores(records: list[dict], metric_names: tuple[str, ...] = METRIC_FIELDS) -> dict:
    """各指标均值（None=该条打分失败，不参与均值）+ 失败计数。

    只统计本次实际运行的指标（metric_names），避免"未启用"被误报为失败。
    """
    summary: dict[str, float] = {}
    failed: dict[str, int] = {}
    for metric in metric_names:
        values = [row["scores"].get(metric) for row in records]
        ok = [value for value in values if isinstance(value, (int, float))]
        if ok:
            summary[metric] = round(statistics.mean(ok), 4)
        if len(ok) < len(values):
            failed[metric] = len(values) - len(ok)
    return {"metrics": summary, "failed_counts": failed}


def build_markdown_report(retrieval_summary: dict | None, ragas_summary: dict | None) -> str:
    """把两份评测的 summary 合成 markdown 总报告。"""
    lines = ["# RAG 评测流水线报告", ""]
    if retrieval_summary is not None:
        lines += [
            "## 检索层（run_eval.py，确定性指标）",
            "",
            "| 指标 | 数值 |",
            "|---|---|",
            f"| 总样本 | {retrieval_summary.get('total', '-')} |",
            f"| Hit@k | {retrieval_summary.get('overall_hit_rate', '-')} |",
            f"| MRR | {retrieval_summary.get('overall_mrr', '-')} |",
            "",
            "| intent | 数量 | Hit | MRR | Recall |",
            "|---|---|---|---|---|",
        ]
        for row in retrieval_summary.get("by_intent", []):
            lines.append(
                f"| {row['intent']} | {row['count']} | {row['hit_rate']} "
                f"| {row['mrr']} | {row['recall']} |"
            )
        lines.append("")
    if ragas_summary is not None:
        metrics = ragas_summary.get("metrics", {})
        failed = ragas_summary.get("failed_counts", {})
        lines += [
            "## 生成层（RAGAS，LLM-as-judge）",
            "",
            "| 指标 | 均值 | 失败条数 |",
            "|---|---|---|",
        ]
        for metric in METRIC_FIELDS:
            if metric in metrics or metric in failed:
                lines.append(f"| {metric} | {metrics.get(metric, '-')} | {failed.get(metric, 0)} |")
        lines += [
            "",
            f"- 样本量: {ragas_summary.get('sample_size', '-')}，"
            f"judge 模型: {ragas_summary.get('judge_model', '-')}",
            f"- 流水线: {ragas_summary.get('pipeline', '-')}，"
            f"reference answers: {'启用' if ragas_summary.get('has_reference') else '未合成'}",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------- 管线采集


def collect_classic_sample(system: Any, item: dict, role: str) -> dict:
    """复刻 _ask_question_inner 的完整路径，同时捕获生成实际所见的 parent。"""
    from rag_modules.domain_config import get_domain

    question = item["query"]
    route_type = system.generation_module.query_router(question)
    rewritten = (
        question if route_type == "list" else system.generation_module.query_rewrite(question)
    )
    chunks = system.retrieve(
        rewritten, filters=system._extract_filters_from_query(question), role=role
    )
    base = {"user_input": question, "route": route_type, "rewritten": rewritten, "pipeline": "classic"}
    if not chunks:
        return {**base, "response": get_domain().classic_no_results, "contexts": []}
    parents = system.data_module.get_parent_documents(chunks)
    image_paths = system._collect_image_paths(chunks)
    if route_type == "list":
        answer = system.generation_module.generate_list_answer(question, parents)
    elif route_type == "detail":
        answer = system.generation_module.generate_step_by_step_answer(
            question, parents, image_paths=image_paths
        )
    else:
        answer = system.generation_module.generate_basic_answer(
            question, parents, image_paths=image_paths
        )
    return {**base, "response": strip_citations(str(answer)), "contexts": parent_texts(parents)}


def collect_agent_sample(system: Any, item: dict, role: str) -> dict:
    """Agent（LangGraph）管线：ask_agent 自带 answer 与 parents。"""
    question = item["query"]
    result = system.ask_agent(question, role=role)
    answer = str(result.get("answer", ""))
    return {
        "user_input": question,
        "route": result.get("route"),
        "rewritten": None,
        "pipeline": "agent",
        "response": strip_citations(answer),
        "contexts": parent_texts(result.get("parents", [])),
    }


# ---------------------------------------------------------------- RAGAS 打分


def build_judge_llm(config: Any):  # noqa: ANN202 - ragas 类型运行期才有
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    base_url = os.environ.get("DEEPSEEK_BASE_URL") or None
    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=base_url)
    # llm_factory 默认 max_tokens=1024；faithfulness 的中文 claim 拆解 JSON
    # 很占 token，实测 4096 仍会截断，8192 是 deepseek-chat 的输出上限
    return llm_factory(config.llm_model, provider="openai", client=client, max_tokens=8192)


def build_embeddings(config: Any):  # noqa: ANN202
    """ragas 原生 HuggingFaceEmbeddings 加载项目同款 bge-small-zh（modern 接口）。

    collections 指标要求 modern embeddings（BaseRagasEmbedding），拒绝
    Langchain 包装类。先按本地缓存离线加载（与 embedding 模型同源、已在缓存），
    失败再允许联网；仍失败返回 None，跳过依赖嵌入的指标。
    """
    from ragas.embeddings import HuggingFaceEmbeddings

    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return HuggingFaceEmbeddings(
            model=config.embedding_model, device=config.embedding_device
        )
    except Exception as offline_error:  # noqa: BLE001 - 缓存冷时降级联网
        logger.warning("离线加载 ragas embeddings 失败（%s），尝试联网下载", offline_error)
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous
        try:
            return HuggingFaceEmbeddings(
                model=config.embedding_model, device=config.embedding_device
            )
        except Exception as error:  # noqa: BLE001 - embeddings 缺失只降级不致命
            logger.warning("ragas embeddings 初始化失败，跳过依赖嵌入的指标: %s", error)
            return None


def build_metrics(judge_llm: Any, embeddings: Any, has_reference: bool) -> dict:
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    metrics: dict[str, Any] = {"faithfulness": Faithfulness(llm=judge_llm)}
    if embeddings is not None:
        metrics["answer_relevancy"] = AnswerRelevancy(llm=judge_llm, embeddings=embeddings)
        if has_reference:
            metrics["answer_correctness"] = AnswerCorrectness(
                llm=judge_llm, embeddings=embeddings
            )
    if has_reference:
        metrics["context_precision"] = ContextPrecision(llm=judge_llm)
        metrics["context_recall"] = ContextRecall(llm=judge_llm)
    return metrics


def _metric_kwargs(sample: dict, fields: tuple[str, ...]) -> dict:
    """把样本字段映射成 ascore 关键字参数（样本里存的是 contexts）。"""
    kwargs: dict[str, Any] = {}
    for field in fields:
        if field == "retrieved_contexts":
            kwargs[field] = sample.get("contexts") or []
        else:
            value = sample.get(field)
            if value is None:
                raise KeyError(field)
            kwargs[field] = value
    return kwargs


async def score_sample(sample: dict, metrics: dict, semaphore: asyncio.Semaphore) -> dict:
    """单样本逐指标打分；单指标失败记 None 并保留错误信息，不拖垮整批。

    judge 输出截断（finish_reason=length）在并发下偶发（约 1%）且重跑即好，
    故每条指标调用最多重试 2 次。
    """
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for name, metric in metrics.items():
        needs_reference = "reference" in METRIC_FIELDS[name]
        if needs_reference and not sample.get("reference"):
            continue
        try:
            kwargs = _metric_kwargs(sample, METRIC_FIELDS[name])
        except KeyError as error:
            scores[name] = None
            errors[name] = f"样本缺少字段 {error}（采集失败）"
            continue
        last_error = ""
        for attempt in (1, 2, 3):
            try:
                async with semaphore:
                    result = await metric.ascore(**kwargs)
                scores[name] = float(getattr(result, "value", result))
                last_error = ""
                break
            except Exception as error:  # noqa: BLE001 - 单点失败不拖垮整批
                last_error = str(error)[:200]
                if attempt < 3:
                    await asyncio.sleep(1.5 * attempt)
        if last_error:
            scores[name] = None
            errors[name] = last_error
    return {**sample, "scores": scores, "errors": errors}


async def score_all(samples: list[dict], metrics: dict, max_workers: int) -> list[dict]:
    semaphore = asyncio.Semaphore(max_workers)
    return list(await asyncio.gather(*(score_sample(s, metrics, semaphore) for s in samples)))


# ---------------------------------------------------------------- 入口


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 生成质量评测（LLM-as-judge）")
    parser.add_argument(
        "--golden-set", default=str(Path(__file__).resolve().parent / "golden_set.json")
    )
    parser.add_argument("--top-k", type=int, default=None, help="检索条数（默认 RAG_TOP_K）")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条")
    parser.add_argument("--sample", type=int, default=None, help="随机抽样 N 条")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pipeline", choices=("classic", "both"), default="classic")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / ".artifacts" / "eval"))
    parser.add_argument("--skip-missing-reference", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT.parents[1] / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("错误：未设置 DEEPSEEK_API_KEY，无法进行生成评测", file=sys.stderr)
        return 1

    install_ragas_compat_shims()

    from config import RAGConfig
    from main import RecipeRAGSystem

    config = RAGConfig.from_env()
    if args.top_k:
        config.top_k = args.top_k
    system = RecipeRAGSystem(config)
    system.initialize_system(load_generation=True)
    system.build_knowledge_base()

    items = filter_golden_items(json.loads(Path(args.golden_set).read_text(encoding="utf-8")))
    items = sample_items(items, limit=args.limit, sample=args.sample, seed=args.seed)

    reference_map: dict = {}
    if REFERENCE_FILE.exists():
        reference_map = json.loads(REFERENCE_FILE.read_text(encoding="utf-8")).get("answers", {})
    samples, missing_reference = merge_reference(items, reference_map)
    has_reference = any(s["reference"] for s in samples)
    if args.skip_missing_reference:
        samples = [s for s in samples if s["reference"]]
    if not samples:
        print("错误：没有可评测的样本", file=sys.stderr)
        return 1

    roles = {item["query"]: item.get("role", "user") for item in items}
    records = []
    for index, sample in enumerate(samples, start=1):
        role = sample.get("role") or roles.get(sample["query"], "user")
        try:
            collected = collect_classic_sample(system, sample, role)
            if args.pipeline == "both":
                records.append({**sample, **collect_agent_sample(system, sample, role)})
            records.append({**sample, **collected})
            print(f"[{index}/{len(samples)}] 已采集: {sample['query'][:30]}")
        except Exception as error:  # noqa: BLE001 - 采集失败记录后继续
            logger.warning("采集失败（%s）: %s", sample["query"][:30], error)
            records.append(
                {
                    **sample,
                    "user_input": sample["query"],
                    "pipeline": "classic",
                    "response": "",
                    "contexts": [],
                    "scores": {},
                    "errors": {"collect": str(error)[:200]},
                }
            )

    judge_llm = build_judge_llm(config)
    embeddings = build_embeddings(config)
    metrics = build_metrics(judge_llm, embeddings, has_reference)
    print(f"\n开始 RAGAS 打分：{len(records)} 条 × {len(metrics)} 指标（workers={args.max_workers}）")
    records = asyncio.run(score_all(records, metrics, args.max_workers))

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "judge_model": config.llm_model,
        "pipeline": args.pipeline,
        "top_k": config.top_k,
        "sample_size": len(records),
        "has_reference": has_reference,
        "missing_reference": missing_reference,
        **summarize_scores(records, tuple(metrics)),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    (output_dir / f"ragas_eval_{stamp}.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== RAGAS 生成质量评测（judge={config.llm_model}, n={len(records)}）===")
    for metric, value in summary["metrics"].items():
        failed = summary["failed_counts"].get(metric, 0)
        suffix = f"（失败 {failed}）" if failed else ""
        print(f"  {metric:<20} {value}{suffix}")
    if not has_reference:
        print("  提示：未找到 reference_answers.json，上下文/正确性指标未启用；"
              "先运行 eval/generate_reference_answers.py")
    print(f"报告已写入 {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
