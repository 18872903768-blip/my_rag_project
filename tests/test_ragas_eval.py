"""Offline regressions for the RAGAS eval pipeline helpers (no network, no ragas)."""

from __future__ import annotations

from eval.run_ragas_eval import (
    build_markdown_report,
    filter_golden_items,
    merge_reference,
    parent_texts,
    sample_items,
    strip_citations,
    summarize_scores,
)


def _item(query: str, intent: str = "detail", expect_empty: bool = False) -> dict:
    return {"query": query, "intent": intent, "expect_empty": expect_empty}


def test_filter_keeps_detail_and_keyword_only() -> None:
    items = [
        _item("红烧肉怎么做"),
        _item("红烧菜品有哪些", intent="keyword"),
        _item("推荐几道主食", intent="browse"),
        _item("半成品工艺", intent="detail", expect_empty=True),
    ]

    kept = filter_golden_items(items)

    assert [row["query"] for row in kept] == ["红烧肉怎么做", "红烧菜品有哪些"]


def test_sample_items_is_seed_deterministic_and_respects_limit() -> None:
    items = [_item(f"q{i}") for i in range(10)]

    first = sample_items(items, limit=None, sample=5, seed=7)
    again = sample_items(items, limit=None, sample=5, seed=7)
    limited = sample_items(items, limit=2, sample=None, seed=7)

    assert [row["query"] for row in first] == [row["query"] for row in again]
    assert len(first) == 5
    assert len(limited) == 2 and limited[0]["query"] == "q0"


def test_parent_texts_extracts_page_content() -> None:
    class FakeDoc:
        def __init__(self, content: str) -> None:
            self.page_content = content

    assert parent_texts([FakeDoc("a"), FakeDoc("b")]) == ["a", "b"]
    assert parent_texts([]) == []


def test_strip_citations_removes_appendix() -> None:
    answer = "红烧肉需要五花肉。\n\n——\n📚 以上回答参考自：\n- 红烧肉"

    assert strip_citations(answer) == "红烧肉需要五花肉。"
    assert strip_citations("无引用的回答") == "无引用的回答"


def test_merge_reference_attaches_and_counts_missing() -> None:
    items = [_item("有答案"), _item("没答案")]
    reference_map = {"有答案": {"reference": "标准答案"}}

    merged, missing = merge_reference(items, reference_map)

    assert merged[0]["reference"] == "标准答案"
    assert merged[1]["reference"] is None
    assert missing == 1


def test_summarize_scores_ignores_failures_and_counts_them() -> None:
    records = [
        {"scores": {"faithfulness": 0.8, "context_recall": None}, "errors": {}},
        {"scores": {"faithfulness": 1.0, "context_recall": None}, "errors": {}},
    ]

    summary = summarize_scores(records, ("faithfulness", "context_recall"))

    assert summary["metrics"]["faithfulness"] == 0.9
    assert summary["failed_counts"]["context_recall"] == 2


def test_summarize_scores_only_counts_run_metrics() -> None:
    records = [{"scores": {"faithfulness": 0.5}, "errors": {}}]

    summary = summarize_scores(records, ("faithfulness",))

    # context_precision 等未启用的指标不应被误报为失败
    assert summary["metrics"] == {"faithfulness": 0.5}
    assert summary["failed_counts"] == {}


def test_markdown_report_contains_both_layers() -> None:
    retrieval = {
        "total": 73,
        "overall_hit_rate": 1.0,
        "overall_mrr": 0.9315,
        "by_intent": [
            {"intent": "detail", "count": 40, "hit_rate": 1.0, "mrr": 1.0, "recall": 1.0}
        ],
    }
    ragas = {
        "metrics": {"faithfulness": 0.92, "answer_relevancy": 0.85},
        "failed_counts": {"answer_relevancy": 1},
        "sample_size": 10,
        "judge_model": "deepseek-chat",
        "pipeline": "classic",
        "has_reference": False,
    }

    report = build_markdown_report(retrieval, ragas)

    assert "## 检索层" in report and "0.9315" in report
    assert "## 生成层" in report and "faithfulness" in report
    assert "未合成" in report  # reference answers 未启用的标注
