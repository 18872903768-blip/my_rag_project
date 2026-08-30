"""Compare two or more retrieval_eval_*.json reports side by side.

Usage:
    python eval/compare_eval.py .artifacts/eval/retrieval_eval_A.json \
        .artifacts/eval/retrieval_eval_B.json [report_C.json ...]

Prints overall + per-intent Hit Rate / MRR / Recall for each report, then
lists queries where the runs disagree (hit in one, missed in another), which
is where to look for the real quality delta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _labels(paths: list[Path]) -> list[str]:
    return [path.stem for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval eval reports")
    parser.add_argument("reports", nargs="+", help="retrieval_eval_*.json 路径，至少两个")
    parser.add_argument("--diff", type=int, default=20, help="逐 query 差异最多显示条数")
    args = parser.parse_args()

    paths = [Path(p) for p in args.reports]
    if len(paths) < 2:
        parser.error("至少需要两个报告文件")
    reports = [_load(path) for path in paths]
    summaries = [report["summary"] for report in reports]
    labels = _labels(paths)

    overall_keys = ["overall_hit_rate", "overall_mrr"]
    print(f"{'指标':<16}" + "".join(f"{label:>28}" for label in labels))
    for key in overall_keys:
        row = f"{key:<16}" + "".join(f"{summary[key]:>28.4f}" for summary in summaries)
        print(row)
    print(f"{'top_k / backend':<16}" + "".join(
        f"{'%s / %s' % (s['top_k'], s['backend']):>28}" for s in summaries
    ))
    print()

    intents = sorted({row["intent"] for summary in summaries for row in summary["by_intent"]})
    print("== 按 intent ==")
    print(f"{'intent':<22}" + "".join(f"{label:>28}" for label in labels))
    for intent in intents:
        cells = []
        for summary in summaries:
            row = next((r for r in summary["by_intent"] if r["intent"] == intent), None)
            if row is None:
                cells.append("n/a")
            else:
                cells.append(
                    f"hit={row['hit_rate']:.3f} mrr={row['mrr']:.3f} recall={row['recall']:.3f}"
                )
        print(f"{intent:<22}" + "".join(f"{cell:>28}" for cell in cells))
    print()

    # 逐 query 差异：按 (query, role) 对齐各报告的 records，
    # 找 hit 不一致或 MRR 差异明显的记录
    aligned: dict[tuple[str, str], list[dict]] = {}
    for report in reports:
        for record in report["records"]:
            aligned.setdefault((record["query"], record["role"]), []).append(record)

    diffs = []
    for key, rows in aligned.items():
        if len(rows) != len(reports):
            continue
        hits = [bool(row.get("hit")) for row in rows]
        mrrs = [float(row.get("reciprocal_rank", 0.0)) for row in rows]
        if len(set(hits)) > 1 or (max(mrrs) - min(mrrs)) > 0.3:
            diffs.append((key, rows))
    if diffs:
        print(f"== 逐 query 差异（hit 或 MRR 不一致，共 {len(diffs)} 条，最多显示 {args.diff}）==")
        for (query, role), rows in diffs[: args.diff]:
            cells = " | ".join(
                f"{label}: hit={int(bool(row.get('hit')))} mrr={row.get('reciprocal_rank', 0):.2f}"
                for label, row in zip(labels, rows, strict=True)
            )
            print(f"  [{role}] {query}\n    {cells}")
    else:
        print("== 逐 query 差异 ==")
        print("  无：所有配置在每条 query 上的表现一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
