"""一条命令跑完整评测流水线：检索层（run_eval）→ 生成层（run_ragas_eval）。

串行执行两个子评测，读取各自最新的 summary，合成 markdown 总报告
pipeline_report_{stamp}.md 到 .artifacts/eval/。子进程继承当前环境，
可用 RAG_RERANK_* 等环境变量切换被测配置（与单跑 run_eval 一致）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = Path(__file__).resolve().parent


def _latest(pattern: str, output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob(pattern))
    return candidates[-1] if candidates else None


def _run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(str(part) for part in cmd)}")
    subprocess.run([str(part) for part in cmd], check=True, cwd=str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="检索 + 生成 全量评测流水线")
    parser.add_argument(
        "--golden-set", default=str(EVAL_DIR / "golden_set.json"),
        help="两阶段共用的评测集",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ragas-limit", type=int, default=None, help="RAGAS 阶段只评前 N 条")
    parser.add_argument("--ragas-sample", type=int, default=None, help="RAGAS 阶段抽样 N 条")
    parser.add_argument("--ragas-max-workers", type=int, default=4)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / ".artifacts" / "eval"))
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    python = sys.executable
    run_eval_cmd = [python, str(EVAL_DIR / "run_eval.py"), "--top-k", str(args.top_k),
                    "--golden-set", args.golden_set, "--output-dir", str(output_dir)]
    ragas_cmd = [python, str(EVAL_DIR / "run_ragas_eval.py"), "--top-k", str(args.top_k),
                 "--golden-set", args.golden_set, "--output-dir", str(output_dir),
                 "--max-workers", str(args.ragas_max_workers)]
    if args.ragas_limit:
        ragas_cmd += ["--limit", str(args.ragas_limit)]
    if args.ragas_sample:
        ragas_cmd += ["--sample", str(args.ragas_sample)]

    retrieval_before = _latest("retrieval_eval_*.json", output_dir)
    ragas_before = _latest("ragas_eval_*.json", output_dir)
    if not args.skip_retrieval:
        _run(run_eval_cmd)
    if not args.skip_ragas:
        _run(ragas_cmd)

    retrieval_report = _latest("retrieval_eval_*.json", output_dir)
    ragas_report = _latest("ragas_eval_*.json", output_dir)
    retrieval_summary = (
        json.loads(retrieval_report.read_text(encoding="utf-8"))["summary"]
        if retrieval_report and retrieval_report != retrieval_before
        else None
    )
    ragas_summary = (
        json.loads(ragas_report.read_text(encoding="utf-8"))["summary"]
        if ragas_report and ragas_report != ragas_before
        else None
    )
    if retrieval_summary is None and retrieval_report:
        retrieval_summary = json.loads(retrieval_report.read_text(encoding="utf-8"))["summary"]
    if ragas_summary is None and ragas_report:
        ragas_summary = json.loads(ragas_report.read_text(encoding="utf-8"))["summary"]

    from eval.run_ragas_eval import build_markdown_report

    report = build_markdown_report(retrieval_summary, ragas_summary)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"pipeline_report_{stamp}.md"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\n=== 流水线完成 ===\n总报告已写入 {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
