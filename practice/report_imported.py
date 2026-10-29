"""Build a readable report from local validation of the original website submissions."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    archive = args.manifest.resolve().parent
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = json.loads((archive / "validation/summary.json").read_text(encoding="utf-8"))
    diagnostic_file = archive / "validation/zero_output_diagnostic.json"
    diagnostics = (
        json.loads(diagnostic_file.read_text(encoding="utf-8")) if diagnostic_file.exists() else []
    )
    zero_passed = {d["key"] for d in diagnostics if d["zero_initialized_outputs_pass"]}
    by_submission = {sid: r for r in summary["results"] for sid in r["submission_ids"]}
    rows, failures = [], []
    questions_with_pass = 0
    for question in manifest["questions"]:
        versions = [s for s in manifest["submissions"] if s["challengeId"] == question["id"]]
        results = {by_submission[s["id"]]["key"]: by_submission[s["id"]] for s in versions}
        passing = [r for r in results.values() if r["status"] == "passed"]
        questions_with_pass += bool(passing)
        selected = [by_submission[s["submission_id"]] for s in question["selected"]]
        default_ok = all(r["status"] == "passed" for r in selected)
        times = " / ".join(
            f"{r['median_wall_ms']:.4f}" for r in selected if r["status"] == "passed"
        )
        state = "通过" if default_ok else "默认版本未通过"
        rows.append(
            f"| {question['id']} | {question['title']} | {state} | "
            f"{len(passing)}/{len(results)} | {times or '—'} |"
        )
    for result in summary["results"]:
        if result["status"] == "passed":
            continue
        log = (ROOT / result["log"]).read_text(encoding="utf-8")
        reason = next(
            (line for line in log.splitlines() if line.startswith("AssertionError:")), None
        )
        if not reason:
            reason = next(
                (line.strip() for line in log.splitlines() if "error:" in line.lower()),
                f"{result['status']}, exit code {result['exit_code']}",
            )
        if result["key"] in zero_passed:
            reason = "依赖输出缓冲区初始为零；明确清零后全部用例通过，严格检查未通过"
        elif "invalid configuration argument" in log:
            reason = "CUDA kernel 启动配置无效；反转数组在 N=1 时启动了 0 个 block"
        elif result["challenge_id"] == 37:
            reason = "功能用例通过；512×512、P=3 的性能用例与参考结果超出 1e-4 误差阈值"
        failures.append(
            {
                "challenge_id": result["challenge_id"],
                "source": result["source"],
                "reason": reason,
                "log": result["log"],
                "key": result["key"],
            }
        )
    report = {
        "completed_questions": manifest["completed_count"],
        "questions_with_passing_version": questions_with_pass,
        "unique_solutions": summary["unique_solutions"],
        "accepted_submissions": manifest["accepted_submission_count"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "timeout": summary["timeout"],
        "zero_output_compatible": len(zero_passed),
        "other_failed": summary["failed"] - len(zero_passed),
        "failures": failures,
    }
    (archive / "REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    text = [
        "# 已通过题目的 4090 本地验证报告",
        "",
        f"导入 {manifest['completed_count']} 题、{manifest['accepted_submission_count']} 份网站成功提交；"
        f"去重后 {summary['unique_solutions']} 份代码全部执行本地测试。",
        "",
        f"通过 {summary['passed']}，失败 {summary['failed']}，超时 {summary['timeout']}。"
        f"其中 {questions_with_pass} 道题至少有一个本地通过版本。",
        "",
        "环境：RTX 4090 D，PyTorch 2.5.1+cu124，CUDA Toolkit 12.4，GCC 13。",
        "测试使用仓库当前示例、功能用例和性能用例；通过后预热 2 次、测量 15 次。",
        "严格检查额外使用非零/NaN 初始化 out 缓冲区，检查实现能否独立覆盖输出。",
        "此初始化条件不等于网站运行时的初始化方式；兼容性诊断另行将 out 明确清零。",
        f"有 {len(zero_passed)} 份严格未通过的代码在输出预先清零后通过全部用例；"
        f"另有 {summary['failed'] - len(zero_passed)} 份需要排查边界或数值误差。",
        "不同 GPU 各运行一个任务，GPU 0 保留给已有任务。",
        "墙钟时间包含 solve、同步和事件记录开销；属于本机比较结果，不是网站成绩。",
        "",
        "默认解法保留每种语言最近一次网站成功提交，未改写任何原始算法。",
        "",
        "| 题号 | 题目 | 默认解法 | 通过版本/去重版本 | 默认耗时中位数 ms |",
        "|---|---|---|---:|---:|",
        *rows,
        "",
        "## 未通过的历史版本",
        "",
    ]
    for failure in failures:
        source = failure["source"]
        log = failure["log"]
        text.extend(
            [
                f"- #{failure['challenge_id']}：`{failure['reason']}`",
                f"  [原始代码](../../{source}) · [完整日志](../../{log})",
            ]
        )
    if not failures:
        text.append("全部通过。")
    text.extend(
        [
            "",
            "所有原始提交都保存在 solutions/ 各题的 submissions/ 中。",
            "validation/summary.json 包含每个版本、原提交 ID 和运行结果。",
            "仅网站曾成功、当前测试未通过的版本仍完整保留，便于排查测试变化和边界问题。",
        ]
    )
    (archive / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
