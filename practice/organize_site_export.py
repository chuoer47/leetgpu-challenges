"""Organize cached, authenticated LeetGPU exports without storing credentials."""

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--date", default="2026-10-28")
    args = parser.parse_args()
    cache = args.cache
    archive = ROOT / "site-archive" / args.date
    progress = json.loads((cache / "progress.json").read_text(encoding="utf-8"))
    completed = sorted(
        int(k) for k, v in progress["progressByChallengeId"].items() if v == "completed"
    )
    catalog = json.loads((cache / "catalog.json").read_text(encoding="utf-8"))
    titles = {c["id"]: c["title"] for c in catalog["challenges"]}
    folders = {
        int(p.parent.name.split("_")[0]): p.parent
        for p in (ROOT / "challenges").glob("*/*/challenge.py")
    }
    versions = defaultdict(list)
    for source in sorted((cache / "codes").glob("*.json")):
        submission = json.loads(source.read_text(encoding="utf-8"))
        if submission["status"] != "success" or submission["challengeId"] not in completed:
            continue
        folder = folders[submission["challengeId"]]
        destination = ROOT / "solutions" / folder.parent.name / folder.name
        stamp = submission["createdAt"].replace("-", "").replace(":", "").split(".")[0]
        version = destination / "submissions" / f"{stamp}_{submission['id'][:8]}"
        version.mkdir(parents=True, exist_ok=True)
        file_records = []
        for file in submission["files"]:
            name = Path(file["name"])
            if name.is_absolute() or ".." in name.parts or ":" in str(name):
                raise ValueError(f"Unsafe submission filename: {name}")
            target = version / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = file["content"].encode("utf-8")
            if target.exists() and target.read_bytes() != content:
                raise FileExistsError(f"Archived source changed: {target}")
            target.write_bytes(content)
            file_records.append(
                {
                    "path": target.relative_to(ROOT).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        metadata = {k: v for k, v in submission.items() if k != "files"}
        metadata["files"] = file_records
        metadata["source"] = "https://leetgpu.com/"
        write_json(version / "submission.json", metadata)
        versions[submission["challengeId"]].append(metadata)

    questions, all_versions = [], []
    for number in completed:
        if number not in versions:
            raise ValueError(f"No accepted source exported for completed challenge {number}")
        folder = folders[number]
        destination = ROOT / "solutions" / folder.parent.name / folder.name
        by_language = defaultdict(list)
        for version in versions[number]:
            by_language[version["language"]].append(version)
            all_versions.append(version)
        selected = []
        for language, candidates in sorted(by_language.items()):
            candidates.sort(key=lambda v: v["createdAt"], reverse=True)
            latest = candidates[0]
            extension = {
                "cuda": "cu",
                "pytorch": "pytorch.py",
                "triton": "triton.py",
                "jax": "jax.py",
                "mojo": "mojo",
                "cute": "cute.py",
            }[language]
            source = ROOT / latest["files"][0]["path"]
            target = destination / f"solution.{extension}"
            if target.exists() and target.read_bytes() != source.read_bytes():
                target = destination / f"site_latest.{extension}"
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise FileExistsError(f"Will not overwrite {target}")
            shutil.copyfile(source, target)
            selected.append(
                {
                    "language": language,
                    "submission_id": latest["id"],
                    "path": target.relative_to(ROOT).as_posix(),
                }
            )
        detail = cache / "challenges" / f"{number}.json"
        if detail.exists():
            target = archive / "challenges" / detail.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(detail, target)
        question = {
            "id": number,
            "title": titles[number],
            "challenge": folder.relative_to(ROOT).as_posix(),
            "solutions": destination.relative_to(ROOT).as_posix(),
            "selected": selected,
            "accepted_submissions": len(versions[number]),
        }
        questions.append(question)
        lines = [
            f"# {number}. {titles[number]}",
            "",
            "从本人 LeetGPU 账号导入；归档保留网站成功提交的原始内容。",
            "",
            f"题目：`{question['challenge']}/challenge.html`",
            "",
            "默认 solution 是每种语言最近一次网站成功提交；submissions/ 保留历史版本。",
            "网站曾通过不代表一定通过当前仓库新增的测试；本地结果见总测试报告。",
            "",
            "| 提交时间（UTC） | 语言 | 原 GPU | 提交 ID |",
            "|---|---|---|---|",
        ]
        for version in sorted(versions[number], key=lambda v: v["createdAt"], reverse=True):
            lines.append(
                f"| {version['createdAt']} | {version['language']} | "
                f"{version['accelerator']} | {version['id']} |"
            )
        (destination / "SITE_IMPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    histories = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((cache / "histories").glob("*.json"))
    ]
    manifest = {
        "source": "https://leetgpu.com/",
        "export_date": args.date,
        "completed_count": len(completed),
        "accepted_submission_count": len(all_versions),
        "questions": questions,
        "submissions": all_versions,
        "history_filters": [
            {k: h[k] for k in ("id", "language", "accelerator")} for h in histories
        ],
    }
    write_json(archive / "manifest.json", manifest)
    write_json(archive / "submission_history.json", histories)
    write_json(archive / "progress.json", progress)
    index = [
        "# 网站已通过题目导入",
        "",
        f"导入日期：{args.date}（Asia/Shanghai）",
        "",
        f"共 {len(completed)} 题，{len(all_versions)} 份成功提交；每题保留原始代码及提交元数据。",
        "",
        "| 题号 | 题目 | 成功提交数 | 默认解法 |",
        "|---|---|---:|---|",
    ]
    for q in questions:
        links = " / ".join(f"[{s['language']}](../../{s['path']})" for s in q["selected"])
        index.append(f"| {q['id']} | {q['title']} | {q['accepted_submissions']} | {links} |")
    index.extend(
        [
            "",
            "批量测试：",
            "",
            "```bash",
            "source practice/env.sh",
            f"python practice/test_imported.py --manifest site-archive/{args.date}/manifest.json",
            "```",
            "",
            "批量工具对每份不同的代码单独运行；重复代码共用结果，并保留对应关系。",
            "原始代码不自动修改。结果和失败日志写入同一归档目录的 validation/。",
        ]
    )
    (archive / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"Organized {len(completed)} challenges, {len(all_versions)} accepted submissions")
    print(archive / "manifest.json")


if __name__ == "__main__":
    main()
