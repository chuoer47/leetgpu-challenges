"""Validate each distinct imported solution and save per-submission results."""

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", default="1,2,3", help="One independent worker per physical GPU")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.manifest.resolve().parent / "validation"
    output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for submission in manifest["submissions"]:
        # Hash contents/names, not the per-submission parent directory.
        digest = hashlib.sha256(
            json.dumps(
                [(Path(f["path"]).name, f["sha256"]) for f in submission["files"]], sort_keys=True
            ).encode()
        ).hexdigest()
        key = f"{submission['challengeId']}_{submission['language']}_{digest[:16]}"
        groups.setdefault(key, {"key": key, "submission": submission, "equivalent": []})[
            "equivalent"
        ].append(submission["id"])
    jobs = queue.Queue()
    for group in groups.values():
        jobs.put(group)
    results = []

    def worker(gpu):
        while True:
            try:
                group = jobs.get_nowait()
            except queue.Empty:
                return
            submission = group["submission"]
            key = group["key"]
            record_path = output / f"{key}.result.json"
            if args.resume and record_path.exists():
                results.append(json.loads(record_path.read_text(encoding="utf-8")))
                continue
            solution = ROOT / submission["files"][0]["path"]
            timing = (output / f"{key}.timing.json").resolve()
            command = [
                sys.executable,
                str(ROOT / "practice/run.py"),
                str(submission["challengeId"]),
                "--language",
                submission["language"],
                "--solution",
                str(solution),
                "--runs",
                str(args.runs),
                "--json",
                str(timing),
            ]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1")
            start = time.monotonic()
            try:
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                log = process.stdout
                status = "passed" if process.returncode == 0 else "failed"
                code = process.returncode
            except subprocess.TimeoutExpired as error:
                log = error.stdout or b""
                if isinstance(log, bytes):
                    log = log.decode("utf-8", errors="replace")
                status, code = "timeout", None
            log_path = output / f"{key}.log"
            log_path.write_text(log, encoding="utf-8")
            result = {
                "key": key,
                "challenge_id": submission["challengeId"],
                "language": submission["language"],
                "submission_ids": group["equivalent"],
                "source": solution.relative_to(ROOT).as_posix(),
                "status": status,
                "exit_code": code,
                "gpu": gpu,
                "duration_seconds": time.monotonic() - start,
                "log": log_path.relative_to(ROOT).as_posix(),
            }
            if status == "passed":
                report = json.loads(timing.read_text(encoding="utf-8"))
                result.update(
                    {
                        "passed_cases": report["passed_cases"],
                        "median_wall_ms": report["median_wall_ms"],
                        "median_cuda_event_ms": report["median_cuda_event_ms"],
                    }
                )
            record_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            results.append(result)
            print(
                f"[{len(results)}/{len(groups)}] #{submission['challengeId']} "
                f"{status} GPU={gpu} {key}",
                flush=True,
            )

    gpus = args.gpus.split(",")
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    results.sort(key=lambda r: (r["challenge_id"], r["key"]))
    summary = {
        "unique_solutions": len(results),
        "accepted_submissions": len(manifest["submissions"]),
        "passed": sum(r["status"] == "passed" for r in results),
        "failed": sum(r["status"] == "failed" for r in results),
        "timeout": sum(r["status"] == "timeout" for r in results),
        "results": results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
