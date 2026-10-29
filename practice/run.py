#!/usr/bin/env python3
"""Run single-GPU LeetGPU CUDA/PyTorch/Triton solutions locally."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "challenges"))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def challenge_path(value):
    path = Path(value)
    if (path / "challenge.py").is_file():
        return path.resolve()
    matches = [p for p in (ROOT / "challenges").glob("*/*") if p.name.split("_")[0] == value]
    if len(matches) != 1:
        raise ValueError("Pass a challenge number or directory containing challenge.py")
    return matches[0]


def materialize(case, torch, device):
    from core.challenge_base import FullTensor, OutTensor, RandIntTensor, RandnTensor, RandTensor

    result = {}
    for name, value in case.items():
        if isinstance(value, (RandTensor, RandnTensor, RandIntTensor, FullTensor, OutTensor)):
            dtype = getattr(torch, value.dtype)
            if isinstance(value, RandIntTensor):
                value = torch.randint(
                    value.low, value.high, value.shape, device=device, dtype=dtype
                )
            else:
                tensor = torch.empty(value.shape, device=device, dtype=dtype)
                if isinstance(value, RandTensor):
                    tensor.uniform_(value.low, value.high)
                elif isinstance(value, RandnTensor):
                    tensor.normal_(value.mean, value.std)
                elif isinstance(value, FullTensor):
                    tensor.fill_(value.value)
                value = tensor
        result[name] = value
    return result


def copy_case(case, torch):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in case.items()}


def poison_outputs(case, signature, torch):
    # Empty starters must fail even when a correct output happens to be all zeros.
    for name, (_, direction) in signature.items():
        value = case[name]
        if direction == "out" and isinstance(value, torch.Tensor):
            value.fill_(float("nan") if value.is_floating_point() else 123)


def build_solver(path, language, signature, torch):
    if language != "cuda":
        return load_module(path, "user_solution").solve
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc is missing; activate a CUDA toolkit environment first")
    major, minor = torch.cuda.get_device_capability()
    key = hashlib.sha256(
        str(path).encode() + path.read_bytes() + f"{major}{minor}".encode()
    ).hexdigest()[:16]
    library_path = ROOT / "build" / f"solution_{key}.so"
    library_path.parent.mkdir(exist_ok=True)
    compiler_flags = ["-ccbin", os.environ["NVCC_CCBIN"]] if os.getenv("NVCC_CCBIN") else []
    cuda_home = Path(os.environ.get("CUDA_HOME", str(Path(nvcc).resolve().parent.parent)))
    # Compile every invocation so changes in included headers cannot reuse a stale library.
    subprocess.run(
        [
            nvcc,
            "-O3",
            "-std=c++17",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "--cudart=shared",
            "--cudadevrt=none",
            f"-L{cuda_home / 'lib'}",
            f"-L{cuda_home / 'lib64'}",
            f"-arch=sm_{major}{minor}",
            *compiler_flags,
            str(path),
            "-o",
            str(library_path),
        ],
        check=True,
    )
    library = ctypes.CDLL(str(library_path))
    function = library.solve
    function.argtypes = [kind for kind, _ in signature.values()]
    function.restype = None

    def solve(*values):
        converted = [
            ctypes.cast(value.data_ptr(), kind) if isinstance(value, torch.Tensor) else value
            for value, (kind, _) in zip(values, signature.values())
        ]
        function(*converted)

    return solve


def verify(case, challenge, solve, signature, torch):
    expected = copy_case(case, torch)
    actual = copy_case(case, torch)
    poison_outputs(expected, signature, torch)
    poison_outputs(actual, signature, torch)
    names = list(signature)
    challenge.reference_impl(*(expected[k] for k in names))
    solve(*(actual[k] for k in names))
    torch.cuda.synchronize()
    for name, (_, direction) in signature.items():
        if direction in ("out", "inout"):
            torch.testing.assert_close(
                actual[name],
                expected[name],
                atol=challenge.atol,
                rtol=challenge.rtol,
                equal_nan=False,
                msg=lambda msg: f"Output {name}: {msg}",
            )
        elif isinstance(case[name], torch.Tensor):
            torch.testing.assert_close(actual[name], case[name], atol=0, rtol=0)


def git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def select_gpu(gpu):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    return os.environ["CUDA_VISIBLE_DEVICES"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("challenge", help="Challenge number or directory")
    parser.add_argument("--language", choices=["cuda", "pytorch", "triton"], default="cuda")
    parser.add_argument("--solution", type=Path)
    parser.add_argument("--init", action="store_true", help="Copy starter into solutions/")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Physical GPU index; overrides CUDA_VISIBLE_DEVICES (otherwise env, then 1)",
    )
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path, help="Also save machine-readable results here")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or (args.gpu is not None and args.gpu < 0):
        parser.error("runs must be positive; warmup and gpu must be nonnegative")
    folder = challenge_path(args.challenge)
    extension = {"cuda": "cu", "pytorch": "pytorch.py", "triton": "triton.py"}[args.language]
    default = ROOT / "solutions" / folder.parent.name / folder.name / f"solution.{extension}"
    solution = (args.solution or default).resolve()
    if args.init:
        solution.parent.mkdir(parents=True, exist_ok=True)
        if solution.exists():
            raise FileExistsError(f"Will not overwrite {solution}")
        shutil.copyfile(folder / "starter" / f"starter.{extension}", solution)
        print(f"Starter copied to {solution}; implement solve, then run without --init")
        return
    if not solution.is_file():
        raise FileNotFoundError(f"Missing {solution}; use --init first")
    visible_devices = select_gpu(args.gpu)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    device = "cuda:0"
    torch.cuda.set_device(0)
    print(
        f"CUDA_VISIBLE_DEVICES={visible_devices}; using logical cuda:0 "
        f"({torch.cuda.get_device_name(0)}, UUID={torch.cuda.get_device_properties(0).uuid})",
        flush=True,
    )
    torch.manual_seed(args.seed)
    challenge = load_module(folder / "challenge.py", "local_challenge").Challenge(device=device)
    if challenge.num_gpus != 1:
        raise ValueError("This runner currently supports single-GPU challenges only")
    signature = challenge.get_solve_signature()
    solve = build_solver(solution, args.language, signature, torch)
    passed = 0
    with torch.no_grad():
        example = challenge.generate_example_test()
        examples = example if isinstance(example, list) else [example]
        for case in examples + challenge.generate_functional_test():
            verify(materialize(case, torch, device), challenge, solve, signature, torch)
            passed += 1
        # Validate the large performance case too: small functional tests miss size-dependent bugs.
        case = materialize(challenge.generate_performance_test(), torch, device)
        verify(case, challenge, solve, signature, torch)
        passed += 1
        del case
        print(f"PASS: {passed} cases (example + functional + performance)")
        if args.check_only:
            return
        properties = torch.cuda.get_device_properties(0)
        flush_bytes = max(64 * 1024 * 1024, 2 * getattr(properties, "L2_cache_size", 0))
        flush = torch.empty(flush_bytes, device=device, dtype=torch.uint8)
        wall_ms, event_ms = [], []
        for iteration in range(args.warmup + args.runs):
            # Fresh inputs handle inout correctly; allocation/generation/flush are outside timing.
            case = materialize(challenge.generate_performance_test(), torch, device)
            poison_outputs(case, signature, torch)
            values = [case[name] for name in signature]
            flush.zero_()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin = time.perf_counter()
            start.record()
            solve(*values)
            end.record()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - begin) * 1000
            if iteration >= args.warmup:
                wall_ms.append(elapsed)
                event_ms.append(start.elapsed_time(end))
            del values, case
    stamp = datetime.now(timezone.utc)
    report = {
        "timestamp_utc": stamp.isoformat(),
        "challenge": str(folder.relative_to(ROOT)),
        "solution": str(solution),
        "solution_sha256": hashlib.sha256(solution.read_bytes()).hexdigest(),
        "language": args.language,
        "git_commit": git_revision(),
        "gpu": properties.name,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seed": args.seed,
        "passed_cases": passed,
        "runs": args.runs,
        "warmup": args.warmup,
        "l2_flush_bytes": flush_bytes,
        "wall_ms": wall_ms,
        "cuda_event_ms": event_ms,
        "median_wall_ms": statistics.median(wall_ms),
        "median_cuda_event_ms": statistics.median(event_ms),
        "timing_note": "Local solve + sync wall time and CUDA event time; compilation excluded. "
        "Wall time includes event instrumentation; not an official platform score.",
    }
    output = args.json or ROOT / "benchmarks" / "local" / (
        f"{folder.name}_{args.language}_{stamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"GPU: {properties.name}; median solve + sync: {report['median_wall_ms']:.4f} ms; "
        f"CUDA event: {report['median_cuda_event_ms']:.4f} ms"
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
