#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def _parse_float_list(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [float(x) for x in parts]


def _parse_int_list(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(x) for x in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a matrix of vllm bench serve cases and capture one JSON per case."
        ))
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-name", default="sharegpt")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--request-rates",
        required=True,
        help="Comma-separated list, e.g. 5,10,20,40",
    )
    parser.add_argument(
        "--num-prompts-list",
        required=True,
        help="Comma-separated list, e.g. 500,1000,2000,4000",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--runner-script",
        default="tools/run_vllm_bench_and_capture.py",
        help="Path to single-run capture script.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to call runner script.",
    )
    parser.add_argument(
        "--free-report-json",
        default="/root/kvcached/kvcached_free_debug_report.json",
    )
    parser.add_argument(
        "--out-dir",
        default="/root/kvcached/results/bench_runs",
    )
    parser.add_argument(
        "--wait-report-timeout-seconds",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--extra-bench-args",
        default="",
        help="Extra args passed through to vllm bench serve.",
    )
    parser.add_argument(
        "--tag-prefix",
        default="matrix",
        help="Tag prefix to identify this batch in output files.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one case fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_rates = _parse_float_list(args.request_rates)
    num_prompts_list = _parse_int_list(args.num_prompts_list)
    runner = Path(args.runner_script).resolve()

    if not runner.exists():
        print(f"Runner script not found: {runner}")
        return 2

    total_cases = len(request_rates) * len(num_prompts_list) * max(args.repeats, 1)
    case_idx = 0
    failures = 0

    for repeat_idx in range(1, max(args.repeats, 1) + 1):
        for rate in request_rates:
            for num_prompts in num_prompts_list:
                case_idx += 1
                tag = f"{args.tag_prefix}_rep{repeat_idx}"
                cmd = [
                    args.python_exe,
                    str(runner),
                    "--model",
                    args.model,
                    "--dataset-name",
                    args.dataset_name,
                    "--dataset-path",
                    args.dataset_path,
                    "--request-rate",
                    str(rate),
                    "--num-prompts",
                    str(num_prompts),
                    "--port",
                    str(args.port),
                    "--free-report-json",
                    args.free_report_json,
                    "--out-dir",
                    args.out_dir,
                    "--wait-report-timeout-seconds",
                    str(args.wait_report_timeout_seconds),
                    "--tag",
                    tag,
                ]
                if args.extra_bench_args.strip():
                    cmd.extend([
                        "--extra-bench-args",
                        args.extra_bench_args.strip(),
                    ])

                print(f"[{case_idx}/{total_cases}] rate={rate} num_prompts={num_prompts} repeat={repeat_idx}")
                print("[cmd] " + " ".join(shlex.quote(x) for x in cmd))
                ret = subprocess.call(cmd)
                if ret != 0:
                    failures += 1
                    print(f"[warn] case failed with exit code={ret}")
                    if args.stop_on_error:
                        return ret

    print(f"Matrix finished. total_cases={total_cases} failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

