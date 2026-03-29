#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


METRIC_PATTERNS: Dict[str, re.Pattern[str]] = {
    "successful_requests": re.compile(r"Successful requests:\s*([0-9]+)"),
    "benchmark_duration_s": re.compile(
        r"Benchmark duration \(s\):\s*([0-9]+(?:\.[0-9]+)?)"),
    "request_throughput_req_s": re.compile(
        r"Request throughput \(req/s\):\s*([0-9]+(?:\.[0-9]+)?)"),
    "input_token_throughput_tok_s": re.compile(
        r"Input token throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)"),
    "output_token_throughput_tok_s": re.compile(
        r"Output token throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)"),
    "total_token_throughput_tok_s": re.compile(
        r"Total token throughput \(tok/s\):\s*([0-9]+(?:\.[0-9]+)?)"),
    "mean_e2e_latency_ms": re.compile(
        r"Mean E2E Latency \(ms\):\s*([0-9]+(?:\.[0-9]+)?)"),
    "mean_ttft_ms": re.compile(
        r"Mean TTFT \(ms\):\s*([0-9]+(?:\.[0-9]+)?)"),
    "p99_ttft_ms": re.compile(
        r"P99 TTFT \(ms\):\s*([0-9]+(?:\.[0-9]+)?)"),
    "mean_itl_ms": re.compile(
        r"Mean ITL \(ms\):\s*([0-9]+(?:\.[0-9]+)?)"),
    "p99_itl_ms": re.compile(
        r"P99 ITL \(ms\):\s*([0-9]+(?:\.[0-9]+)?)"),
}


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_metrics(output_text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, pat in METRIC_PATTERNS.items():
        m = pat.search(output_text)
        if not m:
            continue
        raw = m.group(1)
        if raw.isdigit():
            metrics[key] = float(int(raw))
        else:
            metrics[key] = float(raw)
    return metrics


def _load_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _diff(after: Dict[str, Any], before: Optional[Dict[str, Any]], key: str) -> float:
    a = _to_float(after.get(key))
    b = _to_float((before or {}).get(key))
    return a - b


def _compute_delta_report(
    before_report: Optional[Dict[str, Any]],
    after_report: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not after_report:
        return None

    before_calls = _to_int((before_report or {}).get("total_free_calls"), default=0)
    after_calls = _to_int(after_report.get("total_free_calls"), default=0)
    if after_calls < before_calls:
        # Server likely restarted; treat this run as a fresh window.
        before_calls = 0

    all_samples = after_report.get("memory_trend_samples", []) or []
    run_samples = [s for s in all_samples if _to_int(s.get("call")) > before_calls]
    blocks_per_call = [_to_float(s.get("num_blocks")) for s in run_samples]
    unmap_samples = [s for s in run_samples if _to_int(s.get("num_unmapped")) > 0]

    if run_samples:
        total_free_calls = len(run_samples)
        total_blocks_freed = int(sum(_to_int(s.get("num_blocks")) for s in run_samples))
        total_pages_returned = int(sum(_to_int(s.get("pages_returned")) for s in run_samples))
        total_free_time_ms = float(sum(_to_float(s.get("elapsed_ms")) for s in run_samples))
    else:
        # Fallback if sample list is unavailable.
        total_free_calls = max(0, after_calls - before_calls)
        total_blocks_freed = int(max(0.0, _diff(after_report, before_report, "total_blocks_freed")))
        total_pages_returned = int(max(0.0, _diff(after_report, before_report, "total_pages_returned")))
        total_free_time_ms = max(0.0, _diff(after_report, before_report, "total_free_time_ms"))

    total_pages_unmapped = int(sum(_to_int(s.get("num_unmapped")) for s in unmap_samples))
    total_unmap_time_ms = float(sum(_to_float(s.get("unmap_time_ms")) for s in unmap_samples))

    returned_ids = sorted({
        _to_int(pid) for s in run_samples for pid in (s.get("returned_page_ids", []) or [])
    })
    unmapped_ids = sorted({
        _to_int(pid) for s in run_samples for pid in (s.get("unmapped_page_ids", []) or [])
    })

    block_mem_size = _to_int((after_report.get("allocator_meta") or {}).get("block_mem_size_bytes"))
    block_mapped_size = _to_int((after_report.get("allocator_meta") or {}).get("block_mapped_total_bytes"))
    total_freed_logical_bytes = int(sum(_to_int(s.get("freed_logical_bytes")) for s in run_samples))
    total_freed_mapped_bytes_est = int(sum(_to_int(s.get("freed_mapped_bytes_est")) for s in run_samples))
    if total_freed_logical_bytes == 0 and block_mem_size > 0 and total_blocks_freed > 0:
        total_freed_logical_bytes = total_blocks_freed * block_mem_size
    if total_freed_mapped_bytes_est == 0 and block_mapped_size > 0 and total_blocks_freed > 0:
        total_freed_mapped_bytes_est = total_blocks_freed * block_mapped_size

    unmap_events = [{
        "call": _to_int(s.get("call")),
        "elapsed_s": _to_float(s.get("elapsed_s")),
        "num_blocks": _to_int(s.get("num_blocks")),
        "pages_returned": _to_int(s.get("pages_returned")),
        "pages_unmapped": _to_int(s.get("num_unmapped")),
        "unmap_time_ms": _to_float(s.get("unmap_time_ms")),
    } for s in unmap_samples]

    return {
        "call_window": {
            "before_total_free_calls": before_calls,
            "after_total_free_calls": after_calls,
            "start_call_exclusive": before_calls,
            "end_call_inclusive": after_calls,
        },
        "total_free_calls": total_free_calls,
        "total_blocks_freed": total_blocks_freed,
        "total_pages_returned": total_pages_returned,
        "total_free_time_ms": round(total_free_time_ms, 3),
        "avg_time_per_call_ms": round(total_free_time_ms / total_free_calls, 4)
        if total_free_calls else 0.0,
        "allocator_meta": after_report.get("allocator_meta"),
        "free_block_stats": {
            "mean_blocks_per_call": round(total_blocks_freed / total_free_calls, 4)
            if total_free_calls else 0.0,
            "p50_blocks_per_call": round(_percentile(blocks_per_call, 50), 4)
            if blocks_per_call else 0.0,
            "p95_blocks_per_call": round(_percentile(blocks_per_call, 95), 4)
            if blocks_per_call else 0.0,
            "p99_blocks_per_call": round(_percentile(blocks_per_call, 99), 4)
            if blocks_per_call else 0.0,
            "max_blocks_per_call": max(blocks_per_call) if blocks_per_call else 0.0,
            "total_freed_logical_bytes": total_freed_logical_bytes,
            "total_freed_mapped_bytes_est": total_freed_mapped_bytes_est,
        },
        "unmap_stats": {
            "total_unmap_calls": len(unmap_samples),
            "total_pages_unmapped": total_pages_unmapped,
            "total_unmap_time_ms": round(total_unmap_time_ms, 3),
            "avg_unmap_time_ms": round(total_unmap_time_ms / len(unmap_samples), 4)
            if unmap_samples else 0.0,
            "avg_unmap_time_per_page_ms": round(total_unmap_time_ms / total_pages_unmapped, 4)
            if total_pages_unmapped > 0 else 0.0,
        },
        "page_id_stats": {
            "unique_returned_pages": len(returned_ids),
            "unique_unmapped_pages": len(unmapped_ids),
            "returned_page_ids": returned_ids,
            "unmapped_page_ids": unmapped_ids,
        },
        "unmap_events": unmap_events,
        "num_samples_in_run_window": len(run_samples),
    }


def _wait_for_new_report(path: Path,
                         min_mtime: float,
                         timeout_s: float,
                         poll_s: float = 0.5) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            try:
                if path.stat().st_mtime >= min_mtime:
                    return True
            except OSError:
                pass
        time.sleep(poll_s)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one `vllm bench serve` case and save a per-case JSON report "
            "that includes both bench metrics and kvcached free debug report."
        ))
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-name", default="sharegpt")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument(
        "--free-report-json",
        default="/root/kvcached/kvcached_free_debug_report.json",
        help="Path where server writes KVCACHED_FREE_DEBUG_REPORT JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default="/root/kvcached/results/bench_runs",
        help="Directory to save per-run JSON/log files.",
    )
    parser.add_argument(
        "--wait-report-timeout-seconds",
        type=float,
        default=20.0,
        help="Max wait time for idle report JSON refresh after bench exits.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag appended to output filename.",
    )
    parser.add_argument(
        "--extra-bench-args",
        default="",
        help="Extra args passed to `vllm bench serve` as one shell-style string.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    free_report_path = Path(args.free_report_json).resolve()

    before_report = _load_json_if_exists(free_report_path)

    pre_mtime = 0.0
    if free_report_path.exists():
        try:
            pre_mtime = free_report_path.stat().st_mtime
        except OSError:
            pre_mtime = 0.0

    cmd: List[str] = [
        "vllm",
        "bench",
        "serve",
        "--model",
        args.model,
        "--dataset-name",
        args.dataset_name,
        "--dataset-path",
        args.dataset_path,
        "--request-rate",
        str(args.request_rate),
        "--num-prompts",
        str(args.num_prompts),
        "--port",
        str(args.port),
    ]
    if args.extra_bench_args.strip():
        cmd.extend(shlex.split(args.extra_bench_args.strip()))

    start_ts = datetime.now().isoformat(timespec="seconds")
    run_id = f"r{str(args.request_rate).replace('.', 'p')}_n{args.num_prompts}_{_now_tag()}"
    if args.tag:
        run_id = f"{run_id}_{args.tag}"

    print(f"[run] {run_id}")
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        output_lines.append(line)

    ret = proc.wait()
    end_ts = datetime.now().isoformat(timespec="seconds")
    bench_output = "".join(output_lines)
    bench_metrics = _parse_metrics(bench_output)

    # Wait for refreshed report from server side (idle dump is periodic).
    min_mtime = pre_mtime + 1e-6
    report_refreshed = _wait_for_new_report(
        free_report_path,
        min_mtime=min_mtime,
        timeout_s=args.wait_report_timeout_seconds,
    )
    free_report_after = _load_json_if_exists(free_report_path)
    post_mtime = pre_mtime
    if free_report_path.exists():
        try:
            post_mtime = free_report_path.stat().st_mtime
        except OSError:
            post_mtime = pre_mtime
    free_report_delta = _compute_delta_report(before_report, free_report_after)

    log_path = out_dir / f"{run_id}.bench.log"
    run_json_path = out_dir / f"{run_id}.json"
    log_path.write_text(bench_output, encoding="utf-8")

    run_record = {
        "run_id": run_id,
        "start_time": start_ts,
        "end_time": end_ts,
        "bench_cmd": cmd,
        "bench_exit_code": ret,
        "run_meta": {
            "model": args.model,
            "dataset_name": args.dataset_name,
            "dataset_path": args.dataset_path,
            "port": args.port,
            "request_rate": args.request_rate,
            "num_prompts": args.num_prompts,
            "tag": args.tag,
        },
        "bench_metrics": bench_metrics,
        "kvcached_free_report_path": str(free_report_path),
        "kvcached_free_report_found": free_report_after is not None,
        "kvcached_free_report_refreshed": bool(report_refreshed),
        "kvcached_free_report_mtime_before": pre_mtime,
        "kvcached_free_report_mtime_after": post_mtime,
        "kvcached_free_report_before": before_report,
        "kvcached_free_report_after": free_report_after,
        "kvcached_free_report_delta": free_report_delta,
        "bench_log_path": str(log_path),
    }
    run_json_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")

    print(f"[saved] {run_json_path}")
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
