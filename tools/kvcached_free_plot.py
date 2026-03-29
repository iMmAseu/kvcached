#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError(
        "matplotlib is required for plotting. Install it with: pip install matplotlib"
    ) from e


def _bytes_to_gb(v: Optional[int]) -> float:
    if v is None:
        return 0.0
    return float(v) / (1024.0**3)


def _linear_fit(xs: List[float], ys: List[float]) -> Optional[Tuple[float, float]]:
    """Return (slope, intercept) for y = slope*x + intercept."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    n = float(len(xs))
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _savefig(out_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_line_trend(samples: List[Dict[str, Any]], out_path: Path) -> None:
    calls = [int(s.get("call", 0)) for s in samples]
    num_blocks = [int(s.get("num_blocks", 0)) for s in samples]
    pages_returned = [int(s.get("pages_returned", 0)) for s in samples]
    num_unmapped = [int(s.get("num_unmapped", 0)) for s in samples]

    plt.figure(figsize=(11, 5))
    plt.plot(calls, num_blocks, label="free blocks per call", linewidth=1.7)
    plt.plot(calls, pages_returned, label="pages returned per call", linewidth=1.7)
    plt.plot(calls, num_unmapped, label="pages unmapped per call", linewidth=1.7)
    plt.xlabel("free() call #")
    plt.ylabel("count")
    plt.title("free()/unmap Trend by Call")
    plt.grid(alpha=0.25)
    plt.legend()
    _savefig(out_path)


def _plot_unmap_time_line(samples: List[Dict[str, Any]], out_path: Path) -> None:
    unmap_samples = [s for s in samples if int(s.get("num_unmapped", 0)) > 0]
    calls = [int(s.get("call", 0)) for s in unmap_samples]
    unmap_time_ms = [float(s.get("unmap_time_ms", 0.0)) for s in unmap_samples]
    num_unmapped = [int(s.get("num_unmapped", 0)) for s in unmap_samples]

    plt.figure(figsize=(11, 5))
    plt.plot(calls, unmap_time_ms, label="unmap time (ms)", linewidth=1.8, marker="o")
    plt.plot(calls, num_unmapped, label="unmapped pages", linewidth=1.5, marker="o")
    plt.xlabel("free() call # (UNMAP events only)")
    plt.ylabel("value")
    plt.title("UNMAP Event Trend")
    plt.grid(alpha=0.25)
    plt.legend()
    _savefig(out_path)


def _plot_memory_line(samples: List[Dict[str, Any]], out_path: Path) -> None:
    calls = [int(s.get("call", 0)) for s in samples]
    kv_used = [_bytes_to_gb(s.get("kvcached_used_bytes")) for s in samples]
    kv_reserved = [_bytes_to_gb(s.get("kvcached_reserved_bytes")) for s in samples]
    kv_total = [_bytes_to_gb(s.get("kvcached_total_mapped_bytes")) for s in samples]
    cuda_used = [_bytes_to_gb(s.get("cuda_used_bytes")) for s in samples]

    plt.figure(figsize=(11, 5))
    plt.plot(calls, kv_used, label="KV active (GB)", linewidth=1.7)
    plt.plot(calls, kv_reserved, label="KV reserved (GB)", linewidth=1.7)
    plt.plot(calls, kv_total, label="KV total mapped (GB)", linewidth=1.7)
    plt.plot(calls, cuda_used, label="CUDA used (GB)", linewidth=1.7)
    plt.xlabel("free() call #")
    plt.ylabel("GB")
    plt.title("Memory Trend by Call")
    plt.grid(alpha=0.25)
    plt.legend()
    _savefig(out_path)


def _plot_cumulative_line(samples: List[Dict[str, Any]], out_path: Path) -> None:
    calls = [int(s.get("call", 0)) for s in samples]

    cum_blocks: List[int] = []
    cum_pages_ret: List[int] = []
    cum_pages_unmap: List[int] = []
    b_acc = 0
    p_acc = 0
    u_acc = 0
    for s in samples:
        b_acc += int(s.get("num_blocks", 0))
        p_acc += int(s.get("pages_returned", 0))
        u_acc += int(s.get("num_unmapped", 0))
        cum_blocks.append(b_acc)
        cum_pages_ret.append(p_acc)
        cum_pages_unmap.append(u_acc)

    plt.figure(figsize=(11, 5))
    plt.plot(calls, cum_blocks, label="cumulative free blocks", linewidth=1.8)
    plt.plot(calls, cum_pages_ret, label="cumulative returned pages", linewidth=1.8)
    plt.plot(calls, cum_pages_unmap, label="cumulative unmapped pages", linewidth=1.8)
    plt.xlabel("free() call #")
    plt.ylabel("cumulative count")
    plt.title("Cumulative Trend")
    plt.grid(alpha=0.25)
    plt.legend()
    _savefig(out_path)


def _plot_unmap_scatter(samples: List[Dict[str, Any]], out_path: Path) -> None:
    unmap_samples = [s for s in samples if int(s.get("num_unmapped", 0)) > 0]
    xs = [float(s.get("num_unmapped", 0)) for s in unmap_samples]
    ys = [float(s.get("unmap_time_ms", 0.0)) for s in unmap_samples]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, alpha=0.75, label="UNMAP events")
    fit = _linear_fit(xs, ys)
    if fit is not None:
        slope, intercept = fit
        x_min, x_max = min(xs), max(xs)
        x_line = [x_min, x_max]
        y_line = [slope * x + intercept for x in x_line]
        plt.plot(x_line, y_line, color="red", linewidth=1.8,
                 label=f"linear fit: y={slope:.3f}x+{intercept:.3f}")
    plt.xlabel("unmapped pages")
    plt.ylabel("unmap time (ms)")
    plt.title("UNMAP Time vs Unmapped Pages")
    plt.grid(alpha=0.25)
    plt.legend()
    _savefig(out_path)


def _plot_hist_blocks(samples: List[Dict[str, Any]], out_path: Path) -> None:
    values = [int(s.get("num_blocks", 0)) for s in samples]
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30, alpha=0.85)
    plt.xlabel("free blocks per call")
    plt.ylabel("frequency")
    plt.title("Distribution of Free Blocks per Call")
    plt.grid(alpha=0.2)
    _savefig(out_path)


def _plot_hist_unmap_time(samples: List[Dict[str, Any]], out_path: Path) -> None:
    values = [
        float(s.get("unmap_time_ms", 0.0))
        for s in samples
        if int(s.get("num_unmapped", 0)) > 0
    ]
    if not values:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=24, alpha=0.85)
    plt.xlabel("unmap time (ms)")
    plt.ylabel("frequency")
    plt.title("Distribution of UNMAP Time")
    plt.grid(alpha=0.2)
    _savefig(out_path)


def _build_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    free_block_stats = report.get("free_block_stats", {}) or {}
    allocator_meta = report.get("allocator_meta", {}) or {}
    unmap_stats = report.get("unmap_stats", {}) or {}

    return {
        "elapsed_total_s": report.get("elapsed_total_s", 0),
        "total_free_calls": report.get("total_free_calls", 0),
        "total_blocks_freed": report.get("total_blocks_freed", 0),
        "total_pages_returned": report.get("total_pages_returned", 0),
        "total_free_time_ms": report.get("total_free_time_ms", 0),
        "total_unmap_calls": unmap_stats.get("total_unmap_calls", 0),
        "total_pages_unmapped": unmap_stats.get("total_pages_unmapped", 0),
        "total_unmap_time_ms": unmap_stats.get("total_unmap_time_ms", 0),
        "mean_blocks_per_call": free_block_stats.get("mean_blocks_per_call", 0),
        "p95_blocks_per_call": free_block_stats.get("p95_blocks_per_call", 0),
        "block_mem_size_bytes": allocator_meta.get("block_mem_size_bytes"),
        "block_mapped_total_bytes": allocator_meta.get("block_mapped_total_bytes"),
        "total_freed_logical_bytes": free_block_stats.get("total_freed_logical_bytes", 0),
        "total_freed_mapped_bytes_est": free_block_stats.get(
            "total_freed_mapped_bytes_est", 0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trend plots from kvcached free debug JSON report.")
    parser.add_argument(
        "--report-json",
        default="kvcached_free_debug_report.json",
        help="Path to kvcached free debug JSON report.")
    parser.add_argument(
        "--out-dir",
        default="kvcached_free_plots",
        help="Output directory for generated figures.")
    parser.add_argument(
        "--prefix",
        default="kvcached_free",
        help="Output filename prefix.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not report_path.exists():
        raise FileNotFoundError(f"Report JSON not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    samples = list(report.get("memory_trend_samples", []) or [])
    if not samples:
        print(
            "No memory_trend_samples found in report. "
            "Wait for an idle report or verify KVCACHED_FREE_DEBUG=1."
        )
        return 1

    prefix = args.prefix
    _plot_line_trend(samples, out_dir / f"{prefix}_line_trend.png")
    _plot_unmap_time_line(samples, out_dir / f"{prefix}_line_unmap_events.png")
    _plot_memory_line(samples, out_dir / f"{prefix}_line_memory_gb.png")
    _plot_cumulative_line(samples, out_dir / f"{prefix}_line_cumulative.png")
    _plot_unmap_scatter(samples, out_dir / f"{prefix}_scatter_unmap_vs_pages.png")
    _plot_hist_blocks(samples, out_dir / f"{prefix}_hist_blocks_per_call.png")
    _plot_hist_unmap_time(samples, out_dir / f"{prefix}_hist_unmap_time_ms.png")

    summary = _build_summary(report)
    summary_path = out_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Generated plots under: {out_dir}")
    print(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
