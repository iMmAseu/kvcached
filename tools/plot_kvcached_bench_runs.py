#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError(
        "matplotlib is required for plotting. Install it with: pip install matplotlib"
    ) from e


def _safe_get(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _load_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = obj.get("run_meta", {}) or {}
        free = obj.get("kvcached_free_report_delta")
        if not free:
            # Backward compatibility with older run records.
            free = obj.get("kvcached_free_report_after")
        if not free:
            free = obj.get("kvcached_free_report")
        free = free or {}
        unmap = free.get("unmap_stats", {}) or {}
        free_block_stats = free.get("free_block_stats", {}) or {}
        call_window = free.get("call_window", {}) or {}

        rows.append({
            "file": str(p),
            "run_id": obj.get("run_id", p.stem),
            "request_rate": _safe_get(meta, "request_rate"),
            "num_prompts": _safe_get(meta, "num_prompts"),
            "start_call_exclusive": _safe_get(call_window, "start_call_exclusive"),
            "end_call_inclusive": _safe_get(call_window, "end_call_inclusive"),
            "total_free_calls": _safe_get(free, "total_free_calls"),
            "total_blocks_freed": _safe_get(free, "total_blocks_freed"),
            "total_pages_returned": _safe_get(free, "total_pages_returned"),
            "total_unmap_calls": _safe_get(unmap, "total_unmap_calls"),
            "total_pages_unmapped": _safe_get(unmap, "total_pages_unmapped"),
            "total_unmap_time_ms": _safe_get(unmap, "total_unmap_time_ms"),
            "avg_free_time_ms": _safe_get(free, "avg_time_per_call_ms"),
            "avg_blocks_per_call": _safe_get(free_block_stats, "mean_blocks_per_call"),
            "p95_blocks_per_call": _safe_get(free_block_stats, "p95_blocks_per_call"),
            "total_freed_logical_bytes": _safe_get(
                free_block_stats, "total_freed_logical_bytes"),
            "total_freed_mapped_bytes_est": _safe_get(
                free_block_stats, "total_freed_mapped_bytes_est"),
        })
    return rows


def _load_unmap_events(runs_dir: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = obj.get("run_meta", {}) or {}
        free = obj.get("kvcached_free_report_delta")
        if not free:
            free = obj.get("kvcached_free_report_after")
        if not free:
            free = obj.get("kvcached_free_report")
        free = free or {}
        for ev in free.get("unmap_events", []) or []:
            events.append({
                "file": str(p),
                "run_id": obj.get("run_id", p.stem),
                "request_rate": _safe_get(meta, "request_rate"),
                "num_prompts": _safe_get(meta, "num_prompts"),
                "call": _safe_get(ev, "call"),
                "pages_unmapped": _safe_get(ev, "pages_unmapped"),
                "unmap_time_ms": _safe_get(ev, "unmap_time_ms"),
            })
    return events


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _group_by(rows: List[Dict[str, Any]], key: str) -> Dict[float, List[Dict[str, Any]]]:
    groups: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[float(r[key])].append(r)
    return groups


def _plot_by_rate(rows: List[Dict[str, Any]], out_path: Path) -> None:
    groups = _group_by(rows, "request_rate")
    plt.figure(figsize=(11, 5))
    for rate, items in sorted(groups.items()):
        items_sorted = sorted(items, key=lambda x: x["num_prompts"])
        xs = [x["num_prompts"] for x in items_sorted]
        ys = [x["total_pages_unmapped"] for x in items_sorted]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=f"rate={rate:g}")
    plt.xlabel("num_prompts")
    plt.ylabel("total_pages_unmapped")
    plt.title("Trend: Unmapped Pages vs num_prompts (grouped by request_rate)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_by_prompts(rows: List[Dict[str, Any]], out_path: Path) -> None:
    groups = _group_by(rows, "num_prompts")
    plt.figure(figsize=(11, 5))
    for prompts, items in sorted(groups.items()):
        items_sorted = sorted(items, key=lambda x: x["request_rate"])
        xs = [x["request_rate"] for x in items_sorted]
        ys = [x["total_pages_unmapped"] for x in items_sorted]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=f"prompts={prompts:g}")
    plt.xlabel("request_rate")
    plt.ylabel("total_pages_unmapped")
    plt.title("Trend: Unmapped Pages vs request_rate (grouped by num_prompts)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_returned_pages(rows: List[Dict[str, Any]], out_path: Path) -> None:
    groups = _group_by(rows, "request_rate")
    plt.figure(figsize=(11, 5))
    for rate, items in sorted(groups.items()):
        items_sorted = sorted(items, key=lambda x: x["num_prompts"])
        xs = [x["num_prompts"] for x in items_sorted]
        ys = [x["total_pages_returned"] for x in items_sorted]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=f"rate={rate:g}")
    plt.xlabel("num_prompts")
    plt.ylabel("total_pages_returned")
    plt.title("Trend: Returned Pages vs num_prompts (grouped by request_rate)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_unmap_time_scatter(rows: List[Dict[str, Any]], out_path: Path) -> None:
    xs = [r["total_pages_unmapped"] for r in rows]
    ys = [r["total_unmap_time_ms"] for r in rows]
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, alpha=0.75)
    plt.xlabel("total_pages_unmapped")
    plt.ylabel("total_unmap_time_ms")
    plt.title("Scatter: total_unmap_time_ms vs total_pages_unmapped")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_free_blocks(rows: List[Dict[str, Any]], out_path: Path) -> None:
    groups = _group_by(rows, "request_rate")
    plt.figure(figsize=(11, 5))
    for rate, items in sorted(groups.items()):
        items_sorted = sorted(items, key=lambda x: x["num_prompts"])
        xs = [x["num_prompts"] for x in items_sorted]
        ys = [x["total_blocks_freed"] for x in items_sorted]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=f"rate={rate:g}")
    plt.xlabel("num_prompts")
    plt.ylabel("total_blocks_freed")
    plt.title("Trend: Free Blocks vs num_prompts (grouped by request_rate)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_unmap_event_scatter(events: List[Dict[str, Any]], out_path: Path) -> None:
    if not events:
        return
    xs = [e["pages_unmapped"] for e in events]
    ys = [e["unmap_time_ms"] for e in events]
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, alpha=0.7)
    plt.xlabel("pages_unmapped (per free() event)")
    plt.ylabel("unmap_time_ms (per free() event)")
    plt.title("UNMAP Event Scatter: unmap_time_ms vs pages_unmapped")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_unmap_event_line(events: List[Dict[str, Any]], out_path: Path) -> None:
    if not events:
        return
    grouped: Dict[int, List[float]] = defaultdict(list)
    for e in events:
        pages = int(e["pages_unmapped"])
        if pages <= 0:
            continue
        grouped[pages].append(float(e["unmap_time_ms"]))
    if not grouped:
        return
    points: List[Tuple[int, float]] = sorted(
        (pages, sum(times) / len(times)) for pages, times in grouped.items())
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=1.8)
    plt.xlabel("pages_unmapped (per free() event)")
    plt.ylabel("mean unmap_time_ms")
    plt.title("UNMAP Event Trend: mean unmap_time_ms vs pages_unmapped")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multiple bench run JSON files and plot trends.")
    parser.add_argument(
        "--runs-dir",
        default="/root/kvcached/results/bench_runs",
        help="Directory containing per-run JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        default="/root/kvcached/results/bench_runs_plots",
        help="Output directory for summary CSV and figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_runs(runs_dir)
    if not rows:
        print(f"No valid run JSON files found in {runs_dir}")
        return 1
    event_rows = _load_unmap_events(runs_dir)

    csv_path = out_dir / "bench_runs_summary.csv"
    _write_csv(rows, csv_path)
    if event_rows:
        _write_csv(event_rows, out_dir / "unmap_events_summary.csv")

    _plot_by_rate(rows, out_dir / "line_unmapped_vs_prompts_group_by_rate.png")
    _plot_by_prompts(rows, out_dir / "line_unmapped_vs_rate_group_by_prompts.png")
    _plot_returned_pages(rows, out_dir / "line_returned_vs_prompts_group_by_rate.png")
    _plot_unmap_time_scatter(rows, out_dir / "scatter_total_unmap_time_vs_pages.png")
    _plot_free_blocks(rows, out_dir / "line_free_blocks_vs_prompts_group_by_rate.png")
    _plot_unmap_event_scatter(event_rows, out_dir / "scatter_unmap_event_time_vs_pages.png")
    _plot_unmap_event_line(event_rows, out_dir / "line_unmap_event_mean_time_vs_pages.png")

    print(f"Saved summary CSV: {csv_path}")
    if event_rows:
        print(f"Saved event CSV: {out_dir / 'unmap_events_summary.csv'}")
    print(f"Saved plots under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
