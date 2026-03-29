#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Render a one-page PNG report from results/bench_memory_used outputs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional


METRIC_PATTERNS = {
    "total_free_calls": re.compile(r"Total free\(\) Calls:\s+([0-9]+)"),
    "total_blocks_freed": re.compile(r"Total Blocks Freed:\s+([0-9]+)"),
    "total_pages_returned": re.compile(r"Total Pages Returned:\s+([0-9]+)"),
    "total_free_time_ms": re.compile(r"Total free\(\) Time \(ms\):\s+([0-9.]+)"),
    "avg_free_time_ms": re.compile(r"Avg Time / Call \(ms\):\s+([0-9.]+)"),
    "unmap_calls": re.compile(r"Calls that triggered UNMAP:\s+([0-9]+)\s*/\s*([0-9]+)"),
    "total_pages_unmapped": re.compile(r"Total Pages Unmapped:\s+([0-9]+)"),
    "total_unmap_time_ms": re.compile(r"Total UNMAP Time \(ms\):\s+([0-9.]+)"),
    "reserved_pages": re.compile(r"Reserved Pages:\s+([0-9]+)\s*/\s*([0-9]+)"),
    "inuse_pages": re.compile(r"In-use Pages:\s+([0-9]+)"),
    "cuda_used_gb": re.compile(r"CUDA Used:\s+([0-9.]+)\s+GB"),
    "cuda_free_gb": re.compile(r"CUDA Free:\s+([0-9.]+)\s+GB"),
}

PROOF_RE = re.compile(
    r"^\(.*?\)\s+"
    r"(?P<label>baseline|running_peak|logical_free_no_unmap|unmap_event):\s+"
    r"reason=(?P<reason>.*?)\s+"
    r"t=(?P<t>[0-9.]+)s\s+"
    r"inuse=(?P<inuse>[0-9]+)\s+"
    r"reserved=(?P<reserved>[0-9]+)\s+"
    r"returned=(?P<returned>[0-9]+)\s+"
    r"unmapped=(?P<unmapped>None|[0-9]+)\s+"
    r"kv_total=(?P<kv_total>[0-9.]+)GB\s+"
    r"cuda_used=(?P<cuda_used>[0-9.]+)GB"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a PNG report for results/bench_memory_used.",
    )
    parser.add_argument(
        "--input",
        default="results/bench_memory_used/bench_memory_used.txt",
        help="Input text report path.",
    )
    parser.add_argument(
        "--trend-image",
        default="results/bench_memory_used/bench_memory_used.png",
        help="Optional trend image to embed.",
    )
    parser.add_argument(
        "--output",
        default="results/bench_memory_used/bench_memory_used_report.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def parse_report(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    metrics: Dict[str, object] = {}

    for key, pattern in METRIC_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        if key == "unmap_calls":
            metrics["unmap_calls"] = int(match.group(1))
            metrics["unmap_total_calls"] = int(match.group(2))
        elif key == "reserved_pages":
            metrics["reserved_pages"] = int(match.group(1))
            metrics["reserved_pages_cap"] = int(match.group(2))
        elif key in {
                "total_free_calls",
                "total_blocks_freed",
                "total_pages_returned",
                "total_pages_unmapped",
                "inuse_pages",
        }:
            metrics[key] = int(match.group(1))
        else:
            metrics[key] = float(match.group(1))

    proof: Dict[str, Dict[str, object]] = {}
    for line in text.splitlines():
        match = PROOF_RE.match(line)
        if not match:
            continue
        gd = match.groupdict()
        proof[gd["label"]] = {
            "label": gd["label"],
            "reason": gd["reason"],
            "t": float(gd["t"]),
            "inuse": int(gd["inuse"]),
            "reserved": int(gd["reserved"]),
            "returned": int(gd["returned"]),
            "unmapped": gd["unmapped"],
            "kv_total": float(gd["kv_total"]),
            "cuda_used": float(gd["cuda_used"]),
        }
    metrics["proof"] = proof
    return metrics


def _fmt_int(value: object) -> str:
    return f"{int(value):,}"


def _metric_card(ax, x: float, y: float, w: float, h: float, title: str,
                 value: str, accent: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=12",
        linewidth=0,
        facecolor="#ffffff",
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.add_patch(FancyBboxPatch(
        (x, y),
        0.012,
        h,
        boxstyle="round,pad=0.0,rounding_size=12",
        linewidth=0,
        facecolor=accent,
        transform=ax.transAxes,
    ))
    ax.text(
        x + 0.03,
        y + h * 0.63,
        title,
        transform=ax.transAxes,
        fontsize=15,
        color="#475569",
        va="center",
    )
    ax.text(
        x + 0.03,
        y + h * 0.28,
        value,
        transform=ax.transAxes,
        fontsize=24,
        color="#0f172a",
        fontweight="bold",
        va="center",
    )


def _section_title(ax, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color="#0f172a",
        va="top",
    )


def _body_text(ax, x: float, y: float, text: str, size: int = 14,
               color: str = "#334155") -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=size,
        color=color,
        va="top",
    )


def _proof_line(ax, x: float, y: float, label: str, payload: Optional[Dict[str, object]]) -> None:
    if payload is None:
        _body_text(ax, x, y, f"{label}: n/a", size=13)
        return
    text = (
        f"{label}: t={payload['t']:.1f}s, "
        f"in-use={payload['inuse']}, reserved={payload['reserved']}, "
        f"KV total={payload['kv_total']:.2f} GB, CUDA used={payload['cuda_used']:.2f} GB"
    )
    _body_text(ax, x, y, text, size=13)


def render_report(metrics: Dict[str, object], trend_image: Optional[Path],
                  output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed in the current Python environment.",
            file=sys.stderr,
        )
        raise

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.unicode_minus": False,
    })

    fig = plt.figure(figsize=(14, 16), dpi=180, facecolor="#f8fafc")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()

    canvas.text(
        0.05,
        0.965,
        "KV Cache Memory Lifecycle Report",
        fontsize=28,
        fontweight="bold",
        color="#0f172a",
        va="top",
    )
    canvas.text(
        0.05,
        0.935,
        "Single-run summary rendered from results/bench_memory_used/bench_memory_used.txt",
        fontsize=15,
        color="#475569",
        va="top",
    )

    _section_title(canvas, 0.05, 0.885, "Experiment Setup")
    _body_text(
        canvas,
        0.05,
        0.855,
        "One online serving run with kvcached memory tracing enabled. "
        "The report tracks request execution, logical free, reserved-pool retention, and physical unmap.",
    )
    _body_text(canvas, 0.05, 0.825, f"Total requests / free() calls: {_fmt_int(metrics.get('total_free_calls', 0))}")
    _body_text(canvas, 0.05, 0.798, "Observation targets: GPU memory usage, page reclaim behavior, reserved pool usage, and unmap behavior.")

    _metric_card(
        canvas,
        0.05,
        0.695,
        0.27,
        0.09,
        "Blocks Freed",
        _fmt_int(metrics.get("total_blocks_freed", 0)),
        "#2563eb",
    )
    _metric_card(
        canvas,
        0.36,
        0.695,
        0.27,
        0.09,
        "Pages Returned",
        _fmt_int(metrics.get("total_pages_returned", 0)),
        "#f59e0b",
    )
    _metric_card(
        canvas,
        0.67,
        0.695,
        0.28,
        0.09,
        "UNMAP Calls",
        f"{metrics.get('unmap_calls', 0)} / {metrics.get('unmap_total_calls', 0)}",
        "#dc2626",
    )

    _section_title(canvas, 0.05, 0.655, "Key Metrics")
    _body_text(
        canvas,
        0.05,
        0.625,
        f"Total free() time: {metrics.get('total_free_time_ms', 0):.3f} ms    "
        f"Avg free() time / call: {metrics.get('avg_free_time_ms', 0):.4f} ms",
    )
    _body_text(
        canvas,
        0.05,
        0.598,
        f"Pages unmapped: {_fmt_int(metrics.get('total_pages_unmapped', 0))}    "
        f"Total UNMAP time: {metrics.get('total_unmap_time_ms', 0):.3f} ms",
    )
    _body_text(
        canvas,
        0.05,
        0.571,
        f"Final state: reserved pages {metrics.get('reserved_pages', 0)} / {metrics.get('reserved_pages_cap', 0)}, "
        f"in-use pages {metrics.get('inuse_pages', 0)}, "
        f"CUDA used {metrics.get('cuda_used_gb', 0):.2f} GB, CUDA free {metrics.get('cuda_free_gb', 0):.2f} GB",
    )

    _section_title(canvas, 0.05, 0.53, "Key Findings")
    _body_text(
        canvas,
        0.05,
        0.500,
        "1. During request execution, KV cache allocation increases GPU memory usage.",
        size=15,
    )
    _body_text(
        canvas,
        0.05,
        0.472,
        "2. After free(), blocks/pages can be logically released while GPU memory remains high because pages stay mapped or reserved.",
        size=15,
    )
    _body_text(
        canvas,
        0.05,
        0.444,
        "3. A visible GPU-memory drop happens only after trim()/unmap triggers real physical reclamation.",
        size=15,
    )

    _section_title(canvas, 0.05, 0.402, "Lifecycle Evidence")
    proof = metrics.get("proof", {})
    assert isinstance(proof, dict)
    _proof_line(canvas, 0.05, 0.374, "Baseline", proof.get("baseline"))
    _proof_line(canvas, 0.05, 0.348, "Running peak", proof.get("running_peak"))
    _proof_line(canvas, 0.05, 0.322, "Logical free, no unmap", proof.get("logical_free_no_unmap"))
    _proof_line(canvas, 0.05, 0.296, "Unmap event", proof.get("unmap_event"))

    if trend_image is not None and trend_image.exists():
        _section_title(canvas, 0.05, 0.255, "Memory Trend Figure")
        img = mpimg.imread(trend_image)
        ax_img = fig.add_axes([0.05, 0.05, 0.90, 0.18])
        ax_img.imshow(img)
        ax_img.set_axis_off()

    fig.savefig(output_path, bbox_inches="tight")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    trend_image = Path(args.trend_image).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    metrics = parse_report(input_path)
    render_report(
        metrics=metrics,
        trend_image=trend_image if trend_image.exists() else None,
        output_path=output_path,
    )
    print(f"Saved report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
