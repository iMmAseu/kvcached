#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


SINGLE_RE = re.compile(
    r"^cuMemUnmap\s+(?P<avg_us>[0-9.]+)\s+(?P<p50_us>[0-9.]+)\s+"
    r"(?P<p90_us>[0-9.]+)\s+(?P<p99_us>[0-9.]+)\s+(?P<max_us>[0-9.]+)\s*$"
)

MULTI_RE = re.compile(
    r"\[result\]\[(?P<mode>loop|range)\]\s+pages=\s*(?P<pages>\d+)\s+"
    r"mean_unmapped=(?P<mean_unmapped>[0-9.]+)/(?P<pages2>\d+)\s+"
    r"mean_unmap=(?P<mean_unmap_ms>[0-9.]+)ms\s+"
    r"p95=(?P<p95_ms>[0-9.]+)ms\s+"
    r"per_page=(?P<per_page_ms>[0-9.]+)ms"
)


def parse_single_page_bench(path: Path) -> Dict[str, float]:
    for line in path.read_text(encoding="utf-8").splitlines():
        m = SINGLE_RE.search(line.strip())
        if m:
            return {
                "avg_ms": float(m.group("avg_us")) / 1000.0,
                "p50_ms": float(m.group("p50_us")) / 1000.0,
                "p90_ms": float(m.group("p90_us")) / 1000.0,
                "p99_ms": float(m.group("p99_us")) / 1000.0,
                "max_ms": float(m.group("max_us")) / 1000.0,
            }
    raise RuntimeError(f"Failed to parse cuMemUnmap row from {path}")


def parse_multi_mode_bench(path: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    rows: Dict[str, Dict[int, Dict[str, float]]] = {"loop": {}, "range": {}}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = MULTI_RE.search(line)
        if not m:
            continue
        mode = m.group("mode")
        pages = int(m.group("pages"))
        rows[mode][pages] = {
            "mean_unmap_ms": float(m.group("mean_unmap_ms")),
            "per_page_ms": float(m.group("per_page_ms")),
            "p95_ms": float(m.group("p95_ms")),
        }
    if not rows["loop"] or not rows["range"]:
        raise RuntimeError(f"Failed to parse loop/range rows from {path}")
    return rows


def write_csv(
    pages: List[int],
    single: Dict[str, float],
    multi: Dict[str, Dict[int, Dict[str, float]]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pages",
                "single_page_total_ms",
                "single_page_per_page_ms",
                "loop_total_ms",
                "loop_per_page_ms",
                "range_total_ms",
                "range_per_page_ms",
            ],
        )
        writer.writeheader()
        for pages_count in pages:
            writer.writerow({
                "pages": pages_count,
                "single_page_total_ms": single["avg_ms"],
                "single_page_per_page_ms": single["avg_ms"],
                "loop_total_ms": multi["loop"][pages_count]["mean_unmap_ms"],
                "loop_per_page_ms": multi["loop"][pages_count]["per_page_ms"],
                "range_total_ms": multi["range"][pages_count]["mean_unmap_ms"],
                "range_per_page_ms": multi["range"][pages_count]["per_page_ms"],
            })


def plot(
    pages: List[int],
    single: Dict[str, float],
    multi: Dict[str, Dict[int, Dict[str, float]]],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    single_total = [single["avg_ms"]] * len(pages)
    loop_total = [multi["loop"][p]["mean_unmap_ms"] for p in pages]
    range_total = [multi["range"][p]["mean_unmap_ms"] for p in pages]

    single_per = [single["avg_ms"]] * len(pages)
    loop_per = [multi["loop"][p]["per_page_ms"] for p in pages]
    range_per = [multi["range"][p]["per_page_ms"] for p in pages]

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(pages, single_total, marker="o", linewidth=2.0,
                 color="#7f8c8d", label="Single-page cuMemUnmap")
    axes[0].plot(pages, loop_total, marker="o", linewidth=2.0,
                 color="#2E86AB", label="Looped page-wise cuMemUnmap")
    axes[0].plot(pages, range_total, marker="o", linewidth=2.0,
                 color="#F18F01", label="Contiguous-range cuMemUnmap(ptr, size)")
    axes[0].set_ylabel("Total Time per Event (ms)")
    axes[0].set_title("CUDA VMM UNMAP Comparison")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(pages, single_per, marker="o", linewidth=2.0,
                 color="#7f8c8d", label="Single-page cuMemUnmap")
    axes[1].plot(pages, loop_per, marker="o", linewidth=2.0,
                 color="#2E86AB", label="Looped page-wise cuMemUnmap")
    axes[1].plot(pages, range_per, marker="o", linewidth=2.0,
                 color="#F18F01", label="Contiguous-range cuMemUnmap(ptr, size)")
    axes[1].set_ylabel("Average Time per Page (ms)")
    axes[1].set_xlabel("Pages Unmapped per Event")
    axes[1].grid(alpha=0.25)

    for ax in axes:
      ax.spines["top"].set_visible(False)
      ax.spines["right"].set_visible(False)

    axes[1].set_xticks(pages)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot three CUDA VMM unmap modes: single-page cuMemUnmap, looped "
            "multi-page cuMemUnmap, and contiguous-range cuMemUnmap(ptr, size)."
        )
    )
    parser.add_argument(
        "--single-input",
        default="results/bench_vmm_single.txt",
        help="Raw stdout from ./bench_vmm.bin",
    )
    parser.add_argument(
        "--multi-input",
        default="results/bench_vmm_unmap_compare.txt",
        help="Raw stdout from ./bench_vmm_unmap_compare.bin",
    )
    parser.add_argument(
        "--csv-out",
        default="results/bench_vmm_unmap_compare.csv",
        help="Output merged CSV path.",
    )
    parser.add_argument(
        "--svg-out",
        default="results/bench_vmm_unmap_compare.svg",
        help="Output SVG path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    single_path = Path(args.single_input).resolve()
    multi_path = Path(args.multi_input).resolve()
    csv_path = Path(args.csv_out).resolve()
    svg_path = Path(args.svg_out).resolve()

    single = parse_single_page_bench(single_path)
    multi = parse_multi_mode_bench(multi_path)
    pages = sorted(set(multi["loop"]) & set(multi["range"]))
    if not pages:
        raise RuntimeError("No overlapping pages values between loop and range results")

    write_csv(pages, single, multi, csv_path)
    plot(pages, single, multi, svg_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved SVG: {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
