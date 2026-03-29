#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import re
from pathlib import Path
from typing import Dict, List


SINGLE_RE = re.compile(
    r"^cuMemUnmap\s+(?P<avg_us>[0-9.]+)\s+(?P<p50_us>[0-9.]+)\s+"
    r"(?P<p90_us>[0-9.]+)\s+(?P<p99_us>[0-9.]+)\s+(?P<max_us>[0-9.]+)\s*$"
)

RESULT_RE = re.compile(
    r"\[result\](?:\[(?P<mode>loop|range)\])?\s+pages=\s*(?P<pages>\d+)\s+"
    r"mean_unmapped=(?P<mean_unmapped>[0-9.]+)/(?P<pages2>\d+)\s+"
    r"mean_unmap=(?P<mean_unmap_ms>[0-9.]+)ms\s+"
    r"p95=(?P<p95_ms>[0-9.]+)ms\s+"
    r"per_page=(?P<per_page_ms>[0-9.]+)ms"
)


def parse_single(path: Path) -> Dict[str, float]:
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


def parse_results(path: Path, default_mode: str = "plain") -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = RESULT_RE.search(line)
        if not m:
            continue
        mode = m.group("mode") or default_mode
        pages = int(m.group("pages"))
        out.setdefault(mode, {})[pages] = {
            "mean_unmap_ms": float(m.group("mean_unmap_ms")),
            "per_page_ms": float(m.group("per_page_ms")),
            "p95_ms": float(m.group("p95_ms")),
        }
    if not out:
        raise RuntimeError(f"No [result] rows found in {path}")
    return out


def _apply_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "svg.fonttype": "none",
    })


def _save(fig, stem: Path):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".svg"))


def plot_three_modes(
    pages: List[int],
    single: Dict[str, float],
    loop_rows: Dict[int, Dict[str, float]],
    range_rows: Dict[int, Dict[str, float]],
    out_stem: Path,
) -> None:
    import matplotlib.pyplot as plt

    _apply_style()
    single_total = [single["avg_ms"]] * len(pages)
    single_per = [single["avg_ms"]] * len(pages)
    loop_total = [loop_rows[p]["mean_unmap_ms"] for p in pages]
    loop_per = [loop_rows[p]["per_page_ms"] for p in pages]
    range_total = [range_rows[p]["mean_unmap_ms"] for p in pages]
    range_per = [range_rows[p]["per_page_ms"] for p in pages]

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8), sharex=True)

    axes[0].plot(pages, single_total, marker="o", linewidth=2.2,
                 color="#7f8c8d", label="Single-page cuMemUnmap")
    axes[0].plot(pages, loop_total, marker="o", linewidth=2.2,
                 color="#2E86AB", label="Looped page-wise cuMemUnmap")
    axes[0].plot(pages, range_total, marker="o", linewidth=2.2,
                 color="#F18F01", label="Contiguous-range cuMemUnmap(ptr, size)")
    axes[0].set_ylabel("Total Time per Event (ms)")
    axes[0].set_title("CUDA VMM UNMAP Comparison")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(pages, single_per, marker="o", linewidth=2.2,
                 color="#7f8c8d", label="Single-page cuMemUnmap")
    axes[1].plot(pages, loop_per, marker="o", linewidth=2.2,
                 color="#2E86AB", label="Looped page-wise cuMemUnmap")
    axes[1].plot(pages, range_per, marker="o", linewidth=2.2,
                 color="#F18F01", label="Contiguous-range cuMemUnmap(ptr, size)")
    axes[1].set_ylabel("Average Time per Page (ms)")
    axes[1].set_xlabel("Pages Unmapped per Event")
    axes[1].grid(alpha=0.25)
    axes[1].set_xticks(pages)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _save(fig, out_stem)
    plt.close(fig)


def plot_loop_consistency(
    pages: List[int],
    plain_rows: Dict[int, Dict[str, float]],
    loop_rows: Dict[int, Dict[str, float]],
    out_stem: Path,
) -> None:
    import matplotlib.pyplot as plt

    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8), sharex=True)

    axes[0].plot(pages, [plain_rows[p]["mean_unmap_ms"] for p in pages],
                 marker="o", linewidth=2.2, color="#2563eb",
                 label="bench_vmm_batch_unmap")
    axes[0].plot(pages, [loop_rows[p]["mean_unmap_ms"] for p in pages],
                 marker="o", linewidth=2.2, color="#dc2626",
                 label="bench_vmm_unmap_compare [loop]")
    axes[0].set_ylabel("Total Time per Event (ms)")
    axes[0].set_title("Looped Multi-page cuMemUnmap: Consistency Check")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(pages, [plain_rows[p]["per_page_ms"] for p in pages],
                 marker="o", linewidth=2.2, color="#2563eb",
                 label="bench_vmm_batch_unmap")
    axes[1].plot(pages, [loop_rows[p]["per_page_ms"] for p in pages],
                 marker="o", linewidth=2.2, color="#dc2626",
                 label="bench_vmm_unmap_compare [loop]")
    axes[1].set_ylabel("Average Time per Page (ms)")
    axes[1].set_xlabel("Pages Unmapped per Event")
    axes[1].grid(alpha=0.25)
    axes[1].set_xticks(pages)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _save(fig, out_stem)
    plt.close(fig)


def plot_range_speedup(
    pages: List[int],
    loop_rows: Dict[int, Dict[str, float]],
    range_rows: Dict[int, Dict[str, float]],
    out_stem: Path,
) -> None:
    import matplotlib.pyplot as plt

    _apply_style()
    total_speedup = [
        loop_rows[p]["mean_unmap_ms"] / range_rows[p]["mean_unmap_ms"] for p in pages
    ]
    per_page_speedup = [
        loop_rows[p]["per_page_ms"] / range_rows[p]["per_page_ms"] for p in pages
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8), sharex=True)

    axes[0].bar(pages, total_speedup, width=0.72, color="#10b981")
    axes[0].set_ylabel("Speedup (x)")
    axes[0].set_title("Contiguous-range UNMAP Speedup over Looped Page-wise UNMAP")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(pages, per_page_speedup, width=0.72, color="#f59e0b")
    axes[1].set_ylabel("Per-page Speedup (x)")
    axes[1].set_xlabel("Pages Unmapped per Event")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(pages)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _save(fig, out_stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all figures for results/bench_unmap from raw benchmark text files."
    )
    parser.add_argument(
        "--single-input",
        default="results/bench_unmap/bench_vmm_single.txt",
        help="Raw stdout from bench_vmm.bin",
    )
    parser.add_argument(
        "--compare-input",
        default="results/bench_unmap/bench_vmm_unmap_compare.txt",
        help="Raw stdout from bench_vmm_unmap_compare.bin",
    )
    parser.add_argument(
        "--loop-input",
        default="results/bench_unmap/unmap_vmm_results.txt",
        help="Raw stdout from bench_vmm_batch_unmap.bin",
    )
    parser.add_argument(
        "--out-dir",
        default="results/bench_unmap",
        help="Output directory for figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    single = parse_single(Path(args.single_input).resolve())
    compare = parse_results(Path(args.compare_input).resolve())
    plain = parse_results(Path(args.loop_input).resolve())

    loop_rows = compare["loop"]
    range_rows = compare["range"]
    plain_rows = plain["plain"]

    common_pages = sorted(set(loop_rows) & set(range_rows))
    if not common_pages:
        raise RuntimeError("No overlapping pages between loop and range results")

    consistency_pages = sorted(set(common_pages) & set(plain_rows))
    if not consistency_pages:
        raise RuntimeError("No overlapping pages between loop results and batch benchmark")

    plot_three_modes(
        common_pages, single, loop_rows, range_rows,
        out_dir / "bench_unmap_fig1_three_modes"
    )
    plot_loop_consistency(
        consistency_pages, plain_rows, loop_rows,
        out_dir / "bench_unmap_fig2_loop_consistency"
    )
    plot_range_speedup(
        common_pages, loop_rows, range_rows,
        out_dir / "bench_unmap_fig3_range_speedup"
    )

    print(f"Saved figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
