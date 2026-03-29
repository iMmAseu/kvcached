#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List


RESULT_RE = re.compile(
    r"\[result\]\s+pages=\s*(?P<pages>\d+)\s+"
    r"mean_unmapped=(?P<mean_unmapped>[0-9.]+)/(?P<pages2>\d+)\s+"
    r"mean_unmap=(?P<mean_unmap_ms>[0-9.]+)ms\s+"
    r"p95=(?P<p95_ms>[0-9.]+)ms\s+"
    r"per_page=(?P<per_page_ms>[0-9.]+)ms"
)


def parse_result_txt(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = RESULT_RE.search(line)
        if not match:
            continue
        rows.append({
            "pages": float(match.group("pages")),
            "mean_unmap_ms": float(match.group("mean_unmap_ms")),
            "per_page_ms": float(match.group("per_page_ms")),
            "p95_ms": float(match.group("p95_ms")),
        })
    rows.sort(key=lambda x: x["pages"])
    return rows


def index_by_pages(rows: List[Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    return {int(row["pages"]): row for row in rows}


def write_merged_csv(
    pages: List[int],
    pageallocator_rows: Dict[int, Dict[str, float]],
    vmm_rows: Dict[int, Dict[str, float]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pages",
                "pageallocator_total_ms",
                "pageallocator_per_page_ms",
                "cuda_vmm_total_ms",
                "cuda_vmm_per_page_ms",
            ],
        )
        writer.writeheader()
        for pages_count in pages:
            pa = pageallocator_rows[pages_count]
            vmm = vmm_rows[pages_count]
            writer.writerow({
                "pages": pages_count,
                "pageallocator_total_ms": pa["mean_unmap_ms"],
                "pageallocator_per_page_ms": pa["per_page_ms"],
                "cuda_vmm_total_ms": vmm["mean_unmap_ms"],
                "cuda_vmm_per_page_ms": vmm["per_page_ms"],
            })


def plot_matplotlib(
    pages: List[int],
    pageallocator_rows: Dict[int, Dict[str, float]],
    vmm_rows: Dict[int, Dict[str, float]],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs = np.arange(len(pages))
    width = 0.38
    pa_total = [pageallocator_rows[p]["mean_unmap_ms"] for p in pages]
    vmm_total = [vmm_rows[p]["mean_unmap_ms"] for p in pages]
    pa_avg = [pageallocator_rows[p]["per_page_ms"] for p in pages]
    vmm_avg = [vmm_rows[p]["per_page_ms"] for p in pages]

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].bar(xs - width / 2, pa_total, width=width,
                label="PageAllocator.free_pages()", color="#2E86AB")
    axes[0].bar(xs + width / 2, vmm_total, width=width,
                label="Pure CUDA VMM", color="#F18F01")
    axes[0].set_ylabel("Total Time per Call (ms)")
    axes[0].set_title("UNMAP Time vs Pages: PageAllocator.free_pages() vs Pure CUDA VMM")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncols=2, loc="upper left")

    axes[1].bar(xs - width / 2, pa_avg, width=width,
                label="PageAllocator.free_pages()", color="#5DADE2")
    axes[1].bar(xs + width / 2, vmm_avg, width=width,
                label="Pure CUDA VMM", color="#F5B041")
    axes[1].set_ylabel("Average Time per Page (ms)")
    axes[1].set_xlabel("Pages Unmapped per Call")
    axes[1].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([str(p) for p in pages], rotation=0)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PageAllocator.free_pages() unmap timing against a pure CUDA "
            "VMM batch-unmap benchmark."
        )
    )
    parser.add_argument(
        "--pageallocator-input",
        default="results/unmap_results.txt",
        help="Path to PageAllocator.free_pages() raw benchmark text output.",
    )
    parser.add_argument(
        "--cuda-vmm-input",
        default="results/unmap_vmm_results.txt",
        help="Path to pure CUDA VMM batch-unmap raw benchmark text output.",
    )
    parser.add_argument(
        "--csv-out",
        default="results/unmap_pageallocator_vs_vmm.csv",
        help="Output merged CSV path.",
    )
    parser.add_argument(
        "--svg-out",
        default="results/unmap_pageallocator_vs_vmm.svg",
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pa_path = Path(args.pageallocator_input).resolve()
    vmm_path = Path(args.cuda_vmm_input).resolve()
    csv_path = Path(args.csv_out).resolve()
    out_path = Path(args.svg_out).resolve()

    pa_rows = parse_result_txt(pa_path)
    vmm_rows = parse_result_txt(vmm_path)
    if not pa_rows:
        raise RuntimeError(f"No valid [result] rows found in {pa_path}")
    if not vmm_rows:
        raise RuntimeError(f"No valid [result] rows found in {vmm_path}")

    pa_by_pages = index_by_pages(pa_rows)
    vmm_by_pages = index_by_pages(vmm_rows)
    common_pages = sorted(set(pa_by_pages) & set(vmm_by_pages))
    if not common_pages:
        raise RuntimeError("No overlapping pages values between the two benchmarks")

    write_merged_csv(common_pages, pa_by_pages, vmm_by_pages, csv_path)
    plot_matplotlib(common_pages, pa_by_pages, vmm_by_pages, out_path)

    print(f"Saved merged CSV: {csv_path}")
    print(f"Saved figure: {out_path}")
    print(f"Compared pages: {common_pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
