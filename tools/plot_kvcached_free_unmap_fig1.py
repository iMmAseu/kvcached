#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Redraw fig1 for kvcached free/unmap benchmark with matplotlib."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw 500reqs_different_rates_fig1_returned_free_time.png with matplotlib.",
    )
    parser.add_argument(
        "--csv",
        default="results/bench_kvached_free_unmap/500reqs_different_rates.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="results/bench_kvached_free_unmap/500reqs_different_rates_fig1_returned_free_time.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> Tuple[List[float], List[float], List[float]]:
    req_rate: List[float] = []
    returned_pages: List[float] = []
    avg_free_ms: List[float] = []

    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            req_rate.append(float(row["req_rate"]))
            returned_pages.append(float(row["Total Pages Returned"]))
            avg_free_ms.append(float(row["Avg Time / Call (ms)"]))
    return req_rate, returned_pages, avg_free_ms


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed in the current Python environment.",
            file=sys.stderr,
        )
        return 1

    x, returned_pages, avg_free_ms = load_csv(csv_path)

    font_size = 13
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    })

    fig, ax1 = plt.subplots(figsize=(11.5, 6.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    color_pages = "#1f4e79"
    color_time = "#c2410c"

    line1 = ax1.plot(
        x,
        returned_pages,
        color=color_pages,
        linewidth=2.8,
        marker="o",
        markersize=7,
        label="Total Pages Returned",
    )
    ax1.set_xlabel("Request Rate (req/s)")
    ax1.set_ylabel("Returned Pages")
    ax1.grid(axis="y", color="#d9e2f3", linewidth=0.9, alpha=0.85)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(min(x) - 1, max(x) + 1)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        x,
        avg_free_ms,
        color=color_time,
        linewidth=2.8,
        marker="s",
        markersize=6.5,
        label="Avg Free Time / Call (ms)",
    )
    ax2.set_ylabel("Avg Free Time / Call (ms)")
    ax2.spines["top"].set_visible(False)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper left",
        frameon=False,
    )

    ax1.set_title("500 Requests: Returned Pages and Avg Free Time by Request Rate")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
