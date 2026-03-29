#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Redraw fig2 and fig3 for kvcached free/unmap benchmark with matplotlib."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw fig2 and fig3 for 500reqs_different_rates with matplotlib.",
    )
    parser.add_argument(
        "--csv",
        default="results/bench_kvached_free_unmap/500reqs_different_rates.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--fig2-output",
        default="results/bench_kvached_free_unmap/500reqs_different_rates_fig2_unmap_calls_pages_time.png",
        help="Output path for fig2.",
    )
    parser.add_argument(
        "--fig3-output",
        default="results/bench_kvached_free_unmap/500reqs_different_rates_fig3_unmap_time_ratio_pct.png",
        help="Output path for fig3.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> Dict[str, List[float]]:
    cols: Dict[str, List[float]] = {
        "req_rate": [],
        "unmap_calls": [],
        "unmap_pages": [],
        "total_unmap_time": [],
        "avg_unmap_time": [],
        "total_free_time": [],
    }

    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            cols["req_rate"].append(float(row["req_rate"]))
            cols["unmap_calls"].append(float(row["Calls that triggered UNMAP"]))
            cols["unmap_pages"].append(float(row["Total Pages Unmapped"]))
            cols["total_unmap_time"].append(float(row["Total UNMAP Time"]))
            cols["avg_unmap_time"].append(float(row["Avg UNMAP Time"]))
            cols["total_free_time"].append(float(row["Total free() Time (ms)"]))
    return cols


def configure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed in the current Python environment.",
            file=sys.stderr,
        )
        raise

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
    return plt


def redraw_fig2(data: Dict[str, List[float]], output_path: Path, plt) -> None:
    x = data["req_rate"]
    fig, ax1 = plt.subplots(figsize=(11.5, 6.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    line1 = ax1.plot(
        x,
        data["unmap_calls"],
        color="#0f766e",
        linewidth=2.6,
        marker="^",
        markersize=7,
        label="Calls Triggering UNMAP",
    )
    line2 = ax1.plot(
        x,
        data["unmap_pages"],
        color="#2563eb",
        linewidth=2.6,
        marker="o",
        markersize=7,
        label="Total Pages Unmapped",
    )
    ax1.set_xlabel("Request Rate (req/s)")
    ax1.set_ylabel("Calls / Pages")
    ax1.grid(axis="y", color="#d9e2f3", linewidth=0.9, alpha=0.85)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(min(x) - 1, max(x) + 1)

    ax2 = ax1.twinx()
    line3 = ax2.plot(
        x,
        data["total_unmap_time"],
        color="#d97706",
        linewidth=2.6,
        marker="s",
        markersize=6.5,
        label="Total UNMAP Time (ms)",
    )
    line4 = ax2.plot(
        x,
        data["avg_unmap_time"],
        color="#7c3aed",
        linewidth=2.6,
        marker="D",
        markersize=6.0,
        linestyle="--",
        label="Avg UNMAP Time (ms)",
    )
    ax2.set_ylabel("Time (ms)")
    ax2.spines["top"].set_visible(False)

    lines = line1 + line2 + line3 + line4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=False, ncol=2)

    ax1.set_title("500 Requests: UNMAP Calls, Pages, and Time by Request Rate")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def redraw_fig3(data: Dict[str, List[float]], output_path: Path, plt) -> None:
    x = data["req_rate"]
    ratio_pct: List[float] = []
    for total_unmap, total_free in zip(
            data["total_unmap_time"], data["total_free_time"]):
        ratio_pct.append((total_unmap / total_free) * 100.0 if total_free > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        x,
        ratio_pct,
        color="#dc2626",
        linewidth=2.8,
        marker="o",
        markersize=7,
        label="UNMAP Time Share (%)",
    )
    ax.fill_between(x, ratio_pct, color="#fecaca", alpha=0.35)
    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("UNMAP Time / Total Free Time (%)")
    ax.set_title("500 Requests: UNMAP Time Share in Total Free Time")
    ax.grid(axis="y", color="#d9e2f3", linewidth=0.9, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    fig2_output = Path(args.fig2_output).resolve()
    fig3_output = Path(args.fig3_output).resolve()
    fig2_output.parent.mkdir(parents=True, exist_ok=True)
    fig3_output.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        plt = configure_matplotlib()
    except ImportError:
        return 1

    data = load_csv(csv_path)
    redraw_fig2(data, fig2_output, plt)
    redraw_fig3(data, fig3_output, plt)
    print(f"Saved figure to: {fig2_output}")
    print(f"Saved figure to: {fig3_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
