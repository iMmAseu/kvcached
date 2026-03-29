#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _read_summary_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "pages_to_free": float(row["pages_to_free"]),
                "mean_unmap_time_ms": float(row["mean_unmap_time_ms"]),
                "mean_unmap_time_per_page_ms": float(row["mean_unmap_time_per_page_ms"]),
                "p95_unmap_time_ms": float(row["p95_unmap_time_ms"]),
                "mean_wall_time_ms": float(row["mean_wall_time_ms"]),
            })
    rows.sort(key=lambda x: x["pages_to_free"])
    if not rows:
        raise RuntimeError(f"No rows found in summary CSV: {path}")
    return rows


def _read_raw_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def _read_meta(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("meta", {}) or {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a grouped bar chart from PageAllocator exact-unmap benchmark outputs."
    )
    parser.add_argument(
        "--summary-csv",
        required=True,
        help="Path to *_summary.csv",
    )
    parser.add_argument(
        "--raw-csv",
        required=True,
        help="Path to *_raw.csv",
    )
    parser.add_argument(
        "--report-json",
        required=True,
        help="Path to *.json benchmark report",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output image path. Defaults to *_total_vs_avg_bar.png beside summary CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary_csv).resolve()
    raw_path = Path(args.raw_csv).resolve()
    json_path = Path(args.report_json).resolve()

    rows = _read_summary_csv(summary_path)
    raw_count = _read_raw_csv(raw_path)
    meta = _read_meta(json_path)

    out_path = Path(args.out).resolve() if args.out else summary_path.with_name(
        summary_path.stem.replace("_summary", "") + "_total_vs_avg_bar.png"
    )
    svg_path = out_path.with_suffix(".svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "svg.fonttype": "none",
    })

    pages = [int(r["pages_to_free"]) for r in rows]
    total_ms = [float(r["mean_unmap_time_ms"]) for r in rows]
    avg_ms = [float(r["mean_unmap_time_per_page_ms"]) for r in rows]

    xs = np.arange(len(pages))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.bar(xs - width / 2, total_ms, width=width,
           label="Total Time per Call (ms)", color="#2E86AB")
    ax.bar(xs + width / 2, avg_ms, width=width,
           label="Average Time per Page (ms)", color="#F18F01")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(p) for p in pages])
    ax.set_xlabel("Pages Unmapped per Call")
    ax.set_ylabel("Time (ms)")
    ax.set_title("UNMAP Time vs Pages (Total vs Average)", pad=18)

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    fig.savefig(svg_path)
    plt.close(fig)

    print(f"Saved PNG: {out_path}")
    print(f"Saved SVG: {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
