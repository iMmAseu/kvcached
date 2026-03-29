#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import re
from html import escape
from pathlib import Path
from typing import Dict, List

RESULT_RE = re.compile(
    r"\[result\]\s+pages=\s*(?P<pages>\d+)\s+"
    r"mean_unmapped=(?P<mean_unmapped>[0-9.]+)/(?P<pages2>\d+)\s+"
    r"mean_unmap=(?P<mean_unmap_ms>[0-9.]+)ms\s+"
    r"p95=(?P<p95_ms>[0-9.]+)ms\s+"
    r"per_page=(?P<per_page_ms>[0-9.]+)ms"
)


def parse_lines(text: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for line in text.splitlines():
        m = RESULT_RE.search(line)
        if not m:
            continue
        pages = int(m.group("pages"))
        rows.append({
            "pages": float(pages),
            "mean_unmapped": float(m.group("mean_unmapped")),
            "mean_unmap_ms": float(m.group("mean_unmap_ms")),
            "p95_ms": float(m.group("p95_ms")),
            "per_page_ms": float(m.group("per_page_ms")),
        })
    rows.sort(key=lambda x: x["pages"])
    return rows


def write_csv(rows: List[Dict[str, float]], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pages", "mean_unmapped", "mean_unmap_ms", "p95_ms", "per_page_ms"])
        writer.writeheader()
        writer.writerows(rows)


def plot_grouped_bar_matplotlib(rows: List[Dict[str, float]], png_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pages = [int(r["pages"]) for r in rows]
    total_ms = [r["mean_unmap_ms"] for r in rows]
    avg_ms = [r["per_page_ms"] for r in rows]

    xs = list(range(len(pages)))
    width = 0.42

    plt.figure(figsize=(14, 6))
    plt.bar(
        [x - width / 2 for x in xs],
        total_ms,
        width=width,
        label="Total Time per Call (ms)",
        color="#2E86AB",
    )
    plt.bar(
        [x + width / 2 for x in xs],
        avg_ms,
        width=width,
        label="Average Time per Page (ms)",
        color="#F18F01",
    )

    plt.xticks(xs, pages)
    plt.xlabel("Pages Unmapped per Call")
    plt.ylabel("Time (ms)")
    plt.title("UNMAP Time vs Pages (Total vs Average)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


def plot_grouped_bar_svg(rows: List[Dict[str, float]], svg_path: Path) -> None:
    pages = [int(r["pages"]) for r in rows]
    total_ms = [r["mean_unmap_ms"] for r in rows]
    avg_ms = [r["per_page_ms"] for r in rows]

    width = 1400
    height = 650
    left = 80
    right = 30
    top = 60
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_y = max(max(total_ms), max(avg_ms)) * 1.1
    n = len(pages)
    group_w = plot_w / max(1, n)
    bar_w = group_w * 0.35
    gap = group_w * 0.1

    def y_to_px(y: float) -> float:
        return top + (plot_h - (y / max_y) * plot_h)

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20" font-family="Arial">UNMAP Time vs Pages (Total vs Average)</text>')

    # Axes
    x0, y0 = left, top + plot_h
    x1, y1 = left + plot_w, top
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333" stroke-width="1.5"/>')

    # Y ticks
    ticks = 6
    for i in range(ticks + 1):
        v = max_y * i / ticks
        y = y_to_px(v)
        parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{x0 - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{v:.1f}</text>')

    # Bars
    for i, p in enumerate(pages):
        cx = left + (i + 0.5) * group_w
        t = total_ms[i]
        a = avg_ms[i]

        tx = cx - gap / 2 - bar_w
        ax = cx + gap / 2
        ty = y_to_px(t)
        ay = y_to_px(a)
        th = y0 - ty
        ah = y0 - ay

        parts.append(f'<rect x="{tx:.2f}" y="{ty:.2f}" width="{bar_w:.2f}" height="{th:.2f}" fill="#2E86AB"/>')
        parts.append(f'<rect x="{ax:.2f}" y="{ay:.2f}" width="{bar_w:.2f}" height="{ah:.2f}" fill="#F18F01"/>')
        parts.append(f'<text x="{cx:.2f}" y="{y0 + 18:.2f}" text-anchor="middle" font-size="10" font-family="Arial">{p}</text>')

    # Axis labels
    parts.append(f'<text x="{left + plot_w/2:.2f}" y="{height - 20}" text-anchor="middle" font-size="13" font-family="Arial">Pages Unmapped per Call</text>')
    parts.append(
        f'<text x="20" y="{top + plot_h/2:.2f}" transform="rotate(-90 20,{top + plot_h/2:.2f})" text-anchor="middle" font-size="13" font-family="Arial">Time (ms)</text>'
    )

    # Legend
    lx = width - 320
    ly = 50
    parts.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="#2E86AB"/>')
    parts.append(f'<text x="{lx + 22}" y="{ly + 12}" font-size="12" font-family="Arial">{escape("Total Time per Call (ms)")}</text>')
    parts.append(f'<rect x="{lx}" y="{ly + 22}" width="14" height="14" fill="#F18F01"/>')
    parts.append(f'<text x="{lx + 22}" y="{ly + 34}" font-size="12" font-family="Arial">{escape("Average Time per Page (ms)")}</text>')

    parts.append("</svg>")
    svg_path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse unmap benchmark txt and plot grouped bars.")
    parser.add_argument(
        "--input",
        default="results/unmap_results.txt",
        help="Path to raw txt log.",
    )
    parser.add_argument(
        "--csv-out",
        default="results/unmap_results_parsed.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--png-out",
        default="results/unmap_pages_total_vs_avg_bar.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input).resolve()
    csv_path = Path(args.csv_out).resolve()
    png_path = Path(args.png_out).resolve()
    svg_path = png_path.with_suffix(".svg")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    text = in_path.read_text(encoding="utf-8")
    rows = parse_lines(text)
    if not rows:
        raise RuntimeError(f"No valid [result] rows found in: {in_path}")

    write_csv(rows, csv_path)
    rendered = ""
    try:
        plot_grouped_bar_matplotlib(rows, png_path)
        rendered = str(png_path)
    except ModuleNotFoundError:
        plot_grouped_bar_svg(rows, svg_path)
        rendered = str(svg_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Figure: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
