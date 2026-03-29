#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List


HEADING_RE = re.compile(r"^###\s*(?P<prompts>\d+)\s*/\s*(?P<rate>[0-9]+(?:\.[0-9]+)?)\s*$")
RET_RE = re.compile(r"Total Pages Returned:\s*(\d+)")
UNMAP_RE = re.compile(r"Total Pages Unmapped:\s*(\d+)")


def parse_sections(text: str) -> List[Dict[str, float]]:
    lines = text.splitlines()
    sections: List[Dict[str, float]] = []
    current_header = None
    current_block: List[str] = []

    def flush_block() -> None:
        nonlocal current_header, current_block
        if current_header is None:
            return
        block_text = "\n".join(current_block)
        ret_m = RET_RE.search(block_text)
        unmap_m = UNMAP_RE.search(block_text)
        if ret_m and unmap_m:
            sections.append({
                "prompts": float(current_header["prompts"]),
                "request_rate": float(current_header["rate"]),
                "pages_returned": float(ret_m.group(1)),
                "pages_unmapped": float(unmap_m.group(1)),
            })
        current_header = None
        current_block = []

    for line in lines:
        m = HEADING_RE.match(line.strip())
        if m:
            flush_block()
            current_header = {
                "prompts": m.group("prompts"),
                "rate": m.group("rate"),
            }
            current_block = []
            continue
        if current_header is not None:
            current_block.append(line)
    flush_block()
    return sections


def write_csv(rows: List[Dict[str, float]], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompts", "request_rate", "pages_returned", "pages_unmapped"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_prompt_matplotlib(
    prompt: int,
    rows: List[Dict[str, float]],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda x: x["request_rate"])
    rates = [r["request_rate"] for r in rows]
    returned = [r["pages_returned"] for r in rows]
    unmapped = [r["pages_unmapped"] for r in rows]
    xs = list(range(len(rates)))
    width = 0.42

    plt.figure(figsize=(10, 5))
    plt.bar(
        [x - width / 2 for x in xs],
        returned,
        width=width,
        color="#2E86AB",
        label="Pages Returned",
    )
    plt.bar(
        [x + width / 2 for x in xs],
        unmapped,
        width=width,
        color="#F18F01",
        label="Pages Unmapped",
    )
    plt.xticks(xs, [f"{x:g}" for x in rates])
    plt.xlabel("Request Rate")
    plt.ylabel("Pages")
    plt.title(f"Prompts={prompt}: Returned vs Unmapped by Request Rate")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_prompt_svg(
    prompt: int,
    rows: List[Dict[str, float]],
    out_path: Path,
) -> None:
    rows = sorted(rows, key=lambda x: x["request_rate"])
    rates = [r["request_rate"] for r in rows]
    returned = [r["pages_returned"] for r in rows]
    unmapped = [r["pages_unmapped"] for r in rows]

    width = 1200
    height = 560
    left = 80
    right = 30
    top = 60
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_y = max(max(returned, default=0), max(unmapped, default=0), 1) * 1.12
    n = len(rates)
    group_w = plot_w / max(1, n)
    bar_w = group_w * 0.34
    gap = group_w * 0.08

    def y_px(v: float) -> float:
        return top + (plot_h - (v / max_y) * plot_h)

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20" font-family="Arial">'
        f'Prompts={prompt}: Returned vs Unmapped by Request Rate</text>')

    x0, y0 = left, top + plot_h
    x1, y1 = left + plot_w, top
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333" stroke-width="1.5"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333" stroke-width="1.5"/>')

    ticks = 6
    for i in range(ticks + 1):
        v = max_y * i / ticks
        y = y_px(v)
        parts.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{x0 - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial">{v:.1f}</text>')

    for i, rate in enumerate(rates):
        cx = left + (i + 0.5) * group_w
        rv = returned[i]
        uv = unmapped[i]
        rx = cx - gap / 2 - bar_w
        ux = cx + gap / 2
        ry = y_px(rv)
        uy = y_px(uv)
        rh = y0 - ry
        uh = y0 - uy
        parts.append(f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{bar_w:.2f}" height="{rh:.2f}" fill="#2E86AB"/>')
        parts.append(f'<rect x="{ux:.2f}" y="{uy:.2f}" width="{bar_w:.2f}" height="{uh:.2f}" fill="#F18F01"/>')
        parts.append(f'<text x="{cx:.2f}" y="{y0 + 18:.2f}" text-anchor="middle" font-size="10" font-family="Arial">{rate:g}</text>')

    parts.append(f'<text x="{left + plot_w/2:.2f}" y="{height - 20}" text-anchor="middle" font-size="13" font-family="Arial">Request Rate</text>')
    parts.append(
        f'<text x="20" y="{top + plot_h/2:.2f}" transform="rotate(-90 20,{top + plot_h/2:.2f})" text-anchor="middle" font-size="13" font-family="Arial">Pages</text>'
    )
    lx = width - 260
    ly = 48
    parts.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="#2E86AB"/>')
    parts.append(f'<text x="{lx + 22}" y="{ly + 12}" font-size="12" font-family="Arial">Pages Returned</text>')
    parts.append(f'<rect x="{lx}" y="{ly + 22}" width="14" height="14" fill="#F18F01"/>')
    parts.append(f'<text x="{lx + 22}" y="{ly + 34}" font-size="12" font-family="Arial">Pages Unmapped</text>')
    parts.append("</svg>")

    out_path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot returned/unmapped pages by request rate for each prompt size.")
    parser.add_argument("--input", default="results/bench_results.txt")
    parser.add_argument("--out-dir", default="results/bench_pages_by_prompt")
    parser.add_argument("--csv-out", default="results/bench_pages_parsed.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    csv_out = Path(args.csv_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    rows = parse_sections(in_path.read_text(encoding="utf-8"))
    if not rows:
        raise RuntimeError(f"No valid prompt/rate sections found in {in_path}")
    write_csv(rows, csv_out)

    grouped: Dict[int, List[Dict[str, float]]] = {}
    for row in rows:
        p = int(row["prompts"])
        grouped.setdefault(p, []).append(row)

    use_matplotlib = True
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        use_matplotlib = False

    for prompt, prompt_rows in sorted(grouped.items()):
        if use_matplotlib:
            out_file = out_dir / f"prompts_{prompt}_returned_vs_unmapped_by_rate.png"
            plot_prompt_matplotlib(prompt, prompt_rows, out_file)
        else:
            out_file = out_dir / f"prompts_{prompt}_returned_vs_unmapped_by_rate.svg"
            plot_prompt_svg(prompt, prompt_rows, out_file)
        print(f"Saved: {out_file}")

    print(f"Saved parsed CSV: {csv_out}")
    print(f"Prompts found: {', '.join(str(k) for k in sorted(grouped.keys()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

