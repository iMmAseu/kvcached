#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List


HEADER_RE = re.compile(
    r"^\s*total\s+reqs\s*:\s*(?P<reqs>\d+)\s*/\s*rates\s*:\s*(?P<rate>[0-9]+(?:\.[0-9]+)?)\s*$",
    re.IGNORECASE,
)
RETURNED_RE = re.compile(r"Total Pages Returned:\s*(\d+)", re.IGNORECASE)
UNMAP_RE = re.compile(r"UNMAP Pages:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
UNIQUE_RETURNED_RE = re.compile(r"Unique Returned Pages:\s*(\d+)", re.IGNORECASE)


def parse_easy_text(text: str) -> List[Dict[str, float]]:
    lines = text.splitlines()
    rows: List[Dict[str, float]] = []

    cur_reqs = None
    cur_rate = None
    cur_returned = None
    cur_unmap = None
    cur_unmap_den = None
    cur_unique = None

    def flush_if_ready() -> None:
        nonlocal cur_reqs, cur_rate, cur_returned, cur_unmap, cur_unmap_den, cur_unique
        if (cur_reqs is None or cur_rate is None or cur_returned is None
                or cur_unmap is None or cur_unique is None):
            return
        rows.append({
            "total_reqs": float(cur_reqs),
            "request_rate": float(cur_rate),
            "pages_returned": float(cur_returned),
            "pages_unmapped": float(cur_unmap),
            "unique_returned_pages": float(cur_unique),
            "unmap_denominator": float(cur_unmap_den if cur_unmap_den is not None else 0),
        })
        cur_reqs = None
        cur_rate = None
        cur_returned = None
        cur_unmap = None
        cur_unmap_den = None
        cur_unique = None

    for line in lines:
        hm = HEADER_RE.match(line.strip())
        if hm:
            flush_if_ready()
            cur_reqs = int(hm.group("reqs"))
            cur_rate = float(hm.group("rate"))
            continue

        rm = RETURNED_RE.search(line)
        if rm:
            cur_returned = int(rm.group(1))
            continue

        um = UNMAP_RE.search(line)
        if um:
            cur_unmap = int(um.group(1))
            cur_unmap_den = int(um.group(2))
            continue

        uq = UNIQUE_RETURNED_RE.search(line)
        if uq:
            cur_unique = int(uq.group(1))
            flush_if_ready()
            continue

    flush_if_ready()
    rows.sort(key=lambda x: (x["total_reqs"], x["request_rate"]))
    return rows


def write_csv(rows: List[Dict[str, float]], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "total_reqs",
                "request_rate",
                "pages_returned",
                "pages_unmapped",
                "unique_returned_pages",
                "unmap_denominator",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_returned_only_matplotlib(
    title: str,
    x_label: str,
    x_values: List[float],
    rows: List[Dict[str, float]],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(x_values)))
    width = 0.16
    returned = [r["pages_returned"] for r in rows]

    plt.figure(figsize=(10, 5))
    plt.bar(xs, returned, width=width, color="#2E86AB", label="Total Pages Returned")

    plt.xticks(xs, [f"{v:g}" for v in x_values])
    plt.xlabel(x_label)
    plt.ylabel("Pages")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_unmap_unique_matplotlib(
    title: str,
    x_label: str,
    x_values: List[float],
    rows: List[Dict[str, float]],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(x_values)))
    width = 0.18
    unmapped = [r["pages_unmapped"] for r in rows]
    unique_ret = [r["unique_returned_pages"] for r in rows]

    plt.figure(figsize=(10, 5))
    plt.bar([x - width / 2 for x in xs], unmapped, width=width, color="#F18F01", label="UNMAP Pages")
    plt.bar([x + width / 2 for x in xs], unique_ret, width=width, color="#4CAF50", label="Unique Returned Pages")

    plt.xticks(xs, [f"{v:g}" for v in x_values])
    plt.xlabel(x_label)
    plt.ylabel("Pages")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_returned_only_svg(
    title: str,
    x_label: str,
    x_values: List[float],
    rows: List[Dict[str, float]],
    out_path: Path,
) -> None:
    returned = [r["pages_returned"] for r in rows]

    width = 1200
    height = 560
    left = 80
    right = 30
    top = 60
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_y = max(max(returned, default=0), 1) * 1.12
    n = len(x_values)
    group_w = plot_w / max(1, n)
    bar_w = group_w * 0.18

    def y_px(v: float) -> float:
        return top + (plot_h - (v / max_y) * plot_h)

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>')

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

    for i, xv in enumerate(x_values):
        cx = left + (i + 0.5) * group_w
        val = returned[i]
        bx = cx - bar_w / 2
        by = y_px(val)
        bh = y0 - by
        parts.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" fill="#2E86AB"/>')
        parts.append(f'<text x="{cx:.2f}" y="{y0 + 18:.2f}" text-anchor="middle" font-size="10" font-family="Arial">{xv:g}</text>')

    parts.append(f'<text x="{left + plot_w/2:.2f}" y="{height - 20}" text-anchor="middle" font-size="13" font-family="Arial">{x_label}</text>')
    parts.append(
        f'<text x="20" y="{top + plot_h/2:.2f}" transform="rotate(-90 20,{top + plot_h/2:.2f})" text-anchor="middle" font-size="13" font-family="Arial">Pages</text>'
    )
    lx = width - 320
    ly = 46
    parts.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="#2E86AB"/>')
    parts.append(f'<text x="{lx + 22}" y="{ly + 12}" font-size="12" font-family="Arial">Total Pages Returned</text>')
    parts.append("</svg>")

    out_path.write_text("\n".join(parts), encoding="utf-8")


def _plot_unmap_unique_svg(
    title: str,
    x_label: str,
    x_values: List[float],
    rows: List[Dict[str, float]],
    out_path: Path,
) -> None:
    unmapped = [r["pages_unmapped"] for r in rows]
    unique_ret = [r["unique_returned_pages"] for r in rows]

    width = 1200
    height = 560
    left = 80
    right = 30
    top = 60
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_y = max(max(unmapped, default=0), max(unique_ret, default=0), 1) * 1.12
    n = len(x_values)
    group_w = plot_w / max(1, n)
    bar_w = group_w * 0.16
    gap = group_w * 0.05

    def y_px(v: float) -> float:
        return top + (plot_h - (v / max_y) * plot_h)

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>')

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

    for i, xv in enumerate(x_values):
        cx = left + (i + 0.5) * group_w
        vals = [unmapped[i], unique_ret[i]]
        colors = ["#F18F01", "#4CAF50"]
        for j, (val, color) in enumerate(zip(vals, colors)):
            bx = cx + (j - 0.5) * (bar_w + gap) - bar_w / 2
            by = y_px(val)
            bh = y0 - by
            parts.append(f'<rect x="{bx:.2f}" y="{by:.2f}" width="{bar_w:.2f}" height="{bh:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{cx:.2f}" y="{y0 + 18:.2f}" text-anchor="middle" font-size="10" font-family="Arial">{xv:g}</text>')

    parts.append(f'<text x="{left + plot_w/2:.2f}" y="{height - 20}" text-anchor="middle" font-size="13" font-family="Arial">{x_label}</text>')
    parts.append(
        f'<text x="20" y="{top + plot_h/2:.2f}" transform="rotate(-90 20,{top + plot_h/2:.2f})" text-anchor="middle" font-size="13" font-family="Arial">Pages</text>'
    )
    lx = width - 320
    ly = 46
    parts.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="#F18F01"/>')
    parts.append(f'<text x="{lx + 22}" y="{ly + 12}" font-size="12" font-family="Arial">UNMAP Pages</text>')
    parts.append(f'<rect x="{lx}" y="{ly + 22}" width="14" height="14" fill="#4CAF50"/>')
    parts.append(f'<text x="{lx + 22}" y="{ly + 34}" font-size="12" font-family="Arial">Unique Returned Pages</text>')
    parts.append("</svg>")

    out_path.write_text("\n".join(parts), encoding="utf-8")


def _plot_split(
    title_prefix: str,
    x_label: str,
    x_values: List[float],
    rows: List[Dict[str, float]],
    out_base: Path,
    use_matplotlib: bool,
) -> List[Path]:
    outputs: List[Path] = []
    if use_matplotlib:
        p1 = out_base.parent / f"{out_base.name}_returned_only.png"
        p2 = out_base.parent / f"{out_base.name}_unmap_unique.png"
        _plot_returned_only_matplotlib(
            title=f"{title_prefix}: Total Pages Returned",
            x_label=x_label,
            x_values=x_values,
            rows=rows,
            out_path=p1,
        )
        _plot_unmap_unique_matplotlib(
            title=f"{title_prefix}: UNMAP vs Unique Returned",
            x_label=x_label,
            x_values=x_values,
            rows=rows,
            out_path=p2,
        )
        outputs.extend([p1, p2])
        return outputs

    p1 = out_base.parent / f"{out_base.name}_returned_only.svg"
    p2 = out_base.parent / f"{out_base.name}_unmap_unique.svg"
    _plot_returned_only_svg(
        title=f"{title_prefix}: Total Pages Returned",
        x_label=x_label,
        x_values=x_values,
        rows=rows,
        out_path=p1,
    )
    _plot_unmap_unique_svg(
        title=f"{title_prefix}: UNMAP vs Unique Returned",
        x_label=x_label,
        x_values=x_values,
        rows=rows,
        out_path=p2,
    )
    outputs.extend([p1, p2])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot three metrics (returned/unmapped/unique returned) from "
            "bench_results_easy.txt by fixed-reqs and fixed-rate views."
        ))
    parser.add_argument("--input", default="results/bench_results_easy.txt")
    parser.add_argument("--out-dir", default="results/bench_easy_plots")
    parser.add_argument("--csv-out", default="results/bench_results_easy_parsed.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    csv_out = Path(args.csv_out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    rows = parse_easy_text(in_path.read_text(encoding="utf-8"))
    if not rows:
        raise RuntimeError(f"No valid records parsed from {in_path}")
    write_csv(rows, csv_out)

    use_matplotlib = True
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        use_matplotlib = False

    # Round 1: fixed total_reqs, compare rates.
    by_reqs: Dict[int, List[Dict[str, float]]] = {}
    for r in rows:
        by_reqs.setdefault(int(r["total_reqs"]), []).append(r)

    for reqs, req_rows in sorted(by_reqs.items()):
        req_rows = sorted(req_rows, key=lambda x: x["request_rate"])
        x_values = [r["request_rate"] for r in req_rows]
        title_prefix = f"Fixed Total Reqs={reqs}"
        out_paths = _plot_split(
            title_prefix=title_prefix,
            x_label="Request Rate",
            x_values=x_values,
            rows=req_rows,
            out_base=out_dir / f"fixed_reqs_{reqs}_by_rate",
            use_matplotlib=use_matplotlib,
        )
        for p in out_paths:
            print(f"Saved: {p}")

    # Round 2: fixed request_rate, compare total_reqs.
    by_rate: Dict[float, List[Dict[str, float]]] = {}
    for r in rows:
        by_rate.setdefault(float(r["request_rate"]), []).append(r)

    for rate, rate_rows in sorted(by_rate.items(), key=lambda x: x[0]):
        rate_rows = sorted(rate_rows, key=lambda x: x["total_reqs"])
        x_values = [r["total_reqs"] for r in rate_rows]
        title_prefix = f"Fixed Request Rate={rate:g}"
        out_paths = _plot_split(
            title_prefix=title_prefix,
            x_label="Total Reqs",
            x_values=x_values,
            rows=rate_rows,
            out_base=out_dir / f"fixed_rate_{rate:g}_by_reqs",
            use_matplotlib=use_matplotlib,
        )
        for p in out_paths:
            print(f"Saved: {p}")

    print(f"Saved parsed CSV: {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
