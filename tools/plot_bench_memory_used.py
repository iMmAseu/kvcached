#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Plot GPU/KV memory trend from a kvcached benchmark text report."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


FREE_ROW_RE = re.compile(
    r"^\(.*?\)\s+"
    r"(?P<t>[0-9]+(?:\.[0-9]+)?)\s+\|\s+"
    r"(?P<call>[0-9]+)\s+\|\s+"
    r"(?P<blks>[0-9]+)\s+\|\s+"
    r"(?P<pg_ret>[0-9]+)\s+\|\s+"
    r"(?P<unmap>[0-9]+)\s+\|\s+"
    r"(?P<unmap_ms>[0-9]+(?:\.[0-9]+)?)\s+\|\s+"
    r"(?P<rsvd_pg>[0-9]+)\s+\|\s+"
    r"(?P<inuse_pg>[0-9]+)\s+\|\s+"
    r"(?P<free_pg>[0-9]+)\s+\|\s+"
    r"(?P<avail_blk>[0-9]+)\s+\|\s+"
    r"(?P<cuda_used>[0-9]+(?:\.[0-9]+)?)\s+\|\s+"
    r"(?P<cuda_free>[0-9]+(?:\.[0-9]+)?)\s*$"
)

STATE_ROW_RE = re.compile(
    r"^\(.*?\)\s+"
    r"(?P<t>[0-9]+(?:\.[0-9]+)?)\s+\|\s+"
    r"(?P<kind>\w+)\s+\|\s+"
    r"(?P<reason>.+?)\s+\|\s+"
    r"(?P<inuse_pg>[0-9]+)\s+\|\s+"
    r"(?P<rsvd_pg>[0-9]+)\s+\|\s+"
    r"(?P<kv_total>[0-9]+(?:\.[0-9]+)?)\s+\|\s+"
    r"(?P<cuda_used>[0-9]+(?:\.[0-9]+)?)\s*$"
)

MEM_RE = re.compile(
    r"^\(.*?\)\s+"
    r"(?P<name>KV Active \(in-use\)|KV Reserved \(prealloc\)|KV Total Mapped|CUDA Used Total):\s+"
    r"(?P<value>(?:n/a|[0-9]+(?:\.[0-9]+)?))\s+GB"
)

PROOF_SAMPLE_RE = re.compile(
    r"^\(.*?\)\s+(?P<label>baseline|running_peak|logical_free_no_unmap|unmap_event):\s+"
    r"reason=(?P<reason>.*?)\s+"
    r"t=(?P<t>[0-9]+(?:\.[0-9]+)?)s.*?"
    r"kv_total=(?P<kv_total>[0-9]+(?:\.[0-9]+)?)GB\s+"
    r"cuda_used=(?P<cuda_used>[0-9]+(?:\.[0-9]+)?)GB"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot memory trend from results/bench_memory_used/bench_memory_used.txt",
    )
    parser.add_argument(
        "--input",
        default="results/bench_memory_used/bench_memory_used.txt",
        help="Path to the benchmark memory text report.",
    )
    parser.add_argument(
        "--output",
        default="results/bench_memory_used/bench_memory_used.png",
        help="Path to the output PNG.",
    )
    parser.add_argument(
        "--title",
        default="kvcached GPU Memory Trend",
        help="Chart title.",
    )
    return parser.parse_args()


def _to_float(raw: str) -> Optional[float]:
    if raw == "n/a":
        return None
    return float(raw)


def parse_report(path: Path) -> Dict[str, Any]:
    free_rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []
    proof_samples: Dict[str, Dict[str, Any]] = {}
    mem_summary: Dict[str, Optional[float]] = {}

    lines = path.read_text(encoding="utf-8").splitlines()

    for line in lines:
        free_match = FREE_ROW_RE.match(line)
        if free_match:
            gd = free_match.groupdict()
            free_rows.append({
                "t": float(gd["t"]),
                "call": int(gd["call"]),
                "blks": int(gd["blks"]),
                "pg_ret": int(gd["pg_ret"]),
                "unmap": int(gd["unmap"]),
                "unmap_ms": float(gd["unmap_ms"]),
                "rsvd_pg": int(gd["rsvd_pg"]),
                "inuse_pg": int(gd["inuse_pg"]),
                "free_pg": int(gd["free_pg"]),
                "avail_blk": int(gd["avail_blk"]),
                "cuda_used": float(gd["cuda_used"]),
                "cuda_free": float(gd["cuda_free"]),
            })
            continue

        state_match = STATE_ROW_RE.match(line)
        if state_match:
            gd = state_match.groupdict()
            state_rows.append({
                "t": float(gd["t"]),
                "kind": gd["kind"],
                "reason": gd["reason"].strip(),
                "inuse_pg": int(gd["inuse_pg"]),
                "rsvd_pg": int(gd["rsvd_pg"]),
                "kv_total": float(gd["kv_total"]),
                "cuda_used": float(gd["cuda_used"]),
            })
            continue

        mem_match = MEM_RE.match(line)
        if mem_match:
            mem_summary[mem_match.group("name")] = _to_float(mem_match.group("value"))
            continue

        proof_match = PROOF_SAMPLE_RE.match(line)
        if proof_match:
            gd = proof_match.groupdict()
            proof_samples[gd["label"]] = {
                "label": gd["label"],
                "reason": gd["reason"].strip(),
                "t": float(gd["t"]),
                "kv_total": float(gd["kv_total"]),
                "cuda_used": float(gd["cuda_used"]),
            }

    if not state_rows:
        raise ValueError(f"No state trend rows found in {path}")

    kv_active = mem_summary.get("KV Active (in-use)")
    kv_reserved = mem_summary.get("KV Reserved (prealloc)")

    return {
        "free_rows": free_rows,
        "state_rows": state_rows,
        "proof_samples": proof_samples,
        "kv_active_last": kv_active,
        "kv_reserved_last": kv_reserved,
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print(
            "matplotlib is not installed in the current Python environment.\n"
            "Install it with: python -m pip install matplotlib",
            file=sys.stderr,
        )
        return 1

    report = parse_report(input_path)
    state_rows = report["state_rows"]
    free_rows = report["free_rows"]
    proof_samples = report["proof_samples"]

    xs = [row["t"] for row in state_rows]
    cuda_used = [row["cuda_used"] for row in state_rows]
    kv_total = [row["kv_total"] for row in state_rows]

    if report["kv_active_last"] is not None and report["kv_reserved_last"] is not None:
        last_kv_total = kv_total[-1] if kv_total else 0.0
        last_active = float(report["kv_active_last"])
        last_reserved = float(report["kv_reserved_last"])
        kv_active = [
            max(v - last_reserved, 0.0)
            if i < len(kv_total) - 1 else last_active
            for i, v in enumerate(kv_total)
        ]
        kv_reserved = [
            min(v, last_reserved)
            if i < len(kv_total) - 1 else last_reserved
            for i, v in enumerate(kv_total)
        ]
        if last_kv_total == 0.0:
            kv_active = [row["inuse_pg"] * 0.0 for row in state_rows]
            kv_reserved = kv_total[:]
    else:
        kv_active = [0.0 for _ in state_rows]
        kv_reserved = kv_total[:]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )

    ax0 = axes[0]
    ax1 = axes[1]

    ax0.plot(xs, cuda_used, color="#203864", linewidth=2.4, label="CUDA Used (GB)")
    ax0.plot(xs, kv_total, color="#d97706", linewidth=2.2, label="KV Total Mapped (GB)")
    ax0.plot(xs, kv_active, color="#059669", linewidth=1.8, linestyle="--", label="KV Active Est. (GB)")
    ax0.plot(xs, kv_reserved, color="#b91c1c", linewidth=1.8, linestyle=":", label="KV Reserved Est. (GB)")

    for row in free_rows:
        if row["unmap"] > 0:
            ax0.axvline(row["t"], color="#dc2626", alpha=0.25, linewidth=1.1)

    colors = {
        "baseline": "#475569",
        "running_peak": "#2563eb",
        "logical_free_no_unmap": "#ea580c",
        "unmap_event": "#dc2626",
    }
    for label, sample in proof_samples.items():
        c = colors.get(label, "#111827")
        ax0.scatter([sample["t"]], [sample["cuda_used"]], color=c, s=55, zorder=5)

    ax0.set_ylabel("Memory (GB)")
    ax0.set_title(args.title)
    ax0.grid(alpha=0.18, axis="y")
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax0.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(loc="best", ncol=2)

    inuse_pages = [row["inuse_pg"] for row in state_rows]
    reserved_pages = [row["rsvd_pg"] for row in state_rows]
    ax1.plot(xs, inuse_pages, color="#0f766e", linewidth=2.0, label="In-use Pages")
    ax1.plot(xs, reserved_pages, color="#7c3aed", linewidth=2.0, label="Reserved Pages")
    ax1.set_xlabel("Time Since Process Start (s)")
    ax1.set_ylabel("Pages")
    ax1.grid(alpha=0.22)
    ax1.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    print(f"Saved plot to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
