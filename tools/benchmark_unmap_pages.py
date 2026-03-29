#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _parse_int_list(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    vals = [v for v in vals if v > 0]
    if not vals:
        raise ValueError("pages list is empty after parsing")
    return vals


def _dtype_to_bytes(dtype_str: str) -> int:
    s = (dtype_str or "").lower().strip()
    table = {
        "float16": 2,
        "half": 2,
        "bfloat16": 2,
        "float32": 4,
        "float": 4,
        "float64": 8,
        "double": 8,
        "int8": 1,
        "uint8": 1,
    }
    return table.get(s, 2)


def _load_model_config(model_path: str) -> Dict[str, Any]:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found under model path: {model_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    num_layers = cfg.get("num_hidden_layers", cfg.get("n_layer"))
    torch_dtype = cfg.get("torch_dtype")
    return {
        "config_path": str(cfg_path),
        "num_layers": int(num_layers) if num_layers is not None else None,
        "torch_dtype": torch_dtype,
        "dtype_bytes": _dtype_to_bytes(str(torch_dtype)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Micro-benchmark PageAllocator.free_pages() unmap latency with "
            "different page counts per free call."
        ))
    parser.add_argument(
        "--pages-list",
        default="1,2,4,8,16,24,32",
        help="Comma-separated page counts to free per call.",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-kv-buffers", type=int, default=2)
    parser.add_argument(
        "--total-pages",
        type=int,
        default=256,
        help="Virtual pages per layer used by PageAllocator.",
    )
    parser.add_argument(
        "--dtype-bytes",
        type=int,
        default=2,
        help="Element size for create_kv_tensors (2 for fp16/bf16).",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Optional HF model directory (reads config.json).",
    )
    parser.add_argument(
        "--use-model-config",
        action="store_true",
        help="Override num_layers/dtype_bytes from --model-path config.json.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--force-zero-reserved-pages",
        action="store_true",
        help=(
            "Set KVCACHED_MIN_RESERVED_PAGES=0 and KVCACHED_MAX_RESERVED_PAGES=0 "
            "to force every free_pages to go through unmap path."
        ),
    )
    parser.add_argument(
        "--prepared-batch-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pre-allocate one fixed page batch per pages_to_free value, then "
            "measure only free_pages()/unmap while re-mapping the same batch "
            "outside the timing window."
        ),
    )
    parser.add_argument(
        "--async-sched",
        action="store_true",
        help="Initialize kvcached with async scheduling.",
    )
    parser.add_argument(
        "--sync-before-free",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call torch.cuda.synchronize() before each free_pages timing.",
    )
    parser.add_argument(
        "--sync-after-free",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call torch.cuda.synchronize() after each free_pages timing.",
    )
    parser.add_argument(
        "--sync-after-remap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --prepared-batch-reuse is enabled, synchronize after the "
            "out-of-window re-map step before the next timed free_pages()."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="/root/kvcached/results/unmap_microbench",
        help="Directory for JSON/CSV/plots.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix in output filename.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate line/scatter plots (requires matplotlib).",
    )
    return parser.parse_args()


def _maybe_plot(summary_rows: List[Dict[str, Any]], out_dir: Path, stem: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib not found; skip plotting")
        return

    xs = [float(r["pages_to_free"]) for r in summary_rows]
    ys = [float(r["mean_unmap_time_ms"]) for r in summary_rows]
    per_page = [float(r["mean_unmap_time_per_page_ms"]) for r in summary_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=1.8)
    plt.xlabel("pages_to_free per free_pages() call")
    plt.ylabel("mean unmap_time_ms")
    plt.title("UNMAP latency vs pages_to_free")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_line_mean_unmap_ms.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(xs, per_page, marker="o", linewidth=1.8)
    plt.xlabel("pages_to_free per free_pages() call")
    plt.ylabel("mean unmap_time_per_page_ms")
    plt.title("Per-page UNMAP latency vs pages_to_free")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_line_mean_unmap_per_page_ms.png", dpi=180)
    plt.close()


def _restore_page_batch(allocator: Any, page_ids: List[int], torch_mod: Any,
                        sync_after_remap: bool) -> None:
    """Restore a fixed page batch outside the timed window.

    This re-maps the same pages and marks them as in-use again without going
    through alloc_page(), so allocation/map latency is not mixed into the next
    free_pages() timing sample.
    """
    with allocator._lock:
        for page_id in page_ids:
            if page_id in allocator.reserved_page_list:
                raise RuntimeError(
                    "prepared-batch-reuse requires reserved pool to be zero; "
                    f"page {page_id} unexpectedly ended up in reserved_page_list")
            try:
                allocator.free_page_list.remove(page_id)
            except ValueError as e:
                raise RuntimeError(
                    f"page {page_id} was not found in free_page_list during restore"
                ) from e
        allocator.num_free_pages -= len(page_ids)

    allocator._map_pages(list(page_ids))
    if sync_after_remap:
        torch_mod.cuda.synchronize()
    allocator._update_memory_usage()


def main() -> int:
    args = parse_args()

    if args.prepared_batch_reuse:
        args.force_zero_reserved_pages = True

    if args.force_zero_reserved_pages:
        os.environ["KVCACHED_MIN_RESERVED_PAGES"] = "0"
        os.environ["KVCACHED_MAX_RESERVED_PAGES"] = "0"
        os.environ["KVCACHED_PAGE_PREALLOC_ENABLED"] = "false"

    model_cfg_meta: Dict[str, Any] = {}
    if args.use_model_config:
        if not args.model_path:
            raise ValueError("--use-model-config requires --model-path")
        model_cfg_meta = _load_model_config(args.model_path)
        if model_cfg_meta.get("num_layers") is not None:
            args.num_layers = int(model_cfg_meta["num_layers"])
        args.dtype_bytes = int(model_cfg_meta["dtype_bytes"])
        print(
            "[setup] model config loaded: "
            f"layers={args.num_layers} torch_dtype={model_cfg_meta.get('torch_dtype')} "
            f"dtype_bytes={args.dtype_bytes}"
        )

    import torch
    from kvcached.integration.vllm.interfaces import init_kvcached, shutdown_kvcached
    from kvcached.page_allocator import PageAllocator
    from kvcached.utils import CONTIGUOUS_LAYOUT, PAGE_SIZE
    from kvcached.vmm_ops import create_kv_tensors

    pages_list = _parse_int_list(args.pages_list)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"unmap_microbench_{now_tag}"
    if args.tag:
        stem = f"{stem}_{args.tag}"

    mem_size_per_layer = int(args.total_pages) * int(PAGE_SIZE)
    compound_page_bytes = int(PAGE_SIZE) * int(args.num_layers) * int(args.num_kv_buffers)

    print(
        "[setup] "
        f"page_size={PAGE_SIZE}B total_pages={args.total_pages} "
        f"num_layers={args.num_layers} num_kv_buffers={args.num_kv_buffers} "
        f"contiguous_layout={CONTIGUOUS_LAYOUT} "
        f"compound_page_bytes={compound_page_bytes}B ({compound_page_bytes / (1024**2):.2f}MB)"
    )
    if args.prepared_batch_reuse:
        print("[setup] prepared_batch_reuse=on (measure free/unmap only; remap happens out-of-window)")

    kv_tensors = None
    allocator = None
    raw_samples: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    try:
        init_kvcached(
            tp_rank=0,
            tp_size=1,
            is_worker=False,
            device=args.device,
            async_sched=args.async_sched,
        )

        # Keep a reference alive to ensure mapped virtual memory backing exists.
        kv_tensors = create_kv_tensors(
            mem_size_per_layer * int(args.num_kv_buffers),
            int(args.dtype_bytes),
            args.device,
            int(args.num_layers),
            num_kv_buffers=int(args.num_kv_buffers),
        )
        if not kv_tensors:
            raise RuntimeError("create_kv_tensors returned empty list")

        allocator = PageAllocator(
            num_layers=int(args.num_layers),
            mem_size_per_layer=mem_size_per_layer,
            page_size=PAGE_SIZE,
            tp_size=1,
            async_sched=bool(args.async_sched),
            contiguous_layout=bool(CONTIGUOUS_LAYOUT),
            enable_page_prealloc=False,
            num_kv_buffers=int(args.num_kv_buffers),
        )
        if args.force_zero_reserved_pages:
            # Extra runtime safety: enforce zero reserved pages even if env was
            # not honored due prior imports in this Python process.
            allocator.min_reserved_pages = 0
            allocator.max_reserved_pages = 0
            allocator.trim()
            print("[setup] enforced reserved pool: min=0 max=0 and trimmed")

        for pages_to_free in pages_list:
            if pages_to_free > args.total_pages:
                raise ValueError(
                    f"pages_to_free={pages_to_free} > total_pages={args.total_pages}")

            total_repeats = int(args.warmup_repeats) + int(args.repeats)
            run_times_ms: List[float] = []
            run_wall_ms: List[float] = []
            run_unmapped_pages: List[int] = []
            prepared_page_ids: List[int] = []

            if args.prepared_batch_reuse:
                if args.force_zero_reserved_pages and allocator.get_num_reserved_pages() > 0:
                    allocator.trim()
                for _ in range(pages_to_free):
                    page = allocator.alloc_page()
                    prepared_page_ids.append(page.page_id)
                if args.sync_before_free:
                    torch.cuda.synchronize()

            for i in range(total_repeats):
                if args.force_zero_reserved_pages and allocator.get_num_reserved_pages() > 0:
                    allocator.trim()
                if args.prepared_batch_reuse:
                    page_ids = list(prepared_page_ids)
                else:
                    page_ids = []
                    for _ in range(pages_to_free):
                        page = allocator.alloc_page()
                        page_ids.append(page.page_id)

                if args.sync_before_free:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                num_unmapped, unmap_time_ms, unmapped_page_ids = allocator.free_pages(
                    page_ids)
                if args.sync_after_free:
                    torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - t0) * 1000.0

                is_warmup = i < int(args.warmup_repeats)
                raw_samples.append({
                    "pages_to_free": pages_to_free,
                    "repeat_index": i + 1,
                    "is_warmup": is_warmup,
                    "num_unmapped": int(num_unmapped),
                    "unmap_time_ms": float(unmap_time_ms),
                    "wall_time_ms": float(wall_ms),
                    "returned_page_ids": list(page_ids),
                    "unmapped_page_ids": list(unmapped_page_ids),
                })
                if not is_warmup:
                    run_times_ms.append(float(unmap_time_ms))
                    run_wall_ms.append(float(wall_ms))
                    run_unmapped_pages.append(int(num_unmapped))

                if args.prepared_batch_reuse and i != total_repeats - 1:
                    _restore_page_batch(
                        allocator,
                        page_ids,
                        torch,
                        bool(args.sync_after_remap),
                    )

            mean_unmap_ms = statistics.fmean(run_times_ms) if run_times_ms else 0.0
            mean_wall_ms = statistics.fmean(run_wall_ms) if run_wall_ms else 0.0
            mean_per_page_ms = (mean_unmap_ms / pages_to_free
                                if pages_to_free > 0 else 0.0)
            mean_unmapped = statistics.fmean(run_unmapped_pages) if run_unmapped_pages else 0.0
            full_unmap_ratio = (
                sum(1 for x in run_unmapped_pages if x == pages_to_free) / len(run_unmapped_pages)
                if run_unmapped_pages else 0.0
            )

            summary_rows.append({
                "pages_to_free": pages_to_free,
                "repeats": int(args.repeats),
                "warmup_repeats": int(args.warmup_repeats),
                "mean_num_unmapped": round(mean_unmapped, 6),
                "full_unmap_ratio": round(full_unmap_ratio, 6),
                "mean_unmap_time_ms": round(mean_unmap_ms, 6),
                "p50_unmap_time_ms": round(_percentile(run_times_ms, 50), 6),
                "p95_unmap_time_ms": round(_percentile(run_times_ms, 95), 6),
                "p99_unmap_time_ms": round(_percentile(run_times_ms, 99), 6),
                "min_unmap_time_ms": round(min(run_times_ms) if run_times_ms else 0.0, 6),
                "max_unmap_time_ms": round(max(run_times_ms) if run_times_ms else 0.0, 6),
                "mean_unmap_time_per_page_ms": round(mean_per_page_ms, 6),
                "mean_wall_time_ms": round(mean_wall_ms, 6),
            })
            print(
                f"[result] pages={pages_to_free:>3d} "
                f"mean_unmapped={mean_unmapped:.2f}/{pages_to_free} "
                f"mean_unmap={mean_unmap_ms:.4f}ms "
                f"p95={_percentile(run_times_ms, 95):.4f}ms "
                f"per_page={mean_per_page_ms:.4f}ms "
                f"batch_unmap_bytes={pages_to_free * compound_page_bytes / (1024**2):.2f}MB"
            )

    finally:
        # Keep references until shutdown.
        _ = kv_tensors
        _ = allocator
        try:
            shutdown_kvcached()
        except Exception:
            pass

    result_json = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "device": args.device,
            "pages_list": pages_list,
            "repeats": int(args.repeats),
            "warmup_repeats": int(args.warmup_repeats),
            "num_layers": int(args.num_layers),
            "num_kv_buffers": int(args.num_kv_buffers),
            "total_pages": int(args.total_pages),
            "page_size_bytes": int(PAGE_SIZE),
            "dtype_bytes": int(args.dtype_bytes),
            "contiguous_layout": bool(CONTIGUOUS_LAYOUT),
            "async_sched": bool(args.async_sched),
            "sync_before_free": bool(args.sync_before_free),
            "sync_after_free": bool(args.sync_after_free),
            "sync_after_remap": bool(args.sync_after_remap),
            "force_zero_reserved_pages": bool(args.force_zero_reserved_pages),
            "prepared_batch_reuse": bool(args.prepared_batch_reuse),
            "compound_page_bytes": compound_page_bytes,
            "model_config_meta": model_cfg_meta,
        },
        "summary": summary_rows,
        "raw_samples": raw_samples,
    }

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result_json, indent=2), encoding="utf-8")

    summary_csv_path = out_dir / f"{stem}_summary.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    raw_csv_path = out_dir / f"{stem}_raw.csv"
    with raw_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_samples[0].keys()))
        writer.writeheader()
        writer.writerows(raw_samples)

    if args.plot:
        _maybe_plot(summary_rows, out_dir, stem)

    print(f"[saved] {json_path}")
    print(f"[saved] {summary_csv_path}")
    print(f"[saved] {raw_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
