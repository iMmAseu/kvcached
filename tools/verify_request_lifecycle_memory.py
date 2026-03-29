#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Verify GPU memory changes across one kvcached request lifecycle.

This script is designed to prove three statements:

1. During a request, KV cache allocation makes GPU memory go up.
2. After request completion and ``free()``, blocks/pages can be logically
   released while GPU memory does not immediately go down.
3. Only after ``trim()`` triggers unmap (eventually ``cuMemUnmap``) does GPU
   memory actually go down.

The script keeps the reserved-page pool empty before the request, then makes
``free()`` return all fully freed pages into the reserved pool instead of
unmapping them. Finally it calls ``trim()`` to force unmap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

DTYPE_NAMES = ("float16", "half", "bfloat16", "float32")


@dataclass
class Snapshot:
    name: str
    cuda_used_bytes: int
    cuda_free_bytes: int
    torch_allocated_bytes: int
    torch_reserved_bytes: int
    inuse_pages: int
    reserved_pages: int
    free_pages: int
    active_mapped_bytes: int
    reserved_mapped_bytes: int
    total_mapped_bytes: int
    available_blocks: int


def format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):,.1f} MiB"


def dtype_from_name(name: str) -> torch.dtype:
    if torch is None:
        raise ImportError(
            "PyTorch is required to run this script. Please use a Python "
            "environment where `import torch` succeeds.")
    key = name.lower().strip()
    dtype_table: Dict[str, torch.dtype] = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if key not in dtype_table:
        raise ValueError(
            f"Unsupported dtype '{name}'. Choose from: "
            f"{', '.join(sorted(DTYPE_NAMES))}"
        )
    return dtype_table[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one request lifecycle in kvcached: alloc raises GPU memory, "
            "free() keeps memory mapped, trim()/unmap releases it."
        ))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-kv-buffers", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--request-pages",
        type=int,
        default=4,
        help="How many kvcached physical pages one request should occupy.",
    )
    parser.add_argument(
        "--total-pages",
        type=int,
        default=32,
        help="Total kvcached physical pages managed by the experiment.",
    )
    parser.add_argument(
        "--max-reserved-pages",
        type=int,
        default=None,
        help=(
            "Reserved page capacity after free(). "
            "Defaults to request-pages so free() does not unmap."
        ),
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=50,
        help="Sleep this long after each stage before sampling memory.",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional path to write a JSON report.",
    )
    return parser.parse_args()


def configure_env(max_reserved_pages: int) -> None:
    os.environ["KVCACHED_CONTIGUOUS_LAYOUT"] = "true"
    os.environ["KVCACHED_PAGE_PREALLOC_ENABLED"] = "false"
    os.environ["KVCACHED_MIN_RESERVED_PAGES"] = "0"
    os.environ["KVCACHED_MAX_RESERVED_PAGES"] = str(max_reserved_pages)


def capture_snapshot(name: str, manager: Any, device: torch.device,
                     compound_page_bytes: int,
                     settle_ms: int) -> Snapshot:
    torch.cuda.synchronize(device)
    if settle_ms > 0:
        time.sleep(settle_ms / 1000.0)
        torch.cuda.synchronize(device)

    cuda_free_bytes, cuda_total_bytes = torch.cuda.mem_get_info(device)
    page_allocator = manager.page_allocator
    reserved_pages = page_allocator.get_num_reserved_pages()
    active_mapped_bytes = int(manager.get_mapped_memory_size("bytes"))
    reserved_mapped_bytes = reserved_pages * compound_page_bytes

    return Snapshot(
        name=name,
        cuda_used_bytes=int(cuda_total_bytes - cuda_free_bytes),
        cuda_free_bytes=int(cuda_free_bytes),
        torch_allocated_bytes=int(torch.cuda.memory_allocated(device)),
        torch_reserved_bytes=int(torch.cuda.memory_reserved(device)),
        inuse_pages=int(page_allocator.get_num_inuse_pages()),
        reserved_pages=int(reserved_pages),
        free_pages=int(page_allocator.get_num_free_pages()),
        active_mapped_bytes=active_mapped_bytes,
        reserved_mapped_bytes=reserved_mapped_bytes,
        total_mapped_bytes=active_mapped_bytes + reserved_mapped_bytes,
        available_blocks=int(manager.available_size()),
    )


def print_snapshot(snapshot: Snapshot, baseline: Snapshot | None = None) -> None:
    print(f"\n[{snapshot.name}]")
    print(
        f"  cuda_used={format_bytes(snapshot.cuda_used_bytes)}"
        f"  cuda_free={format_bytes(snapshot.cuda_free_bytes)}"
    )
    print(
        f"  torch_allocated={format_bytes(snapshot.torch_allocated_bytes)}"
        f"  torch_reserved={format_bytes(snapshot.torch_reserved_bytes)}"
    )
    print(
        f"  inuse_pages={snapshot.inuse_pages}"
        f"  reserved_pages={snapshot.reserved_pages}"
        f"  free_pages={snapshot.free_pages}"
        f"  available_blocks={snapshot.available_blocks}"
    )
    print(
        f"  active_mapped={format_bytes(snapshot.active_mapped_bytes)}"
        f"  reserved_mapped={format_bytes(snapshot.reserved_mapped_bytes)}"
        f"  total_mapped={format_bytes(snapshot.total_mapped_bytes)}"
    )
    if baseline is not None:
        print(
            f"  delta_vs_{baseline.name}:"
            f" cuda_used={format_bytes(snapshot.cuda_used_bytes - baseline.cuda_used_bytes)}"
            f"  total_mapped={format_bytes(snapshot.total_mapped_bytes - baseline.total_mapped_bytes)}"
        )


def touch_allocated_pages(raw_tensor: torch.Tensor, page_ids: List[int],
                          compound_page_bytes: int) -> None:
    element_size = raw_tensor.element_size()
    flat = raw_tensor.view(-1)
    for i, page_id in enumerate(sorted(set(page_ids))):
        elem_idx = (page_id * compound_page_bytes) // element_size
        flat[elem_idx] = flat.new_tensor((i % 13) + 1)


def evaluate(
    baseline: Snapshot,
    running: Snapshot,
    after_free: Snapshot,
    after_trim: Snapshot,
    expected_pages: int,
    compound_page_bytes: int,
) -> Dict[str, Dict[str, Any]]:
    expected_total = expected_pages * compound_page_bytes
    delta_alloc = running.cuda_used_bytes - baseline.cuda_used_bytes
    delta_free = after_free.cuda_used_bytes - running.cuda_used_bytes
    delta_trim = after_trim.cuda_used_bytes - after_free.cuda_used_bytes

    rise_threshold = max(expected_total // 2, compound_page_bytes)
    stable_tolerance = max(compound_page_bytes // 2, 64 * 1024 * 1024)
    drop_threshold = max(expected_total // 2, compound_page_bytes)

    return {
        "claim_1_alloc_increases_gpu_memory": {
            "ok": (
                running.inuse_pages == expected_pages
                and running.active_mapped_bytes == expected_total
                and delta_alloc >= rise_threshold
            ),
            "details": {
                "expected_pages": expected_pages,
                "expected_total_mapped_bytes": expected_total,
                "cuda_used_delta_bytes": delta_alloc,
                "threshold_bytes": rise_threshold,
            },
        },
        "claim_2_free_is_logical_release_but_memory_stays": {
            "ok": (
                after_free.inuse_pages == 0
                and after_free.reserved_pages == expected_pages
                and after_free.active_mapped_bytes == 0
                and after_free.reserved_mapped_bytes == expected_total
                and abs(delta_free) <= stable_tolerance
            ),
            "details": {
                "cuda_used_delta_from_running_bytes": delta_free,
                "stable_tolerance_bytes": stable_tolerance,
                "reserved_pages_after_free": after_free.reserved_pages,
            },
        },
        "claim_3_only_unmap_makes_gpu_memory_drop": {
            "ok": (
                after_trim.reserved_pages == 0
                and after_trim.total_mapped_bytes == 0
                and (-delta_trim) >= drop_threshold
            ),
            "details": {
                "cuda_used_delta_from_after_free_bytes": delta_trim,
                "drop_threshold_bytes": drop_threshold,
            },
        },
    }


def print_verdicts(verdicts: Dict[str, Dict[str, Any]]) -> None:
    print("\n[verdict]")
    for name, verdict in verdicts.items():
        status = "PASS" if verdict["ok"] else "FAIL"
        print(f"  {status}  {name}")
        details = verdict["details"]
        print(
            "        "
            + ", ".join(
                f"{key}={format_bytes(val) if 'bytes' in key else val}"
                for key, val in details.items()
            )
        )


def build_report(args: argparse.Namespace, baseline: Snapshot,
                 running: Snapshot, after_free: Snapshot,
                 after_trim: Snapshot, verdicts: Dict[str, Dict[str, Any]],
                 meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "args": vars(args),
        "meta": meta,
        "snapshots": {
            "baseline": asdict(baseline),
            "running": asdict(running),
            "after_free": asdict(after_free),
            "after_trim": asdict(after_trim),
        },
        "verdicts": verdicts,
    }


def main() -> int:
    args = parse_args()
    if torch is None:
        print(
            "PyTorch is not installed in the current Python environment.",
            file=sys.stderr,
        )
        return 1
    if not torch.cuda.is_available():
        print("CUDA is not available.", file=sys.stderr)
        return 1

    dtype = dtype_from_name(args.dtype)
    max_reserved_pages = (args.max_reserved_pages
                          if args.max_reserved_pages is not None
                          else args.request_pages)
    if max_reserved_pages < args.request_pages:
        raise ValueError(
            "--max-reserved-pages must be >= --request-pages, otherwise free() "
            "will already unmap some pages and the experiment becomes mixed.")

    configure_env(max_reserved_pages)

    from kvcached.integration.vllm.interfaces import init_kvcached, shutdown_kvcached
    from kvcached.kv_cache_manager import KVCacheManager
    from kvcached.utils import PAGE_SIZE
    from kvcached.vmm_ops import create_kv_tensors

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    cell_size = args.num_heads * args.head_dim * dtype.itemsize
    block_mem_size = args.block_size * cell_size
    if PAGE_SIZE % block_mem_size != 0:
        raise ValueError(
            "For this verification script, PAGE_SIZE must be divisible by "
            "block_mem_size exactly.\n"
            f"PAGE_SIZE={PAGE_SIZE}, block_mem_size={block_mem_size}. "
            "Adjust block-size/num-heads/head-dim/dtype or "
            "KVCACHED_PAGE_SIZE_MB.")

    blocks_per_page = PAGE_SIZE // block_mem_size
    request_blocks = args.request_pages * blocks_per_page
    num_blocks = args.total_pages * blocks_per_page
    mem_size_per_layer = num_blocks * block_mem_size
    compound_page_bytes = PAGE_SIZE * args.num_layers * args.num_kv_buffers

    print("[setup]")
    print(f"  device={device}")
    print(f"  dtype={dtype}")
    print(f"  num_layers={args.num_layers}  num_kv_buffers={args.num_kv_buffers}")
    print(
        f"  block_size={args.block_size}  num_heads={args.num_heads}  head_dim={args.head_dim}"
    )
    print(
        f"  page_size={PAGE_SIZE} ({format_bytes(PAGE_SIZE)})"
        f"  block_mem_size={block_mem_size} ({format_bytes(block_mem_size)})"
    )
    print(
        f"  blocks_per_page={blocks_per_page}"
        f"  request_pages={args.request_pages}"
        f"  request_blocks={request_blocks}"
    )
    print(
        f"  total_pages={args.total_pages}"
        f"  num_blocks={num_blocks}"
        f"  compound_page_bytes={format_bytes(compound_page_bytes)}"
    )
    print(
        f"  reserved_pool_before_request=0"
        f"  reserved_pool_after_free<={max_reserved_pages}"
    )

    raw_kv_tensors: List[torch.Tensor] = []
    manager = None

    try:
        init_kvcached(
            tp_rank=0,
            tp_size=1,
            is_worker=False,
            device=args.device,
            async_sched=False,
        )
        raw_kv_tensors = create_kv_tensors(
            mem_size_per_layer * args.num_kv_buffers,
            dtype.itemsize,
            args.device,
            args.num_layers,
            num_kv_buffers=args.num_kv_buffers,
        )
        if not raw_kv_tensors:
            raise RuntimeError("create_kv_tensors returned no tensors")

        manager = KVCacheManager(
            num_blocks=num_blocks,
            block_size=args.block_size,
            cell_size=cell_size,
            num_layers=args.num_layers,
            tp_size=1,
            async_sched=False,
            reserve_null_block=False,
            num_kv_buffers=args.num_kv_buffers,
        )
        manager._post_init_done.wait(timeout=10.0)
        if not manager._post_init_done.is_set():
            raise TimeoutError("KVCacheManager post-init did not finish within 10s")

        manager.trim()
        baseline = capture_snapshot(
            "baseline", manager, device, compound_page_bytes, args.settle_ms)
        print_snapshot(baseline)

        block_ids = manager.alloc(request_blocks)
        if block_ids is None or len(block_ids) != request_blocks:
            raise RuntimeError(
                f"alloc({request_blocks}) failed, available={manager.available_size()}")

        page_ids = sorted({
            manager.page_allocator.get_page_id(block_id, manager.block_mem_size)
            for block_id in block_ids
        })
        if len(page_ids) != args.request_pages:
            raise RuntimeError(
                f"Expected {args.request_pages} pages, got {len(page_ids)} pages: {page_ids}")

        touch_allocated_pages(raw_kv_tensors[0], page_ids, compound_page_bytes)
        running = capture_snapshot(
            "request_running", manager, device, compound_page_bytes, args.settle_ms)
        print_snapshot(running, baseline)

        manager.free(block_ids)
        after_free = capture_snapshot(
            "after_free", manager, device, compound_page_bytes, args.settle_ms)
        print_snapshot(after_free, baseline)

        manager.trim()
        after_trim = capture_snapshot(
            "after_trim_unmap", manager, device, compound_page_bytes, args.settle_ms)
        print_snapshot(after_trim, baseline)

        verdicts = evaluate(
            baseline=baseline,
            running=running,
            after_free=after_free,
            after_trim=after_trim,
            expected_pages=args.request_pages,
            compound_page_bytes=compound_page_bytes,
        )
        print_verdicts(verdicts)

        report = build_report(
            args=args,
            baseline=baseline,
            running=running,
            after_free=after_free,
            after_trim=after_trim,
            verdicts=verdicts,
            meta={
                "page_size_bytes": PAGE_SIZE,
                "cell_size_bytes": cell_size,
                "block_mem_size_bytes": block_mem_size,
                "blocks_per_page": blocks_per_page,
                "request_blocks": request_blocks,
                "compound_page_bytes": compound_page_bytes,
            },
        )
        if args.report_json:
            with open(args.report_json, "w", encoding="utf-8") as fp:
                json.dump(report, fp, indent=2)
            print(f"\n[report] saved to {args.report_json}")

        all_ok = all(v["ok"] for v in verdicts.values())
        return 0 if all_ok else 1
    finally:
        raw_kv_tensors.clear()
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
        except Exception:
            pass
        try:
            shutdown_kvcached()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
