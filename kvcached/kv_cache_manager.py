# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
kvcached Memory Manager

This module implements a hierarchical memory management system for KV cache:
- Pages: Large memory chunks (e.g., 2MB) that are mapped/unmapped to physical memory
- Blocks: Smaller units within pages that are allocated to store KV cache data
"""

import atexit
import functools
import json
import os
import signal
import threading
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import torch
from kvcached.locks import NoOpLock
from kvcached.page_allocator import Page, PageAllocator
from kvcached.runtime_stats import get_model_weight_bytes
from kvcached.tp_ipc_util import broadcast_kv_tensors_created
from kvcached.utils import PAGE_SIZE, SANITY_CHECK, get_kvcached_logger
from kvcached.vmm_ops import kv_tensors_created

logger = get_kvcached_logger()

# ═══════════════════════════════════════════════════════════════════════════════
# Free Debug Stats (controlled by KVCACHED_FREE_DEBUG=1)
# ═══════════════════════════════════════════════════════════════════════════════
KVCACHED_FREE_DEBUG = os.environ.get("KVCACHED_FREE_DEBUG", "0") == "1"
_FREE_TREND_MAX_ROWS = 12
_FREE_IDLE_REPORT_SECONDS = float(
    os.environ.get("KVCACHED_FREE_DEBUG_IDLE_REPORT_SECONDS", "5"))
_FREE_PAGE_ID_PREVIEW = max(
    0, int(os.environ.get("KVCACHED_FREE_DEBUG_PAGE_ID_PREVIEW", "16")))
_FREE_DUMP_JSON_ON_IDLE = os.environ.get(
    "KVCACHED_FREE_DEBUG_DUMP_JSON_ON_IDLE", "1").lower() in ("1", "true")
_RESERVED_LOG_INTERVAL_SECONDS = float(
    os.environ.get("KVCACHED_RESERVED_LOG_INTERVAL_SECONDS", "0"))
_TRIM_ON_IDLE = os.environ.get(
    "KVCACHED_TRIM_ON_IDLE", "0").lower() in ("1", "true")


def _format_bytes_gb(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes / (1024**3):.2f}"


def _format_percent(part_bytes: Optional[int], total_bytes: Optional[int]) -> str:
    if part_bytes is None or total_bytes is None or total_bytes <= 0:
        return "n/a"
    return f"{(100.0 * part_bytes / total_bytes):.2f}%"


def _format_page_id_preview(page_ids: List[int], max_items: int = _FREE_PAGE_ID_PREVIEW) -> str:
    if not page_ids:
        return "[]"
    if max_items <= 0 or len(page_ids) <= max_items:
        return "[" + ",".join(str(x) for x in page_ids) + "]"
    shown = ",".join(str(x) for x in page_ids[:max_items])
    return f"[{shown},...](total={len(page_ids)})"


def _percentile(values: List[float], q: float) -> float:
    """Return percentile q in [0, 100] using linear interpolation."""
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))

    sorted_vals = sorted(float(v) for v in values)
    idx = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class _FreeDebugStats:
    """Thread-safe stats collector for memory free operations."""

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.time()
        self.total_free_calls = 0
        self.total_blocks_freed = 0
        self.total_free_time_ms = 0.0
        self.total_pages_returned = 0
        self.unique_returned_page_ids: set[int] = set()
        self.unique_unmapped_page_ids: set[int] = set()
        self.samples: List[dict] = []
        self.state_samples: List[dict] = []
        self._last_reported_call = 0
        self._last_reported_state_seq = 0
        self._state_seq = 0
        self._idle_report_timer: Optional[threading.Timer] = None

    def record(self, num_blocks: int, elapsed_ms: float,
               pages_returned: int,
               returned_page_ids: Optional[List[int]] = None,
               unmapped_page_ids: Optional[List[int]] = None,
               memory_snapshot: Optional[dict] = None):
        with self._lock:
            self.total_free_calls += 1
            self.total_blocks_freed += num_blocks
            self.total_free_time_ms += elapsed_ms
            self.total_pages_returned += pages_returned
            if returned_page_ids:
                self.unique_returned_page_ids.update(returned_page_ids)
            if unmapped_page_ids:
                self.unique_unmapped_page_ids.update(unmapped_page_ids)
            if memory_snapshot is not None:
                sample = dict(memory_snapshot)
                sample.setdefault("sample_type", "free")
                sample.setdefault("reason", "free()")
                sample["call"] = self.total_free_calls
                self.samples.append(sample)

    def record_state(self, reason: str, snapshot: dict):
        with self._lock:
            self._state_seq += 1
            sample = dict(snapshot)
            sample["sample_type"] = "state"
            sample["reason"] = reason
            sample["state_seq"] = self._state_seq
            self.state_samples.append(sample)

    def _select_trend_samples(self,
                              samples: Optional[List[dict]] = None,
                              max_rows: int = _FREE_TREND_MAX_ROWS) -> List[dict]:
        samples = list(self.samples if samples is None else samples)
        if len(samples) <= max_rows:
            return samples

        last_idx = len(samples) - 1
        raw_indices = [
            round(i * last_idx / (max_rows - 1)) for i in range(max_rows)
        ]
        seen = set()
        selected = []
        for idx in raw_indices:
            if idx not in seen:
                selected.append(samples[idx])
                seen.add(idx)
        return selected

    def _format_trend_table(self, samples: Optional[List[dict]] = None) -> Optional[str]:
        samples = self._select_trend_samples(samples=samples)
        if not samples:
            return None

        header = (
            " t(s)  | call | blks | pg_ret | unmap | unmap_ms | "
            "rsvd_pg | inuse_pg | free_pg | avail_blk | "
            "cuda_used(GB) | cuda_free(GB)"
        )
        rows = []
        for sa in samples:
            unmap_n = sa.get("num_unmapped", 0)
            unmap_t = sa.get("unmap_time_ms", 0.0)
            rows.append(
                f"{sa['elapsed_s']:>6.2f} | "
                f"{sa['call']:>4d} | "
                f"{sa['num_blocks']:>4d} | "
                f"{sa['pages_returned']:>6d} | "
                f"{unmap_n:>5d} | "
                f"{unmap_t:>8.3f} | "
                f"{sa.get('reserved_pages', 0):>7d} | "
                f"{sa.get('inuse_pages', 0):>8d} | "
                f"{sa.get('free_pages', 0):>7d} | "
                f"{sa.get('avail_blocks', 0):>9d} | "
                f"{_format_bytes_gb(sa.get('cuda_used_bytes')):>13} | "
                f"{_format_bytes_gb(sa.get('cuda_free_bytes')):>13}"
            )
        return "\n".join([
            "",
            "KVCacheManager.free Memory Trend (sampled)",
            header,
            "-" * len(header),
            *rows,
        ])

    def _build_delta_report(self, mark_reported: bool) -> Optional[dict]:
        with self._lock:
            samples = [s for s in self.samples if s["call"] > self._last_reported_call]
            state_samples = [
                s for s in self.state_samples
                if s["state_seq"] > self._last_reported_state_seq
            ]
            if not samples:
                if not state_samples:
                    return None

            first_call = self._last_reported_call + 1 if samples else 0
            last_call = samples[-1]["call"] if samples else self._last_reported_call
            count = last_call - self._last_reported_call if samples else 0
            delta_blocks = sum(sample["num_blocks"] for sample in samples)
            delta_pages = sum(sample["pages_returned"] for sample in samples)
            delta_time_ms = sum(sample["elapsed_ms"] for sample in samples)
            if mark_reported:
                if samples:
                    self._last_reported_call = last_call
                if state_samples:
                    self._last_reported_state_seq = state_samples[-1]["state_seq"]

        return {
            "first_call": first_call,
            "last_call": last_call,
            "count": count,
            "delta_blocks": delta_blocks,
            "delta_pages": delta_pages,
            "delta_time_ms": delta_time_ms,
            "avg_time_ms": delta_time_ms / count if count else 0.0,
            "avg_blocks": delta_blocks / count if count else 0.0,
            "samples": samples,
            "state_samples": state_samples,
        }

    def _format_state_trend_table(
            self,
            state_samples: Optional[List[dict]] = None) -> Optional[str]:
        state_samples = list(
            self.state_samples if state_samples is None else state_samples)
        if not state_samples:
            return None

        selected = self._select_trend_samples(
            samples=state_samples,
            max_rows=_FREE_TREND_MAX_ROWS,
        )
        header = (
            " t(s)  | type  | reason                         | "
            "inuse_pg | rsvd_pg | kv_total(GB) | cuda_used(GB)"
        )
        rows = []
        for sa in selected:
            reason = str(sa.get("reason", ""))[:28]
            rows.append(
                f"{sa.get('elapsed_s', 0.0):>6.2f} | "
                f"{sa.get('sample_type', 'state'):>5s} | "
                f"{reason:<28} | "
                f"{sa.get('inuse_pages', 0):>8d} | "
                f"{sa.get('reserved_pages', 0):>7d} | "
                f"{_format_bytes_gb(sa.get('kvcached_total_mapped_bytes')):>12} | "
                f"{_format_bytes_gb(sa.get('cuda_used_bytes')):>13}"
            )
        return "\n".join([
            "",
            "KVCacheManager State Trend (sampled)",
            header,
            "-" * len(header),
            *rows,
        ])

    def _build_memory_lifecycle_proof(
            self,
            free_samples: List[dict],
            state_samples: List[dict]) -> Optional[dict]:
        timeline: List[dict] = []
        for sa in state_samples:
            item = dict(sa)
            item["_sort_kind"] = 0
            timeline.append(item)
        for sa in free_samples:
            item = dict(sa)
            item["_sort_kind"] = 1
            timeline.append(item)
        if not timeline:
            return None

        timeline.sort(
            key=lambda sa: (
                float(sa.get("elapsed_s", 0.0)),
                int(sa.get("_sort_kind", 0)),
                int(sa.get("state_seq", 0)),
                int(sa.get("call", 0)),
            ))

        prev = None
        for sa in timeline:
            for key in (
                    "kvcached_used_bytes",
                    "kvcached_reserved_bytes",
                    "kvcached_total_mapped_bytes",
                    "cuda_used_bytes",
                    "inuse_pages",
                    "reserved_pages",
            ):
                cur = sa.get(key)
                old = prev.get(key) if prev is not None else None
                if cur is None or old is None:
                    sa[f"delta_{key}"] = None
                else:
                    sa[f"delta_{key}"] = int(cur) - int(old)
            prev = sa

        baseline = next(
            (sa for sa in timeline if sa.get("reason") == "post_init"),
            timeline[0],
        )
        running_candidates = [
            sa for sa in timeline
            if int(sa.get("inuse_pages", 0)) > 0
            or int(sa.get("kvcached_used_bytes", 0)) > 0
        ]
        running_peak = max(
            running_candidates,
            key=lambda sa: (
                int(sa.get("kvcached_used_bytes", 0)),
                int(sa.get("cuda_used_bytes", 0) or 0),
            ),
            default=None,
        )

        no_unmap_free_candidates = [
            sa for sa in free_samples
            if int(sa.get("pages_returned", 0)) > 0
            and int(sa.get("num_unmapped", 0)) == 0
        ]
        logical_release = max(
            no_unmap_free_candidates,
            key=lambda sa: (
                int(sa.get("pages_returned", 0)),
                int(sa.get("reserved_pages", 0)),
                int(sa.get("call", 0)),
            ),
            default=None,
        )

        unmap_candidates = [
            sa for sa in timeline
            if int(sa.get("num_unmapped", 0)) > 0
            or str(sa.get("reason", "")) == "trim()"
        ]
        unmap_event = min(
            unmap_candidates,
            key=lambda sa: (
                int(sa.get("delta_cuda_used_bytes", 0) or 0),
                int(sa.get("delta_kvcached_total_mapped_bytes", 0) or 0),
            ),
            default=None,
        )

        compound_page_bytes = 0
        for sa in timeline:
            page_size = sa.get("page_size_bytes")
            num_layers = sa.get("num_layers")
            num_kv_buffers = sa.get("num_kv_buffers")
            if page_size is not None and num_layers is not None and num_kv_buffers is not None:
                compound_page_bytes = int(page_size) * int(num_layers) * int(
                    num_kv_buffers)
                break
        if compound_page_bytes <= 0:
            compound_page_bytes = 256 * 1024 * 1024
        cuda_stable_tolerance = max(compound_page_bytes // 2, 256 * 1024 * 1024)

        no_unmap_cuda_drops = [
            abs(int(sa.get("delta_cuda_used_bytes")))
            for sa in no_unmap_free_candidates
            if sa.get("delta_cuda_used_bytes") is not None
            and int(sa.get("delta_cuda_used_bytes")) < 0
        ]
        unmap_cuda_drops = [
            abs(int(sa.get("delta_cuda_used_bytes")))
            for sa in unmap_candidates
            if sa.get("delta_cuda_used_bytes") is not None
            and int(sa.get("delta_cuda_used_bytes")) < 0
        ]
        max_drop_without_unmap = max(no_unmap_cuda_drops, default=0)
        max_drop_with_unmap = max(unmap_cuda_drops, default=0)

        def _sample_view(sa: Optional[dict]) -> Optional[dict]:
            if sa is None:
                return None
            return {
                "reason": sa.get("reason"),
                "sample_type": sa.get("sample_type"),
                "elapsed_s": sa.get("elapsed_s"),
                "call": sa.get("call"),
                "inuse_pages": sa.get("inuse_pages"),
                "reserved_pages": sa.get("reserved_pages"),
                "pages_returned": sa.get("pages_returned"),
                "num_unmapped": sa.get("num_unmapped"),
                "kvcached_used_bytes": sa.get("kvcached_used_bytes"),
                "kvcached_reserved_bytes": sa.get("kvcached_reserved_bytes"),
                "kvcached_total_mapped_bytes":
                sa.get("kvcached_total_mapped_bytes"),
                "cuda_used_bytes": sa.get("cuda_used_bytes"),
                "delta_kvcached_used_bytes":
                sa.get("delta_kvcached_used_bytes"),
                "delta_kvcached_total_mapped_bytes":
                sa.get("delta_kvcached_total_mapped_bytes"),
                "delta_cuda_used_bytes": sa.get("delta_cuda_used_bytes"),
            }

        claim_1 = False
        if running_peak is not None:
            claim_1 = (
                int(running_peak.get("kvcached_used_bytes", 0)) >
                int(baseline.get("kvcached_used_bytes", 0))
                and int(running_peak.get("cuda_used_bytes", 0) or 0) >
                int(baseline.get("cuda_used_bytes", 0) or 0)
            )

        claim_2 = False
        if logical_release is not None:
            delta_total = int(
                logical_release.get("delta_kvcached_total_mapped_bytes") or 0)
            delta_cuda = int(logical_release.get("delta_cuda_used_bytes") or 0)
            delta_kv_used = int(
                logical_release.get("delta_kvcached_used_bytes") or 0)
            claim_2 = (
                delta_kv_used <= 0
                and abs(delta_total) <= compound_page_bytes
                and delta_cuda >= -cuda_stable_tolerance
                and int(logical_release.get("num_unmapped", 0)) == 0
            )

        claim_3 = False
        if unmap_event is not None:
            claim_3 = (
                int(unmap_event.get("delta_kvcached_total_mapped_bytes") or 0)
                < 0
                and max_drop_with_unmap >= max_drop_without_unmap
                and max_drop_with_unmap >= cuda_stable_tolerance
            )

        return {
            "selected_samples": {
                "baseline": _sample_view(baseline),
                "running_peak": _sample_view(running_peak),
                "logical_free_no_unmap": _sample_view(logical_release),
                "unmap_event": _sample_view(unmap_event),
            },
            "thresholds": {
                "compound_page_bytes": compound_page_bytes,
                "cuda_stable_tolerance_bytes": cuda_stable_tolerance,
            },
            "drop_stats": {
                "max_cuda_drop_without_unmap_bytes": max_drop_without_unmap,
                "max_cuda_drop_with_unmap_bytes": max_drop_with_unmap,
            },
            "claims": {
                "request_running_kv_cache_increases_gpu_memory": claim_1,
                "free_logically_releases_but_gpu_memory_does_not_drop_immediately":
                claim_2,
                "only_unmap_can_make_gpu_memory_truly_drop": claim_3,
            },
        }

    def print_summary(self, report: Optional[dict] = None, title: str = "KVCacheManager.free  Debug Summary"):
        if report is None:
            with self._lock:
                elapsed = time.time() - self._start_time
                avg_ms = (self.total_free_time_ms / self.total_free_calls
                          if self.total_free_calls else 0)
                avg_blocks = (self.total_blocks_freed / self.total_free_calls
                              if self.total_free_calls else 0)
                total_calls = self.total_free_calls
                total_blocks = self.total_blocks_freed
                total_pages = self.total_pages_returned
                total_time_ms = self.total_free_time_ms
                samples = list(self.samples)
                state_samples = list(self.state_samples)
        else:
            elapsed = samples[-1]["elapsed_s"] - samples[0]["elapsed_s"] if (
                samples := report["samples"]) else 0.0
            avg_ms = report["avg_time_ms"]
            avg_blocks = report["avg_blocks"]
            total_calls = report["count"]
            total_blocks = report["delta_blocks"]
            total_pages = report["delta_pages"]
            total_time_ms = report["delta_time_ms"]
            state_samples = list(report.get("state_samples", []))

        # Compute unmap aggregates from samples
        unmap_samples = [sa for sa in samples if sa.get("num_unmapped", 0) > 0]
        total_unmap_calls = len(unmap_samples)
        total_pages_unmapped = sum(sa.get("num_unmapped", 0) for sa in unmap_samples)
        total_unmap_time_ms = sum(sa.get("unmap_time_ms", 0.0) for sa in unmap_samples)
        avg_unmap_ms = total_unmap_time_ms / total_unmap_calls if total_unmap_calls else 0.0

        # Latest memory state from last sample
        last = samples[-1] if samples else {}
        returned_unique_ids = {
            pid for sa in samples for pid in sa.get("returned_page_ids", [])
        }
        unmapped_unique_ids = {
            pid for sa in samples for pid in sa.get("unmapped_page_ids", [])
        }

        w = 66  # box inner width
        logger.info(
            "\n"
            f"┏{'━' * w}┓\n"
            f"┃  {title:<{w-2}}┃\n"
            f"┣{'━' * w}┫\n"
            f"┃  Duration:                {elapsed:>10.2f} s               ┃\n"
            f"┃  Total free() Calls:      {total_calls:>10d}                ┃\n"
            f"┃  Total Blocks Freed:      {total_blocks:>10d}                ┃\n"
            f"┃  Total Pages Returned:    {total_pages:>10d}                ┃\n"
            f"┃  Total free() Time (ms):  {total_time_ms:>10.3f}                ┃\n"
            f"┃  Avg Time / Call (ms):    {avg_ms:>10.4f}                ┃\n"
            f"┃  Avg Blocks / Call:       {avg_blocks:>10.1f}                ┃\n"
            f"┣{'━' * w}┫\n"
            f"┃  UNMAP Statistics                                          ┃\n"
            f"┣{'━' * w}┫\n"
            f"┃  Calls that triggered UNMAP:  {total_unmap_calls:>6d} / {total_calls:<6d}      ┃\n"
            f"┃  Total Pages Unmapped:    {total_pages_unmapped:>10d}                ┃\n"
            f"┃  Total UNMAP Time (ms):   {total_unmap_time_ms:>10.3f}                ┃\n"
            f"┃  Avg UNMAP Time (ms):     {avg_unmap_ms:>10.4f}                ┃\n"
            f"┣{'━' * w}┫\n"
            f"┃  Current Memory State (last sample)                        ┃\n"
            f"┣{'━' * w}┫\n"
            f"┃  Reserved Pages:    {last.get('reserved_pages', 0):>5d} / {last.get('max_reserved_pages', 0):<5d}"
            f"                        ┃\n"
            f"┃  In-use Pages:            {last.get('inuse_pages', 0):>10d}                ┃\n"
            f"┃  Free Pages:              {last.get('free_pages', 0):>10d}                ┃\n"
            f"┃  Total Pages:             {last.get('total_pages', 0):>10d}                ┃\n"
            f"┃  Full Pages:              {last.get('full_pages', 0):>10d}                ┃\n"
            f"┃  Avail Pages:             {last.get('avail_pages', 0):>10d}                ┃\n"
            f"┃  Avail Blocks:            {last.get('avail_blocks', 0):>10d}                ┃\n"
            f"┃  CUDA Used:         {_format_bytes_gb(last.get('cuda_used_bytes')):>8s} GB"
            f"                           ┃\n"
            f"┃  CUDA Free:         {_format_bytes_gb(last.get('cuda_free_bytes')):>8s} GB"
            f"                           ┃\n"
            f"┗{'━' * w}┛"
        )
        trend_table = self._format_trend_table(samples=samples if report is not None else None)
        if trend_table is not None:
            logger.info(trend_table)
        state_trend_table = self._format_state_trend_table(
            state_samples=state_samples if report is not None else None)
        if state_trend_table is not None:
            logger.info(state_trend_table)

        if samples:
            logger.info(
                "\nKVCacheManager.free Page-ID Stats\n"
                f"  Unique Returned Pages: {len(returned_unique_ids)}\n"
                f"  Unique Unmapped Pages: {len(unmapped_unique_ids)}\n"
                f"  Returned Page IDs (preview): {_format_page_id_preview(sorted(returned_unique_ids))}\n"
                f"  Unmapped Page IDs (preview): {_format_page_id_preview(sorted(unmapped_unique_ids))}"
            )

            cuda_used = last.get("cuda_used_bytes")
            model_weight = last.get("model_weight_bytes")
            kv_active = last.get("kvcached_used_bytes")
            kv_reserved = last.get("kvcached_reserved_bytes")
            kv_total = last.get("kvcached_total_mapped_bytes")
            non_kv = last.get("non_kvcached_bytes")
            runtime_other = last.get("runtime_other_bytes")
            torch_alloc = last.get("torch_memory_allocated_bytes")
            torch_reserved = last.get("torch_memory_reserved_bytes")
            torch_max_alloc = last.get("torch_max_memory_allocated_bytes")
            torch_max_reserved = last.get("torch_max_memory_reserved_bytes")
            torch_active = last.get("torch_active_bytes")
            torch_inactive_split = last.get("torch_inactive_split_bytes")
            torch_cached = last.get("torch_allocator_cached_bytes")
            non_kv_torch_reserved = last.get("non_kv_torch_reserved_bytes")
            non_kv_torch_active = last.get("non_kv_torch_active_bytes")
            non_kv_torch_cached = last.get("non_kv_torch_cached_bytes")
            non_kv_non_torch = last.get("non_kv_non_torch_bytes")
            model_in_non_kv = last.get("model_weight_in_non_kv_bytes")
            non_kv_runtime_minus_model = last.get(
                "non_kv_runtime_minus_model_bytes")
            logger.info(
                "\nCUDA Memory Breakdown (last sample)\n"
                f"  CUDA Used Total:        {_format_bytes_gb(cuda_used)} GB (100%)\n"
                f"  Model Weights:          {_format_bytes_gb(model_weight)} GB ({_format_percent(model_weight, cuda_used)})\n"
                f"  KV Active (in-use):     {_format_bytes_gb(kv_active)} GB ({_format_percent(kv_active, cuda_used)})\n"
                f"  KV Reserved (prealloc): {_format_bytes_gb(kv_reserved)} GB ({_format_percent(kv_reserved, cuda_used)})\n"
                f"  KV Total Mapped:        {_format_bytes_gb(kv_total)} GB ({_format_percent(kv_total, cuda_used)})\n"
                f"  Non-KV Total:           {_format_bytes_gb(non_kv)} GB ({_format_percent(non_kv, cuda_used)})\n"
                f"  Runtime Other (non-KV - model): {_format_bytes_gb(runtime_other)} GB "
                f"({_format_percent(runtime_other, cuda_used)})"
            )
            logger.info(
                "\nNon-KV Detailed Breakdown (last sample)\n"
                "  [Torch allocator raw]\n"
                f"    torch.memory_allocated:   {_format_bytes_gb(torch_alloc)} GB\n"
                f"    torch.memory_reserved:    {_format_bytes_gb(torch_reserved)} GB\n"
                f"    torch.max_allocated:      {_format_bytes_gb(torch_max_alloc)} GB\n"
                f"    torch.max_reserved:       {_format_bytes_gb(torch_max_reserved)} GB\n"
                f"    torch.active_bytes:       {_format_bytes_gb(torch_active)} GB\n"
                f"    torch.inactive_split:     {_format_bytes_gb(torch_inactive_split)} GB\n"
                f"    torch.cached(resv-alloc): {_format_bytes_gb(torch_cached)} GB\n"
                "  [Non-KV partition]\n"
                f"    Non-KV in Torch reserved: {_format_bytes_gb(non_kv_torch_reserved)} GB ({_format_percent(non_kv_torch_reserved, non_kv)})\n"
                f"    Non-KV Torch active:      {_format_bytes_gb(non_kv_torch_active)} GB ({_format_percent(non_kv_torch_active, non_kv)})\n"
                f"    Non-KV Torch cached:      {_format_bytes_gb(non_kv_torch_cached)} GB ({_format_percent(non_kv_torch_cached, non_kv)})\n"
                f"    Non-KV non-Torch:         {_format_bytes_gb(non_kv_non_torch)} GB ({_format_percent(non_kv_non_torch, non_kv)})\n"
                "  [Model attribution]\n"
                f"    Model in Non-KV:          {_format_bytes_gb(model_in_non_kv)} GB ({_format_percent(model_in_non_kv, non_kv)})\n"
                f"    Runtime minus model:      {_format_bytes_gb(non_kv_runtime_minus_model)} GB ({_format_percent(non_kv_runtime_minus_model, non_kv)})"
            )

        # Print unmap detail for calls that triggered unmap
        if unmap_samples:
            unmap_lines = [
                "",
                "=== UNMAP Events Detail ===",
                f"Total free() calls that triggered UNMAP: {total_unmap_calls} / {total_calls}",
                (
                    "call# |  t(s)  | blks | pg_ret | unmapped | unmap_ms |"
                    " rsvd_pg | inuse_pg | avail_blk | cuda_used(GB) | cuda_free(GB)"
                ),
                "-" * 120,
            ]
            for sa in unmap_samples:
                unmap_lines.append(
                    f"{sa['call']:>5d} | "
                    f"{sa['elapsed_s']:>6.2f} | "
                    f"{sa['num_blocks']:>4d} | "
                    f"{sa['pages_returned']:>6d} | "
                    f"{sa['num_unmapped']:>8d} | "
                    f"{sa.get('unmap_time_ms', 0.0):>8.3f} | "
                    f"{sa.get('reserved_pages', 0):>7d} | "
                    f"{sa.get('inuse_pages', 0):>8d} | "
                    f"{sa.get('avail_blocks', 0):>9d} | "
                    f"{_format_bytes_gb(sa.get('cuda_used_bytes')):>13} | "
                    f"{_format_bytes_gb(sa.get('cuda_free_bytes')):>13}"
                )
            logger.info("\n".join(unmap_lines))
        else:
            logger.info("\n=== No UNMAP events occurred during this period ===")

        lifecycle_proof = self._build_memory_lifecycle_proof(
            free_samples=samples,
            state_samples=state_samples,
        )
        if lifecycle_proof is not None:
            selected = lifecycle_proof["selected_samples"]
            thresholds = lifecycle_proof["thresholds"]
            drop_stats = lifecycle_proof["drop_stats"]
            claims = lifecycle_proof["claims"]

            def _fmt_sample(name: str, sa: Optional[dict]) -> str:
                if sa is None:
                    return f"  {name}: n/a"
                return (
                    f"  {name}: reason={sa.get('reason')}  "
                    f"t={sa.get('elapsed_s')}s  "
                    f"inuse={sa.get('inuse_pages')}  "
                    f"reserved={sa.get('reserved_pages')}  "
                    f"returned={sa.get('pages_returned')}  "
                    f"unmapped={sa.get('num_unmapped')}  "
                    f"kv_total={_format_bytes_gb(sa.get('kvcached_total_mapped_bytes'))}GB  "
                    f"cuda_used={_format_bytes_gb(sa.get('cuda_used_bytes'))}GB  "
                    f"delta_kv_total={_format_bytes_gb(sa.get('delta_kvcached_total_mapped_bytes'))}GB  "
                    f"delta_cuda={_format_bytes_gb(sa.get('delta_cuda_used_bytes'))}GB"
                )

            logger.info(
                "\n=== Memory Lifecycle Proof (sampled) ===\n"
                f"{_fmt_sample('baseline', selected.get('baseline'))}\n"
                f"{_fmt_sample('running_peak', selected.get('running_peak'))}\n"
                f"{_fmt_sample('logical_free_no_unmap', selected.get('logical_free_no_unmap'))}\n"
                f"{_fmt_sample('unmap_event', selected.get('unmap_event'))}\n"
                f"  compound_page_bytes={_format_bytes_gb(thresholds.get('compound_page_bytes'))}GB  "
                f"cuda_stable_tolerance={_format_bytes_gb(thresholds.get('cuda_stable_tolerance_bytes'))}GB\n"
                f"  max_cuda_drop_without_unmap={_format_bytes_gb(drop_stats.get('max_cuda_drop_without_unmap_bytes'))}GB  "
                f"max_cuda_drop_with_unmap={_format_bytes_gb(drop_stats.get('max_cuda_drop_with_unmap_bytes'))}GB\n"
                f"  [{'PASS' if claims['request_running_kv_cache_increases_gpu_memory'] else 'FAIL'}] "
                "request running: KV cache makes GPU memory go up\n"
                f"  [{'PASS' if claims['free_logically_releases_but_gpu_memory_does_not_drop_immediately'] else 'FAIL'}] "
                "after free(): logical release can happen while GPU memory stays high\n"
                f"  [{'PASS' if claims['only_unmap_can_make_gpu_memory_truly_drop'] else 'FAIL'}] "
                "only unmap/cuMemUnmap makes a real GPU-memory drop observable"
            )

    def schedule_idle_report(self,
                             idle_callback: Optional[Callable[[], None]] = None):
        if _FREE_IDLE_REPORT_SECONDS <= 0:
            return

        with self._lock:
            if self._idle_report_timer is not None:
                self._idle_report_timer.cancel()
            self._idle_report_timer = threading.Timer(
                _FREE_IDLE_REPORT_SECONDS,
                self._emit_idle_report,
                kwargs={"idle_callback": idle_callback},
            )
            self._idle_report_timer.daemon = True
            self._idle_report_timer.start()

    def _emit_idle_report(self,
                          idle_callback: Optional[Callable[[], bool]] = None):
        if idle_callback is not None:
            try:
                should_emit = idle_callback()
                if should_emit is False:
                    return
            except Exception as e:
                logger.warning(f"[FREE][IDLE] idle callback failed: {e}")
        report = self._build_delta_report(mark_reported=True)
        if report is None:
            return

        logger.info(
            "[FREE][IDLE] No new free() activity for %.1fs, emitting interval report for calls %d-%d",
            _FREE_IDLE_REPORT_SECONDS,
            report["first_call"],
            report["last_call"],
        )
        self.print_summary(report=report, title="KVCacheManager.free  Idle Report")
        if _FREE_DUMP_JSON_ON_IDLE:
            report_path = os.environ.get(
                "KVCACHED_FREE_DEBUG_REPORT",
                "kvcached_free_debug_report.json")
            self.dump_json(report_path)

    def dump_json(self, path: str):
        with self._lock:
            elapsed = time.time() - self._start_time
            samples = list(self.samples)
            unmap_samples = [sa for sa in samples if sa.get("num_unmapped", 0) > 0]
            blocks_per_call = [
                int(sa.get("num_blocks", 0)) for sa in samples
                if sa.get("num_blocks") is not None
            ]
            returned_unique_ids = sorted({
                pid for sa in samples for pid in sa.get("returned_page_ids", [])
            })
            unmapped_unique_ids = sorted({
                pid for sa in samples for pid in sa.get("unmapped_page_ids", [])
            })
            state_samples = list(self.state_samples)
            last_sample = samples[-1] if samples else {}
            block_mem_size_bytes = last_sample.get("block_mem_size_bytes")
            block_mapped_total_bytes = last_sample.get("block_mapped_total_bytes")
            total_freed_logical_bytes = sum(
                int(sa.get("freed_logical_bytes", 0)) for sa in samples
            )
            total_freed_mapped_bytes_est = sum(
                int(sa.get("freed_mapped_bytes_est", 0)) for sa in samples
            )
            if (total_freed_logical_bytes == 0
                    and self.total_blocks_freed > 0
                    and block_mem_size_bytes is not None):
                total_freed_logical_bytes = (
                    self.total_blocks_freed * int(block_mem_size_bytes)
                )
            if (total_freed_mapped_bytes_est == 0
                    and self.total_blocks_freed > 0
                    and block_mapped_total_bytes is not None):
                total_freed_mapped_bytes_est = (
                    self.total_blocks_freed * int(block_mapped_total_bytes)
                )
            lifecycle_proof = self._build_memory_lifecycle_proof(
                free_samples=samples,
                state_samples=state_samples,
            )
            report = {
                "elapsed_total_s": round(elapsed, 2),
                "total_free_calls": self.total_free_calls,
                "total_blocks_freed": self.total_blocks_freed,
                "total_pages_returned": self.total_pages_returned,
                "total_free_time_ms": round(self.total_free_time_ms, 3),
                "avg_time_per_call_ms": round(
                    self.total_free_time_ms / self.total_free_calls, 4
                ) if self.total_free_calls else 0,
                "allocator_meta": {
                    "block_size_cells": last_sample.get("block_size_cells"),
                    "cell_size_bytes": last_sample.get("cell_size_bytes"),
                    "block_mem_size_bytes": block_mem_size_bytes,
                    "block_mapped_total_bytes": block_mapped_total_bytes,
                    "page_size_bytes": last_sample.get("page_size_bytes"),
                    "num_layers": last_sample.get("num_layers"),
                    "num_kv_buffers": last_sample.get("num_kv_buffers"),
                },
                "free_block_stats": {
                    "mean_blocks_per_call": round(
                        self.total_blocks_freed / self.total_free_calls, 4
                    ) if self.total_free_calls else 0.0,
                    "p50_blocks_per_call": round(
                        _percentile(blocks_per_call, 50), 4
                    ) if blocks_per_call else 0.0,
                    "p95_blocks_per_call": round(
                        _percentile(blocks_per_call, 95), 4
                    ) if blocks_per_call else 0.0,
                    "p99_blocks_per_call": round(
                        _percentile(blocks_per_call, 99), 4
                    ) if blocks_per_call else 0.0,
                    "max_blocks_per_call": max(blocks_per_call, default=0),
                    "total_freed_logical_bytes": total_freed_logical_bytes,
                    "total_freed_mapped_bytes_est": total_freed_mapped_bytes_est,
                },
                "unmap_stats": {
                    "total_unmap_calls": len(unmap_samples),
                    "total_pages_unmapped": sum(
                        sa.get("num_unmapped", 0) for sa in unmap_samples),
                    "total_unmap_time_ms": round(sum(
                        sa.get("unmap_time_ms", 0.0) for sa in unmap_samples), 3),
                    "unmap_call_numbers": [sa["call"] for sa in unmap_samples],
                },
                "page_id_stats": {
                    "unique_returned_pages": len(returned_unique_ids),
                    "unique_unmapped_pages": len(unmapped_unique_ids),
                    "returned_page_ids": returned_unique_ids,
                    "unmapped_page_ids": unmapped_unique_ids,
                },
                "memory_lifecycle_proof": lifecycle_proof,
                "last_memory_breakdown": {
                    "cuda_used_bytes": last_sample.get("cuda_used_bytes"),
                    "model_weight_bytes": last_sample.get("model_weight_bytes"),
                    "kvcached_used_bytes": last_sample.get("kvcached_used_bytes"),
                    "kvcached_reserved_bytes": last_sample.get("kvcached_reserved_bytes"),
                    "kvcached_total_mapped_bytes": last_sample.get("kvcached_total_mapped_bytes"),
                    "non_kvcached_bytes": last_sample.get("non_kvcached_bytes"),
                    "runtime_other_bytes": last_sample.get("runtime_other_bytes"),
                    "torch_memory_allocated_bytes": last_sample.get("torch_memory_allocated_bytes"),
                    "torch_memory_reserved_bytes": last_sample.get("torch_memory_reserved_bytes"),
                    "torch_max_memory_allocated_bytes": last_sample.get("torch_max_memory_allocated_bytes"),
                    "torch_max_memory_reserved_bytes": last_sample.get("torch_max_memory_reserved_bytes"),
                    "torch_active_bytes": last_sample.get("torch_active_bytes"),
                    "torch_inactive_split_bytes": last_sample.get("torch_inactive_split_bytes"),
                    "torch_allocator_cached_bytes": last_sample.get("torch_allocator_cached_bytes"),
                    "non_kv_torch_reserved_bytes": last_sample.get("non_kv_torch_reserved_bytes"),
                    "non_kv_torch_active_bytes": last_sample.get("non_kv_torch_active_bytes"),
                    "non_kv_torch_cached_bytes": last_sample.get("non_kv_torch_cached_bytes"),
                    "non_kv_non_torch_bytes": last_sample.get("non_kv_non_torch_bytes"),
                    "model_weight_in_non_kv_bytes": last_sample.get("model_weight_in_non_kv_bytes"),
                    "non_kv_runtime_minus_model_bytes": last_sample.get("non_kv_runtime_minus_model_bytes"),
                },
                "memory_trend_samples": samples,
                "state_samples": state_samples,
            }
        try:
            with open(path, "w") as fp:
                json.dump(report, fp, indent=2)
            logger.info(f"Free debug report saved to: {path}")
        except Exception as e:
            logger.warning(f"Failed to save free debug report: {e}")


_free_debug_stats: Optional[_FreeDebugStats] = None
_free_debug_exit_lock = threading.Lock()
_free_debug_exit_done = False

if KVCACHED_FREE_DEBUG:
    _free_debug_stats = _FreeDebugStats()

    def _emit_free_debug_summary() -> None:
        global _free_debug_exit_done
        with _free_debug_exit_lock:
            if _free_debug_exit_done:
                return
            _free_debug_exit_done = True

        if _free_debug_stats is not None:
            _free_debug_stats.print_summary()
            report_path = os.environ.get(
                "KVCACHED_FREE_DEBUG_REPORT",
                "kvcached_free_debug_report.json")
            _free_debug_stats.dump_json(report_path)

    def _on_exit(*args):
        _emit_free_debug_summary()

        if args and isinstance(args[0], int):
            signum = args[0]
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    atexit.register(_on_exit)
    for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP,
                 signal.SIGQUIT):
        try:
            signal.signal(_sig, _on_exit)
        except Exception:
            pass

KV_TENSOR_WAIT_TIMEOUT: float = 10.0  # seconds


def synchronized(method):
    """
    A helper decorator to synchronize access to a method.
    """

    @functools.wraps(method)
    def synchronized_method(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return synchronized_method


class KVCacheManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        cell_size: int,
        num_layers: int,
        tp_size: int = 1,
        async_sched: bool = False,
        reserve_null_block: bool = False,
        num_kv_buffers: int = 2,
    ):
        """
        Args:
            num_blocks: Number of blocks.
            block_size: Size of each block in bytes.
            cell_size: Size of each cell in bytes.
            num_layers: Number of layers.
            tp_size: Number of tensor parallel processes.
            async_sched: Whether asynchronous scheduling is enabled.
            reserve_null_block: Whether to reserve the first block as null block
                for padding tokens. This is required by SGLang which assumes the
                first block is always reserved as padded tokens.
            num_kv_buffers: Number of KV buffers per layer (2 for MHA K+V,
                1 for MLA combined KV).
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.cell_size = cell_size
        self.block_mem_size = block_size * cell_size
        self.num_layers = num_layers
        self.num_kv_buffers = num_kv_buffers
        self.reserve_null_block = reserve_null_block

        # The physical page size used by kvcached page allocator.
        self.page_size = PAGE_SIZE
        # NOTE: this is the memory size of the K or V tensor in one layer
        self.mem_size = self.num_blocks * self.block_mem_size
        self.tp_size = tp_size
        self.page_allocator = PageAllocator(
            self.num_layers,
            self.mem_size,
            self.page_size,
            self.tp_size,
            async_sched=async_sched,
            num_kv_buffers=self.num_kv_buffers,
        )

        self.num_avail_blocks = 0  # Only count free blocks in avail_pages
        self.avail_pages: Dict[int, Page] = {}
        self.full_pages: Dict[int, Page] = {}

        self.reserved_blocks: List[int] = []
        self.null_block: Optional[list[int]] = None

        self.in_shrink: bool = False
        self.target_num_blocks: Optional[int] = None
        # NOTE: we use a no-op lock for sync scheduling to avoid overhead
        self._lock = threading.RLock() if async_sched else NoOpLock()
        self._last_state_log_ts: float = 0.0
        self._last_state_log_snapshot: Optional[dict] = None

        # Event used to signal that _post_init() has finished.
        self._post_init_done = threading.Event()
        # Launch _post_init in the background; it will block until KV tensors
        # exist, then complete the remaining setup (reserve null block, start
        # pre-alloc thread) and finally set the event.
        threading.Thread(target=self._post_init, daemon=True).start()

    def _post_init(self):
        if self.null_block is not None:
            return

        def _check_kv_tensors_created():
            if self.tp_size > 1:
                return broadcast_kv_tensors_created(self.tp_size)
            else:
                return kv_tensors_created()

        try:
            total_wait = 0.0
            while not _check_kv_tensors_created():
                if total_wait >= KV_TENSOR_WAIT_TIMEOUT:
                    raise TimeoutError("KV tensors not created after "
                                       f"{KV_TENSOR_WAIT_TIMEOUT} seconds")
                time.sleep(0.001)  # 1ms
                total_wait += 0.001
            # KV tensors created now
            # Possibly reserve the first block as null block for padding tokens
            self._reserve_null_block()

            self.page_allocator.start_prealloc_thread()
            self._log_memory_state("post_init", force=True)
        except Exception as e:
            logger.error(
                f"Error during KVCacheManager post-initialization: {e}")
            # Set the event even on error to unblock waiting threads
            raise
        finally:
            self._post_init_done.set()

    def _wait_post_init(self):
        if not self._post_init_done.is_set():
            self._post_init_done.wait()

    def _reserve_null_block(self) -> None:
        """
        Reserve the first block as null block for padding tokens.
        """
        if self.reserve_null_block:
            self.null_block = self._alloc(1, _skip_wait=True)
            if self.null_block != [0]:
                logger.error(f"Failed to reserve null block, got {self.null_block}")
                raise RuntimeError("Failed to reserve null block at index 0")
        else:
            self.null_block = None

    def _get_current_memory_snapshot(self, num_blocks: int, elapsed_ms: float,
                                     pages_returned: int) -> dict:
        pa = self.page_allocator
        kvcached_used_bytes = int(self.get_mapped_memory_size("bytes"))
        reserved_pages = pa.get_num_reserved_pages()
        kvcached_reserved_bytes = (
            reserved_pages * self.num_layers *
            self.page_size * self.num_kv_buffers
        )
        total_pages = pa.get_num_total_pages()
        free_pages = pa.get_num_free_pages()
        inuse_pages = pa.get_num_inuse_pages()
        block_mapped_total_bytes = (
            self.block_mem_size * self.num_layers * self.num_kv_buffers
        )
        snapshot = {
            "elapsed_s": round(time.time() - _free_debug_stats._start_time, 3)
            if _free_debug_stats is not None else 0.0,
            "num_blocks": num_blocks,
            "pages_returned": pages_returned,
            "elapsed_ms": round(elapsed_ms, 3),
            "avail_blocks": self.num_avail_blocks + len(self.reserved_blocks),
            "full_pages": len(self.full_pages),
            "avail_pages": len(self.avail_pages),
            "inuse_pages": inuse_pages,
            "free_pages": free_pages,
            "reserved_pages": reserved_pages,
            "max_reserved_pages": pa.max_reserved_pages,
            "total_pages": total_pages,
            "kvcached_used_bytes": kvcached_used_bytes,
            "kvcached_reserved_bytes": kvcached_reserved_bytes,
            "kvcached_total_mapped_bytes":
            kvcached_used_bytes + kvcached_reserved_bytes,
            "block_size_cells": self.block_size,
            "cell_size_bytes": self.cell_size,
            "block_mem_size_bytes": self.block_mem_size,
            "block_mapped_total_bytes": block_mapped_total_bytes,
            "page_size_bytes": self.page_size,
            "num_layers": self.num_layers,
            "num_kv_buffers": self.num_kv_buffers,
            "freed_logical_bytes": num_blocks * self.block_mem_size,
            "freed_mapped_bytes_est": num_blocks * block_mapped_total_bytes,
        }

        try:
            if torch.cuda.is_available():
                cuda_free_bytes, cuda_total_bytes = torch.cuda.mem_get_info()
                snapshot["cuda_free_bytes"] = int(cuda_free_bytes)
                snapshot["cuda_total_bytes"] = int(cuda_total_bytes)
                snapshot["cuda_used_bytes"] = int(cuda_total_bytes -
                                                   cuda_free_bytes)
                snapshot["torch_memory_allocated_bytes"] = int(
                    torch.cuda.memory_allocated())
                snapshot["torch_memory_reserved_bytes"] = int(
                    torch.cuda.memory_reserved())
                snapshot["torch_max_memory_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated())
                snapshot["torch_max_memory_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved())
                torch_stats = torch.cuda.memory_stats()
                snapshot["torch_active_bytes"] = int(
                    torch_stats.get("active_bytes.all.current", 0))
                snapshot["torch_inactive_split_bytes"] = int(
                    torch_stats.get("inactive_split_bytes.all.current", 0))
            else:
                snapshot["cuda_free_bytes"] = None
                snapshot["cuda_total_bytes"] = None
                snapshot["cuda_used_bytes"] = None
                snapshot["torch_memory_allocated_bytes"] = None
                snapshot["torch_memory_reserved_bytes"] = None
                snapshot["torch_max_memory_allocated_bytes"] = None
                snapshot["torch_max_memory_reserved_bytes"] = None
                snapshot["torch_active_bytes"] = None
                snapshot["torch_inactive_split_bytes"] = None
        except Exception:
            snapshot["cuda_free_bytes"] = None
            snapshot["cuda_total_bytes"] = None
            snapshot["cuda_used_bytes"] = None
            snapshot["torch_memory_allocated_bytes"] = None
            snapshot["torch_memory_reserved_bytes"] = None
            snapshot["torch_max_memory_allocated_bytes"] = None
            snapshot["torch_max_memory_reserved_bytes"] = None
            snapshot["torch_active_bytes"] = None
            snapshot["torch_inactive_split_bytes"] = None

        model_weight_bytes = get_model_weight_bytes()
        snapshot["model_weight_bytes"] = (int(model_weight_bytes)
                                          if model_weight_bytes is not None
                                          else None)
        cuda_used_bytes = snapshot.get("cuda_used_bytes")
        kvc_total = snapshot["kvcached_total_mapped_bytes"]
        if cuda_used_bytes is None:
            snapshot["non_kvcached_bytes"] = None
            snapshot["runtime_other_bytes"] = None
            snapshot["torch_allocator_cached_bytes"] = None
            snapshot["non_kv_torch_reserved_bytes"] = None
            snapshot["non_kv_torch_active_bytes"] = None
            snapshot["non_kv_torch_cached_bytes"] = None
            snapshot["non_kv_non_torch_bytes"] = None
            snapshot["model_weight_in_non_kv_bytes"] = None
            snapshot["non_kv_runtime_minus_model_bytes"] = None
        else:
            non_kv_bytes = max(int(cuda_used_bytes) - int(kvc_total), 0)
            snapshot["non_kvcached_bytes"] = non_kv_bytes
            if model_weight_bytes is None:
                snapshot["runtime_other_bytes"] = None
            else:
                snapshot["runtime_other_bytes"] = max(
                    non_kv_bytes - int(model_weight_bytes), 0)

            torch_reserved_bytes = snapshot.get("torch_memory_reserved_bytes")
            torch_alloc_bytes = snapshot.get("torch_memory_allocated_bytes")
            if torch_reserved_bytes is None or torch_alloc_bytes is None:
                snapshot["non_kv_torch_reserved_bytes"] = None
                snapshot["non_kv_torch_active_bytes"] = None
                snapshot["non_kv_torch_cached_bytes"] = None
                snapshot["non_kv_non_torch_bytes"] = None
                snapshot["torch_allocator_cached_bytes"] = None
            else:
                torch_reserved = int(torch_reserved_bytes)
                torch_alloc = int(torch_alloc_bytes)
                torch_cached = max(torch_reserved - torch_alloc, 0)
                non_kv_torch_reserved = min(non_kv_bytes, torch_reserved)
                non_kv_torch_active = min(non_kv_torch_reserved, torch_alloc)
                non_kv_torch_cached = max(
                    non_kv_torch_reserved - non_kv_torch_active, 0)
                non_kv_non_torch = max(non_kv_bytes - non_kv_torch_reserved, 0)
                snapshot["torch_allocator_cached_bytes"] = torch_cached
                snapshot["non_kv_torch_reserved_bytes"] = non_kv_torch_reserved
                snapshot["non_kv_torch_active_bytes"] = non_kv_torch_active
                snapshot["non_kv_torch_cached_bytes"] = non_kv_torch_cached
                snapshot["non_kv_non_torch_bytes"] = non_kv_non_torch

            if model_weight_bytes is None:
                snapshot["model_weight_in_non_kv_bytes"] = None
                snapshot["non_kv_runtime_minus_model_bytes"] = None
            else:
                model_in_non_kv = min(non_kv_bytes, int(model_weight_bytes))
                snapshot["model_weight_in_non_kv_bytes"] = model_in_non_kv
                snapshot["non_kv_runtime_minus_model_bytes"] = max(
                    non_kv_bytes - model_in_non_kv, 0)

        return snapshot

    def _log_memory_state(self, reason: str, force: bool = False) -> None:
        snapshot = self._get_current_memory_snapshot(0, 0.0, 0)
        if KVCACHED_FREE_DEBUG and _free_debug_stats is not None:
            _free_debug_stats.record_state(reason, snapshot)

        if not force and _RESERVED_LOG_INTERVAL_SECONDS <= 0:
            return

        now = time.time()
        if (not force and self._last_state_log_ts > 0
                and now - self._last_state_log_ts < _RESERVED_LOG_INTERVAL_SECONDS):
            return

        prev = self._last_state_log_snapshot or {}

        def _delta(key: str) -> str:
            cur = snapshot.get(key)
            old = prev.get(key)
            if cur is None or old is None:
                return "n/a"
            return _format_bytes_gb(int(cur) - int(old))

        def _delta_int(key: str) -> str:
            cur = snapshot.get(key)
            old = prev.get(key)
            if cur is None or old is None:
                return "n/a"
            return str(int(cur) - int(old))

        logger.info(
            "[KVSTATE] reason=%s  "
            "inuse_pages=%s  reserved_pages=%s/%s  free_pages=%s  avail_blocks=%s  "
            "kv_active=%sGB  kv_reserved=%sGB  kv_total=%sGB  "
            "cuda_used=%sGB  cuda_free=%sGB  "
            "delta_kv_total=%sGB  delta_cuda_used=%sGB  "
            "delta_inuse_pages=%s  delta_reserved_pages=%s",
            reason,
            snapshot.get("inuse_pages"),
            snapshot.get("reserved_pages"),
            snapshot.get("max_reserved_pages"),
            snapshot.get("free_pages"),
            snapshot.get("avail_blocks"),
            _format_bytes_gb(snapshot.get("kvcached_used_bytes")),
            _format_bytes_gb(snapshot.get("kvcached_reserved_bytes")),
            _format_bytes_gb(snapshot.get("kvcached_total_mapped_bytes")),
            _format_bytes_gb(snapshot.get("cuda_used_bytes")),
            _format_bytes_gb(snapshot.get("cuda_free_bytes")),
            _delta("kvcached_total_mapped_bytes"),
            _delta("cuda_used_bytes"),
            _delta_int("inuse_pages"),
            _delta_int("reserved_pages"),
        )
        self._last_state_log_ts = now
        self._last_state_log_snapshot = snapshot

    def _on_free_idle(self) -> bool:
        self._log_memory_state("idle_before_report", force=True)
        inuse_pages = self.page_allocator.get_num_inuse_pages()
        if inuse_pages > 0:
            logger.info(
                "[FREE][IDLE] Skip report because requests are still active: inuse_pages=%d",
                inuse_pages,
            )
            return False
        if not _TRIM_ON_IDLE:
            return True

        reserved_pages = self.page_allocator.get_num_reserved_pages()
        if reserved_pages <= 0:
            return True

        logger.info(
            "[FREE][IDLE] Triggering trim() on idle to unmap %d reserved pages",
            reserved_pages,
        )
        self.trim()
        return True


    def alloc(self, need_size: int) -> Optional[List[int]]:
        return self._alloc(need_size)

    @synchronized
    def _alloc(self,
               need_size: int,
               _skip_wait: bool = False) -> Optional[List[int]]:
        if not _skip_wait:
            # Normal callers must wait until background initialisation is
            # finished and then perform the usual capacity check.
            self._wait_post_init()

        new_mem_size = self.page_allocator.mem_info_tracker.check_and_get_resize_target(
            self.mem_size, self.num_layers, self.num_kv_buffers)
        if new_mem_size is not None:
            self.resize(new_mem_size)

        if self.available_size() < need_size:
            logger.warning(f"available_size()={self.available_size()} < "
                           f"need_size={need_size}")
            return None

        ret_index = []
        page: Optional[Page] = None

        remaining_need = need_size

        if self.reserved_blocks:  # Try to allocate from reserved blocks first
            num_from_reserved = min(len(self.reserved_blocks), remaining_need)
            # ret_index is empty before so we directly assign it
            ret_index = self.reserved_blocks[:num_from_reserved]
            self.reserved_blocks = self.reserved_blocks[num_from_reserved:]
            remaining_need -= num_from_reserved

        while remaining_need > 0:  # Allocate the remaining blocks from pages
            if not self.avail_pages:
                page = self.page_allocator.alloc_page()
                page.init(self.block_mem_size)
                self.num_avail_blocks += page.num_free_blocks()
            else:
                _, page = self.avail_pages.popitem()
            num_from_page = min(page.num_free_blocks(), remaining_need)
            alloced_index = page.alloc(num_from_page)
            ret_index.extend(alloced_index)
            if page.full():
                self.full_pages[page.page_id] = page
            else:
                self.avail_pages[page.page_id] = page

            self.num_avail_blocks -= num_from_page
            remaining_need -= num_from_page

        self._log_memory_state(f"alloc({need_size})")
        return ret_index

    @synchronized
    def free(self, indices: List[int]):
        self._wait_post_init()

        if len(indices) == 0:
            return  # Nothing to free

        if KVCACHED_FREE_DEBUG:
            t0 = time.perf_counter()

        if SANITY_CHECK:
            for idx in indices:
                if idx in self.reserved_blocks:
                    raise ValueError(f"Freed index {idx} is in "
                                     " reserved_blocks, which is not allowed.")

        # Group indices by page_id
        idx_dict = defaultdict(list)
        for idx in indices:
            page_id = self.page_allocator.get_page_id(idx, self.block_mem_size)
            idx_dict[page_id].append(idx)

        pages_to_free: List[int] = []
        for page_id, idxs in idx_dict.items():
            # Find the page - it must be in either full_pages or avail_pages
            page = None
            if page_id in self.full_pages:
                page = self.full_pages.pop(page_id)
            elif page_id in self.avail_pages:
                page = self.avail_pages.pop(page_id)
            else:
                if SANITY_CHECK:
                    # This is a serious error - the page should exist
                    raise ValueError(
                        f"Page {page_id} not found in avail_pages or full_pages. "
                        f"This indicates a serious state inconsistency.")
                else:
                    logger.error(
                        f"Page {page_id} not found in avail_pages or full_pages. "
                        f"Skipping to avoid crash, but this indicates a serious bug."
                    )
                    continue

            self.num_avail_blocks += len(idxs)
            page.free_batch(idxs)

            if page.empty():
                pages_to_free.append(page.page_id)
                self.num_avail_blocks -= page.num_free_blocks()
            else:
                self.avail_pages[page_id] = page

        num_unmapped = 0
        unmap_time_ms = 0.0
        unmapped_page_ids: List[int] = []
        if pages_to_free:
            num_unmapped, unmap_time_ms, unmapped_page_ids = self.page_allocator.free_pages(
                pages_to_free)

        if KVCACHED_FREE_DEBUG and _free_debug_stats is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            memory_snapshot = self._get_current_memory_snapshot(
                len(indices), elapsed_ms, len(pages_to_free))
            memory_snapshot["num_unmapped"] = num_unmapped
            memory_snapshot["unmap_time_ms"] = round(unmap_time_ms, 3)
            memory_snapshot["returned_page_ids"] = list(sorted(pages_to_free))
            memory_snapshot["unmapped_page_ids"] = list(sorted(unmapped_page_ids))
            _free_debug_stats.record(
                len(indices),
                elapsed_ms,
                len(pages_to_free),
                returned_page_ids=pages_to_free,
                unmapped_page_ids=unmapped_page_ids,
                memory_snapshot=memory_snapshot,
            )
            call_num = _free_debug_stats.total_free_calls
            snap = memory_snapshot
            if num_unmapped > 0:
                logger.info(
                    f"[FREE][UNMAP] *** call #{call_num} triggered UNMAP ***  "
                    f"blocks_freed={len(indices)}  time={elapsed_ms:.3f}ms  "
                    f"unmap_time={unmap_time_ms:.3f}ms  "
                    f"pages_returned={len(pages_to_free)}  "
                    f"pages_unmapped={num_unmapped}  "
                    f"returned_ids={_format_page_id_preview(sorted(pages_to_free))}  "
                    f"unmapped_ids={_format_page_id_preview(sorted(unmapped_page_ids))}  "
                    f"pages_reserved_to_pool={len(pages_to_free) - num_unmapped}  "
                    f"reserved_pool={snap['reserved_pages']}/{snap['max_reserved_pages']}  "
                    f"inuse_pages={snap['inuse_pages']}  "
                    f"free_pages={snap['free_pages']}  "
                    f"avail_blocks={snap['avail_blocks']}  "
                    f"full_pages={snap['full_pages']}  avail_pages={snap['avail_pages']}  "
                    f"cuda_used={_format_bytes_gb(snap['cuda_used_bytes'])}GB  "
                    f"cuda_free={_format_bytes_gb(snap['cuda_free_bytes'])}GB")
            else:
                logger.info(
                    f"[FREE] call #{call_num}  "
                    f"blocks_freed={len(indices)}  time={elapsed_ms:.3f}ms  "
                    f"pages_returned={len(pages_to_free)}  "
                    f"returned_ids={_format_page_id_preview(sorted(pages_to_free))}  "
                    f"reserved_pool={snap['reserved_pages']}/{snap['max_reserved_pages']}  "
                    f"inuse_pages={snap['inuse_pages']}  "
                    f"free_pages={snap['free_pages']}  "
                    f"avail_blocks={snap['avail_blocks']}  "
                    f"full_pages={snap['full_pages']}  avail_pages={snap['avail_pages']}  "
                    f"kvc_used={_format_bytes_gb(snap['kvcached_used_bytes'])}GB  "
                    f"cuda_used={_format_bytes_gb(snap['cuda_used_bytes'])}GB  "
                    f"cuda_free={_format_bytes_gb(snap['cuda_free_bytes'])}GB")
            _free_debug_stats.schedule_idle_report(
                idle_callback=self._on_free_idle)
        self._log_memory_state(
            f"free(blocks={len(indices)}, pages={len(pages_to_free)}, unmapped={num_unmapped})",
            force=(num_unmapped > 0))

        if self.in_shrink:
            assert self.target_num_blocks is not None
            if self._get_num_alloced_blocks() <= self.target_num_blocks:
                self.page_allocator.resize(self.target_num_blocks *
                                           self.block_mem_size)
                self.in_shrink = False
                self.target_num_blocks = None

    @synchronized
    def try_to_reserve(self, need_size: int) -> bool:
        self._wait_post_init()
        if self.available_size() < need_size:
            return False
        reserved = self.alloc(need_size)
        if reserved is None:
            logger.warning("Failed to reserve blocks.")
            return False
        self.reserved_blocks.extend(reserved)
        return True

    @synchronized
    def free_reserved(self):
        if self.reserved_blocks:
            self.free(self.reserved_blocks)
            self.reserved_blocks.clear()

    @synchronized
    def resize(self, new_mem_size: int):
        """
        Reset the limit of the K or V tensor in one layer.
        new_mem_size: the memory size of the K or V tensor in one layer
        """
        self._wait_post_init()
        assert new_mem_size > 0, "new_mem_size must be positive"
        if self.page_allocator.resize(new_mem_size):
            if self.in_shrink:
                self.in_shrink = False
                self.target_num_blocks = None
            return True  # Successfully resized.
        # Failed to resize due to too many in-use blocks.
        assert (len(self.reserved_blocks) == 0
                ), "Reserved blocks must be freed before resizing."
        # NOTE: we can support resizing with reserved blocks, but we want to
        # enforce this check for now to ensure correctness.
        self.in_shrink = True
        self.target_num_blocks = new_mem_size // self.block_mem_size
        self.free_reserved()
        return False

    @synchronized
    def trim(self) -> None:
        """
        Trim the reserved pages to free up physical memory.
        """
        self._wait_post_init()
        self.page_allocator.trim()
        self._log_memory_state("trim()", force=True)

    @synchronized
    def available_size(self) -> int:
        avail_blocks = self.num_avail_blocks + len(self.reserved_blocks)
        if self.in_shrink:
            blocks_from_free_pages = 0
        else:
            virtual_free_pages = self.page_allocator.get_num_free_pages()
            physical_free_pages = self.page_allocator.get_avail_physical_pages(
            ) + self.page_allocator.get_num_reserved_pages()
            free_pages = min(virtual_free_pages, physical_free_pages)
            blocks_from_free_pages = free_pages * Page.get_num_blocks(
                self.page_size, self.block_mem_size)
        return avail_blocks + blocks_from_free_pages

    @synchronized
    def get_mapped_memory_size(self, unit='bytes') -> float:
        """Get memory usage in specified unit (bytes, kb, mb, gb)."""
        memory_bytes = (self.page_allocator.get_num_inuse_pages() *
                        self.num_layers * self.page_size *
                        self.num_kv_buffers)

        if unit == 'bytes':
            return memory_bytes
        elif unit == 'kb':
            return memory_bytes / 1024
        elif unit == 'mb':
            return memory_bytes / (1024**2)
        elif unit == 'gb':
            return memory_bytes / (1024**3)
        else:
            raise ValueError(f"Unknown unit: {unit}")

    @synchronized
    def clear(self):
        """
        Free all allocated blocks and reset the allocator to initial state.
        """

        self._wait_post_init()

        # Clear reserved blocks
        self.free_reserved()

        # Free all blocks from avail_pages and full_pages
        pages_to_free: List[int] = []
        for page in self.avail_pages.values():
            pages_to_free.append(page.page_id)
        for page in self.full_pages.values():
            pages_to_free.append(page.page_id)
        if pages_to_free:
            self.page_allocator.free_pages(pages_to_free)
        self.avail_pages.clear()
        self.full_pages.clear()

        # Trim the page allocator to free up reserved pages
        self.trim()

        self.target_num_blocks = None
        self.in_shrink = False
        self.num_avail_blocks = 0

        # Possibly reserve the first block as null block for padding tokens
        self._reserve_null_block()

    # Private methods
    @synchronized
    def _get_num_alloced_blocks(self) -> int:
        # Blocks from fully allocated pages
        blocks_from_full_pages = len(self.full_pages) * Page.get_num_blocks(
            self.page_size, self.block_mem_size)
        # Blocks from partially allocated pages. num_avail_blocks is the number
        # of free blocks in the partially allocated pages so the number of
        # allocated blocks is the total number of blocks in the partially
        # allocated pages minus the number of free blocks.
        blocks_from_avail_pages = len(self.avail_pages) * Page.get_num_blocks(
            self.page_size, self.block_mem_size) - self.num_avail_blocks
        # Blocks from reserved blocks
        blocks_from_reserved_blocks = len(self.reserved_blocks)
        return (blocks_from_full_pages + blocks_from_avail_pages +
                blocks_from_reserved_blocks)
