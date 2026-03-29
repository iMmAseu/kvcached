#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import signal
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import re


# Newer FREE log format emitted by KVCacheManager.free():
# [FREE] call #x blocks_freed=... time=...ms pages_returned=... ... avail_blocks=...
# [FREE][UNMAP] ... blocks_freed=... time=...ms unmap_time=...ms pages_returned=... pages_unmapped=...
FREE_RE_NEW = re.compile(
    r"\[FREE\](?:\[UNMAP\])?.*?blocks_freed=(?P<blocks>\d+)\s+"
    r"time=(?P<time_ms>[0-9]+(?:\.[0-9]+)?)ms\s+"
    r".*?pages_returned=(?P<pages_returned>\d+)\s+"
    r".*?avail_blocks=(?P<avail_blocks>\d+)"
)
# Backward compatibility for older FREE format:
# [FREE] blocks=... time=...ms pages_returned=... avail_blocks=...
FREE_RE_OLD = re.compile(
    r"\[FREE\]\s+blocks=(?P<blocks>\d+)\s+time=(?P<time_ms>[0-9]+(?:\.[0-9]+)?)ms\s+"
    r"pages_returned=(?P<pages_returned>\d+)\s+avail_blocks=(?P<avail_blocks>\d+)"
)
# Inline UNMAP section carried in [FREE][UNMAP] lines.
UNMAP_INLINE_RE = re.compile(
    r"\[FREE\]\[UNMAP\].*?unmap_time=(?P<time_ms>[0-9]+(?:\.[0-9]+)?)ms\s+"
    r".*?pages_unmapped=(?P<pages>\d+)"
)
# Backward compatibility for PageAllocator debug lines:
# [UNMAP] pages=... time=...ms
UNMAP_RE_OLD = re.compile(
    r"\[UNMAP\]\s+pages=(?P<pages>\d+)\s+time=(?P<time_ms>[0-9]+(?:\.[0-9]+)?)ms"
)
TS_RE = re.compile(r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


@dataclass
class FreeStats:
    free_calls: int = 0
    total_blocks_freed: int = 0
    total_pages_returned: int = 0
    total_free_time_ms: float = 0.0
    unmap_calls: int = 0
    total_pages_unmapped: int = 0
    total_unmap_time_ms: float = 0.0

    @property
    def avg_free_time_ms(self) -> float:
        if self.free_calls == 0:
            return 0.0
        return self.total_free_time_ms / self.free_calls

    @property
    def avg_blocks_per_free(self) -> float:
        if self.free_calls == 0:
            return 0.0
        return self.total_blocks_freed / self.free_calls

    @property
    def avg_unmap_time_ms(self) -> float:
        if self.unmap_calls == 0:
            return 0.0
        return self.total_unmap_time_ms / self.unmap_calls


@dataclass
class TimelineEvent:
    kind: str
    ts: Optional[str]
    blocks: Optional[int] = None
    pages_returned: Optional[int] = None
    avail_blocks: Optional[int] = None
    pages: Optional[int] = None
    time_ms: float = 0.0


class FreeLogFilter:

    def __init__(
        self,
        log_path: Path,
        report_json: Path,
        timeline_mermaid: Path,
        max_timeline_events: int,
    ) -> None:
        self.log_path = log_path
        self.report_json = report_json
        self.timeline_mermaid = timeline_mermaid
        self.max_timeline_events = max_timeline_events

        self.stats = FreeStats()
        self.events: List[TimelineEvent] = []
        self._start_time = datetime.now().isoformat(timespec="seconds")
        self._stopped = False

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_json.parent.mkdir(parents=True, exist_ok=True)
        self.timeline_mermaid.parent.mkdir(parents=True, exist_ok=True)

        self.log_fp = self.log_path.open("w", encoding="utf-8")

    def close(self) -> None:
        if not self.log_fp.closed:
            self.log_fp.close()

    def _extract_ts(self, line: str) -> Optional[str]:
        m = TS_RE.search(line)
        if not m:
            return None
        return m.group("ts")

    def _record_event(self, event: TimelineEvent) -> None:
        if len(self.events) < self.max_timeline_events:
            self.events.append(event)

    def process_line(self, line: str) -> None:
        line = line.rstrip("\n")

        free_m = FREE_RE_NEW.search(line) or FREE_RE_OLD.search(line)
        if free_m:
            blocks = int(free_m.group("blocks"))
            time_ms = float(free_m.group("time_ms"))
            pages_returned = int(free_m.group("pages_returned"))
            avail_blocks = int(free_m.group("avail_blocks"))

            self.stats.free_calls += 1
            self.stats.total_blocks_freed += blocks
            self.stats.total_pages_returned += pages_returned
            self.stats.total_free_time_ms += time_ms

            self._record_event(
                TimelineEvent(
                    kind="free",
                    ts=self._extract_ts(line),
                    blocks=blocks,
                    pages_returned=pages_returned,
                    avail_blocks=avail_blocks,
                    time_ms=time_ms,
                ))

            unmap_inline_m = UNMAP_INLINE_RE.search(line)
            if unmap_inline_m:
                pages = int(unmap_inline_m.group("pages"))
                unmap_time_ms = float(unmap_inline_m.group("time_ms"))
                self.stats.unmap_calls += 1
                self.stats.total_pages_unmapped += pages
                self.stats.total_unmap_time_ms += unmap_time_ms
                self._record_event(
                    TimelineEvent(
                        kind="unmap",
                        ts=self._extract_ts(line),
                        pages=pages,
                        time_ms=unmap_time_ms,
                    ))

            self._emit(line)
            return

        unmap_m = UNMAP_RE_OLD.search(line)
        if unmap_m:
            pages = int(unmap_m.group("pages"))
            time_ms = float(unmap_m.group("time_ms"))

            self.stats.unmap_calls += 1
            self.stats.total_pages_unmapped += pages
            self.stats.total_unmap_time_ms += time_ms

            self._record_event(
                TimelineEvent(
                    kind="unmap",
                    ts=self._extract_ts(line),
                    pages=pages,
                    time_ms=time_ms,
                ))
            self._emit(line)
            return

    def _emit(self, line: str) -> None:
        print(line, flush=True)
        self.log_fp.write(line + "\n")
        self.log_fp.flush()

    def _final_summary_text(self) -> str:
        return (
            "\n"
            "========== KVCACHED FREE SUMMARY ==========\n"
            f"free_calls:            {self.stats.free_calls}\n"
            f"total_blocks_freed:    {self.stats.total_blocks_freed}\n"
            f"total_pages_returned:  {self.stats.total_pages_returned}\n"
            f"total_free_time_ms:    {self.stats.total_free_time_ms:.6f}\n"
            f"avg_free_time_ms:      {self.stats.avg_free_time_ms:.6f}\n"
            f"avg_blocks_per_free:   {self.stats.avg_blocks_per_free:.6f}\n"
            f"unmap_calls:           {self.stats.unmap_calls}\n"
            f"total_pages_unmapped:  {self.stats.total_pages_unmapped}\n"
            f"total_unmap_time_ms:   {self.stats.total_unmap_time_ms:.6f}\n"
            f"avg_unmap_time_ms:     {self.stats.avg_unmap_time_ms:.6f}\n"
            f"log_path:              {self.log_path}\n"
            f"report_json:           {self.report_json}\n"
            f"timeline_mermaid:      {self.timeline_mermaid}\n"
            "==========================================="
        )

    def _write_report_json(self) -> None:
        report = {
            "start_time": self._start_time,
            "end_time": datetime.now().isoformat(timespec="seconds"),
            "stats": {
                **asdict(self.stats),
                "avg_free_time_ms": round(self.stats.avg_free_time_ms, 6),
                "avg_blocks_per_free": round(self.stats.avg_blocks_per_free, 6),
                "avg_unmap_time_ms": round(self.stats.avg_unmap_time_ms, 6),
            },
            "events_captured": len(self.events),
            "events_captured_limit": self.max_timeline_events,
            "events": [asdict(e) for e in self.events],
        }
        self.report_json.write_text(json.dumps(report, indent=2),
                                    encoding="utf-8")

    def _write_timeline_mermaid(self) -> None:
        lines = [
            "sequenceDiagram",
            "autonumber",
            "participant C as RequestFlow",
            "participant K as KVCacheManager",
            "participant P as PageAllocator",
            "participant G as GPU(VMM)",
        ]
        for event in self.events:
            if event.kind == "free":
                ts_note = f" @ {event.ts}" if event.ts else ""
                lines.append(
                    f"C->>K: free(blocks={event.blocks}){ts_note}")
                lines.append(
                    "Note over K: "
                    f"dt={event.time_ms:.3f}ms, pages_returned={event.pages_returned}, "
                    f"avail_blocks={event.avail_blocks}")
                if event.pages_returned and event.pages_returned > 0:
                    lines.append(
                        f"K->>P: free_pages(pages={event.pages_returned})")
            elif event.kind == "unmap":
                ts_note = f" @ {event.ts}" if event.ts else ""
                lines.append(f"P->>G: unmap(pages={event.pages}){ts_note}")
                lines.append(
                    f"Note over P,G: dt={event.time_ms:.3f}ms")
        self.timeline_mermaid.write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")

    def finish(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        summary = self._final_summary_text()
        print(summary, flush=True)
        self.log_fp.write(summary + "\n")
        self.log_fp.flush()

        self._write_report_json()
        self._write_timeline_mermaid()
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter server logs to keep only kvcached FREE/UNMAP lines, "
            "print final stats summary, and emit JSON + Mermaid timeline."
        ))
    parser.add_argument(
        "--log-path",
        default="kvcached_free_only.log",
        help="Path to filtered text log output.",
    )
    parser.add_argument(
        "--report-json",
        default="kvcached_free_summary.json",
        help="Path to final summary JSON report.",
    )
    parser.add_argument(
        "--timeline-mermaid",
        default="kvcached_free_timeline.mmd",
        help="Path to Mermaid sequence diagram output.",
    )
    parser.add_argument(
        "--max-timeline-events",
        type=int,
        default=400,
        help="Maximum number of FREE/UNMAP events to keep in timeline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    filterer = FreeLogFilter(
        log_path=Path(args.log_path).resolve(),
        report_json=Path(args.report_json).resolve(),
        timeline_mermaid=Path(args.timeline_mermaid).resolve(),
        max_timeline_events=args.max_timeline_events,
    )

    def _handle_signal(_signum, _frame):
        filterer.finish()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        for raw_line in sys.stdin:
            filterer.process_line(raw_line)
    except KeyboardInterrupt:
        pass
    finally:
        filterer.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
