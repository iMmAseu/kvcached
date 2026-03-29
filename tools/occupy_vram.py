# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import argparse
import signal
import sys
import time
from typing import List

import torch


def mib_to_bytes(mib: int) -> int:
    return mib * 1024 * 1024


def format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Occupy a fixed amount of GPU memory on a target CUDA device."
    )
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="CUDA device index to use")
    parser.add_argument("--mib",
                        type=int,
                        default=None,
                        help="Target amount of memory to occupy in MiB")
    parser.add_argument(
        "--leave-free-mib",
        type=int,
        default=None,
        help="Instead of --mib, allocate all currently free memory except this amount in MiB",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=256,
        help="Chunk size in MiB for each allocation to reduce one-shot OOM risk",
    )
    parser.add_argument(
        "--touch",
        action="store_true",
        help="Write into each allocation chunk so the memory is eagerly committed",
    )
    return parser.parse_args()


def get_target_bytes(args: argparse.Namespace) -> int:
    if (args.mib is None) == (args.leave_free_mib is None):
        raise ValueError("Specify exactly one of --mib or --leave-free-mib")

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if args.mib is not None:
        target_bytes = mib_to_bytes(args.mib)
    else:
        target_bytes = free_bytes - mib_to_bytes(args.leave_free_mib)

    if target_bytes <= 0:
        raise ValueError(
            f"Target allocation must be positive, got {target_bytes} bytes")
    if target_bytes >= total_bytes:
        raise ValueError(
            f"Target allocation {format_mib(target_bytes)} exceeds total device memory {format_mib(total_bytes)}"
        )
    return target_bytes


def allocate_chunks(target_bytes: int, chunk_bytes: int,
                    touch: bool) -> List[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    allocated = 0
    while allocated < target_bytes:
        current = min(chunk_bytes, target_bytes - allocated)
        tensor = torch.empty(current, dtype=torch.uint8, device="cuda")
        if touch:
            tensor.fill_(1)
        chunks.append(tensor)
        allocated += current
    return chunks


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available.", file=sys.stderr)
        return 1

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    free_before, total_bytes = torch.cuda.mem_get_info(device)
    target_bytes = get_target_bytes(args)
    chunk_bytes = mib_to_bytes(args.chunk_mib)

    print(
        f"[occupy_vram] device={device} total={format_mib(total_bytes)} free_before={format_mib(free_before)} target={format_mib(target_bytes)} chunk={format_mib(chunk_bytes)}"
    )

    chunks = allocate_chunks(target_bytes, chunk_bytes, args.touch)
    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)

    print(
        f"[occupy_vram] allocated={format_mib(target_bytes)} free_after={format_mib(free_after)} held_chunks={len(chunks)}"
    )
    print("[occupy_vram] holding memory, press Ctrl+C to release")

    stop = False

    def _handle_signal(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while not stop:
        time.sleep(1)

    del chunks
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        free_final, _ = torch.cuda.mem_get_info(device)
        print(f"[occupy_vram] released, free_final={format_mib(free_final)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
