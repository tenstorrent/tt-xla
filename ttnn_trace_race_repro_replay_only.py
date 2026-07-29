#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Even more minimal variant: capture N traces ONCE, then continuously
# replay (execute_trace, blocking=False) ALL of them concurrently from
# multiple threads, forever -- NO release_trace, NO recapture, NO eviction
# at all. Isolates whether concurrent REPLAY of different traces alone
# (no capture/eviction machinery involved at all) is sufficient to trigger
# a crash/hang, or whether eviction/recapture must be involved.

import argparse
import threading
import time

import torch
import ttnn


def build_op_chain(a, b):
    return ttnn.add(ttnn.gelu(a), ttnn.relu(b))


def capture_trace(device, shape, dtype, cq_id=0):
    a = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    b = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    build_op_chain(a, b)
    trace_id = ttnn.begin_trace_capture(device, cq_id=cq_id)
    out = build_op_chain(a, b)
    ttnn.end_trace_capture(device, trace_id, cq_id=cq_id)
    return trace_id, (a, b, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-traces", type=int, default=4)
    parser.add_argument("--duration-seconds", type=int, default=20)
    args = parser.parse_args()

    shape = (1, 1, 256, 256)
    dtype = ttnn.bfloat16

    print("[repro-ro] opening device, trace_region_size=0", flush=True)
    device = ttnn.open_device(device_id=0, trace_region_size=0)

    try:
        print(f"[repro-ro] capturing {args.num_traces} distinct traces (once, no recapture ever)...", flush=True)
        traces = [capture_trace(device, shape, dtype) for _ in range(args.num_traces)]
        print(f"[repro-ro] captured {len(traces)} traces, resident for the whole run", flush=True)

        stop = threading.Event()
        counts = [0] * args.num_traces

        def replayer(i):
            tid, _tensors = traces[i]
            while not stop.is_set():
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                counts[i] += 1

        threads = [threading.Thread(target=replayer, args=(i,), daemon=True) for i in range(args.num_traces)]
        print(f"[repro-ro] starting {len(threads)} concurrent replay threads, running for {args.duration_seconds}s...", flush=True)
        for t in threads:
            t.start()

        time.sleep(args.duration_seconds)
        stop.set()
        for t in threads:
            t.join(timeout=10)

        print(f"[repro-ro] replay counts per trace: {counts}", flush=True)
        print("[repro-ro] draining device...", flush=True)
        ttnn.synchronize_device(device)
        print("[repro-ro] SUCCESS -- no crash, no hang, pure concurrent replay with no eviction.", flush=True)

        for tid, _tensors in traces:
            ttnn.release_trace(device, tid)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
