#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Single-threaded variant of ttnn_trace_race_repro.py: NO background replay
# thread, NO concurrency at all -- purely sequential release_trace() +
# recapture (begin_trace_capture/end_trace_capture) on N traces, one at a
# time. Isolates whether the crash discovered in the concurrent-replay
# version requires overlap with another trace's in-flight replay, or is a
# plain sequential logic bug in trace eviction/recapture bookkeeping
# (e.g. RingbufferCacheManager) with no timing/race component at all.

import argparse

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
    parser.add_argument("--recapture-rounds", type=int, default=2)
    args = parser.parse_args()

    shape = (1, 1, 256, 256)
    dtype = ttnn.bfloat16

    print("[repro-st] opening device, trace_region_size=0", flush=True)
    device = ttnn.open_device(device_id=0, trace_region_size=0)

    try:
        print(f"[repro-st] capturing {args.num_traces} distinct traces...", flush=True)
        traces = []
        for i in range(args.num_traces):
            traces.append(capture_trace(device, shape, dtype))
        print(f"[repro-st] captured {len(traces)} traces", flush=True)

        for round_num in range(args.recapture_rounds):
            print(f"[repro-st] round {round_num}: releasing+recapturing all {args.num_traces} traces sequentially, no concurrency", flush=True)
            for i in range(args.num_traces):
                old_tid, _old_tensors = traces[i]
                ttnn.release_trace(device, old_tid)
                traces[i] = capture_trace(device, shape, dtype)
            print(f"[repro-st] round {round_num} done", flush=True)

        print("[repro-st] draining device...", flush=True)
        ttnn.synchronize_device(device)
        print("[repro-st] SUCCESS -- no crash, no hang, purely sequential.", flush=True)

        for tid, _tensors in traces:
            ttnn.release_trace(device, tid)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
