#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Standalone, pure-ttnn reproducer attempt for a suspected DRAM address-reuse
# race in tt-metal's trace-capture/eviction machinery -- NO tt-mlir runtime,
# NO tt-xla, NO vLLM involved. See tt-xla's FALCON3_SINGLE_LAYER_HANG_DEBUG.md
# "ROOT CAUSE IDENTIFIED" for the full mechanism this targets:
#
#   With trace_region_size=0 (the default -- shared DRAM pool), releasing a
#   trace buffer (release_trace -> ... -> Buffer::deallocate_impl) returns its
#   DRAM address to the allocator with ZERO device-side synchronization. A
#   newly-captured trace can then be allocated at that freed address while a
#   DIFFERENT, still-resident trace's non-blocking replay may still have work
#   in flight nearby.
#
# Strategy: keep half of N captured traces continuously replaying
# (execute_trace(..., blocking=False)) on a background thread, while the main
# thread repeatedly releases+recaptures the other half -- racing DRAM reuse
# against in-flight replay. If the hypothesis is right, this should
# eventually hang inside a completion-queue wait (mirroring the real
# workload's FDMeshCommandQueue::read_completion_queue() stall), or produce
# a golden-compare mismatch (silent corruption instead of a hang).
#
# Status: written from source research (ttnn-nanobind bindings +
# tests/ttnn/unit_tests/base_functionality/test_multi_device_trace.py), not
# yet smoke-tested against real hardware -- expect to need small API-surface
# fixes on the first real run (exact kwarg names, tensor allocation calls).
#
# Usage:
#   TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=... python3 ttnn_trace_race_repro.py \
#       [--num-traces 48] [--recapture-rounds 500] [--replay-workers-per-trace 1] \
#       [--timeout-seconds 300]

import argparse
import sys
import threading
import time

import torch
import ttnn


def build_op_chain(a, b):
    # Deliberately simple/cheap so many traces can be captured without
    # DRAM-OOMing during the initial compile pass -- the interesting variable
    # is trace COUNT/eviction churn, not per-op activation size.
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

    # Compile once eagerly, matching production's "first run compiles, then
    # trace-capture the identical program" pattern.
    build_op_chain(a, b)

    trace_id = ttnn.begin_trace_capture(device, cq_id=cq_id)
    out = build_op_chain(a, b)
    ttnn.end_trace_capture(device, trace_id, cq_id=cq_id)
    return trace_id, (a, b, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-traces", type=int, default=48)
    parser.add_argument("--recapture-rounds", type=int, default=500)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    shape = (1, 1, 256, 256)
    dtype = ttnn.bfloat16

    print(f"[repro] opening device, trace_region_size=0 (shared DRAM pool)", flush=True)
    device = ttnn.open_device(device_id=0, trace_region_size=0)

    try:
        print(f"[repro] capturing {args.num_traces} distinct traces...", flush=True)
        traces = []
        for i in range(args.num_traces):
            traces.append(capture_trace(device, shape, dtype))
        print(f"[repro] captured {len(traces)} traces, resident on device", flush=True)

        even_ids = list(range(0, args.num_traces, 2))
        odd_ids = list(range(1, args.num_traces, 2))

        stop = threading.Event()
        replay_count = [0]

        def replayer():
            while not stop.is_set():
                for i in even_ids:
                    tid, _tensors = traces[i]
                    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                    replay_count[0] += 1

        replay_thread = threading.Thread(target=replayer, daemon=True)
        replay_thread.start()

        print(
            f"[repro] racing: background non-blocking replay of {len(even_ids)} traces "
            f"vs. {args.recapture_rounds} release+recapture rounds on {len(odd_ids)} traces",
            flush=True,
        )
        for round_num in range(args.recapture_rounds):
            for i in odd_ids:
                old_tid, _old_tensors = traces[i]
                ttnn.release_trace(device, old_tid)
                traces[i] = capture_trace(device, shape, dtype)
            if round_num % 50 == 0:
                print(
                    f"[repro] round {round_num}/{args.recapture_rounds}, "
                    f"background replays so far: {replay_count[0]}",
                    flush=True,
                )

        stop.set()
        replay_thread.join(timeout=30)

        print("[repro] draining device (suspected hang manifests here)...", flush=True)

        result_holder = {}

        def drain():
            ttnn.synchronize_device(device)
            result_holder["drained"] = True

        drain_thread = threading.Thread(target=drain, daemon=True)
        start = time.monotonic()
        drain_thread.start()
        drain_thread.join(timeout=args.timeout_seconds)

        if not result_holder.get("drained"):
            elapsed = time.monotonic() - start
            print(
                f"[repro] HANG REPRODUCED: synchronize_device did not return after "
                f"{elapsed:.1f}s (timeout {args.timeout_seconds}s). This process will "
                f"likely need to be killed -- attach gdb to find the spinning thread, "
                f"expect FDMeshCommandQueue::read_completion_queue() or equivalent.",
                flush=True,
            )
            sys.exit(2)

        print("[repro] drain completed cleanly -- no hang this run.", flush=True)

        for tid, _tensors in traces:
            ttnn.release_trace(device, tid)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
