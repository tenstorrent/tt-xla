# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Minimal pure-TTNN repro attempt for the fast-dispatch wedge.
#
# The full generated graph_1.forward reproduces the hang (wedges on a to_device
# in the KV-upload region). The distinguishing factor vs a pure-upload loop
# (which does NOT hang) is that forward runs a batch of const-eval device
# *compute* programs first, then does a long run of to_layout+to_device uploads.
# This script tests whether "dispatch some compute programs, then blast a run of
# resident to_device uploads" is the minimal trigger.
#
#   python repro_min.py            # expect wedge (last "[NNN] up" without "ok")
#   COMPUTE=0 python repro_min.py  # control: skip compute phase (expect pass)
import os

import torch
import ttnn

DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)
COMPUTE = os.environ.get("COMPUTE", "1") == "1"

# KV-cache-ish shape (~64 MiB) + assorted weight shapes, like the upload run.
KV = (1025, 8, 32, 128)
WEIGHTS = [
    (2048, 1024),
    (1024, 1024),
    (1024, 1024),
    (1024, 2048),
    (3072, 1024),
    (3072, 1024),
    (1024, 3072),
    (1024,),
    (1024,),
]


def mib(shape):
    n = 1
    for d in shape:
        n *= d
    return n * 2 / 1024 / 1024


def main():
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape([1, 1]), l1_small_size=1 << 15
    )
    print(f"opened {device}  COMPUTE={COMPUTE}", flush=True)

    # Phase 1: dispatch compute programs (mimic the const-eval subgraphs).
    if COMPUTE:
        a = ttnn.from_torch(
            torch.randn(512, 512, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=DRAM,
        )
        for i in range(40):
            a = ttnn.matmul(a, a, memory_config=DRAM)
            a = ttnn.typecast(a, ttnn.bfloat16, memory_config=DRAM)
        ttnn.synchronize_device(device)
        print("compute phase done (40 programs dispatched)", flush=True)

    # Phase 2: long run of to_layout(TILE)+to_device(DRAM), tensors kept resident,
    # per-op synchronize (matches the instrumented generated forward).
    resident = []
    seq = []
    for l in range(28):  # 28 layers of weights, interleaved
        seq += WEIGHTS
    for l in range(28):  # then the KV-cache run (where the real wedge hits)
        seq += [KV, KV]
    for i, shape in enumerate(seq):
        print(f"[{i:04d}] up {tuple(shape)} {mib(shape):.1f}MiB", flush=True)
        host = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        tiled = ttnn.to_layout(host, ttnn.TILE_LAYOUT)
        dev = ttnn.to_device(tiled, device, memory_config=DRAM)
        ttnn.synchronize_device(device)
        resident.append(dev)
        print(f"[{i:04d}] ok", flush=True)

    ttnn.synchronize_device(device)
    print("ALL DONE (no hang)", flush=True)
    ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
