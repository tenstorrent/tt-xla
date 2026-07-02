# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Minimal: N sequential 64 MiB KV-shape uploads. By op 0238 in the failing run,
# the graph had uploaded 22 KV tensors (23rd wedged). Try 25 back-to-back KV
# uploads with no other shapes.
import os

import ttnn

DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)
N = int(os.environ.get("N", "25"))
KV_PATH = os.environ.get(
    "KV", "qwen_codegen_32k_OOM_attempt2/graph_1/tensors/arg9.tensorbin"
)

device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]), l1_small_size=1 << 15)
print(f"opened; {N} KV uploads from {KV_PATH}", flush=True)
resident = []
for i in range(N):
    host = ttnn.load_tensor(KV_PATH)
    tiled = ttnn.to_layout(host, ttnn.TILE_LAYOUT)
    ttnn.synchronize_device(device)
    print(f"[{i:03d}] to_device (64 MiB)...", flush=True)
    dev = ttnn.to_device(tiled, device, memory_config=DRAM)
    ttnn.synchronize_device(device)
    resident.append(dev)
    print(f"[{i:03d}] ok  resident={(i+1)*64.06:.1f} MiB", flush=True)
print("ALL DONE (no hang)", flush=True)
ttnn.close_mesh_device(device)
