# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Minimal hypothesis: what "primes" the FD wedge is the 297 MiB embedding
# upload early on. After that, ~20 x 64 MiB KV uploads trigger the wedge.
# Uses ttnn.load_tensor from the same graph_1/tensors dir so shapes/dtypes
# are exactly the real inputs.
import os

import ttnn

DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)
BASE = "qwen_codegen_32k_OOM_attempt2/graph_1/tensors"

device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]), l1_small_size=1 << 15)
print("opened", flush=True)
resident = []


def upload(i, note=""):
    host = ttnn.load_tensor(f"{BASE}/arg{i}.tensorbin")
    layout_needed = ttnn.TILE_LAYOUT
    if host.layout != layout_needed:
        host = ttnn.to_layout(host, layout_needed)
    dev = ttnn.to_device(host, device, memory_config=DRAM)
    ttnn.synchronize_device(device)
    resident.append(dev)
    print(f"  input[{i:3d}] {note} OK", flush=True)


# Phase 1: prelude (ops 1..11 in real forward): tiny INT32/FP32 + first weights
print("prelude uploads (input[0..7])", flush=True)
for i in range(8):
    upload(i, f"prelude")

# Phase 2: the big 297 MiB embedding
print("EMBEDDING upload (input[8], 297 MiB)", flush=True)
upload(8, "297 MiB embedding")

# Phase 3: KV uploads until it wedges (up to 25)
KV_IDX = int(os.environ.get("KV_IDX", "9"))  # arg9 is first KV
N_KV = int(os.environ.get("N_KV", "25"))
print(f"KV uploads (input[{KV_IDX}] repeated, {N_KV} times)", flush=True)
for k in range(N_KV):
    # Load the KV tensor fresh each time (same file, but load creates a new host tensor)
    host = ttnn.load_tensor(f"{BASE}/arg{KV_IDX}.tensorbin")
    if host.layout != ttnn.TILE_LAYOUT:
        host = ttnn.to_layout(host, ttnn.TILE_LAYOUT)
    print(f"  KV #{k:03d} to_device (64 MiB)...", flush=True)
    dev = ttnn.to_device(host, device, memory_config=DRAM)
    ttnn.synchronize_device(device)
    resident.append(dev)
    print(f"  KV #{k:03d} OK  cumulative_KV={(k+1)*64.06:.1f} MiB", flush=True)

print("ALL DONE (no hang)", flush=True)
ttnn.close_mesh_device(device)
