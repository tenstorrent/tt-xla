# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Standalone repro: ttnn.to_device of a large host tensor wedges NOC0.
#
# Mirrors the exact failing op from the Qwen3 codegen (graph_1, op 0258):
#   host [1492, 8, 32, 128] bf16 ROW_MAJOR --to_layout(TILE)--> --to_device(DRAM)-->
# The tilize (to_layout) is host-side and completes; the to_device upload hangs.
#
# Uses open_mesh_device([1,1]) on purpose: the hang backtrace was in
# tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue, i.e. the
# mesh command-queue path, so a plain open_device would not be faithful.
#
# Sweeps upload sizes to bracket the threshold. Each size prints "trying" before
# and "OK" after; the size that hangs is the last "trying" with no matching "OK".
#
#   python repro_to_device_hang.py
import torch
import ttnn

# KV-cache-shaped tensors: [num_blocks, kv_heads, block_size, head_dim].
# num_blocks scales the size; the failing config was num_blocks=1492 (~93 MiB).
KV_HEADS, BLOCK_SIZE, HEAD_DIM = 8, 32, 128
BYTES_PER_EL = 2  # bf16


def mib(num_blocks):
    return num_blocks * KV_HEADS * BLOCK_SIZE * HEAD_DIM * BYTES_PER_EL / 1024 / 1024


# ~12 MiB (worked @4k) up to ~93 MiB (hangs @32k), plus a couple beyond.
BLOCK_COUNTS = [186, 512, 768, 1024, 1492, 2048]

DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


# A single large to_device (even 128 MiB) uploads fine, so raw transfer size is
# NOT the trigger. The real forward keeps every uploaded input resident. So here
# we ACCUMULATE: upload KV-sized tensors without deallocating and track the
# cumulative resident total, replaying what the real run does up to op 0258.
# Set DEALLOCATE=1 to revert to the (passing) single-shot-per-size behavior.
import os

DEALLOCATE = os.environ.get("DEALLOCATE") == "1"
NB = 1492  # 93.2 MiB each — the failing KV shape
N_TENSORS = 80  # 80 * 93 MiB ~= 7.3 GiB; covers all 56 KV caches + headroom


def main():
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape([1, 1]), l1_small_size=1 << 15
    )
    print(f"opened {device}  (DEALLOCATE={DEALLOCATE})", flush=True)
    resident = []
    cumulative = 0.0
    try:
        for i in range(N_TENSORS):
            shape = (NB, KV_HEADS, BLOCK_SIZE, HEAD_DIM)
            cumulative += mib(NB)
            print(
                f"\n=== #{i} shape={shape} {mib(NB):.1f} MiB  cumulative~{cumulative/1024:.2f} GiB ===",
                flush=True,
            )
            host = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            tiled = ttnn.to_layout(host, ttnn.TILE_LAYOUT)
            print(f"  trying to_device (#{i}) ...", flush=True)
            dev = ttnn.to_device(tiled, device, memory_config=DRAM)
            ttnn.synchronize_device(device)
            print(f"  OK #{i}  cumulative~{cumulative/1024:.2f} GiB", flush=True)
            if DEALLOCATE:
                ttnn.deallocate(dev)
                cumulative -= mib(NB)
            else:
                resident.append(dev)  # keep it resident, like the real forward
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
