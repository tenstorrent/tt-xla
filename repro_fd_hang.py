# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Standalone TTNN repro attempt for the fast-dispatch to_device wedge seen in the
# Qwen3-0.6B codegen graph_1. DO NOT expect the older repro_to_device_hang.py to
# reproduce: it synchronizes after every to_device, which drains the command
# queue each iteration and hides the bug. The real forward fires a long run of
# to_layout + to_device commands *asynchronously* (no per-op sync), keeps every
# tensor resident, and wedges partway through -- host stuck in
# FDMeshCommandQueue::read_completion_queue while the device sits idle.
#
# This script mirrors that: it replays graph_1's upload phase -- ~170 weight
# tensors followed by a consecutive run of 56 paged-KV-cache tensors -- as
# to_layout(TILE)+to_device(DRAM), tensors kept resident, and (by default) NO
# per-op synchronize. Only a single synchronize at the very end.
#
#   python repro_fd_hang.py                 # faithful: async, no per-op sync
#   SYNC_EACH=1 python repro_fd_hang.py     # A/B control: sync each op (expected to pass)
#   BLOCKS=1025 python repro_fd_hang.py     # KV num_blocks (default 1025 ~ 64 MiB/cache)
#
# Interpreting a run: the last "[NNN] upload ..." line printed with no following
# "[NNN] ok" (or a hang before the final "ALL UPLOADS DONE") pins the wedge. Run
# under TT_METAL_WATCHER=1 to confirm the device is idle-waiting while host spins.
import os

import torch
import ttnn

# --- Qwen3-0.6B dims ---
HIDDEN = 1024
INTERMEDIATE = 3072
N_LAYERS = 28
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
VOCAB = 151936

BLOCKS = int(os.environ.get("BLOCKS", "1025"))  # paged KV cache num_blocks
BLOCK_SIZE = 32
SYNC_EACH = os.environ.get("SYNC_EACH") == "1"

DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


def mib(shape):
    n = 1
    for d in shape:
        n *= d
    return n * 2 / 1024 / 1024  # bf16


def upload_sequence():
    """Flat list of (name, shape) mirroring graph_1's inputs: weights first,
    then a consecutive run of KV caches (the region where the real run wedged)."""
    seq = []
    # Per-layer weights (bf16). Shapes are [out, in]; size is order-independent.
    for l in range(N_LAYERS):
        seq += [
            (f"L{l}.input_ln", (HIDDEN,)),
            (f"L{l}.q_proj", (N_HEADS * HEAD_DIM, HIDDEN)),
            (f"L{l}.q_norm", (HEAD_DIM,)),
            (f"L{l}.k_proj", (N_KV_HEADS * HEAD_DIM, HIDDEN)),
            (f"L{l}.k_norm", (HEAD_DIM,)),
            (f"L{l}.v_proj", (N_KV_HEADS * HEAD_DIM, HIDDEN)),
            (f"L{l}.o_proj", (HIDDEN, N_HEADS * HEAD_DIM)),
            (f"L{l}.post_ln", (HIDDEN,)),
            (f"L{l}.gate_proj", (INTERMEDIATE, HIDDEN)),
            (f"L{l}.up_proj", (INTERMEDIATE, HIDDEN)),
            (f"L{l}.down_proj", (HIDDEN, INTERMEDIATE)),
        ]
    seq.append(("final_norm", (HIDDEN,)))
    # Paged KV caches: K and V per layer, uploaded consecutively (the run that
    # contained the wedge). Shape [num_blocks, n_kv_heads, block_size, head_dim].
    kv_shape = (BLOCKS, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    for l in range(N_LAYERS):
        seq.append((f"L{l}.kv_k", kv_shape))
        seq.append((f"L{l}.kv_v", kv_shape))
    return seq


def main():
    seq = upload_sequence()
    total = sum(mib(s) for _, s in seq)
    print(
        f"SYNC_EACH={SYNC_EACH}  tensors={len(seq)}  total~{total / 1024:.2f} GiB  "
        f"KV={mib((BLOCKS, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)):.1f} MiB each",
        flush=True,
    )
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape([1, 1]), l1_small_size=1 << 15
    )
    print(f"opened {device}", flush=True)
    resident = []  # keep everything on device, like the real forward
    try:
        for i, (name, shape) in enumerate(seq):
            print(
                f"[{i:04d}] upload {name} {tuple(shape)} {mib(shape):.2f}MiB",
                flush=True,
            )
            host = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            tiled = ttnn.to_layout(host, ttnn.TILE_LAYOUT)
            dev = ttnn.to_device(tiled, device, memory_config=DRAM)
            resident.append(dev)
            if SYNC_EACH:
                ttnn.synchronize_device(device)
            print(f"[{i:04d}] ok", flush=True)
        ttnn.synchronize_device(device)  # single drain at the end
        print("ALL UPLOADS DONE (no hang)", flush=True)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
