# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TextEncoder (Qwen3) TTNN inference — clean entry point.

Loads Qwen3Model from HuggingFace (Tongyi-MAI/Z-Image-Turbo),
runs a forward pass on a TTNN (1,4) mesh device with 4-way tensor parallelism,
and compares against a CPU PyTorch reference to report PCC.

Usage:
    python main.py
"""

import os
import sys
import time

import torch
import ttnn
from tracy import signpost

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import utils                             # DeviceGetter singleton
import model_pt                          # load_model, forward
from model_ttnn import TextEncoderTTNN   # LightweightModule

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

# Token IDs used during tracing (7 tokens matching the tensorbin snapshots)
DUMMY_TOKEN_IDS = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64)


def _to_device_ids(input_ids_int64, mesh_device):
    """Convert int64 token IDs to [1, seq] INT32 TTNN tensor on device (replicated)."""
    t = input_ids_int64.unsqueeze(0).to(torch.int32)  # [1, seq_len]
    return ttnn.from_torch(
        t,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def pcc_score(tt_tensor, pt_tensor, mesh_device):
    """Return PCC between TTNN output and PT reference."""
    tt_host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    n = tt_host.shape[0] // 4
    tt_cpu = tt_host[:n].float()
    pt_cpu = pt_tensor.float()
    return torch.corrcoef(torch.stack([tt_cpu.flatten(), pt_cpu.flatten()]))[0, 1].item()


def main():
    torch.manual_seed(42)

    # ── Load CPU reference model ──────────────────────────────────────────────
    tokenizer, text_encoder = model_pt.load_model()

    # ── CPU reference ────────────────────────────────────────────────────────
    print("\nRunning CPU reference (3 iterations) ...")
    for i in range(3):
        start = time.time()
        pt_out = model_pt.forward(text_encoder, DUMMY_TOKEN_IDS)
        duration_ms = (time.time() - start) * 1000
        fps = 1000.0 / duration_ms
        print(f"  Iteration {i}:  {duration_ms:.1f} ms  |  {fps:.2f} FPS")
    print(f"  Output: shape={tuple(pt_out.shape)}  "
          f"mean={pt_out.float().mean():.4f}  std={pt_out.float().std():.4f}")

    # ── Open TTNN mesh device ─────────────────────────────────────────────────
    mesh_device = utils.DeviceGetter.get_device((1, 4))
    print(f"\nMesh device: {mesh_device}")

    # ── Build TTNN model ──────────────────────────────────────────────────────
    print("\nBuilding TTNN model ...")
    tt_model = TextEncoderTTNN(mesh_device, pt_model=text_encoder)

    # ── Upload runtime input (done once, reused across iterations) ───────────
    tt_input_ids = _to_device_ids(DUMMY_TOKEN_IDS, mesh_device)

    # ── Inference loop ────────────────────────────────────────────────────────
    print("\nRunning inference loop (3 iterations) ...")
    for i in range(3):
        start = time.time()
        signpost(f"text_encoder_start_{i}")

        tt_out = tt_model(tt_input_ids)
        _ = ttnn.from_device(tt_out)
        ttnn.synchronize_device(mesh_device)

        signpost(f"text_encoder_end_{i}")
        end = time.time()

        duration_ms = (end - start) * 1000
        fps = 1000.0 / duration_ms
        pcc = pcc_score(tt_out, pt_out, mesh_device)

        print(f"  Iteration {i}:  {duration_ms:.1f} ms  |  {fps:.2f} FPS  |  PCC={pcc:.6f}")

    # ── PCC check on last iteration ───────────────────────────────────────────
    if pcc >= 0.99:
        print(f"\n✓  PCC={pcc:.6f} — within expected range (>= 0.99)")
    else:
        print(f"\n✗  PCC={pcc:.6f} — BELOW expected threshold of 0.99!")
        sys.exit(1)


if __name__ == "__main__":
    main()
