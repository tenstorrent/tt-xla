# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ZImageTransformer TTNN inference — clean entry point.

Loads ZImageTransformer2DModel from HuggingFace (Tongyi-MAI/Z-Image-Turbo),
runs a forward pass on a TTNN (1,4) mesh device with 4-way tensor parallelism,
and compares against a CPU PyTorch reference to report PCC.

Baseline PCC (bfloat16, 4-way TP): ≥ 0.995

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

import utils                    # DeviceGetter singleton
import model_pt                 # load_model, pad_heads, forward
from model_ttnn import ZImageTransformerTTNN  # LightweightModule

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _to_device(pt, mesh_device):
    """Upload a bfloat16 PyTorch tensor to the mesh device (replicated)."""
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
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

    # ── Load and patch transformer ───────────────────────────────────────────
    print(f"\nLoading transformer from {model_pt.MODEL_ID}/transformer ...")
    transformer = model_pt.load_model()

    # ── Dummy inputs ─────────────────────────────────────────────────────────
    latents   = [torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)]
    timestep  = torch.tensor([0.5], dtype=torch.bfloat16)
    cap_feats = torch.randn(32, 2560, dtype=torch.bfloat16)

    # ── CPU reference ────────────────────────────────────────────────────────
    print("\nRunning CPU reference (3 iterations) ...")
    for i in range(3):
        start = time.time()
        pt_outputs = model_pt.forward(transformer, latents, timestep, cap_feats)
        duration_ms = (time.time() - start) * 1000
        fps = 1000.0 / duration_ms
        print(f"  Iteration {i}:  {duration_ms:.1f} ms  |  {fps:.2f} FPS")
    pt_out = pt_outputs[0]
    print(f"  Output: shape={tuple(pt_out.shape)}  "
          f"mean={pt_out.float().mean():.4f}  std={pt_out.float().std():.4f}")

    # ── Head padding (30 → 32) before loading TTNN weights ──────────────────
    print("\nApplying head padding...")
    model_pt.pad_heads(transformer)

    # ── Open TTNN mesh device ────────────────────────────────────────────────
    mesh_device = utils.DeviceGetter.get_device((1, 4))
    print(f"\nMesh device: {mesh_device}")

    # ── Build TTNN model (loads weights + runs consteval) ─────────────────────
    print("\nBuilding TTNN model ...")
    tt_model = ZImageTransformerTTNN(mesh_device, transformer)

    # ── Upload runtime inputs (done once, reused across iterations) ──────────
    cap_3d       = cap_feats.unsqueeze(0)  # [1, 32, 2560]
    tt_timestep  = _to_device(timestep.reshape(1), mesh_device)
    tt_cap_feats = _to_device(cap_3d, mesh_device)
    tt_latent    = _to_device(latents[0], mesh_device)

    # ── Inference loop ────────────────────────────────────────────────────────
    print("\nRunning inference loop (3 iterations) ...")
    for i in range(3):
        start = time.time()
        signpost(f"transformer_start_{i}")

        tt_outputs = tt_model([tt_latent], tt_timestep, tt_cap_feats)
        tt_out = tt_outputs[0]

        # Bring result to host (synchronize included in forward via synchronize_device)
        _ = ttnn.from_device(tt_out)

        signpost(f"transformer_end_{i}")
        end = time.time()

        duration_ms = (end - start) * 1000
        fps = 1.0 / (end - start)
        pcc = pcc_score(tt_out, pt_out, mesh_device)

        print(f"  Iteration {i}:  {duration_ms:.1f} ms  |  {fps:.2f} FPS  |  PCC={pcc:.6f}")

    # ── PCC check on last iteration ───────────────────────────────────────────
    if pcc >= 0.995:
        print(f"\n✓  PCC={pcc:.6f} — within expected range (>= 0.995)")
    else:
        print(f"\n✗  PCC={pcc:.6f} — BELOW expected threshold of 0.995!")
        sys.exit(1)


if __name__ == "__main__":
    main()
