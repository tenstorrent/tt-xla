# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal profiling harness for OPT_FUSED_QKV model.

Run via:
    python -m tracy -r -v -o /tmp/tt_profile profile_run.py

This does N_WARMUP passes (not profiled by the device profiler) then
N_PROFILE passes (captured in the Tracy/device-profiler window).
We call ttnn.ReadDeviceProfiler between warmup and profiled passes so
the CSV only contains the profiled iterations.
"""

import os
import sys
import time

import torch
import ttnn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import utils
import model_pt
from model_ttnn_opt import ZImageTransformerTTNNOpt

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
N_WARMUP  = 2
N_PROFILE = 1   # single pass is enough for op breakdown


def _to_device(pt, mesh_device):
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def main():
    torch.manual_seed(42)

    print("Loading transformer ...")
    transformer = model_pt.load_model()

    latents   = [torch.randn(16, 1, 64, 64, dtype=torch.bfloat16)]
    timestep  = torch.tensor([0.5], dtype=torch.bfloat16)
    cap_feats = torch.randn(32, 2560, dtype=torch.bfloat16)

    model_pt.pad_heads(transformer)

    mesh_device = utils.DeviceGetter.get_device((1, 4))

    tt_timestep  = _to_device(timestep.reshape(1), mesh_device)
    tt_cap_feats = _to_device(cap_feats.unsqueeze(0), mesh_device)
    tt_latent    = _to_device(latents[0], mesh_device)

    # Build model with best config
    ZImageTransformerTTNNOpt.USE_FAST_NORMS      = True
    ZImageTransformerTTNNOpt.USE_FAST_QK_NORM    = True
    ZImageTransformerTTNNOpt.USE_FAST_FINAL_NORM = True
    ZImageTransformerTTNNOpt.USE_DIT_NORM        = False
    ZImageTransformerTTNNOpt.USE_MINIMAL_MATMUL  = True
    ZImageTransformerTTNNOpt.USE_FUSED_QKV       = True

    print("Building model (OPT_FUSED_QKV) ...")
    model = ZImageTransformerTTNNOpt(mesh_device, transformer)
    print("Model built.")

    # ── Warmup (not part of profiled window) ──────────────────────────────────
    print(f"Warming up ({N_WARMUP} iters) ...")
    for i in range(N_WARMUP):
        tt_out = model([tt_latent], tt_timestep, tt_cap_feats)[0]
        _ = ttnn.from_device(tt_out)
        print(f"  warmup {i} done")

    # Flush device profiler data accumulated during warmup so it isn't counted
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
        print("Device profiler flushed after warmup.")
    except Exception as e:
        print(f"Note: ReadDeviceProfiler flush failed ({e}) — warmup data may appear in report")

    # ── Profiled pass ─────────────────────────────────────────────────────────
    print(f"Profiled pass ({N_PROFILE} iter) ...")
    for i in range(N_PROFILE):
        t0 = time.time()
        tt_out = model([tt_latent], tt_timestep, tt_cap_feats)[0]
        _ = ttnn.from_device(tt_out)
        print(f"  profiled iter {i}: {(time.time()-t0)*1000:.1f} ms")

    print("Profile run complete.")


if __name__ == "__main__":
    main()
