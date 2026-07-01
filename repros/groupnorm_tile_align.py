# SPDX-License-Identifier: Apache-2.0
"""
Minimal repro: `ttnn.group_norm` rejects a non-tile-aligned "flattened height".

A plain `nn.GroupNorm` whose normalized spatial length is NOT a multiple of 32
fails during the TTIR->TTNN lowering. This blocks the XTTS-v2 conditioning
encoder (issue #5216), whose AttentionBlock GroupNorm sees data-dependent mel
lengths (e.g. 269) that are essentially never a multiple of 32.

Run:
    python repros/groupnorm_tile_align.py

Expected (fails in the MLIR frontend, ~2.5s, before any kernel build):
    CPU ok: (1, 1024, 200)
    loc("custom-call.100"): error: 'ttnn.group_norm' op flattened height must be
        tile-aligned, got 200
    ERR| Failed to run TTIRToTTNNCommon pipeline
    ValueError: Error code: 13
"""
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm

xr.set_device_type("TT")

gn = torch.nn.GroupNorm(num_groups=32, num_channels=1024).eval()
x = torch.randn(1, 1024, 200)  # length 200 is NOT a multiple of 32

with torch.no_grad():
    cpu_out = gn(x)
print("CPU ok:", tuple(cpu_out.shape), flush=True)

dev = xm.xla_device()
gn_tt = gn.to(dev)
gn_tt.compile(backend="tt")
with torch.no_grad():
    tt_out = gn_tt(x.to(dev)).to("cpu")
print("TT ok:", tuple(tt_out.shape), flush=True)
