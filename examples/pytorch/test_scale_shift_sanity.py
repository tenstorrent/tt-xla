# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Sanity test for the WanTransformerBlock scale_shift_table operation on TT.

The single transformer block test OOMs during the scale_shift_table computation
inside WanTransformerBlock.forward (the `temb.float()` typecast):

    result = (scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)

The spatial_seq is derived from video resolution + frame count:
    spatial_seq = ppf * pph * ppw
    HEIGHT=480, WIDTH=832, patch_size=(1,2,2) -> pph=30, ppw=52 (fixed)
    NUM_FRAMES controls ppf -> directly scales spatial_seq

Frame count sweep and resulting spatial_seq:
    81 frames -> ppf=21 -> spatial_seq = 21*30*52 = 32,760  (full pipeline)
    33 frames -> ppf=9  -> spatial_seq =  9*30*52 = 14,040
    17 frames -> ppf=5  -> spatial_seq =  5*30*52 =  7,800
     5 frames -> ppf=2  -> spatial_seq =  2*30*52 =  3,120
     1 frame  -> ppf=1  -> spatial_seq =  1*30*52 =  1,560

Two test variants per frame count:
  1. WITH .float() cast  -- reproduces the OOM path
  2. WITHOUT .float() cast -- stays bfloat16, proves the cast is the bottleneck

Usage:
    pytest examples/pytorch/test_scale_shift_sanity.py -v -s --forked
    pytest examples/pytorch/test_scale_shift_sanity.py -v -s -k "with_float_cast"
    pytest examples/pytorch/test_scale_shift_sanity.py -v -s -k "no_float_cast"
    pytest examples/pytorch/test_scale_shift_sanity.py -v -s -k "17"
"""

import gc
import time

import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import WanTransformer3DModel

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
SEED = 42

HEIGHT = 480
WIDTH = 832
VAE_SCALE_TEMPORAL = 4
VAE_SCALE_SPATIAL = 8


# ── Helpers ────────────────────────────────────────────────────────────

def _get_model_dims():
    """Load config from the real model to get correct dimensions."""
    config = WanTransformer3DModel.load_config(MODEL_ID, subfolder="transformer")
    num_heads = config.get("num_attention_heads", 40)
    head_dim = config.get("attention_head_dim", 128)
    inner_dim = num_heads * head_dim
    patch_size = tuple(config.get("patch_size", (1, 2, 2)))
    return inner_dim, num_heads, head_dim, patch_size


def _get_spatial_seq_len(num_frames, patch_size=(1, 2, 2)):
    """Compute spatial sequence length from video dims + model patch_size.

    Returns (spatial_seq, ppf, pph, ppw, adjusted_num_frames).
    """
    if num_frames % VAE_SCALE_TEMPORAL != 1:
        num_frames = num_frames // VAE_SCALE_TEMPORAL * VAE_SCALE_TEMPORAL + 1
    num_frames = max(num_frames, 1)

    h_mult = VAE_SCALE_SPATIAL * patch_size[1]
    w_mult = VAE_SCALE_SPATIAL * patch_size[2]
    height = HEIGHT // h_mult * h_mult
    width = WIDTH // w_mult * w_mult

    num_latent_frames = (num_frames - 1) // VAE_SCALE_TEMPORAL + 1
    latent_h = height // VAE_SCALE_SPATIAL
    latent_w = width // VAE_SCALE_SPATIAL

    ppf = num_latent_frames
    pph = latent_h // patch_size[1]
    ppw = latent_w // patch_size[2]
    return ppf * pph * ppw, ppf, pph, ppw, num_frames


def _cleanup_device():
    """Best-effort cleanup of TT device DRAM between tests."""
    try:
        torch_xla.sync(wait=True)
    except Exception:
        pass
    torch._dynamo.reset()
    try:
        xr.clear_computation_cache()
    except Exception:
        pass
    gc.collect()


# ── Modules under test ────────────────────────────────────────────────

class ScaleShiftWithCast(nn.Module):
    """Exact reproduction of the scale_shift_table op from WanTransformerBlock.

    This includes the .float() cast that causes the TypecastOp OOM.
    """

    def __init__(self, dim):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, temb):
        result = (self.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
        return tuple(r.squeeze(2) for r in result)


class ScaleShiftNoCast(nn.Module):
    """Same op but WITHOUT the .float() cast -- stays in bfloat16.

    If this passes where ScaleShiftWithCast OOMs, the float32 promotion
    (and TTNN's tiled layout padding for float32) is the sole bottleneck.
    """

    def __init__(self, dim):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, temb):
        result = (self.scale_shift_table.unsqueeze(0) + temb).chunk(6, dim=2)
        return tuple(r.squeeze(2) for r in result)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def tt_setup():
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})


@pytest.fixture(autouse=True)
def cleanup_between_tests():
    """Runs DRAM cleanup BEFORE and AFTER every test."""
    _cleanup_device()
    yield
    _cleanup_device()


# ── Tests ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("num_frames", [1, 5, 17, 33, 81])
def test_scale_shift_with_float_cast(tt_setup, num_frames):
    """scale_shift_table + temb.float() at full spatial seq_len.

    This reproduces the exact OOM seen in WanTransformerBlock.forward:
      temb: (1, spatial_seq, 6, inner_dim)  bfloat16 -> .float() -> float32

    Parametrized by num_frames to find the OOM boundary:
      81 frames -> spatial_seq=32760  (~1.2 GB bf16, OOM at 12.88 GB tiled)
      33 frames -> spatial_seq=14040  (~0.5 GB bf16, OOM at 5.52 GB tiled)
      17 frames -> spatial_seq=7800   (~0.3 GB bf16, PASSES)
       5 frames -> spatial_seq=3120   (~0.1 GB bf16)
       1 frame  -> spatial_seq=1560   (~0.05 GB bf16)
    """
    inner_dim, _, _, patch_size = _get_model_dims()
    spatial_seq, ppf, pph, ppw, adj_frames = _get_spatial_seq_len(num_frames, patch_size)

    temb_bf16_mb = spatial_seq * 6 * inner_dim * 2 / (1024 * 1024)
    temb_fp32_mb = spatial_seq * 6 * inner_dim * 4 / (1024 * 1024)

    print(f"\n--- scale_shift_table WITH .float() [{num_frames} frames] ---")
    print(f"  inner_dim={inner_dim}  (from model config)")
    print(f"  num_frames={num_frames} -> adjusted={adj_frames}")
    print(f"  spatial_seq = {ppf}*{pph}*{ppw} = {spatial_seq}")
    print(f"  temb shape:        (1, {spatial_seq}, 6, {inner_dim})  bfloat16")
    print(f"  temb bf16 size:    {temb_bf16_mb:.1f} MB")
    print(f"  temb fp32 size:    {temb_fp32_mb:.1f} MB  (after .float())")

    op = ScaleShiftWithCast(inner_dim).to(dtype=torch.bfloat16)
    op.compile(backend="tt")
    op = op.to(xm.xla_device())

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    temb = torch.randn(1, spatial_seq, 6, inner_dim, generator=generator, dtype=torch.bfloat16)
    temb_tt = temb.to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        outputs = op(temb_tt)
    elapsed = time.time() - t0

    outputs_cpu = tuple(o.to("cpu") for o in outputs)
    print(f"  num outputs:       {len(outputs_cpu)}")
    print(f"  output[0] shape:   {outputs_cpu[0].shape}  dtype={outputs_cpu[0].dtype}")
    print(f"  time:              {elapsed:.1f}s")

    assert len(outputs_cpu) == 6
    for i, o in enumerate(outputs_cpu):
        assert o.shape == (1, spatial_seq, inner_dim), (
            f"Chunk {i}: expected (1, {spatial_seq}, {inner_dim}), got {o.shape}"
        )
        assert torch.isfinite(o).all(), f"Chunk {i} contains NaN/Inf"

    del op, temb, temb_tt, outputs, outputs_cpu
    print(f"  PASSED ({num_frames} frames, spatial_seq={spatial_seq})")


@pytest.mark.parametrize("num_frames", [1, 5, 17, 33, 81])
def test_scale_shift_no_float_cast(tt_setup, num_frames):
    """scale_shift_table WITHOUT .float() at full spatial seq_len.

    Same operation but stays in bfloat16.  If this passes where the
    float32 version OOMs, the TypecastOp (bf16->fp32) is the bottleneck.
    """
    inner_dim, _, _, patch_size = _get_model_dims()
    spatial_seq, ppf, pph, ppw, adj_frames = _get_spatial_seq_len(num_frames, patch_size)

    temb_bf16_mb = spatial_seq * 6 * inner_dim * 2 / (1024 * 1024)

    print(f"\n--- scale_shift_table NO .float() [{num_frames} frames] ---")
    print(f"  inner_dim={inner_dim}  (from model config)")
    print(f"  num_frames={num_frames} -> adjusted={adj_frames}")
    print(f"  spatial_seq = {ppf}*{pph}*{ppw} = {spatial_seq}")
    print(f"  temb shape:        (1, {spatial_seq}, 6, {inner_dim})  bfloat16")
    print(f"  intermediate:      bfloat16 -> {temb_bf16_mb:.1f} MB")

    op = ScaleShiftNoCast(inner_dim).to(dtype=torch.bfloat16)
    op.compile(backend="tt")
    op = op.to(xm.xla_device())

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    temb = torch.randn(1, spatial_seq, 6, inner_dim, generator=generator, dtype=torch.bfloat16)
    temb_tt = temb.to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        outputs = op(temb_tt)
    elapsed = time.time() - t0

    outputs_cpu = tuple(o.to("cpu") for o in outputs)
    print(f"  output[0] shape:   {outputs_cpu[0].shape}  dtype={outputs_cpu[0].dtype}")
    print(f"  time:              {elapsed:.1f}s")

    assert len(outputs_cpu) == 6
    for i, o in enumerate(outputs_cpu):
        assert o.shape == (1, spatial_seq, inner_dim)
        assert torch.isfinite(o).all(), f"Chunk {i} contains NaN/Inf"

    del op, temb, temb_tt, outputs, outputs_cpu
    print(f"  PASSED ({num_frames} frames, spatial_seq={spatial_seq}, no float cast)")
