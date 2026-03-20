# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Micro-tests for WanTransformerBlock sub-operations on TT.

A single WanTransformerBlock OOMs on TT even though a full block is
~700-800 MB of weights.  The OOM is triggered by the combined memory
of intermediate tensors (especially the temb float32 upcast) plus the
block weights.

These tests isolate individual stages of the block forward to
pinpoint whether the OOM comes from a single operation or from
collective memory pressure.

Tests (ordered by memory footprint):
  1. test_norm1_alone            – FP32LayerNorm only
  2. test_norm1_with_scale_shift – norm + (norm * (1+scale) + shift)
  3. test_temb_processing        – scale_shift_table + temb.float() + chunk
  4. test_temb_plus_norm         – temb processing + norm + scale/shift

Memory estimates for seq_len=32760, inner_dim=5120:
  hidden_states bf16:  335 MB      hidden_states fp32:  671 MB
  temb bf16:          2010 MB      temb fp32:          4020 MB
  scale_shift_table:    ~0 MB      6 x scale/shift:    4020 MB (views)
  TT DRAM total:     ~12.8 GB

Usage:
    cd /proj_sw/user_dev/akannan_new/19_mar_bgd/tt-xla
    pytest examples/pytorch/test_transformer_block_sanities.py -v -s
    pytest examples/pytorch/test_transformer_block_sanities.py -v -s -k norm1_alone
    pytest examples/pytorch/test_transformer_block_sanities.py -v -s -k temb_plus_norm
"""

import time
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import WanTransformer3DModel
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from wan_t2v_tt_pipeline import TTWanAttnProcessor

# ── Shared configuration ──────────────────────────────────────────────
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
SEED = 42

HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 81
VAE_SCALE_TEMPORAL = 4
VAE_SCALE_SPATIAL = 8


def _get_block_config():
    """Load transformer config and compute derived dimensions."""
    config = WanTransformer3DModel.load_config(MODEL_ID, subfolder="transformer")

    num_heads = config.get("num_attention_heads", 40)
    head_dim = config.get("attention_head_dim", 128)
    inner_dim = num_heads * head_dim
    patch_size = tuple(config.get("patch_size", (1, 2, 2)))
    eps = config.get("eps", 1e-6)

    num_frames = NUM_FRAMES
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
    seq_len = ppf * pph * ppw

    return {
        "inner_dim": inner_dim,
        "seq_len": seq_len,
        "eps": eps,
        "ppf": ppf,
        "pph": pph,
        "ppw": ppw,
    }


# ── Standalone modules ────────────────────────────────────────────────

class Norm1Only(nn.Module):
    """FP32LayerNorm: bf16 input -> float() -> layer_norm -> return fp32."""

    def __init__(self, dim, eps):
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, hidden_states):
        return self.norm1(hidden_states.float())


class Norm1WithScaleShift(nn.Module):
    """FP32LayerNorm + scale/shift arithmetic.

    Replicates:
        (self.norm1(hs.float()) * (1 + scale_msa) + shift_msa).type_as(hs)
    """

    def __init__(self, dim, eps):
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, hidden_states, scale_msa, shift_msa):
        return (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)


class TembProcessing(nn.Module):
    """scale_shift_table + temb.float() -> chunk -> squeeze.

    Replicates the temb decomposition at the start of
    WanTransformerBlock.forward for the TI2V path (temb.ndim == 4).
    """

    def __init__(self, dim):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, temb):
        parts = (self.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
        return tuple(p.squeeze(2) for p in parts)


class TembPlusNorm(nn.Module):
    """Temb processing + norm1 + scale/shift in a single compiled graph.

    This is the combined operation that OOMs in the full block:
      1. temb -> float32 -> add scale_shift_table -> chunk -> squeeze
      2. norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
    """

    def __init__(self, dim, eps):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, hidden_states, temb):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)

        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        return norm_hidden_states


# ── Setup ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def tt_setup():
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})


# ── Tests ─────────────────────────────────────────────────────────────

def _mb(t):
    """Return size of a tensor in MB."""
    return t.nelement() * t.element_size() / 1024**2


def test_norm1_alone(tt_setup):
    """FP32LayerNorm alone on full-size hidden_states.

    Expected memory on TT:
      input  (bf16):  ~335 MB
      float cast:     ~671 MB
      norm output:    ~671 MB
      total:         ~1.7 GB  — should PASS
    """
    cfg = _get_block_config()
    seq_len, dim = cfg["seq_len"], cfg["inner_dim"]
    print(f"\n--- norm1 alone: (1, {seq_len}, {dim}) ---")

    module = Norm1Only(dim, cfg["eps"])
    module.compile(backend="tt")
    module = module.to(xm.xla_device())

    hs = torch.randn(1, seq_len, dim, dtype=torch.bfloat16)
    print(f"  input:  {hs.shape}  {hs.dtype}  {_mb(hs):.0f} MB")

    hs_tt = hs.to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        out = module(hs_tt)
    elapsed = time.time() - t0

    out_cpu = out.to("cpu")
    print(f"  output: {out_cpu.shape}  {out_cpu.dtype}  time={elapsed:.1f}s")
    assert out_cpu.shape == (1, seq_len, dim)
    assert torch.isfinite(out_cpu).all()
    print("  PASSED")


def test_norm1_with_scale_shift(tt_setup):
    """norm1 + (norm * (1 + scale) + shift).type_as(hs).

    Expected memory on TT:
      hs (bf16):      ~335 MB
      scale (fp32):   ~671 MB
      shift (fp32):   ~671 MB
      intermediates:  ~2.7 GB  (float cast, norm, multiply, add)
      total:         ~4.4 GB  — should PASS
    """
    cfg = _get_block_config()
    seq_len, dim = cfg["seq_len"], cfg["inner_dim"]
    print(f"\n--- norm1 + scale/shift: (1, {seq_len}, {dim}) ---")

    module = Norm1WithScaleShift(dim, cfg["eps"])
    module.compile(backend="tt")
    module = module.to(xm.xla_device())

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    hs = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)
    scale = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.float32)
    shift = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.float32)

    print(f"  hs:    {hs.shape}  {hs.dtype}  {_mb(hs):.0f} MB")
    print(f"  scale: {scale.shape}  {scale.dtype}  {_mb(scale):.0f} MB")
    print(f"  shift: {shift.shape}  {shift.dtype}  {_mb(shift):.0f} MB")

    hs_tt = hs.to(xm.xla_device())
    scale_tt = scale.to(xm.xla_device())
    shift_tt = shift.to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        out = module(hs_tt, scale_tt, shift_tt)
    elapsed = time.time() - t0

    out_cpu = out.to("cpu")
    print(f"  output: {out_cpu.shape}  {out_cpu.dtype}  time={elapsed:.1f}s")
    assert out_cpu.shape == (1, seq_len, dim)
    assert out_cpu.dtype == torch.bfloat16
    assert torch.isfinite(out_cpu).all()
    print("  PASSED")


def test_temb_processing(tt_setup):
    """temb -> float32 -> add scale_shift_table -> chunk -> squeeze.

    Expected memory on TT:
      temb (bf16):    ~2010 MB
      temb.float():   ~4020 MB
      addition:       ~4020 MB  (may alias after chunk)
      total:         ~10 GB   — tight, may or may not PASS
    """
    cfg = _get_block_config()
    seq_len, dim = cfg["seq_len"], cfg["inner_dim"]
    print(f"\n--- temb processing: temb (1, {seq_len}, 6, {dim}) ---")

    module = TembProcessing(dim)
    module.compile(backend="tt")
    module = module.to(xm.xla_device())

    temb = torch.randn(1, seq_len, 6, dim, dtype=torch.bfloat16)
    print(f"  temb: {temb.shape}  {temb.dtype}  {_mb(temb):.0f} MB")

    temb_tt = temb.to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        outputs = module(temb_tt)
    elapsed = time.time() - t0

    for i, o in enumerate(outputs):
        o_cpu = o.to("cpu")
        if i == 0:
            print(f"  output[0]: {o_cpu.shape}  {o_cpu.dtype}  (x6 total)")
        assert o_cpu.shape == (1, seq_len, dim)
    print(f"  time={elapsed:.1f}s  PASSED")


def test_temb_plus_norm(tt_setup):
    """Combined temb processing + norm1 + scale/shift in one graph.

    This mirrors the operation sequence that OOMs in the full block.

    Expected memory on TT:
      hs (bf16):      ~335 MB
      temb (bf16):    ~2010 MB
      temb.float():   ~4020 MB
      addition:       ~4020 MB
      hs.float():     ~671 MB
      norm + arith:   ~2000 MB
      total:         ~13 GB   — likely OOM
    """
    cfg = _get_block_config()
    seq_len, dim = cfg["seq_len"], cfg["inner_dim"]
    print(f"\n--- temb + norm1 combined ---")
    print(f"  hs: (1, {seq_len}, {dim})  temb: (1, {seq_len}, 6, {dim})")

    module = TembPlusNorm(dim, cfg["eps"])
    module.compile(backend="tt")
    module = module.to(xm.xla_device())

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    hs = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)
    temb = torch.randn(1, seq_len, 6, dim, generator=gen, dtype=torch.bfloat16)

    print(f"  hs:   {hs.shape}  {hs.dtype}  {_mb(hs):.0f} MB")
    print(f"  temb: {temb.shape}  {temb.dtype}  {_mb(temb):.0f} MB")

    hs_tt = hs.to(xm.xla_device())
    temb_tt = temb.to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        out = module(hs_tt, temb_tt)
    elapsed = time.time() - t0

    out_cpu = out.to("cpu")
    print(f"  output: {out_cpu.shape}  {out_cpu.dtype}  time={elapsed:.1f}s")
    assert out_cpu.shape == (1, seq_len, dim)
    assert out_cpu.dtype == torch.bfloat16
    assert torch.isfinite(out_cpu).all()
    print("  PASSED")


# ======================================================================
# Test 5: Full block on TT with temb processing moved to CPU
# ======================================================================

def _forward_with_precomputed_temb(
    self, hidden_states, encoder_hidden_states,
    shift_msa, scale_msa, gate_msa,
    c_shift_msa, c_scale_msa, c_gate_msa,
    rotary_emb,
):
    """Replacement forward that takes pre-computed modulation tensors.

    The original forward computes scale_shift_table + temb.float() on
    device, which OOMs because the temb float32 typecast alone needs
    ~12.9 GB on TT.  This version receives the 6 modulation tensors
    already computed on CPU.
    """
    # 1. Self-attention
    norm_hidden_states = (
        self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
    ).type_as(hidden_states)
    attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
    hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

    # 2. Cross-attention
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output

    # 3. Feed-forward
    norm_hidden_states = (
        self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
    ).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

    return hidden_states


def _precompute_temb_on_cpu(scale_shift_table, temb):
    """Compute temb decomposition on CPU, return 6 bfloat16 tensors.

    Args:
        scale_shift_table: (1, 6, dim) float32, on CPU
        temb: (1, seq_len, 6, dim) bfloat16, on CPU

    Returns:
        tuple of 6 tensors each (1, seq_len, dim) bfloat16
        Order: shift_msa, scale_msa, gate_msa,
               c_shift_msa, c_scale_msa, c_gate_msa
    """
    combined = scale_shift_table.unsqueeze(0) + temb.float()
    parts = combined.chunk(6, dim=2)
    return tuple(p.squeeze(2).to(torch.bfloat16) for p in parts)


# ======================================================================
# Test 5a: Self-Attention stage on TT
# ======================================================================

class _SelfAttnStage(nn.Module):
    """norm1 + self-attention + gate residual — the self-attn portion of
    WanTransformerBlock with precomputed temb modulations."""

    def __init__(self, norm1, attn1):
        super().__init__()
        self.norm1 = norm1
        self.attn1 = attn1

    def forward(self, hidden_states, shift_msa, scale_msa, gate_msa, rotary_emb):
        norm_hs = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(norm_hs, None, None, rotary_emb)
        return (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)


def _run_self_attention_test(seq_len, dim, num_heads, head_dim, added_kv_proj_dim, config, label):
    """Shared self-attention test logic used by both small and full-size variants."""
    print(f"\n--- Self-Attention on TT ({label}) ---")
    print(f"  dim={dim}  num_heads={num_heads}  head_dim={head_dim}  seq_len={seq_len}")

    t0 = time.time()
    block = WanTransformerBlock(
        dim=dim, ffn_dim=config.get("ffn_dim", 13824), num_heads=num_heads,
        qk_norm=config.get("qk_norm", "rms_norm_across_heads"),
        cross_attn_norm=config.get("cross_attn_norm", True),
        eps=config.get("eps", 1e-6), added_kv_proj_dim=added_kv_proj_dim,
    ).to(dtype=torch.bfloat16)

    stage = _SelfAttnStage(block.norm1, block.attn1)
    stage.attn1.processor = TTWanAttnProcessor()

    stage.compile(backend="tt")
    stage = stage.to(xm.xla_device())
    print(f"  Compiled in {time.time() - t0:.1f}s")

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    hs = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)
    shift = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)
    scale = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)
    gate = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)
    rope = (
        torch.randn(1, seq_len, 1, head_dim, generator=gen, dtype=torch.bfloat16),
        torch.randn(1, seq_len, 1, head_dim, generator=gen, dtype=torch.bfloat16),
    )

    print(f"  hs:   {hs.shape}  {_mb(hs):.0f} MB")
    print(f"  mod:  {shift.shape}  {_mb(shift):.0f} MB each (x3)")
    print(f"  rope: {rope[0].shape}  {_mb(rope[0]):.0f} MB each (x2)")

    hs_tt = hs.to(xm.xla_device())
    shift_tt = shift.to(xm.xla_device())
    scale_tt = scale.to(xm.xla_device())
    gate_tt = gate.to(xm.xla_device())
    rope_tt = (rope[0].to(xm.xla_device()), rope[1].to(xm.xla_device()))

    t0 = time.time()
    with torch.no_grad():
        out = stage(hs_tt, shift_tt, scale_tt, gate_tt, rope_tt)
    elapsed = time.time() - t0

    out_cpu = out.to("cpu")
    print(f"  output: {out_cpu.shape}  {out_cpu.dtype}  time={elapsed:.1f}s")
    assert out_cpu.shape == (1, seq_len, dim)
    assert torch.isfinite(out_cpu).all()
    print("  PASSED")


def test_self_attention_on_tt_small(tt_setup):
    """Self-attention stage on TT with small inputs (128x128, 9 frames).

    seq_len=192 — compiles quickly on TT.
    """
    config = WanTransformer3DModel.load_config(MODEL_ID, subfolder="transformer")
    num_heads = config.get("num_attention_heads", 40)
    head_dim = config.get("attention_head_dim", 128)
    dim = num_heads * head_dim
    added_kv_proj_dim = config.get("added_kv_proj_dim")
    patch_size = tuple(config.get("patch_size", (1, 2, 2)))

    small_h, small_w, small_nf = 128, 128, 9
    nf = small_nf
    if nf % VAE_SCALE_TEMPORAL != 1:
        nf = nf // VAE_SCALE_TEMPORAL * VAE_SCALE_TEMPORAL + 1
    nf = max(nf, 1)
    h = small_h // (VAE_SCALE_SPATIAL * patch_size[1]) * (VAE_SCALE_SPATIAL * patch_size[1])
    w = small_w // (VAE_SCALE_SPATIAL * patch_size[2]) * (VAE_SCALE_SPATIAL * patch_size[2])
    ppf = ((nf - 1) // VAE_SCALE_TEMPORAL + 1)
    pph = (h // VAE_SCALE_SPATIAL) // patch_size[1]
    ppw = (w // VAE_SCALE_SPATIAL) // patch_size[2]
    seq_len = ppf * pph * ppw

    _run_self_attention_test(
        seq_len, dim, num_heads, head_dim, added_kv_proj_dim, config,
        label=f"small {small_h}x{small_w}, {small_nf}f, seq_len={seq_len}",
    )


def test_self_attention_on_tt_full(tt_setup):
    """Self-attention stage on TT with actual full-size inputs (480x832, 81 frames).

    seq_len=32760 — may hang during MLIR compilation for a long time.
    """
    cfg = _get_block_config()
    seq_len, dim = cfg["seq_len"], cfg["inner_dim"]
    config = WanTransformer3DModel.load_config(MODEL_ID, subfolder="transformer")
    num_heads = config.get("num_attention_heads", 40)
    head_dim = config.get("attention_head_dim", 128)
    added_kv_proj_dim = config.get("added_kv_proj_dim")

    _run_self_attention_test(
        seq_len, dim, num_heads, head_dim, added_kv_proj_dim, config,
        label=f"full {HEIGHT}x{WIDTH}, {NUM_FRAMES}f, seq_len={seq_len}",
    )


# ======================================================================
# Test 5b: Cross-Attention stage on TT
# ======================================================================

class _CrossAttnStage(nn.Module):
    """norm2 + cross-attention + residual — the cross-attn portion of
    WanTransformerBlock."""

    def __init__(self, norm2, attn2):
        super().__init__()
        self.norm2 = norm2
        self.attn2 = attn2

    def forward(self, hidden_states, encoder_hidden_states):
        norm_hs = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hs, encoder_hidden_states, None, None)
        return hidden_states + attn_output


def test_cross_attention_on_tt(tt_setup):
    """Cross-attention stage of WanTransformerBlock on TT.

    Isolates: norm2 -> attn2(cross, no rotary) -> residual add.
    Uses small inputs for fast compilation (same as self-attn test).
    """
    config = WanTransformer3DModel.load_config(MODEL_ID, subfolder="transformer")

    num_heads = config.get("num_attention_heads", 40)
    head_dim = config.get("attention_head_dim", 128)
    dim = num_heads * head_dim
    added_kv_proj_dim = config.get("added_kv_proj_dim")
    patch_size = tuple(config.get("patch_size", (1, 2, 2)))
    enc_seq_len = (257 + 512) if added_kv_proj_dim is not None else 512

    # --- Small inputs for fast compilation ---
    small_h, small_w, small_nf = 128, 128, 9
    nf = small_nf
    if nf % VAE_SCALE_TEMPORAL != 1:
        nf = nf // VAE_SCALE_TEMPORAL * VAE_SCALE_TEMPORAL + 1
    nf = max(nf, 1)
    h = small_h // (VAE_SCALE_SPATIAL * patch_size[1]) * (VAE_SCALE_SPATIAL * patch_size[1])
    w = small_w // (VAE_SCALE_SPATIAL * patch_size[2]) * (VAE_SCALE_SPATIAL * patch_size[2])
    ppf = ((nf - 1) // VAE_SCALE_TEMPORAL + 1)
    pph = (h // VAE_SCALE_SPATIAL) // patch_size[1]
    ppw = (w // VAE_SCALE_SPATIAL) // patch_size[2]
    seq_len = ppf * pph * ppw

    # # --- Full-size inputs (hangs during TT compilation) ---
    # cfg = _get_block_config()
    # seq_len, dim = cfg["seq_len"], cfg["inner_dim"]

    print(f"\n--- Cross-Attention on TT ---")
    print(f"  dim={dim}  num_heads={num_heads}  seq_len={seq_len}  enc_seq_len={enc_seq_len}")

    t0 = time.time()
    block = WanTransformerBlock(
        dim=dim, ffn_dim=config.get("ffn_dim", 13824), num_heads=num_heads,
        qk_norm=config.get("qk_norm", "rms_norm_across_heads"),
        cross_attn_norm=config.get("cross_attn_norm", True),
        eps=config.get("eps", 1e-6), added_kv_proj_dim=added_kv_proj_dim,
    ).to(dtype=torch.bfloat16)

    stage = _CrossAttnStage(block.norm2, block.attn2)
    stage.attn2.processor = TTWanAttnProcessor()

    stage.compile(backend="tt")
    stage = stage.to(xm.xla_device())
    print(f"  Compiled in {time.time() - t0:.1f}s")

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    hs = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)
    enc = torch.randn(1, enc_seq_len, dim, generator=gen, dtype=torch.bfloat16)

    print(f"  hs:  {hs.shape}  {_mb(hs):.0f} MB")
    print(f"  enc: {enc.shape}  {_mb(enc):.0f} MB")

    hs_tt = hs.to(xm.xla_device())
    enc_tt = enc.to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        out = stage(hs_tt, enc_tt)
    elapsed = time.time() - t0

    out_cpu = out.to("cpu")
    print(f"  output: {out_cpu.shape}  {out_cpu.dtype}  time={elapsed:.1f}s")
    assert out_cpu.shape == (1, seq_len, dim)
    assert torch.isfinite(out_cpu).all()
    print("  PASSED")


# ======================================================================
# Test 5c: Feed-Forward stage on TT
# ======================================================================

class _FFNStage(nn.Module):
    """norm3 + scale/shift + FFN + gate residual — the feed-forward portion
    of WanTransformerBlock with precomputed temb modulations."""

    def __init__(self, norm3, ffn):
        super().__init__()
        self.norm3 = norm3
        self.ffn = ffn

    def forward(self, hidden_states, c_shift_msa, c_scale_msa, c_gate_msa):
        norm_hs = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hs)
        return (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)


def test_ffn_on_tt(tt_setup):
    """Feed-forward stage of WanTransformerBlock on TT.

    Isolates: norm3 -> scale/shift -> FFN(GELU+Linear) -> gate residual.
    No attention, no rotary — just the MLP path.
    Uses small inputs for fast compilation.
    """
    config = WanTransformer3DModel.load_config(MODEL_ID, subfolder="transformer")

    num_heads = config.get("num_attention_heads", 40)
    head_dim = config.get("attention_head_dim", 128)
    dim = num_heads * head_dim
    added_kv_proj_dim = config.get("added_kv_proj_dim")
    ffn_dim = config.get("ffn_dim", 13824)
    patch_size = tuple(config.get("patch_size", (1, 2, 2)))

    # --- Small inputs for fast compilation ---
    small_h, small_w, small_nf = 128, 128, 9
    nf = small_nf
    if nf % VAE_SCALE_TEMPORAL != 1:
        nf = nf // VAE_SCALE_TEMPORAL * VAE_SCALE_TEMPORAL + 1
    nf = max(nf, 1)
    h = small_h // (VAE_SCALE_SPATIAL * patch_size[1]) * (VAE_SCALE_SPATIAL * patch_size[1])
    w = small_w // (VAE_SCALE_SPATIAL * patch_size[2]) * (VAE_SCALE_SPATIAL * patch_size[2])
    ppf = ((nf - 1) // VAE_SCALE_TEMPORAL + 1)
    pph = (h // VAE_SCALE_SPATIAL) // patch_size[1]
    ppw = (w // VAE_SCALE_SPATIAL) // patch_size[2]
    seq_len = ppf * pph * ppw

    # # --- Full-size inputs (hangs during TT compilation) ---
    # cfg = _get_block_config()
    # seq_len, dim = cfg["seq_len"], cfg["inner_dim"]

    print(f"\n--- Feed-Forward on TT ---")
    print(f"  dim={dim}  ffn_dim={ffn_dim}  seq_len={seq_len}")

    t0 = time.time()
    block = WanTransformerBlock(
        dim=dim, ffn_dim=ffn_dim, num_heads=num_heads,
        qk_norm=config.get("qk_norm", "rms_norm_across_heads"),
        cross_attn_norm=config.get("cross_attn_norm", True),
        eps=config.get("eps", 1e-6), added_kv_proj_dim=added_kv_proj_dim,
    ).to(dtype=torch.bfloat16)

    stage = _FFNStage(block.norm3, block.ffn)

    stage.compile(backend="tt")
    stage = stage.to(xm.xla_device())
    print(f"  Compiled in {time.time() - t0:.1f}s")

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    hs = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)
    c_shift = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)
    c_scale = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)
    c_gate = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)

    print(f"  hs:   {hs.shape}  {_mb(hs):.0f} MB")
    print(f"  mod:  {c_shift.shape}  {_mb(c_shift):.0f} MB each (x3)")

    hs_tt = hs.to(xm.xla_device())
    c_shift_tt = c_shift.to(xm.xla_device())
    c_scale_tt = c_scale.to(xm.xla_device())
    c_gate_tt = c_gate.to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        out = stage(hs_tt, c_shift_tt, c_scale_tt, c_gate_tt)
    elapsed = time.time() - t0

    out_cpu = out.to("cpu")
    print(f"  output: {out_cpu.shape}  {out_cpu.dtype}  time={elapsed:.1f}s")
    assert out_cpu.shape == (1, seq_len, dim)
    assert torch.isfinite(out_cpu).all()
    print("  PASSED")


# ======================================================================
# Test 5d: Full block on TT with temb processing moved to CPU
# ======================================================================

def test_block_with_cpu_temb(tt_setup):
    """Full WanTransformerBlock on TT with temb processing on CPU.

    The scale_shift_table + temb.float() decomposition is pure tensor
    math (typecast, add, chunk) — not NN ops.  By pre-computing the
    6 modulation tensors on CPU and sending them individually to TT,
    we avoid the 12.9 GB temb.float() typecast allocation on device.

    Each modulation tensor is (1, seq_len, dim) bfloat16 = ~192 MB,
    so 6 x 192 = ~1.15 GB total — much less than the original
    (1, seq_len, 6, dim) float32 = ~2.3 GB + 12.9 GB typecast overhead.
    """
    cfg = _get_block_config()
    seq_len, dim = cfg["seq_len"], cfg["inner_dim"]
    config = WanTransformer3DModel.load_config(MODEL_ID, subfolder="transformer")

    num_heads = config.get("num_attention_heads", 40)
    head_dim = config.get("attention_head_dim", 128)
    added_kv_proj_dim = config.get("added_kv_proj_dim")

    print(f"\n--- Full block on TT (temb on CPU) ---")
    print(f"  inner_dim={dim}  num_heads={num_heads}  head_dim={head_dim}")
    print(f"  seq_len={seq_len}  added_kv_proj_dim={added_kv_proj_dim}")

    t0 = time.time()
    block = WanTransformerBlock(
        dim=dim,
        ffn_dim=config.get("ffn_dim", 13824),
        num_heads=num_heads,
        qk_norm=config.get("qk_norm", "rms_norm_across_heads"),
        cross_attn_norm=config.get("cross_attn_norm", True),
        eps=config.get("eps", 1e-6),
        added_kv_proj_dim=added_kv_proj_dim,
    ).to(dtype=torch.bfloat16)

    # Keep scale_shift_table on CPU for temb decomposition
    sst_cpu = block.scale_shift_table.data.clone().float()

    # Patch attention processors for TT
    block.attn1.processor = TTWanAttnProcessor()
    block.attn2.processor = TTWanAttnProcessor()

    # Replace forward to accept pre-computed modulations
    block.forward = types.MethodType(_forward_with_precomputed_temb, block)

    block.compile(backend="tt")
    block = block.to(xm.xla_device())
    print(f"  Block created + compiled in {time.time() - t0:.1f}s")

    # --- Create inputs on CPU ---
    gen = torch.Generator(device="cpu").manual_seed(SEED)

    hidden_states = torch.randn(1, seq_len, dim, generator=gen, dtype=torch.bfloat16)

    enc_seq_len = (257 + 512) if added_kv_proj_dim is not None else 512
    encoder_hidden_states = torch.randn(1, enc_seq_len, dim, generator=gen, dtype=torch.bfloat16)

    temb = torch.randn(1, seq_len, 6, dim, generator=gen, dtype=torch.bfloat16)

    rotary_emb = (
        torch.randn(1, seq_len, 1, head_dim, generator=gen, dtype=torch.bfloat16),
        torch.randn(1, seq_len, 1, head_dim, generator=gen, dtype=torch.bfloat16),
    )

    # --- Compute temb decomposition on CPU ---
    t0_temb = time.time()
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        _precompute_temb_on_cpu(sst_cpu, temb)
    )
    temb_time = time.time() - t0_temb

    print(f"  temb decomposition on CPU: {temb_time:.3f}s")
    print(f"  hidden_states:         {hidden_states.shape}   {_mb(hidden_states):.0f} MB")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"  each modulation:       {shift_msa.shape}   {_mb(shift_msa):.0f} MB  (x6 = {6*_mb(shift_msa):.0f} MB)")
    print(f"  rotary_emb[0]:         {rotary_emb[0].shape}   {_mb(rotary_emb[0]):.0f} MB")

    # --- Move to TT ---
    hs_tt = hidden_states.to(xm.xla_device())
    enc_tt = encoder_hidden_states.to(xm.xla_device())
    shift_tt = shift_msa.to(xm.xla_device())
    scale_tt = scale_msa.to(xm.xla_device())
    gate_tt = gate_msa.to(xm.xla_device())
    c_shift_tt = c_shift_msa.to(xm.xla_device())
    c_scale_tt = c_scale_msa.to(xm.xla_device())
    c_gate_tt = c_gate_msa.to(xm.xla_device())
    rope_tt = (rotary_emb[0].to(xm.xla_device()), rotary_emb[1].to(xm.xla_device()))

    # --- Forward on TT ---
    t0 = time.time()
    with torch.no_grad():
        output = block(
            hs_tt, enc_tt,
            shift_tt, scale_tt, gate_tt,
            c_shift_tt, c_scale_tt, c_gate_tt,
            rope_tt,
        )
    elapsed = time.time() - t0

    output_cpu = output.to("cpu")
    print(f"  output: {output_cpu.shape}  {output_cpu.dtype}  time={elapsed:.1f}s")

    assert output_cpu.shape == (1, seq_len, dim), (
        f"Expected (1, {seq_len}, {dim}), got {output_cpu.shape}"
    )
    assert torch.isfinite(output_cpu).all(), "Output contains NaN/Inf"
    print("  PASSED")
