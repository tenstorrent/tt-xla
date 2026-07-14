# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DSV4 fp8/fp4 -> bf16 vLLM-checkpoint converter.

Runs on a tiny *synthetic* quantized checkpoint (no real weights / no device),
exercising the whole `convert()` pipeline: fp8 + fp4 dequant, scale-key drop,
name preservation, config rewrite, aux-file copy, and n-layers filtering.
"""
import json
import os
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Import the sibling converter script (mirrors test_deepseek_v3_2_exp.py's
# `sys.path.insert(...)` + `from build_weight_cache import ...` pattern).
sys.path.insert(0, os.path.dirname(__file__))
import build_vllm_bf16_checkpoint as build  # noqa: E402


def _make_fp8(out_dim, in_dim, seed):
    """A round-trippable fp8 (weight, scale) pair + its bf16 reference."""
    g = torch.Generator().manual_seed(seed)
    ref = (torch.randn(out_dim, in_dim, generator=g) * 0.05).to(torch.bfloat16)
    nbo, nbi = out_dim // build._FP8_BLOCK, in_dim // build._FP8_BLOCK
    blocks = (
        ref.float()
        .unflatten(0, (nbo, build._FP8_BLOCK))
        .unflatten(-1, (nbi, build._FP8_BLOCK))
    )  # [nbo,128,nbi,128]
    amax = blocks.abs().amax(dim=(1, 3))  # [nbo, nbi]
    scale = (amax / 448.0).clamp(min=1e-6)  # fp8_e4m3 max ~448
    fp8 = (
        (blocks / scale[:, None, :, None])
        .flatten(2, 3)
        .flatten(0, 1)
        .to(torch.float8_e4m3fn)
    )
    return fp8, scale, ref


def _make_fp4(out_dim, in_dim, seed):
    """A packed-fp4 (weight, scale) pair; returns the converter's own dequant
    as the expected result (the packing/table is validated by matching it)."""
    g = torch.Generator().manual_seed(seed)
    nibbles = torch.randint(0, 16, (out_dim, in_dim), generator=g, dtype=torch.uint8)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).contiguous()  # [out, in/2]
    scale = torch.rand(out_dim, in_dim // build._FP4_BLOCK, generator=g) + 0.1
    expected = build._dequant_fp4(packed, scale)
    return packed, scale, expected


def _write_synthetic_ckpt(src):
    os.makedirs(src, exist_ok=True)
    fp8_w, fp8_s, fp8_ref = _make_fp8(128, 256, seed=1)
    fp4_w, fp4_s, fp4_ref = _make_fp4(64, 256, seed=2)  # packed -> [64, 128]
    embed = torch.randn(10, 16).to(torch.bfloat16)
    l1_w, l1_s, _ = _make_fp8(128, 128, seed=3)  # layer 1 -> filtered by n_layers=1

    tensors = {
        # layer 0 fp8 linear (HF fp8 scale suffix)
        "model.layers.0.self_attn.q_a_proj.weight": fp8_w,
        "model.layers.0.self_attn.q_a_proj.weight_scale_inv": fp8_s,
        # layer 0 fp4 MoE expert (native `.scale` suffix)
        "model.layers.0.mlp.experts.0.gate_proj.weight": fp4_w,
        "model.layers.0.mlp.experts.0.gate_proj.scale": fp4_s,
        # passthrough bf16 + a non-float tensor
        "model.embed_tokens.weight": embed,
        "model.layers.0.mlp.gate.some_int_buf": torch.arange(4, dtype=torch.int32),
        # layer 1 fp8 (for n_layers filtering)
        "model.layers.1.self_attn.q_a_proj.weight": l1_w,
        "model.layers.1.self_attn.q_a_proj.weight_scale_inv": l1_s,
    }
    save_file(tensors, os.path.join(src, "model.safetensors"))
    with open(os.path.join(src, "config.json"), "w") as f:
        json.dump(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "torch_dtype": "float8_e4m3fn",
                "quantization_config": {"quant_method": "deepseek_v4_fp8"},
            },
            f,
        )
    with open(os.path.join(src, "tokenizer.json"), "w") as f:
        f.write('{"fake": "tokenizer"}')
    return fp8_w, fp8_s, fp8_ref, fp4_w, fp4_s, fp4_ref, embed


def _load_out(dst):
    with safe_open(
        os.path.join(dst, "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def test_convert_dequants_and_preserves_names(tmp_path):
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    fp8_w, fp8_s, fp8_ref, fp4_w, fp4_s, fp4_ref, embed = _write_synthetic_ckpt(src)

    build.convert(src, dst)
    out = _load_out(dst)

    # Names preserved; scale keys dropped.
    assert "model.layers.0.self_attn.q_a_proj.weight" in out
    assert "model.layers.0.self_attn.q_a_proj.weight_scale_inv" not in out
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in out
    assert "model.layers.0.mlp.experts.0.gate_proj.scale" not in out

    # fp8 weight dequantized to bf16 and matches the helper + the reference.
    w8 = out["model.layers.0.self_attn.q_a_proj.weight"]
    assert w8.dtype == torch.bfloat16 and tuple(w8.shape) == (128, 256)
    assert torch.equal(w8, build._dequant_fp8(fp8_w, fp8_s))
    # round-trip: within fp8 quantization error of the original bf16.
    rel = (w8.float() - fp8_ref.float()).abs().mean() / fp8_ref.float().abs().mean()
    assert rel < 0.1, f"fp8 round-trip rel error too high: {rel}"

    # fp4 expert dequantized to bf16, unpacked to full in-dim.
    w4 = out["model.layers.0.mlp.experts.0.gate_proj.weight"]
    assert w4.dtype == torch.bfloat16 and tuple(w4.shape) == (64, 256)
    assert torch.equal(w4, fp4_ref)

    # Passthroughs: bf16 kept, int kept as-is.
    assert torch.equal(out["model.embed_tokens.weight"], embed)
    assert out["model.layers.0.mlp.gate.some_int_buf"].dtype == torch.int32


def test_convert_rewrites_config_and_copies_aux(tmp_path):
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    _write_synthetic_ckpt(src)
    build.convert(src, dst)

    with open(os.path.join(dst, "config.json")) as f:
        cfg = json.load(f)
    assert "quantization_config" not in cfg
    assert cfg["torch_dtype"] == "bfloat16"
    assert cfg["architectures"] == ["DeepseekV4ForCausalLM"]

    # tokenizer copied through.
    assert os.path.exists(os.path.join(dst, "tokenizer.json"))
    # single-shard output has no index file.
    assert not os.path.exists(os.path.join(dst, "model.safetensors.index.json"))


def test_convert_n_layers_filter(tmp_path):
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    _write_synthetic_ckpt(src)
    build.convert(src, dst, n_layers=1)
    out = _load_out(dst)

    # Only layer 0 kept; layer 1 dropped.
    assert "model.layers.0.self_attn.q_a_proj.weight" in out
    assert "model.layers.1.self_attn.q_a_proj.weight" not in out
    # Non-layer passthroughs survive.
    assert "model.embed_tokens.weight" in out
