#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone test for Attention layer prefill + double decode.
No CPU comparison - just run on TT device and print KV cache states.
"""

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from third_party.tt_forge_models.deepseek_v4.modified_model.model import (
    ModelArgs,
    RMSNorm,
    Transformer,
)


def small_args(**overrides) -> ModelArgs:
    """Small ModelArgs that fit comfortably on a single device."""
    defaults = dict(
        max_batch_size=2,
        max_seq_len=32,
        vocab_size=256,
        dim=128,
        moe_inter_dim=64,
        n_layers=3,
        n_mtp_layers=0,
        n_heads=4,
        q_lora_rank=64,
        head_dim=32,
        rope_head_dim=16,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        o_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(0, 4, 8),
        index_n_heads=4,
        index_head_dim=16,
        index_topk=4,
        hc_mult=2,
        hc_sinkhorn_iters=3,
    )
    defaults.update(overrides)
    return ModelArgs(**defaults)


def make_model(args: ModelArgs) -> Transformer:
    """Create model in eval mode."""
    return Transformer(args).eval()


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """Initialize all uninitialized parameters with a normal distribution."""
    with torch.no_grad():
        for sub in module.modules():
            if isinstance(sub, RMSNorm):
                continue
            for _, param in sub.named_parameters(recurse=False):
                torch.nn.init.normal_(param, mean=0.0, std=std)


class _AttentionPrefillThenDoubleDecode(nn.Module):
    """Wrapper that runs prefill, then two decode steps."""

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn

    def forward(
        self, prefill_x: torch.Tensor, decode_x1: torch.Tensor, decode_x2: torch.Tensor
    ):
        # Run prefill to populate KV caches
        print("\n" + "=" * 80)
        print("PREFILL")
        print("=" * 80)
        prefill_out = self.attn(prefill_x, 0)
        torch_xla.sync()

        print(f"\nPrefill output shape: {prefill_out.shape}")
        print(f"Prefill output:\n{prefill_out}")

        print(f"\nKV cache after prefill:")
        print(f"Shape: {self.attn.kv_cache.shape}")
        # print(self.attn.kv_cache)

        # Run first decode step
        print("\n" + "=" * 80)
        print("DECODE 1")
        print("=" * 80)
        decode_out1 = self.attn(decode_x1, prefill_x.shape[1])
        torch_xla.sync()

        print(f"\nDecode 1 output shape: {decode_out1.shape}")
        print(f"Decode 1 output:\n{decode_out1}")

        print(f"\nKV cache after decode 1:")
        print(f"Shape: {self.attn.kv_cache.shape}")
        print(self.attn.kv_cache)

        # Run second decode step
        print("\n" + "=" * 80)
        print("DECODE 2")
        print("=" * 80)
        decode_out2 = self.attn(decode_x2, prefill_x.shape[1] + 1)
        torch_xla.sync()

        print(f"\nDecode 2 output shape: {decode_out2.shape}")
        print(f"Decode 2 output:\n{decode_out2}")

        print(f"\nKV cache after decode 2:")
        print(f"Shape: {self.attn.kv_cache.shape}")
        print(self.attn.kv_cache)

        return decode_out2


def test_swa_attention():
    """Test SWA (Sliding Window Attention) - layer 0."""
    print("\n" + "=" * 80)
    print("Testing SWA Attention (Layer 0)")
    print("=" * 80)

    xr.set_device_type("TT")
    device = xm.xla_device()

    # Create model
    args = small_args()
    model = make_model(args)

    # Get SWA attention (layer 0)
    attn = model.layers[0].attn.to(device)
    init_weights(attn)

    # Create inputs
    bsz = 2
    prefill_seqlen = 4

    torch.manual_seed(42)
    prefill_x = torch.randn(bsz, prefill_seqlen, args.dim, dtype=torch.bfloat16).to(
        device
    )
    decode_x1 = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16).to(device)
    decode_x2 = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16).to(device)

    print(f"\nInputs:")
    print(f"  Prefill: shape={prefill_x.shape}")
    print(f"  Decode 1: shape={decode_x1.shape}")
    print(f"  Decode 2: shape={decode_x2.shape}")
    print(f"\nModel config:")
    print(f"  dim={args.dim}, n_heads={args.n_heads}, head_dim={args.head_dim}")
    print(f"  window_size={args.window_size}")

    # Run test
    wrapper = _AttentionPrefillThenDoubleDecode(attn)

    # Compile with torch.compile for TT backend
    compiled_wrapper = torch.compile(wrapper, backend="tt")

    result = compiled_wrapper(prefill_x, decode_x1, decode_x2)

    print("\n" + "=" * 80)
    print("Test complete")
    print("=" * 80)


if __name__ == "__main__":
    test_swa_attention()
