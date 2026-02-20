# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CPU-only tests validating correctness of the MLACache implementation against
the reference DynamicCache (full K/V expansion) approach.

No Tenstorrent hardware is required to run these tests.
"""

import pytest
import torch
from transformers import DynamicCache

from .configuration_deepseek import DeepseekV3Config
from .modeling_deepseek import DeepseekV3Attention as MlaDeepseekV3Attention
from .original_modeling_deepseek import DeepseekV3Attention as OrigDeepseekV3Attention
from .utils import MLACache


def _make_small_config() -> DeepseekV3Config:
    """Minimal DeepseekV3Config sized for fast CPU testing."""
    return DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=16,
        v_head_dim=8,
        qk_nope_head_dim=8,
        num_hidden_layers=1,
        max_position_embeddings=64,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
    )


@pytest.mark.push
def test_kimi_k2_mla_cache_kv_parity():
    """
    CPU-only test validating the MLA cache (MLACache) against the original
    expanded cache (DynamicCache) for DeepseekV3Attention.

    The MLA optimisation stores the low-rank compressed KV latent
    (compressed_kv) and the RoPE-encoded decoupled key (k_pe) in the cache
    instead of the full, expanded key and value states.  At attention time the
    full K/V is reconstructed by applying kv_b_proj(kv_a_layernorm(...)) to
    the cached latents.  This is valid because kv_a_layernorm and kv_b_proj
    are both token-independent operations, so applying them to each token at
    write time is identical to applying them to the full cached sequence at
    attention time.

    The test verifies this equivalence by running a prefill followed by a
    single decode step with both cache types and comparing, for the full
    accumulated sequence, the assembled key_states and value_states.
    """
    config = _make_small_config()

    BATCH_SIZE = 2
    PREFILL_LEN = 8
    LAYER_IDX = 0
    # MAX_CACHE_LEN == PREFILL_LEN + 1 so that, during the decode step,
    # MLACache.get_max_cache_shape() equals the DynamicCache's kv_seq_len
    # (1 current token + PREFILL_LEN cached tokens).  This keeps the
    # attention-mask shapes identical between both implementations.
    MAX_CACHE_LEN = PREFILL_LEN + 1

    # Build two attention modules with IDENTICAL weights.
    mla_attn = MlaDeepseekV3Attention(config, layer_idx=LAYER_IDX)
    mla_attn.eval()
    orig_attn = OrigDeepseekV3Attention(config, layer_idx=LAYER_IDX)
    orig_attn.load_state_dict(mla_attn.state_dict())
    orig_attn.eval()

    torch.manual_seed(0)

    # ------------------------------------------------------------------ #
    # Prefill — populate both caches                                      #
    # ------------------------------------------------------------------ #
    mla_cache = MLACache(config, max_cache_len=MAX_CACHE_LEN)
    orig_cache = DynamicCache()

    prefill_hidden = torch.randn(BATCH_SIZE, PREFILL_LEN, config.hidden_size)
    prefill_position_ids = torch.arange(PREFILL_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)

    # New impl: MLACache.get_max_cache_shape() == MAX_CACHE_LEN, so the
    # internal kv_seq_len is MAX_CACHE_LEN.
    mla_prefill_mask = torch.zeros(BATCH_SIZE, 1, PREFILL_LEN, MAX_CACHE_LEN)
    # Original: DynamicCache starts empty, kv_seq_len == PREFILL_LEN.
    orig_prefill_mask = torch.zeros(BATCH_SIZE, 1, PREFILL_LEN, PREFILL_LEN)

    with torch.no_grad():
        mla_attn(
            prefill_hidden,
            mla_prefill_mask,
            prefill_position_ids,
            past_key_value=mla_cache,
            use_cache=True,
            cache_position=torch.arange(PREFILL_LEN),
        )
        orig_attn(
            prefill_hidden,
            orig_prefill_mask,
            prefill_position_ids,
            past_key_value=orig_cache,
            use_cache=True,
        )

    # --- Compare K/V states after prefill ---
    orig_key = orig_cache.layers[LAYER_IDX].keys  # [B, H, PREFILL_LEN, q_head_dim]
    orig_val = orig_cache.layers[LAYER_IDX].values  # [B, H, PREFILL_LEN, v_head_dim]

    compressed_kv = mla_cache.layers[LAYER_IDX].compressed_kv[
        :, 0, :PREFILL_LEN, :
    ]  # [B, PREFILL_LEN, kv_lora_rank]
    mla_k_pe = mla_cache.layers[LAYER_IDX].k_pe[
        :, :, :PREFILL_LEN, :
    ]  # [B, 1, PREFILL_LEN, qk_rope_head_dim]

    with torch.no_grad():
        kv = (
            mla_attn.kv_b_proj(mla_attn.kv_a_layernorm(compressed_kv))
            .view(
                BATCH_SIZE,
                PREFILL_LEN,
                config.num_attention_heads,
                config.qk_nope_head_dim + config.v_head_dim,
            )
            .transpose(1, 2)
        )  # [B, H, PREFILL_LEN, qk_nope_head_dim + v_head_dim]

    mla_k_nope, mla_val = torch.split(
        kv, [config.qk_nope_head_dim, config.v_head_dim], dim=-1
    )
    # Assemble full key_states: [k_nope | k_pe] where k_pe expands from 1 head to all heads.
    mla_key = torch.cat(
        [mla_k_nope, mla_k_pe.expand(-1, config.num_attention_heads, -1, -1)], dim=-1
    )  # [B, H, PREFILL_LEN, q_head_dim]

    torch.testing.assert_close(mla_key, orig_key)
    torch.testing.assert_close(mla_val, orig_val)

    # ------------------------------------------------------------------ #
    # Decode — single new token                                           #
    # This is the critical step: the new impl calls kv_b_proj on the     #
    # entire cached sequence, while the original called it per-token at  #
    # write time.  The reconstructed K/V for all positions must match.   #
    # ------------------------------------------------------------------ #
    decode_hidden = torch.randn(BATCH_SIZE, 1, config.hidden_size)
    decode_position_ids = torch.full((BATCH_SIZE, 1), PREFILL_LEN, dtype=torch.long)

    # After the update both caches hold PREFILL_LEN + 1 tokens:
    #   MLACache.get_max_cache_shape() == MAX_CACHE_LEN == PREFILL_LEN + 1
    #   DynamicCache kv_seq_len == 1 + get_usable_length(1, 0) == 1 + PREFILL_LEN
    # Both are equal, so identical mask shapes can be used.
    decode_mask = torch.zeros(BATCH_SIZE, 1, 1, MAX_CACHE_LEN)

    with torch.no_grad():
        mla_attn(
            decode_hidden,
            decode_mask,
            decode_position_ids,
            past_key_value=mla_cache,
            use_cache=True,
            cache_position=torch.tensor([PREFILL_LEN]),
        )
        orig_attn(
            decode_hidden,
            decode_mask,
            decode_position_ids,
            past_key_value=orig_cache,
            use_cache=True,
        )

    total_len = PREFILL_LEN + 1

    # --- Compare K/V states for the full accumulated sequence ---
    orig_key_full = orig_cache.layers[LAYER_IDX].keys  # [B, H, total_len, q_head_dim]
    orig_val_full = orig_cache.layers[LAYER_IDX].values  # [B, H, total_len, v_head_dim]

    compressed_kv_full = mla_cache.layers[LAYER_IDX].compressed_kv[
        :, 0, :total_len, :
    ]  # [B, total_len, kv_lora_rank]
    mla_k_pe_full = mla_cache.layers[LAYER_IDX].k_pe[
        :, :, :total_len, :
    ]  # [B, 1, total_len, qk_rope_head_dim]

    with torch.no_grad():
        kv_full = (
            mla_attn.kv_b_proj(mla_attn.kv_a_layernorm(compressed_kv_full))
            .view(
                BATCH_SIZE,
                total_len,
                config.num_attention_heads,
                config.qk_nope_head_dim + config.v_head_dim,
            )
            .transpose(1, 2)
        )  # [B, H, total_len, qk_nope_head_dim + v_head_dim]

    mla_k_nope_full, mla_val_full = torch.split(
        kv_full, [config.qk_nope_head_dim, config.v_head_dim], dim=-1
    )
    mla_key_full = torch.cat(
        [mla_k_nope_full, mla_k_pe_full.expand(-1, config.num_attention_heads, -1, -1)],
        dim=-1,
    )  # [B, H, total_len, q_head_dim]

    torch.testing.assert_close(mla_key_full, orig_key_full)
    torch.testing.assert_close(mla_val_full, orig_val_full)
