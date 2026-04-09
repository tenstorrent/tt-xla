# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Text Connectors — standalone bringup script.

Component: LTX2TextConnectors
Memory: 2.67 GiB (bf16)
Sharding: Replicated (single device) — fits comfortably on one p150 (32 GiB)
Hardware: Any single p150 chip (32 GiB DRAM)

Transforms concatenated Gemma3 hidden states into separate video and audio
conditioning embeddings via dual transformer modules with "thinking tokens".

Known workarounds applied:
  - rope_type="interleaved" instead of "split" — avoids non-contiguous reshape
    bug in apply_split_rotary_emb during AOT export tracing
  - num_learnable_registers=None — avoids dynamic list comprehension that
    dynamo cannot trace

Input:  text_hidden_states [B, seq_len, caption_channels * text_proj_in_factor]
        attention_mask [B, seq_len]
Output: video_text [B, seq_len, caption_channels]
        audio_text [B, seq_len, caption_channels]
        attention_mask [B, seq_len]
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.pipelines.ltx2 import LTX2TextConnectors


def _patch_unflatten_graph_breaks():
    """Monkey-patch functions that use tensor.unflatten() with -1 to use view instead.

    unflatten(dim, (-1, n)) causes dynamo graph breaks because -1 requires
    dynamic shape inference. Replacing with view/reshape using explicit dims.

    Patched functions:
      1. apply_interleaved_rotary_emb: x.unflatten(2, (-1, 2)) -> x.view(...)
      2. LTX2AudioVideoAttnProcessor.__call__: query.unflatten(2, (heads, -1)) -> query.view(...)
    """
    import diffusers.models.transformers.transformer_ltx2 as ltx2_module

    # Patch 1: apply_interleaved_rotary_emb
    def _patched_apply_interleaved_rotary_emb(x, freqs):
        cos, sin = freqs
        # Replace x.unflatten(2, (-1, 2)) with explicit view
        b, s = x.shape[0], x.shape[1]
        half_c = x.shape[2] // 2
        x_reshaped = x.view(b, s, half_c, 2)
        x_real = x_reshaped[..., 0]  # [B, S, C // 2]
        x_imag = x_reshaped[..., 1]  # [B, S, C // 2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out

    ltx2_module.apply_interleaved_rotary_emb = _patched_apply_interleaved_rotary_emb

    # Patch 2: LTX2AudioVideoAttnProcessor.__call__ — replace unflatten with view
    _original_call = ltx2_module.LTX2AudioVideoAttnProcessor.__call__

    def _patched_processor_call(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
        query_rotary_emb=None, key_rotary_emb=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if query_rotary_emb is not None:
            query = _patched_apply_interleaved_rotary_emb(query, query_rotary_emb)
            key = _patched_apply_interleaved_rotary_emb(
                key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
            )

        # Replace unflatten(2, (attn.heads, -1)) with explicit view
        head_dim = query.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # Use SDPA directly instead of dispatch_attention_fn
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    ltx2_module.LTX2AudioVideoAttnProcessor.__call__ = _patched_processor_call


def run_ltx2_text_connectors():
    """
    Standalone bringup for LTX-2 Text Connectors.

    Uses text_proj_in_factor=3 (minimal config matching reference branch)
    for initial bringup. Full pipeline uses text_proj_in_factor=49 for
    real Gemma3 hidden states.

    Workarounds applied:
      - num_learnable_registers=None: avoids dynamic list comprehension dynamo can't trace
      - rope_type="interleaved": avoids split rotary emb reshape bug in AOT export
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Patch unflatten -> view to avoid dynamo graph breaks
    _patch_unflatten_graph_breaks()

    # Construct connectors with workarounds for compiler compatibility.
    # text_proj_in_factor=3 for minimal bringup (reference branch pattern).
    # Real pipeline will use text_proj_in_factor=49 with pretrained weights.
    connectors = LTX2TextConnectors(
        caption_channels=3840,
        text_proj_in_factor=3,  # minimal config for bringup (real: 49)
        video_connector_num_attention_heads=30,
        video_connector_attention_head_dim=128,
        video_connector_num_layers=2,
        video_connector_num_learnable_registers=None,  # workaround: disable to avoid dynamo trace failure
        audio_connector_num_attention_heads=30,
        audio_connector_attention_head_dim=128,
        audio_connector_num_layers=2,
        audio_connector_num_learnable_registers=None,  # workaround: disable to avoid dynamo trace failure
        connector_rope_base_seq_len=4096,
        rope_theta=10000.0,
        rope_double_precision=True,
        causal_temporal_positioning=False,
        rope_type="interleaved",  # workaround: avoids split rotary emb reshape bug
    ).to(torch.bfloat16).eval()

    connectors = connectors.to(device)
    connectors = torch.compile(connectors, backend="tt", fullgraph=True)

    # Input: concatenated hidden states [1, 256, 3840 * 3 = 11520]
    seq_len = 256
    text_hidden_states = torch.randn(
        1, seq_len, 3840 * 3, dtype=torch.bfloat16
    ).to(device)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

    # Warm-up pass (compilation)
    print("Text Connectors: warm-up pass (compilation)...")
    with torch.no_grad():
        video_text, audio_text, mask = connectors(text_hidden_states, attention_mask)
    torch_xla.sync(wait=True)
    print(f"  Video text shape: {video_text.shape}")
    print(f"  Audio text shape: {audio_text.shape}")

    # Timed pass
    print("Text Connectors: timed pass...")
    start = time.time()
    with torch.no_grad():
        video_text, audio_text, mask = connectors(text_hidden_states, attention_mask)
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    print(f"  Video text shape: {video_text.shape}")
    print(f"  Audio text shape: {audio_text.shape}")
    print(f"  Inference time: {elapsed:.3f}s")

    return video_text, audio_text, mask


if __name__ == "__main__":
    run_ltx2_text_connectors()
