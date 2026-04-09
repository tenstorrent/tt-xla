# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Partial Pipeline — Text Encoding + Denoising on TT hardware.

Validates the integration of working components:
  Phase 1: Text Encoder (Gemma3 2-layer) -> Text Connectors -> conditioning
  Phase 2: Transformer (4-layer) -> single denoising step

Conv3d-dependent components (VAE encode/decode, latent upsampler) are BLOCKED
by tt-metal Conv3d L1 overflow and are skipped with dummy data.

All patches from standalone bringup are applied:
  - unflatten -> reshape (dynamo graph break fix)
  - prims::view_of -> clone (XLA functionalization fix)
  - pre-computed causal mask (int8 slice overflow fix)
  - fullgraph=True (prevents _guards_fn NameError)
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers import LTX2VideoTransformer3DModel


# ---------------------------------------------------------------------------
# Monkey-patches (same as standalone scripts)
# ---------------------------------------------------------------------------

def _patch_unflatten_graph_breaks():
    """Replace unflatten(-1) with reshape to avoid dynamo graph breaks."""
    import diffusers.models.transformers.transformer_ltx2 as ltx2_module

    def _patched_apply_interleaved_rotary_emb(x, freqs):
        cos, sin = freqs
        b, s = x.shape[0], x.shape[1]
        half_c = x.shape[2] // 2
        x_reshaped = x.reshape(b, s, half_c, 2)
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out

    ltx2_module.apply_interleaved_rotary_emb = _patched_apply_interleaved_rotary_emb

    def _patched_processor_call(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
        query_rotary_emb=None, key_rotary_emb=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.reshape(batch_size, attn.heads, -1, attention_mask.shape[-1])
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
        head_dim = query.shape[-1] // attn.heads
        query = query.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    ltx2_module.LTX2AudioVideoAttnProcessor.__call__ = _patched_processor_call


def _patch_view_of():
    """Replace prims::view_of with clone to avoid XLA functionalization error."""
    import tt_torch.torch_overrides as _overrides
    _orig = _overrides.TorchFunctionOverride.__torch_function__

    def _patched(self, func, types, args, kwargs=None):
        if func is torch.ops.prims.view_of.default:
            return args[0].clone()
        return _orig(self, func, types, args, kwargs)

    _overrides.TorchFunctionOverride.__torch_function__ = _patched


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_partial_pipeline():
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    device = torch_xla.device()

    # Apply all monkey-patches
    _patch_unflatten_graph_breaks()
    _patch_view_of()

    print("=" * 60)
    print("LTX-2 Partial Pipeline (Text Encoding + Denoising)")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Phase 1: Text Encoding
    # ---------------------------------------------------------------
    print("\n--- Phase 1: Text Encoding ---")

    # 1a. Text Encoder (Gemma3 2-layer minimal)
    text_config = Gemma3TextConfig(
        hidden_size=3840, intermediate_size=15360, num_hidden_layers=2,
        num_attention_heads=16, num_key_value_heads=8, head_dim=256, vocab_size=262208,
        sliding_window=None, use_cache=False,
    )
    text_encoder = Gemma3TextModel(text_config).to(torch.bfloat16).eval().to(device)
    text_encoder_compiled = torch.compile(text_encoder, backend="tt")

    seq_len = 128
    input_ids = torch.randint(0, 262208, (1, seq_len), dtype=torch.long).to(device)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16), diagonal=1
    ).unsqueeze(0).unsqueeze(0).to(device)

    print("  Running text encoder...")
    t0 = time.time()
    with torch.no_grad():
        enc_out = text_encoder_compiled(
            input_ids=input_ids, attention_mask=causal_mask, output_hidden_states=True,
        )
    torch_xla.sync(wait=True)
    print(f"  Text encoder done in {time.time()-t0:.1f}s, {len(enc_out.hidden_states)} hidden states")

    # Concatenate hidden states for connectors (text_proj_in_factor=3 -> use 3 layers)
    # Take last 3 hidden states and concat along last dim
    text_hidden = torch.cat(enc_out.hidden_states[-3:], dim=-1)  # [1, 128, 3840*3]
    print(f"  Concatenated hidden states: {text_hidden.shape}")

    # Free text encoder from device
    del text_encoder_compiled, text_encoder, enc_out
    torch_xla.sync(wait=True)

    # 1b. Text Connectors
    connectors = LTX2TextConnectors(
        caption_channels=3840, text_proj_in_factor=3,
        video_connector_num_attention_heads=30, video_connector_attention_head_dim=128,
        video_connector_num_layers=2, video_connector_num_learnable_registers=None,
        audio_connector_num_attention_heads=30, audio_connector_attention_head_dim=128,
        audio_connector_num_layers=2, audio_connector_num_learnable_registers=None,
        connector_rope_base_seq_len=4096, rope_theta=10000.0, rope_double_precision=True,
        causal_temporal_positioning=False, rope_type="interleaved",
    ).to(torch.bfloat16).eval().to(device)
    connectors_compiled = torch.compile(connectors, backend="tt", fullgraph=True)

    attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

    print("  Running text connectors...")
    t0 = time.time()
    with torch.no_grad():
        video_text, audio_text, mask = connectors_compiled(text_hidden, attention_mask)
    torch_xla.sync(wait=True)
    print(f"  Connectors done in {time.time()-t0:.1f}s")
    print(f"  Video text: {video_text.shape}, Audio text: {audio_text.shape}")

    del connectors_compiled, connectors
    torch_xla.sync(wait=True)

    # ---------------------------------------------------------------
    # Phase 2: Denoising (single step)
    # ---------------------------------------------------------------
    print("\n--- Phase 2: Denoising ---")

    # Create minimal transformer
    class TransformerWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, **kwargs):
            out = self.inner(**kwargs)
            return out.sample.clone(), out.audio_sample.clone()

    transformer = LTX2VideoTransformer3DModel(num_layers=4).to(torch.bfloat16).eval().to(device)
    transformer.config.rope_type = "interleaved"
    wrapper = TransformerWrapper(transformer)
    transformer_compiled = torch.compile(wrapper, backend="tt", fullgraph=True)

    # Dummy latents (would come from VAE encoding in full pipeline)
    num_frames, h, w = 2, 4, 4
    n_video = num_frames * h * w
    n_audio = 16

    video_latent = torch.randn(1, n_video, 128, dtype=torch.bfloat16).to(device)
    audio_latent = torch.randn(1, n_audio, 128, dtype=torch.bfloat16).to(device)
    timestep = torch.tensor([999], dtype=torch.long).to(device)
    enc_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

    print("  Running transformer denoising step...")
    t0 = time.time()
    with torch.no_grad():
        video_out, audio_out = transformer_compiled(
            hidden_states=video_latent,
            audio_hidden_states=audio_latent,
            encoder_hidden_states=video_text,
            audio_encoder_hidden_states=audio_text,
            timestep=timestep,
            encoder_attention_mask=enc_mask,
            audio_encoder_attention_mask=enc_mask,
            num_frames=num_frames, height=h, width=w,
            audio_num_frames=n_audio,
        )
    torch_xla.sync(wait=True)
    print(f"  Transformer done in {time.time()-t0:.1f}s")
    print(f"  Denoised video: {video_out.shape}, audio: {audio_out.shape}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PARTIAL PIPELINE COMPLETE")
    print("  Phase 1 (Text Encoding): PASS")
    print("  Phase 2 (Denoising): PASS")
    print("  Phase 3 (Decoding): SKIPPED (Conv3d L1 overflow blocker)")
    print("=" * 60)


if __name__ == "__main__":
    run_partial_pipeline()
