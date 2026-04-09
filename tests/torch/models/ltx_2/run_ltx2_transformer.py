# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Dual-Stream DiT Transformer — standalone bringup script.

Component: LTX2VideoTransformer3DModel

Phase 1 (current): Minimal 4-layer config with random weights, replicated.
  - Validates the dual-stream attention path (video + audio tokens).
  - Same dimensions as full model: video 32 heads x 128 dim, audio 32 heads x 64 dim.

Phase 2 (future): Full 48-layer pretrained model with 4-way TP sharding.
  - Requires fixing SPMD + graph break device count mismatch.

Known workaround applied:
  - unflatten -> view monkey-patch: same as text connectors, avoids dynamo graph break
    in LTX2AudioVideoAttnProcessor.__call__ and apply_interleaved_rotary_emb.
  - rope_type="interleaved": set on model config to avoid split rotary emb tracing bug.

Input:  video tokens [B, n_video, 128], audio tokens [B, n_audio, 128],
        video text [B, text_len, 3840], audio text [B, text_len, 3840],
        timestep, masks, spatial dims
Output: denoised video [B, n_video, 128], denoised audio [B, n_audio, 128]
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import LTX2VideoTransformer3DModel


def _patch_unflatten_graph_breaks():
    """Monkey-patch functions that use tensor.unflatten() with -1 to use view instead.

    unflatten(dim, (-1, n)) causes dynamo graph breaks because -1 requires
    dynamic shape inference. Replacing with view/reshape using explicit dims.
    """
    import diffusers.models.transformers.transformer_ltx2 as ltx2_module

    # Patch 1: apply_interleaved_rotary_emb
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

    # Patch 2: LTX2AudioVideoAttnProcessor.__call__
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
        query = query.reshape(batch_size, -1, attn.heads, head_dim)
        key = key.reshape(batch_size, -1, attn.heads, head_dim)
        value = value.reshape(batch_size, -1, attn.heads, head_dim)

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


def run_ltx2_transformer():
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})
    device = torch_xla.device()

    # Patch unflatten -> reshape to avoid dynamo graph breaks
    _patch_unflatten_graph_breaks()

    # Patch TorchFunctionOverride to handle prims::view_of by cloning instead.
    # The XLA/TT backend doesn't support functionalized view_of with alias annotations.
    import tt_torch.torch_overrides as _overrides
    _original_torch_function = _overrides.TorchFunctionOverride.__torch_function__

    def _patched_torch_function(self, func, types, args, kwargs=None):
        if func is torch.ops.prims.view_of.default:
            return args[0].clone()
        return _original_torch_function(self, func, types, args, kwargs)

    _overrides.TorchFunctionOverride.__torch_function__ = _patched_torch_function


    # Create minimal 4-layer transformer with random weights
    model = LTX2VideoTransformer3DModel(
        num_layers=4,
    ).to(torch.bfloat16)
    model.config.rope_type = "interleaved"  # avoid split rotary emb tracing bug
    model = model.eval()

    model = model.to(device)

    # Wrap model to clone outputs — avoids prims::view_of aliasing error
    # during functionalization. The transformer's outputs are views of internal
    # tensors, which triggers unsupported view_of prim.
    class CloneOutputWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args, **kwargs):
            out = self.inner(*args, **kwargs)
            return out.sample.clone(), out.audio_sample.clone()

    wrapper = CloneOutputWrapper(model)
    compiled = torch.compile(wrapper, backend="tt", fullgraph=True)

    # Small inputs for bringup
    num_frames, h, w = 2, 4, 4
    n_video = num_frames * h * w  # 32
    n_audio = 16
    text_len = 16

    hidden_states = torch.randn(1, n_video, 128, dtype=torch.bfloat16).to(device)
    audio_hidden_states = torch.randn(1, n_audio, 128, dtype=torch.bfloat16).to(device)
    encoder_hidden_states = torch.randn(1, text_len, 3840, dtype=torch.bfloat16).to(device)
    audio_encoder_hidden_states = torch.randn(1, text_len, 3840, dtype=torch.bfloat16).to(device)
    timestep = torch.tensor([500], dtype=torch.long).to(device)
    encoder_attention_mask = torch.ones(1, text_len, dtype=torch.long).to(device)
    audio_encoder_attention_mask = torch.ones(1, text_len, dtype=torch.long).to(device)

    kwargs = dict(
        hidden_states=hidden_states,
        audio_hidden_states=audio_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        audio_encoder_hidden_states=audio_encoder_hidden_states,
        timestep=timestep,
        encoder_attention_mask=encoder_attention_mask,
        audio_encoder_attention_mask=audio_encoder_attention_mask,
        num_frames=num_frames,
        height=h,
        width=w,
        audio_num_frames=n_audio,
    )

    # Warm-up pass (compilation)
    print("Transformer (4-layer DiT, minimal): warm-up pass (compilation)...")
    with torch.no_grad():
        video_out, audio_out = compiled(**kwargs)
    torch_xla.sync(wait=True)
    print(f"  Video output shape: {video_out.shape}")
    print(f"  Audio output shape: {audio_out.shape}")

    # Timed pass
    print("Transformer (4-layer DiT, minimal): timed pass...")
    start = time.time()
    with torch.no_grad():
        video_out, audio_out = compiled(**kwargs)
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    print(f"  Video output shape: {video_out.shape}")
    print(f"  Audio output shape: {audio_out.shape}")
    print(f"  Inference time: {elapsed:.3f}s")

    return video_out, audio_out


if __name__ == "__main__":
    run_ltx2_transformer()
