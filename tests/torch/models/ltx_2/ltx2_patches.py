# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Reusable monkey-patches for LTX-2 on TT hardware.

These patches work around dynamo/XLA compilation issues while preserving
the mathematical correctness of the original model (especially RoPE).

Apply with: apply_all_patches()
"""

import torch


def patch_attention_processor():
    """Replace unflatten(2, (heads, -1)) with reshape in LTX2AudioVideoAttnProcessor.

    The original code uses tensor.unflatten(dim, (-1, n)) which causes dynamo
    graph breaks. We replace ONLY the unflatten calls with equivalent reshape,
    keeping the original RoPE application (split or interleaved) unchanged.
    """
    import diffusers.models.transformers.transformer_ltx2 as ltx2_module

    # Patch apply_interleaved_rotary_emb to replace unflatten with reshape
    # (the "split" version uses reshape already and doesn't need patching)
    def _patched_apply_interleaved_rotary_emb(x, freqs):
        cos, sin = freqs
        half_c = x.shape[-1] // 2
        x_reshaped = x.reshape(*x.shape[:-1], half_c, 2)
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out

    ltx2_module.apply_interleaved_rotary_emb = _patched_apply_interleaved_rotary_emb
    _apply_interleaved = _patched_apply_interleaved_rotary_emb

    # Patch apply_split_rotary_emb to add .contiguous() before final reshape
    # (XLA tensors from swapaxes are non-contiguous and can't be viewed)
    def _patched_apply_split_rotary_emb(x, freqs):
        cos, sin = freqs
        x_dtype = x.dtype
        needs_reshape = False
        if x.ndim != 4 and cos.ndim == 4:
            b, h, t, _ = cos.shape
            x = x.reshape(b, t, h, -1).permute(0, 2, 1, 3)  # swapaxes -> permute
            needs_reshape = True

        last = x.shape[-1]
        r = last // 2
        split_x = x.reshape(*x.shape[:-1], 2, r).float()
        first_x = split_x[..., :1, :]
        second_x = split_x[..., 1:, :]

        cos_u = cos.unsqueeze(-2)
        sin_u = sin.unsqueeze(-2)

        # Use explicit operations instead of addcmul_ (in-place may cause issues)
        out_first = first_x * cos_u - second_x * sin_u
        out_second = second_x * cos_u + first_x * sin_u
        out = torch.cat([out_first, out_second], dim=-2)
        out = out.reshape(*out.shape[:-2], last)

        if needs_reshape:
            # permute back and use contiguous() before reshape
            out = out.permute(0, 2, 1, 3).contiguous().reshape(b, t, -1)

        out = out.to(dtype=x_dtype)
        return out

    ltx2_module.apply_split_rotary_emb = _patched_apply_split_rotary_emb
    _apply_split = _patched_apply_split_rotary_emb

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

        # Apply RoPE using the ORIGINAL function matching the model's rope_type
        if query_rotary_emb is not None:
            if attn.rope_type == "interleaved":
                query = _apply_interleaved(query, query_rotary_emb)
                key = _apply_interleaved(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )
            elif attn.rope_type == "split":
                query = _apply_split(query, query_rotary_emb)
                key = _apply_split(
                    key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
                )

        # Replace unflatten(2, (heads, -1)) with reshape — this is the key fix
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


def patch_view_of():
    """Replace prims::view_of with clone to avoid XLA functionalization error."""
    import tt_torch.torch_overrides as _overrides
    _orig = _overrides.TorchFunctionOverride.__torch_function__

    def _patched(self, func, types, args, kwargs=None):
        if func is torch.ops.prims.view_of.default:
            return args[0].clone()
        return _orig(self, func, types, args, kwargs)

    _overrides.TorchFunctionOverride.__torch_function__ = _patched


def apply_all_patches():
    """Apply all necessary patches for LTX-2 on TT hardware."""
    from conv3d_decompose import patch_conv3d_to_conv2d
    patch_conv3d_to_conv2d()
    patch_attention_processor()
    patch_view_of()
