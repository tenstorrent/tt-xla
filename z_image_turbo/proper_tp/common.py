# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared utilities for Z-Image-Turbo proper_tp inference.

Models loaded from: Tongyi-MAI/Z-Image-Turbo
  - text_encoder: Qwen3Model (Qwen3-2.5B), 36 layers, hidden_size=2560
  - transformer:  ZImageTransformer2DModel, 30 layers, dim=3840, n_heads=30
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


def setup_spmd():
    """Enable XLA SPMD mode. Must be called before xr.global_runtime_device_count()."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def get_mesh(mesh_shape, axis_names):
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)
    print(f"Created mesh: shape={mesh_shape}, axes={axis_names}, devices={num_devices}")
    return mesh


def load_text_encoder():
    """Load Qwen3 text encoder (base model, no LM head) in bf16."""
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading text encoder from {MODEL_ID}/text_encoder ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = AutoModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    text_encoder.eval()
    print(
        f"Text encoder loaded: {sum(p.numel() for p in text_encoder.parameters())/1e9:.2f}B params"
    )
    return tokenizer, text_encoder


def apply_te_sharding_tp(text_encoder, mesh, model_axis="model"):
    """Apply Megatron-style tensor parallel sharding to Qwen3 text encoder.

    Column-parallel: q/k/v_proj, gate_proj, up_proj  -> (model_axis, None)
    Row-parallel:    o_proj, down_proj               -> (None, model_axis)

    Requires: num_attention_heads % mesh[model_axis] == 0  (32 / 4 = 8 ✓)
              num_key_value_heads % mesh[model_axis] == 0  (8 / 4 = 2 ✓)
    """
    for layer in text_encoder.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight,      mesh, (model_axis, None))
        xs.mark_sharding(layer.mlp.gate_proj.weight,    mesh, (model_axis, None))
        xs.mark_sharding(layer.mlp.down_proj.weight,    mesh, (None, model_axis))
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, (model_axis, None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, (model_axis, None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, (model_axis, None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, model_axis))


def load_transformer():
    """Load ZImageTransformer2DModel in bf16."""
    from diffusers import ZImageTransformer2DModel

    print(f"Loading transformer from {MODEL_ID}/transformer ...")
    transformer = ZImageTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    transformer.eval()
    print(
        f"Transformer loaded: {sum(p.numel() for p in transformer.parameters())/1e9:.2f}B params"
    )
    return transformer


def make_dummy_latents(batch_size=1, height=256, width=256, device=None):
    """Create dummy VAE latents for testing.

    Args:
        height, width: spatial resolution in pixels. Latent is (H/8, W/8).
    Returns:
        List of (16, 1, H/8, W/8) bf16 tensors (one per batch item).
    """
    h, w = height // 8, width // 8
    latents = [
        torch.randn(16, 1, h, w, dtype=torch.bfloat16)
        for _ in range(batch_size)
    ]
    if device is not None:
        latents = [lat.to(device) for lat in latents]
    return latents


def apply_transformer_full_sharding_tp(transformer, mesh, model_axis="model"):
    """Apply full tensor parallel sharding (MLP + attention) to ZImageTransformer2DModel.

    Requires: n_heads % mesh[model_axis] == 0
    Transformer heads are padded 30 → 32 before calling this function so that
    32 / 4 = 8 heads per device.

    Attention:
      to_q, to_k, to_v -> column-parallel  ("model", None)
      to_out[0]        -> row-parallel     (None, "model")
    MLP (SwiGLU):
      w1, w3           -> column-parallel
      w2               -> row-parallel
    """
    def _shard_block_full(layer):
        # Attention
        xs.mark_sharding(layer.attention.to_q.weight, mesh, (model_axis, None))
        xs.mark_sharding(layer.attention.to_k.weight, mesh, (model_axis, None))
        xs.mark_sharding(layer.attention.to_v.weight, mesh, (model_axis, None))
        xs.mark_sharding(layer.attention.to_out[0].weight, mesh, (None, model_axis))
        # MLP
        xs.mark_sharding(layer.feed_forward.w1.weight, mesh, (model_axis, None))
        xs.mark_sharding(layer.feed_forward.w3.weight, mesh, (model_axis, None))
        xs.mark_sharding(layer.feed_forward.w2.weight, mesh, (None, model_axis))

    for layer in transformer.layers:
        _shard_block_full(layer)
    for layer in transformer.noise_refiner:
        _shard_block_full(layer)
    for layer in transformer.context_refiner:
        _shard_block_full(layer)


def patch_rope_for_tt():
    """Apply all TT-backend compatibility patches. Call once before any model runs.

    Patches applied:
    1. Real-valued RoPE: TT-MLIR doesn't support complex<f32>. Replaces
       torch.polar/view_as_complex in ZImageTransformer2DModel with equivalent
       real ops (cos/sin multiply).

    2. XLA-compatible _prepare_sequence: replaces boolean-indexed tensor
       assignment with torch.where (no dynamic shape indexing).

    3. XLA-compatible unpatchify: derives reshape dims from tensor shape instead
       of Python ints from the `size` list argument, avoiding a dynamo graph break.

    4. cumsum u8 fix: TT hardware only supports cumsum on INT32/UINT32/BFLOAT16/
       UINT16/FLOAT32. transformers/masking_utils.py calls .cumsum() on a boolean
       tensor (maps to uint8). Patched to cast to int32 first.
    """
    from diffusers.models.transformers.transformer_z_image import (
        RopeEmbedder,
        ZSingleStreamAttnProcessor,
    )
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    @staticmethod
    def _precompute_freqs_cis_real(dim, end, theta=256.0):
        # Avoid `with torch.device("cpu"):` — dynamo treats it as unsupported context manager.
        # Pass device="cpu" explicitly to each tensor creation op instead.
        result = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            timestep = torch.arange(e, dtype=torch.float64, device="cpu")
            freqs = torch.outer(timestep, freqs).float()  # [e, d//2]
            result.append(
                torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # [e, d//2, 2]
            )
        return result

    def _rope_call_real(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device
        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]
        elif self.freqs_cis[0].device != device:
            self.freqs_cis = [f.to(device) for f in self.freqs_cis]
        result = []
        for i in range(len(self.axes_dims)):
            result.append(self.freqs_cis[i][ids[:, i]])  # [N, d//2, 2]
        return torch.cat(result, dim=-2)  # [N, sum_d//2, 2]

    def _attn_call_real(self, attn, hidden_states, encoder_hidden_states=None,
                        attention_mask=None, freqs_cis=None):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # reshape instead of unflatten: tt_torch subclasses Tensor with __torch_function__
        # which routes unflatten through super().unflatten() — dynamo can't trace super() calls.
        query = query.reshape(*query.shape[:-1], attn.heads, -1)
        key = key.reshape(*key.shape[:-1], attn.heads, -1)
        value = value.reshape(*value.shape[:-1], attn.heads, -1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if freqs_cis is not None:
            # freqs_cis: [B, N, D//2, 2]
            def _real_rope(x_in, fc):
                x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)  # [B, N, H, D//2, 2]
                cos = fc[..., 0].unsqueeze(2)  # [B, N, 1, D//2]
                sin = fc[..., 1].unsqueeze(2)  # [B, N, 1, D//2]
                x_r, x_i = x[..., 0], x[..., 1]
                out = torch.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], dim=-1)
                return out.flatten(-2).type_as(x_in)  # [B, N, H, D]

            query = _real_rope(query, freqs_cis)
            key = _real_rope(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output

    RopeEmbedder.precompute_freqs_cis = _precompute_freqs_cis_real
    RopeEmbedder.__call__ = _rope_call_real
    ZSingleStreamAttnProcessor.__call__ = _attn_call_real
    print("Applied real-valued RoPE patch (no complex tensors) for TT backend compatibility")

    # Also patch _prepare_sequence to replace boolean-indexed assignment with
    # torch.where, which is XLA-compatible (no dynamic shape indexing).
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
    from torch.nn.utils.rnn import pad_sequence as _pad_sequence

    def _prepare_sequence_xla(self, feats, pos_ids, inner_pad_mask, pad_token, noise_mask=None, device=None):
        """XLA-compatible _prepare_sequence: uses torch.where instead of boolean indexing."""
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)

        # Replace: feats_cat[torch.cat(inner_pad_mask)] = pad_token
        # With XLA-compatible torch.where (no dynamic-shape boolean indexing)
        feats_cat = torch.cat(feats, dim=0)  # [total_seq_len, dim]
        combined_mask = torch.cat(inner_pad_mask)  # [total_seq_len] bool
        feats_cat = torch.where(
            combined_mask.unsqueeze(-1),            # [total_seq_len, 1]
            pad_token.expand_as(feats_cat),          # [total_seq_len, dim]
            feats_cat,
        )
        feats = list(feats_cat.split(item_seqlens, dim=0))

        # RoPE
        freqs_cis = list(
            self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0)
        )

        # Pad to batch
        feats = _pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = _pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]

        # Attention mask
        attn_mask = torch.zeros((bsz, max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(item_seqlens):
            attn_mask[i, :seq_len] = 1

        # Noise mask
        noise_mask_tensor = None
        if noise_mask is not None:
            noise_mask_tensor = _pad_sequence(
                [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
                batch_first=True,
                padding_value=0,
            )[:, : feats.shape[1]]

        return feats, freqs_cis, attn_mask, item_seqlens, noise_mask_tensor

    ZImageTransformer2DModel._prepare_sequence = _prepare_sequence_xla
    print("Applied XLA-compatible _prepare_sequence patch (torch.where for boolean indexing)")

    # Patch unpatchify to avoid a dynamo graph break at the final reshape.
    #
    # Root cause: unpatchify's `else` branch does:
    #   F, H, W = size[i]           # Python ints from a list[tuple] argument
    #   .reshape(self.out_channels, F, H, W)
    # XLA treats F, H, W as potentially-dynamic values when they come from a
    # Python list argument, causing a graph break between the .permute() and
    # .reshape() calls.
    #
    # Fix: derive the output shape from the tensor's own dimension sizes after
    # the permute, which XLA always knows statically.
    #   t.shape = [out_channels, F//pF, pF, H//pH, pH, W//pW, pW]
    #   → reshape to [out_channels, F//pF*pF, H//pH*pH, W//pW*pW]
    #   This is identical to .reshape(out_channels, F, H, W) but uses tensor
    #   shape arithmetic that dynamo/XLA can fold at trace time.
    def _unpatchify_xla(self, x, size, patch_size, f_patch_size, x_pos_offsets=None):
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz

        if x_pos_offsets is not None:
            # Omni variant path — pass through to original (not used for Z-Image-Turbo base)
            return ZImageTransformer2DModel._unpatchify_original(
                self, x, size, patch_size, f_patch_size, x_pos_offsets
            )

        # Original mode: simple unpatchify with XLA-safe reshape
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            t = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
            )
            # Derive output shape from t.shape to avoid dynamo graph break.
            # t.shape = [out_channels, F//pF, pF, H//pH, pH, W//pW, pW]
            x[i] = t.reshape(
                t.shape[0],
                t.shape[1] * t.shape[2],
                t.shape[3] * t.shape[4],
                t.shape[5] * t.shape[6],
            )
        return x

    ZImageTransformer2DModel._unpatchify_original = ZImageTransformer2DModel.unpatchify
    ZImageTransformer2DModel.unpatchify = _unpatchify_xla
    print("Applied XLA-compatible unpatchify patch (tensor-shape reshape, single graph)")

    # Patch find_packed_sequence_indices to cast bool→int32 before cumsum.
    # (position_diff != 1) produces a bool tensor; .cumsum() on bool → uint8,
    # which TT hardware does not support. Cast to int32 before cumsum.
    from transformers import masking_utils as _masking_utils

    _original_find_packed = _masking_utils.find_packed_sequence_indices

    def _find_packed_sequence_indices_tt(position_ids: torch.Tensor):
        first_dummy_value = position_ids[:, :1] - 1
        position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        # Cast bool to int32 before cumsum — TT does not support cumsum on uint8/bool
        packed_sequence_mask = (position_diff != 1).to(torch.int32).cumsum(-1)
        from transformers.utils.import_utils import is_tracing
        if not is_tracing(packed_sequence_mask) and (packed_sequence_mask[:, -1] == 0).all():
            return None
        return packed_sequence_mask

    _masking_utils.find_packed_sequence_indices = _find_packed_sequence_indices_tt
    print("Applied cumsum u8 fix (bool→int32 cast before cumsum in masking_utils)")
