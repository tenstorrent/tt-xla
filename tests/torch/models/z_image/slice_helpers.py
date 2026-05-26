# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU capture of realistic Z-Image transformer tensors for slice / run_op_test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from third_party.tt_forge_models.z_image.pytorch.src.model_utils import (
    SEED,
    encode_prompt_hidden_states,
    load_text_encoder,
    load_transformer,
    make_latent_inputs,
    shard_transformer_specs,
)

PATCH_SIZE = 2
F_PATCH_SIZE = 1


@dataclass
class TransformerSliceBundle:
    """Tensors captured from the real transformer bring-up path (bf16, CPU)."""

    transformer: nn.Module
    adaln_input: torch.Tensor
    x_pos_ids_cat: torch.Tensor
    cap_pos_ids_cat: torch.Tensor
    x_after_prepare: torch.Tensor
    x_freqs: torch.Tensor
    x_mask: Optional[torch.Tensor]
    cap_after_prepare: torch.Tensor
    cap_freqs: torch.Tensor
    cap_mask: Optional[torch.Tensor]


def load_transformer_slice_bundle(
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> TransformerSliceBundle:
    torch.manual_seed(SEED)
    transformer = load_transformer(dtype).eval()
    latents = make_latent_inputs(dtype)
    timestep = torch.tensor([0.5], dtype=dtype)
    encoder = load_text_encoder(dtype)
    cap_feats = encode_prompt_hidden_states(encoder, dtype=dtype)

    x_list = list(latents.unsqueeze(2).unbind(dim=0))
    cap_list = [cap_feats[i] for i in range(cap_feats.shape[0])]
    t = timestep.reshape(-1)

    adaln_input = transformer.t_embedder(t * transformer.t_scale).type_as(x_list[0])

    (
        x_patches,
        cap_patches,
        _x_size,
        x_pos_ids,
        cap_pos_ids,
        x_pad_mask,
        cap_pad_mask,
    ) = transformer.patchify_and_embed(x_list, cap_list, PATCH_SIZE, F_PATCH_SIZE)

    x_seqlens = [len(xi) for xi in x_patches]
    x_cat = transformer.all_x_embedder[f"{PATCH_SIZE}-{F_PATCH_SIZE}"](
        torch.cat(x_patches, dim=0)
    )
    x_after_prepare, x_freqs, x_mask, _, _ = transformer._prepare_sequence(
        list(x_cat.split(x_seqlens, dim=0)),
        x_pos_ids,
        x_pad_mask,
        transformer.x_pad_token,
        None,
        x_list[0].device,
    )

    cap_seqlens = [len(ci) for ci in cap_patches]
    cap_cat = transformer.cap_embedder(torch.cat(cap_patches, dim=0))
    cap_after_prepare, cap_freqs, cap_mask, _, _ = transformer._prepare_sequence(
        list(cap_cat.split(cap_seqlens, dim=0)),
        cap_pos_ids,
        cap_pad_mask,
        transformer.cap_pad_token,
        None,
        x_list[0].device,
    )

    return TransformerSliceBundle(
        transformer=transformer,
        adaln_input=adaln_input,
        x_pos_ids_cat=torch.cat(x_pos_ids, dim=0),
        cap_pos_ids_cat=torch.cat(cap_pos_ids, dim=0),
        x_after_prepare=x_after_prepare,
        x_freqs=x_freqs,
        x_mask=x_mask,
        cap_after_prepare=cap_after_prepare,
        cap_freqs=cap_freqs,
        cap_mask=cap_mask,
    )


class RopeEmbedderModule(nn.Module):
    def __init__(self, rope_embedder):
        super().__init__()
        self.rope_embedder = rope_embedder

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        return self.rope_embedder(pos_ids)


class RopePolarAxisModule(nn.Module):
    """One axis of RopeEmbedder.precompute_freqs_cis (torch.polar path)."""

    def __init__(self, axes_dims, axes_lens, theta: float, axis_idx: int):
        super().__init__()
        self.axis_idx = axis_idx
        self.theta = theta
        self.d = axes_dims[axis_idx]
        self.e = axes_lens[axis_idx]

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.d, 2, dtype=torch.float64, device="cpu") / self.d)
        )
        timestep = torch.arange(self.e, device=freqs.device, dtype=torch.float64)
        freqs = torch.outer(timestep, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)


def fresh_rope_embedder(transformer: nn.Module):
    """RopeEmbedder with empty cache so TT compile includes polar precompute."""
    from diffusers.models.transformers.transformer_z_image import RopeEmbedder

    re = transformer.rope_embedder
    return RopeEmbedder(theta=re.theta, axes_dims=re.axes_dims, axes_lens=re.axes_lens)


def precompute_rope_freqs_tables(rope_embedder) -> list[torch.Tensor]:
    """CPU precompute — same tables ``RopeEmbedder`` lifts as compile-time constants."""
    from diffusers.models.transformers.transformer_z_image import RopeEmbedder

    re = rope_embedder
    return RopeEmbedder.precompute_freqs_cis(re.axes_dims, re.axes_lens, theta=re.theta)


class RopePrecomputeAllAxesModule(nn.Module):
    """All three ``torch.polar`` tables in one forward (no index / cat)."""

    def __init__(self, rope_embedder):
        super().__init__()
        self._rope = rope_embedder

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        tables = precompute_rope_freqs_tables(self._rope)
        # Flatten-concat so run_op_test has a single tensor output to compare.
        return torch.cat([t.reshape(-1) for t in tables])


class RopeIndexAxisModule(nn.Module):
    """``freqs_cis[axis][pos_ids[:, axis]]`` with precomputed complex tables as buffers."""

    def __init__(self, rope_embedder, axis_idx: int):
        super().__init__()
        tables = precompute_rope_freqs_tables(rope_embedder)
        self.axis_idx = axis_idx
        self.register_buffer("freqs_cis_table", tables[axis_idx])

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        return self.freqs_cis_table[pos_ids[:, self.axis_idx]]


class RopeIndexAndCatModule(nn.Module):
    """Index all axes from precomputed tables + ``torch.cat`` (no runtime polar)."""

    def __init__(self, rope_embedder):
        super().__init__()
        tables = precompute_rope_freqs_tables(rope_embedder)
        self.n_axes = len(tables)
        for i, table in enumerate(tables):
            self.register_buffer(f"freqs_cis_{i}", table)

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        parts = [
            getattr(self, f"freqs_cis_{i}")[pos_ids[:, i]] for i in range(self.n_axes)
        ]
        return torch.cat(parts, dim=-1)


class RopePolarThenIndexAxis1Module(nn.Module):
    """``torch.polar`` for one axis, then gather — polar+index in one forward (same device)."""

    def __init__(self, rope_embedder, axis_idx: int = 1):
        super().__init__()
        re = rope_embedder
        self.theta = re.theta
        self.d = re.axes_dims[axis_idx]
        self.e = re.axes_lens[axis_idx]
        self.axis_idx = axis_idx

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        device = pos_ids.device
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.d, 2, dtype=torch.float64, device=device) / self.d)
        )
        timestep = torch.arange(self.e, device=device, dtype=torch.float64)
        freqs = torch.outer(timestep, freqs).float()
        table = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
        return table[pos_ids[:, self.axis_idx].long()]


class PrepareSequenceRopeXModule(nn.Module):
    """Exact ``_prepare_sequence`` RoPE line: ``rope_embedder(cat(pos_ids))`` for image path."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.rope = RopeEmbedderModule(fresh_rope_embedder(transformer))

    def forward(self, pos_ids_cat: torch.Tensor) -> torch.Tensor:
        return self.rope(pos_ids_cat)


class ApplyRotaryEmbModule(nn.Module):
    """Minimal apply_rotary_emb from ZSingleStreamAttnProcessor."""

    def forward(self, x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class AttentionWithRopeModule(nn.Module):
    def __init__(self, attention, processor):
        super().__init__()
        self.attention = attention
        self.processor = processor

    def forward(
        self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        return self.processor(
            self.attention,
            hidden_states,
            attention_mask=None,
            freqs_cis=freqs_cis,
        )


def shard_spec_for_block(full_transformer: nn.Module, block: nn.Module) -> Callable:
    """Reuse Megatron specs from the full transformer for block parameters."""
    full_specs = shard_transformer_specs(full_transformer)

    def shard_spec_fn(model, args, kwargs):
        return {p: full_specs[p] for p in model.parameters() if p in full_specs}

    return shard_spec_fn
