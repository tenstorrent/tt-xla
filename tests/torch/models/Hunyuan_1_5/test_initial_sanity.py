# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-chip repro for the HunyuanVideo-1.5 encoder drop: transformer_blocks[0]
inlined by hand, using the real submodules off self.block, so ops can be commented
out to bisect. norm1/norm1_context (nn.LayerNorm) are the confirmed trigger."""

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers.models.embeddings import apply_rotary_emb
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.hunyuan_1_5.pytorch.src.model_utils import (
    MESH_NAMES,
    MESH_SHAPES,
    load_transformer,
    shard_transformer_specs,
)

DTYPE = torch.bfloat16
PCC_THRESHOLD = 0.99

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _stats(name, t):
    """PCC=nan comes from _compare_pcc's denom==0 (a zero-variance/constant tensor),
    not FP NaN. std~0 identifies the collapsed tensor; min/max/mean show the value."""
    tf = t.float()
    logger.info(
        "[slice-mc] {:>7}: shape={} has_nan={} has_inf={} min={:.4g} max={:.4g} mean={:.4g} std={:.4g}",
        name, tuple(t.shape),
        bool(torch.isnan(tf).any()), bool(torch.isinf(tf).any()),
        tf.min().item(), tf.max().item(), tf.mean().item(), tf.std().item(),
    )


def _to_xla(x):
    """Move a tensor, or a tuple/list of tensors (freqs_cis is (cos, sin))."""
    if isinstance(x, (tuple, list)):
        return type(x)(_to_xla(t) for t in x)
    return x.to(xm.xla_device())


class Block0Repro(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.block = transformer.transformer_blocks[0]

    def forward(self, hidden_states, encoder_hidden_states, temb, attention_mask, freqs_cis):
        block = self.block
        attn = block.attn

        # norm1 / norm1_context (AdaLayerNormZero): silu -> linear -> chunk(6) -> norm * (1+scale) + shift
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1.linear(F.silu(temb)).chunk(6, dim=1)
        norm_hidden_states = block.norm1.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = block.norm1_context.linear(F.silu(temb)).chunk(6, dim=1)
        norm_encoder_hidden_states = block.norm1_context.norm(encoder_hidden_states) * (1 + c_scale_msa[:, None]) + c_shift_msa[:, None]

        query = attn.to_q(norm_hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(norm_hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(norm_hidden_states).unflatten(2, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis, sequence_dim=1)
            key = apply_rotary_emb(key, freqs_cis, sequence_dim=1)

        encoder_query = attn.add_q_proj(norm_encoder_hidden_states).unflatten(2, (attn.heads, -1))
        encoder_key = attn.add_k_proj(norm_encoder_hidden_states).unflatten(2, (attn.heads, -1))
        encoder_value = attn.add_v_proj(norm_encoder_hidden_states).unflatten(2, (attn.heads, -1))
        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)

        batch_size, seq_len, heads, dim = query.shape
        attention_mask = F.pad(attention_mask, (seq_len - attention_mask.shape[1], 0), value=True).bool()
        self_attn_mask_1 = attention_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
        attention_mask = (self_attn_mask_1 & self_attn_mask_1.transpose(2, 3)).bool()

        # native SDPA backend: [B,S,H,D] -permute-> [B,H,S,D] -> SDPA -> permute back
        q_, k_, v_ = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        attn_out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = attn_out.permute(0, 2, 1, 3).flatten(2, 3).to(query.dtype)

        # the slice under suspicion
        hidden = hidden_states[:, : -norm_encoder_hidden_states.shape[1]]
        encoder = hidden_states[:, -norm_encoder_hidden_states.shape[1] :]
        return hidden, encoder


@pytest.mark.nightly
@pytest.mark.tensor_parallel
def test_slice_multichip():
    xr.set_device_type("TT")

    transformer = load_transformer(DTYPE)
    transformer.eval()
    model = Block0Repro(transformer).eval()

    hidden_states = torch.load("block_hidden_states_tt.pt", map_location="cpu")
    encoder_hidden_states = torch.load("block_encoder_hidden_states_tt.pt", map_location="cpu")
    temb = torch.load("block_temb_tt.pt", map_location="cpu")
    attention_mask = torch.load("block_attention_mask_tt.pt", map_location="cpu")
    freqs_cis = torch.load("block_freqs_cis_tt.pt", map_location="cpu")  # (cos, sin)

    logger.info("hidden_states shape={} dtype={}", tuple(hidden_states.shape), hidden_states.dtype)
    logger.info("hidden_states: {}", hidden_states)
    logger.info("encoder_hidden_states shape={} dtype={}", tuple(encoder_hidden_states.shape), encoder_hidden_states.dtype)
    logger.info("encoder_hidden_states: {}", encoder_hidden_states)
    logger.info("temb shape={} dtype={}", tuple(temb.shape), temb.dtype)
    logger.info("temb: {}", temb)
    logger.info("attention_mask shape={} dtype={}", tuple(attention_mask.shape), attention_mask.dtype)
    logger.info("attention_mask: {}", attention_mask)
    logger.info("freqs_cis shapes={} dtype={}", [tuple(t.shape) for t in freqs_cis], freqs_cis[0].dtype)
    logger.info("freqs_cis: {}", freqs_cis)

    cpu_hidden, cpu_encoder = model(hidden_states, encoder_hidden_states, temb, attention_mask, freqs_cis)

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(num_devices)), MESH_SHAPES[num_devices], MESH_NAMES)

    model = model.to(xm.xla_device())
    specs = shard_transformer_specs(transformer)
    model_param_ids = {id(p) for p in model.parameters()}
    n_sharded = 0
    for tensor, spec in specs.items():
        if id(tensor) in model_param_ids:
            xs.mark_sharding(tensor, mesh, spec)
            n_sharded += 1
    logger.info("[slice-mc] sharded {} block-0 weights", n_sharded)

    for name, w in (("attn.to_q.weight", model.block.attn.to_q.weight), ("attn.to_out.0.weight", model.block.attn.to_out[0].weight)):
        logger.info("[slice-mc] sharding spec {}: {}", name, torch_xla._XLAC._get_xla_sharding_spec(w))

    tt_hidden, tt_encoder = model(
        _to_xla(hidden_states),
        _to_xla(encoder_hidden_states),
        _to_xla(temb),
        _to_xla(attention_mask),
        _to_xla(freqs_cis),
    )
    tt_hidden = tt_hidden.to("cpu")
    tt_encoder = tt_encoder.to("cpu")

    _stats("cpu_hid", cpu_hidden)
    _stats("cpu_enc", cpu_encoder)
    _stats("tt_hid", tt_hidden)
    _stats("tt_enc", tt_encoder)

    pcc_hidden = _pcc(tt_hidden, cpu_hidden)
    pcc_encoder = _pcc(tt_encoder, cpu_encoder)
    logger.info("[slice-mc] pcc latent={:.6f} encoder={:.6f}", pcc_hidden, pcc_encoder)

    assert pcc_hidden >= PCC_THRESHOLD, f"latent PCC {pcc_hidden:.6f} < {PCC_THRESHOLD}"
    assert pcc_encoder >= PCC_THRESHOLD, f"encoder PCC {pcc_encoder:.6f} < {PCC_THRESHOLD} — in-graph slice miscompile"
