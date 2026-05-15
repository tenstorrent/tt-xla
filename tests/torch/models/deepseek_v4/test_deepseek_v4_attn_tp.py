# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from infra.utilities.torch_multichip_utils import enable_spmd
from torch import nn
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp

from third_party.tt_forge_models.deepseek_v4.modified_model.model_decode_opt import (
    Attention,
    Block,
    ModelArgs,
    RMSNorm,
    Transformer,
)

from . import realistic_inputs, utils, weight_loader


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V4-Flash"])
def test_indexer_prefill(model_name):
    enable_spmd()
    xr.set_device_type("TT")

    args = utils.real_args()
    model = utils.make_transformer(args)
    attn = model.layers[2].attn
    indexer = attn.indexer
    weight_loader.init_block_weights(
        model_name, indexer, args, 2, sub_prefix="attn.indexer."
    )

    bsz, seq_len = 1, 8
    x = torch.randn(bsz, seq_len, args.dim, dtype=torch.bfloat16)
    qr = torch.randn(bsz, seq_len, args.q_lora_rank, dtype=torch.bfloat16)
    offset = seq_len
    mesh = utils.make_2d_mesh()

    def shard_spec(indexer):
        return {
            indexer.wq_b.weight: ("model", None),
            indexer.weights_proj.weight: ("model", None),
        }

    run_graph_test(
        indexer,
        [x, qr, torch.tensor(0, dtype=torch.long), offset],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec,
        comparison_config=utils.PCC_99,
    )


def attn_shard_spec_fn(attn: Attention, args, kwargs):
    bsz = args[0].size(0)
    if bsz % 2 != 0:
        return {}
    shard_specs = {
        attn.wq_b.weight: ("model", None),
        attn.wo_a.weight: ("model", None),
        attn.wo_b.weight: (None, "model"),
        attn.kv_cache: ("batch", None, None),
    }
    if attn.compress_ratio:
        shard_specs[attn.compressor.kv_cache] = ("batch", None, None)
        shard_specs[attn.compressor.kv_state] = ("batch", None, None)
        shard_specs[attn.compressor.score_state] = ("batch", None, None)
    if hasattr(attn, "indexer") and attn.indexer is not None:
        shard_specs[attn.indexer.wq_b.weight] = ("model", None)
        shard_specs[attn.indexer.weights_proj.weight] = ("model", None)

        shard_specs[attn.indexer.compressor.kv_cache] = ("batch", None, None)
        shard_specs[attn.indexer.compressor.kv_state] = ("batch", None, None)
        shard_specs[attn.indexer.compressor.score_state] = ("batch", None, None)

    shard_specs[args[0]] = ("batch", None, None)
    return shard_specs


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V4-Flash"])
@pytest.mark.parametrize("bsz", [1, 4, 32])
@pytest.mark.parametrize("seq_len", [32, 64, 128])
@pytest.mark.parametrize("is_compression_layer", [False, True])
def test_attention_prefill(model_name, bsz, seq_len, is_compression_layer):
    enable_spmd()
    xr.set_device_type("TT")

    args = utils.real_args()

    args.max_batch_size = bsz
    if is_compression_layer:
        layer_id = 2
    else:
        layer_id = 1
    attn = utils.make_attention(args, layer_id)
    weight_loader.init_block_weights(
        model_name, attn, args, layer_id, sub_prefix="attn."
    )

    mesh = utils.make_2d_mesh()

    x = torch.randn(bsz, seq_len, args.dim, dtype=torch.bfloat16)

    run_graph_test(
        attn,
        [x, torch.tensor(0, dtype=torch.long)],  # [input_tensor, start_pos]
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=attn_shard_spec_fn,
        comparison_config=utils.PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V4-Flash"])
@pytest.mark.parametrize("bsz", [1, 4, 32])
@pytest.mark.parametrize("prefill_seq_len", [32, 64, 128])
@pytest.mark.parametrize("is_compression_layer", [False, True])
def test_attention_decode(model_name, bsz, prefill_seq_len, is_compression_layer):
    enable_spmd()
    xr.set_device_type("TT")

    args = utils.real_args()

    args.max_batch_size = bsz
    if is_compression_layer:
        layer_id = 2
    else:
        layer_id = 1
    attn = utils.make_attention(args, layer_id)
    weight_loader.init_block_weights(
        model_name, attn, args, layer_id, sub_prefix="attn."
    )

    mesh = utils.make_2d_mesh()

    x = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)

    run_graph_test(
        attn,
        [
            x,
            torch.tensor(prefill_seq_len, dtype=torch.long),
        ],  # [input_tensor, start_pos]
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=attn_shard_spec_fn,
        comparison_config=utils.PCC_99,
    )


def _layer_shard_spec(layer: Block, bsz: int):
    """Per-layer attn + ffn shard entries. Mutates shard_specs in place."""
    attn = layer.attn
    shard_specs = {}
    shard_specs[attn.wq_b.weight] = ("model", None)
    shard_specs[attn.wo_a.weight] = ("model", None)
    shard_specs[attn.wo_b.weight] = (None, "model")
    shard_specs[attn.kv_cache] = ("batch", None, None)
    if attn.compress_ratio:
        shard_specs[attn.compressor.kv_cache] = ("batch", None, None)
        shard_specs[attn.compressor.kv_state] = ("batch", None, None)
        shard_specs[attn.compressor.score_state] = ("batch", None, None)
    if hasattr(attn, "indexer") and attn.indexer is not None:
        shard_specs[attn.indexer.wq_b.weight] = ("model", None)
        shard_specs[attn.indexer.weights_proj.weight] = ("model", None)
        shard_specs[attn.indexer.compressor.kv_cache] = ("batch", None, None)
        shard_specs[attn.indexer.compressor.kv_state] = ("batch", None, None)
        shard_specs[attn.indexer.compressor.score_state] = ("batch", None, None)

    ffn = layer.ffn
    if hasattr(ffn, "mlp") and hasattr(ffn.mlp, "experts"):
        experts = ffn.mlp.experts
        compound = ("batch", "model") if "batch" is not None else "model"
        shard_specs[experts.gate_proj] = (compound, None, None)
        shard_specs[experts.up_proj] = (compound, None, None)
        shard_specs[experts.down_proj] = (compound, None, None)
    return shard_specs


def block_shard_spec(block: Block, args, kwargs):
    bsz = args[0].size(0)
    shard_specs = _layer_shard_spec(block, bsz)
    shard_specs[args[0]] = ("batch", None, None, None)  # x: [b, s, hc, d]
    shard_specs[args[2]] = ("batch", None)  # input_ids: [b, s]
    return shard_specs


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V4-Flash"])
@pytest.mark.parametrize("is_compression_layer", [False, True])
@pytest.mark.parametrize("use_realistic_inputs", [False, True])
def test_block_prefill(model_name, is_compression_layer, use_realistic_inputs):
    enable_spmd()
    xr.set_device_type("TT")

    bsz = 4
    seq_len = 32

    args = utils.real_args()

    args.max_batch_size = bsz
    if is_compression_layer:
        layer_id = 2
    else:
        layer_id = 1
    block = utils.make_block(args, layer_id)
    weight_loader.init_block_weights(model_name, block, args, layer_id)

    if use_realistic_inputs:
        input_ids, hidden_states = realistic_inputs.get_realistic_inputs(
            model_name, layer_id, bsz, seq_len
        )
        x = hidden_states.unsqueeze(2).repeat(1, 1, args.hc_mult, 1).contiguous()
    else:
        x = torch.randn(bsz, seq_len, args.hc_mult, args.dim, dtype=torch.bfloat16)
        input_ids = torch.randint(0, args.vocab_size, (bsz, seq_len), dtype=torch.long)

    mesh = utils.make_2d_mesh()
    enable_sparse_mlp(block, mesh=mesh.mesh_shape, cluster_axis=0, config=args)

    run_graph_test(
        block,
        [x, torch.tensor(0, dtype=torch.long), input_ids],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=block_shard_spec,
        comparison_config=utils.PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V4-Flash"])
@pytest.mark.parametrize("is_compression_layer", [False, True])
@pytest.mark.parametrize("use_realistic_inputs", [False, True])
def test_block_decode(model_name, is_compression_layer, use_realistic_inputs):
    enable_spmd()
    xr.set_device_type("TT")

    bsz = 32
    seq_len = 1
    prefill_seq_len = 32

    args = utils.real_args()

    args.max_batch_size = bsz
    if is_compression_layer:
        layer_id = 2
    else:
        layer_id = 1
    block = utils.make_block(args, layer_id)
    weight_loader.init_block_weights(model_name, block, args, layer_id)

    if use_realistic_inputs:
        input_ids, hidden_states = realistic_inputs.get_realistic_inputs(
            model_name, layer_id, bsz, seq_len
        )
        x = hidden_states.unsqueeze(2).repeat(1, 1, args.hc_mult, 1).contiguous()
    else:
        x = torch.randn(bsz, seq_len, args.hc_mult, args.dim, dtype=torch.bfloat16)
        input_ids = torch.randint(0, args.vocab_size, (bsz, seq_len), dtype=torch.long)

    mesh = utils.make_2d_mesh()
    enable_sparse_mlp(block, mesh=mesh.mesh_shape, cluster_axis=0, config=args)

    run_graph_test(
        block,
        [x, torch.tensor(prefill_seq_len, dtype=torch.long), input_ids],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=block_shard_spec,
        comparison_config=utils.PCC_99,
    )
