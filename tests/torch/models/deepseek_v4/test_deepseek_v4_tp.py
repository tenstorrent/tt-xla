# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import enable_spmd
from tt_torch.sparse_mlp import enable_sparse_mlp

from third_party.tt_forge_models.deepseek_v4.modified_model.model_decode_opt import (
    Attention,
    Block,
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
@pytest.mark.parametrize("seq_len", [64, 128])
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
@pytest.mark.parametrize("prefill_seq_len", [64, 128])
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


def _layer_shard_spec(layer: Block, shard_specs: dict):
    """Per-layer attn + ffn shard entries. Mutates shard_specs in place."""
    attn = layer.attn
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
        compound = ("batch", "model")
        shard_specs[experts.gate_proj] = (compound, None, None)
        shard_specs[experts.up_proj] = (compound, None, None)
        shard_specs[experts.down_proj] = (compound, None, None)
    return shard_specs


def block_shard_spec(block: Block, args, kwargs):
    shard_specs = {}
    _layer_shard_spec(block, shard_specs)
    shard_specs[args[0]] = ("batch", None, None, None)  # x: [b, s, hc, d]
    shard_specs[args[2]] = ("batch", None)  # input_ids: [b, s]
    return shard_specs


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V4-Flash"])
@pytest.mark.parametrize("is_compression_layer", [False, True])
@pytest.mark.parametrize("use_realistic_inputs", [True])
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
@pytest.mark.parametrize("use_realistic_inputs", [True])
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
    prime_decode_kv_buffers(block)

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


def transformer_shard_spec(model, args, kwargs):
    shard_specs: dict = {}
    for layer in model.layers:
        _layer_shard_spec(layer, shard_specs)
    shard_specs[args[0]] = ("batch", None)  # inputs_id: [b, s]
    return shard_specs


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V4-Flash"])
@pytest.mark.parametrize("num_layers", [2, 3])
def test_transformer_prefill(model_name, num_layers):
    enable_spmd()
    xr.set_device_type("TT")

    bsz = 16
    seq_len = 32
    args = utils.real_args(n_layers=num_layers, max_batch_size=bsz)
    args.compress_ratios = args.compress_ratios[:num_layers]

    model = utils.make_transformer(args, True)
    weight_loader.init_transformer_weights(model_name, model, num_layers=num_layers)

    input_ids, _ = realistic_inputs.get_realistic_inputs(
        model_name, layer_id=args.n_hash_layers, batch_size=bsz, seq_len=seq_len
    )

    mesh = utils.make_2d_mesh()
    enable_sparse_mlp(model, mesh=mesh.mesh_shape, cluster_axis=0, config=args)

    run_graph_test(
        model,
        [input_ids, torch.tensor(0, dtype=torch.long)],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=transformer_shard_spec,
        comparison_config=utils.PCC_99,
    )


def prime_decode_kv_buffers(model: Transformer | Block, std: float = 0.02) -> None:
    """Fill KV-cache and compressor state buffers with deterministic random
    values."""
    g = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if not buf.is_floating_point():
                continue
            if name.endswith(".freqs_cis"):
                continue
            noise = torch.empty_like(buf, device="cpu").normal_(
                mean=0.0, std=std, generator=g
            )
            buf.copy_(noise)


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V4-Flash"])
@pytest.mark.parametrize(
    "num_layers,start_pos,expected_pcc",
    [
        # (num_layers, start_pos)
        # - start_pos must satisfy `start_pos + 1 - compress_ratio >= 0` for
        #   every Compressor in the included layers (rope_idx index bounds).
        # - start_pos==127 with num_layers==4 hits `(start_pos + 1) % ratio == 0`
        #   for both layer 2 (ratio=4) and layer 3 (ratio=128) — exercises the
        #   compressor's "compress step" branch (kv_state roll, kv_cache write).
        (1, 4, utils.PCC_99),
        (2, 4, utils.PCC_99),
        (3, 4, utils.PCC_99),
        (3, 128, utils.PCC_99),
        (3, 127, utils.PCC_99),
        (4, 128, utils.PCC_99),
        # https://github.com/tenstorrent/tt-xla/issues/4740
        (4, 127, utils.PCC_97),
    ],
)
def test_transformer_decode(model_name, num_layers, start_pos, expected_pcc):
    enable_spmd()
    xr.set_device_type("TT")

    bsz = 32
    seq_len = 1
    args = utils.real_args(n_layers=num_layers, max_batch_size=bsz)
    args.compress_ratios = args.compress_ratios[:num_layers]

    model = utils.make_transformer(args, True)
    weight_loader.init_transformer_weights(model_name, model, num_layers=num_layers)
    prime_decode_kv_buffers(model)

    input_ids, _ = realistic_inputs.get_realistic_inputs(
        model_name, layer_id=args.n_hash_layers, batch_size=bsz, seq_len=seq_len
    )

    mesh = utils.make_2d_mesh()
    enable_sparse_mlp(model, mesh=mesh.mesh_shape, cluster_axis=0, config=args)

    run_graph_test(
        model,
        [input_ids, torch.tensor(start_pos, dtype=torch.long)],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=transformer_shard_spec,
        comparison_config=expected_pcc,
    )
