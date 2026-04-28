# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
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

from . import realistic_inputs

PCC_99 = ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.99))
PCC_SPARSE = ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.99))


def make_mesh() -> Mesh:
    num_devices = xr.global_runtime_device_count()
    if num_devices == 32:
        mesh_shape = (4, 8)
    elif num_devices == 8:
        mesh_shape = (2, 4)
    else:
        mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, ("batch", "model"))


def small_args(**overrides) -> ModelArgs:
    """Small ModelArgs that fit comfortably on two chips."""
    defaults = dict(
        max_batch_size=2,
        max_seq_len=32,
        vocab_size=256,
        dim=128,
        moe_inter_dim=64,
        n_layers=3,
        n_mtp_layers=0,
        n_heads=4,
        q_lora_rank=64,
        head_dim=32,
        rope_head_dim=16,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        o_groups=2,
        o_lora_rank=32,
        window_size=8,
        compress_ratios=(0, 4, 8),
        index_n_heads=4,
        index_head_dim=16,
        index_topk=4,
        hc_mult=2,
        hc_sinkhorn_iters=3,
    )
    defaults.update(overrides)
    return ModelArgs(**defaults)


def real_args(**overrides) -> ModelArgs:
    """Attention-focused args copied from the real config.json with bounded test sizes."""
    defaults = dict(
        max_batch_size=1,
        max_seq_len=256,
        vocab_size=129280,
        dim=4096,
        moe_inter_dim=2048,
        n_layers=43,
        n_hash_layers=3,
        n_mtp_layers=0,
        n_heads=64,
        n_routed_experts=256,
        n_shared_experts=1,
        n_activated_experts=6,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10.0,
        q_lora_rank=1024,
        head_dim=512,
        rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        window_size=128,
        original_seq_len=65536,
        rope_theta=10000,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        dtype="bf16",
        scale_fmt=None,
        expert_dtype=None,
        compress_rope_theta=160000,
        compress_ratios=(
            0,
            0,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            0,
        ),
    )
    defaults.update(overrides)
    return ModelArgs(**defaults)


ATTENTION_TEST_CONFIGS = [
    pytest.param(small_args(), 0, 1, id="toy"),
    pytest.param(real_args(), 1, 2, id="real"),
]


def make_model(args: ModelArgs) -> Transformer:
    return Transformer(args).eval()


def make_attention(args: ModelArgs, layer_id: int) -> Attention:
    return Attention(layer_id, args).eval()


def make_block(args: ModelArgs, layer_id: int) -> Block:
    return Block(layer_id, args).eval()


_WEIGHTS_CACHE = Path(__file__).parents[4] / "generated" / "weights"


def init_weights(
    module: nn.Module,
    std: float = 0.02,
    *,
    args: ModelArgs = None,
    layer_id: int = None,
) -> None:
    """Initialize a module's weights.

    Two modes:
    1. Real V4-Flash mode (when `args` matches HF V4-Flash dims and
       `layer_id` is provided AND `module` is a Block): pull the real Block
       state dict from HF via `weight_loader.load_block_state_dict` and load
       it. Heavy (~12 GB per layer after bf16 dequant) but reuses the
       standard hf_hub cache so subsequent runs are fast.
    2. Otherwise: random-init each floating param (cached on disk by
       module-path + shape so reruns are deterministic). Integer params
       (e.g. hash gate's `tid2eid`) zeroed.
    """
    if (
        args is not None
        and layer_id is not None
        and args.vocab_size == 129280
        and args.dim == 4096
        and args.hc_mult == 4
    ):
        from . import weight_loader

        state = weight_loader.load_block_state_dict(layer_id)
        # strict=False: weight_loader returns checkpoint-format keys but the
        # Block has a few non-persistent buffers (kv_cache, freqs_cis) that
        # aren't in the dict; those stay at their construction defaults.
        module.load_state_dict(state, strict=False)
        return

    _WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for mod_path, sub in module.named_modules():
            if isinstance(sub, RMSNorm):
                continue
            for name, param in sub.named_parameters(recurse=False):
                if param.is_floating_point():
                    shape_str = "x".join(str(d) for d in param.shape)
                    full_path = f"{mod_path}.{name}" if mod_path else name
                    key = f"{full_path.replace('.', '_')}_{shape_str}.pt"
                    cache_file = _WEIGHTS_CACHE / key
                    if cache_file.exists():
                        param.copy_(torch.load(cache_file, weights_only=True))
                    else:
                        torch.nn.init.normal_(param, mean=0.0, std=std)
                        torch.save(param.data.clone(), cache_file)
                else:
                    param.zero_()


def _attn_shard_spec(attn, args, kwargs):
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
@pytest.mark.dual_chip
def test_indexer_prefill():
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    attn1 = model.layers[1].attn
    indexer = attn1.indexer
    init_weights(indexer)

    bsz, seqlen = 1, 8
    x = torch.randn(bsz, seqlen, args.dim, dtype=torch.bfloat16)
    qr = torch.randn(bsz, seqlen, args.q_lora_rank, dtype=torch.bfloat16)
    offset = seqlen
    mesh = make_mesh()

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
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
@pytest.mark.parametrize(
    "args,no_compression_layer_id,_compression_layer_id", ATTENTION_TEST_CONFIGS
)
def test_attention_prefill_no_compression(
    args: ModelArgs, no_compression_layer_id: int, _compression_layer_id: int
):
    xr.set_device_type("TT")

    bsz = 4
    seq_len = 128

    args.max_batch_size = bsz
    attn = make_attention(args, no_compression_layer_id)
    init_weights(attn)

    x = torch.randn(bsz, seq_len, args.dim, dtype=torch.bfloat16)

    mesh = make_mesh()

    run_graph_test(
        attn,
        [x, torch.tensor(0, dtype=torch.long)],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_attn_shard_spec,
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
@pytest.mark.parametrize(
    "args,no_compression_layer_id,_compression_layer_id", ATTENTION_TEST_CONFIGS
)
def test_attention_decode_no_compression(
    args: ModelArgs, no_compression_layer_id: int, _compression_layer_id: int
):
    bsz = 4
    seq_len = 1

    args.max_batch_size = bsz
    attn = make_attention(args, no_compression_layer_id)
    init_weights(attn)

    x = torch.randn(bsz, seq_len, args.dim, dtype=torch.bfloat16)
    mesh = make_mesh()

    run_graph_test(
        attn,
        [x, torch.tensor(4, dtype=torch.long)],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_attn_shard_spec,
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
@pytest.mark.parametrize(
    "args,_no_compression_layer_id,compression_layer_id", ATTENTION_TEST_CONFIGS
)
def test_attention_prefill_with_compression(
    args: ModelArgs, _no_compression_layer_id: int, compression_layer_id: int
):
    xr.set_device_type("TT")

    bsz = 4
    seq_len = 128

    args.max_batch_size = bsz
    attn = make_attention(args, compression_layer_id)
    init_weights(attn)

    x = torch.randn(bsz, seq_len, args.dim, dtype=torch.bfloat16)
    mesh = make_mesh()

    run_graph_test(
        attn,
        [x, torch.tensor(0, dtype=torch.long)],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_attn_shard_spec,
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
@pytest.mark.parametrize(
    "args,_no_compression_layer_id,compression_layer_id", ATTENTION_TEST_CONFIGS
)
def test_attention_decode_with_compression(
    args: ModelArgs, _no_compression_layer_id: int, compression_layer_id: int
):
    xr.set_device_type("TT")

    bsz = 4
    seq_len = 1

    args.max_batch_size = bsz
    attn = make_attention(args, compression_layer_id)
    init_weights(attn)

    x = torch.randn(bsz, seq_len, args.dim, dtype=torch.bfloat16)
    mesh = make_mesh()

    run_graph_test(
        attn,
        [x, torch.tensor(4, dtype=torch.long)],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_attn_shard_spec,
        comparison_config=PCC_99,
    )


def _block_shard_spec(block, args, kwargs):
    attn = block.attn
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

    ffn = block.ffn
    if hasattr(ffn, "mlp") and hasattr(ffn.mlp, "experts"):
        experts = ffn.mlp.experts
        compound = ("batch", "model")
        shard_specs[experts.gate_proj] = (compound, None, None)
        shard_specs[experts.up_proj] = (compound, None, None)
        shard_specs[experts.down_proj] = (compound, None, None)
    # shared = getattr(ffn, "shared_experts", None)
    # if shared is not None:
    #     shard_specs[shared.w1.weight] = ("model", None)
    #     shard_specs[shared.w3.weight] = ("model", None)
    #     shard_specs[shared.w2.weight] = (None, "model")

    shard_specs[args[0]] = ("batch", None, None, None)  # x: [b, s, hc, d]
    shard_specs[args[2]] = ("batch", None)  # input_ids: [b, s]
    return shard_specs


@pytest.mark.nightly
@pytest.mark.parametrize(
    "args,no_compression_layer_id,compression_layer_id", ATTENTION_TEST_CONFIGS
)
@pytest.mark.parametrize(
    "with_compression", [False, True], ids=["no_compress", "compress"]
)
def test_block_prefill(
    args: ModelArgs,
    no_compression_layer_id: int,
    compression_layer_id: int,
    with_compression: bool,
):
    xr.set_device_type("TT")

    bsz = 4
    seq_len = 32
    args.max_batch_size = bsz

    layer_id = compression_layer_id if with_compression else no_compression_layer_id
    block = make_block(args, layer_id)
    # For V4-Flash dims, init_weights pulls real HF weights for this Block
    # (the (args, layer_id) opt-in path). For toy, falls back to disk-cached
    # random init.
    init_weights(block, args=args, layer_id=layer_id)

    # Use cached realistic inputs when the parametrize config matches V4-Flash
    # dims (the cache is generated with vocab/dim/hc_mult pinned to the real
    # config). For the toy config the cache shape doesn't fit, so fall back to
    # random — regenerating a toy-sized cache isn't worth the complexity.
    if args.vocab_size == 129280 and args.dim == 4096 and args.hc_mult == 4:
        input_ids, hidden_states = realistic_inputs.get_realistic_inputs(
            layer_id=args.n_hash_layers, batch_size=bsz, seq_len=seq_len
        )
        # The cache stores single-stream `[b, s, d]` (post-ffn_norm of layer
        # n_hash_layers). Block.forward expects hc-expanded `[b, s, hc, d]`.
        # Replicating into hc copies mirrors Transformer.forward's initial
        # `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)` — degenerate hc streams
        # but identical to how the real model enters layer 0, so the per-token
        # activation distribution stays realistic.
        x = hidden_states.unsqueeze(2).repeat(1, 1, args.hc_mult, 1).contiguous()
    else:
        x = torch.randn(bsz, seq_len, args.hc_mult, args.dim, dtype=torch.bfloat16)
        input_ids = torch.randint(0, args.vocab_size, (bsz, seq_len), dtype=torch.long)
    mesh = make_mesh()

    enable_sparse_mlp(block, mesh=mesh.mesh_shape, cluster_axis=0, config=args)

    run_graph_test(
        block,
        [x, torch.tensor(0, dtype=torch.long), input_ids],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_block_shard_spec,
        comparison_config=PCC_SPARSE,
    )


@pytest.mark.nightly
@pytest.mark.parametrize(
    "args,no_compression_layer_id,compression_layer_id", ATTENTION_TEST_CONFIGS
)
@pytest.mark.parametrize(
    "with_compression", [False, True], ids=["no_compress", "compress"]
)
def test_block_decode(
    args: ModelArgs,
    no_compression_layer_id: int,
    compression_layer_id: int,
    with_compression: bool,
):
    xr.set_device_type("TT")

    bsz = 32
    seq_len = 1
    args.max_batch_size = bsz

    layer_id = compression_layer_id if with_compression else no_compression_layer_id
    block = make_block(args, layer_id)
    # For V4-Flash dims, init_weights pulls real HF weights for this Block
    # (the (args, layer_id) opt-in path). For toy, falls back to disk-cached
    # random init.
    init_weights(block, args=args, layer_id=layer_id)

    # Use cached realistic inputs when the parametrize config matches V4-Flash
    # dims. For the toy config the cache shape doesn't fit, fall back to random.
    if args.vocab_size == 129280 and args.dim == 4096 and args.hc_mult == 4:
        input_ids, hidden_states = realistic_inputs.get_realistic_inputs(
            layer_id=args.n_hash_layers, batch_size=bsz, seq_len=seq_len
        )
        # Cache stores single-stream `[b, s, d]`; Block.forward expects
        # hc-expanded `[b, s, hc, d]`. Replicating into hc copies mirrors
        # Transformer.forward's initial expansion.
        x = hidden_states.unsqueeze(2).repeat(1, 1, args.hc_mult, 1).contiguous()
    else:
        x = torch.randn(bsz, seq_len, args.hc_mult, args.dim, dtype=torch.bfloat16)
        input_ids = torch.randint(0, args.vocab_size, (bsz, seq_len), dtype=torch.long)
    mesh = make_mesh()

    enable_sparse_mlp(block, mesh=mesh.mesh_shape, cluster_axis=0, config=args)

    run_graph_test(
        block,
        [x, torch.tensor(4, dtype=torch.long), input_ids],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_block_shard_spec,
        comparison_config=PCC_SPARSE,
    )
