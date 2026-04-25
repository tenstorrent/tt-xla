# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch import nn
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.deepseek_v4.modified_model.model import (
    ModelArgs,
    RMSNorm,
    Transformer,
)

PCC_99 = ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.99))


def make_mesh() -> Mesh:
    num_devices = xr.global_runtime_device_count()
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


def make_model(args: ModelArgs) -> Transformer:
    return Transformer(args).eval()


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    with torch.no_grad():
        for sub in module.modules():
            if isinstance(sub, RMSNorm):
                continue
            for _, param in sub.named_parameters(recurse=False):
                torch.nn.init.normal_(param, mean=0.0, std=std)


def _attn_shard_spec(attn):
    def spec_fn(module):
        shard_specs = {
            module.wq_b.weight: ("model", None),
            module.wo_a.weight: ("model", None),
            module.wo_b.weight: (None, "model"),
        }
        if hasattr(module, "indexer") and module.indexer is not None:
            shard_specs[module.indexer.wq_b.weight] = ("model", None)
            shard_specs[module.indexer.weights_proj.weight] = ("model", None)
        return shard_specs

    return spec_fn


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
        [x, qr, 0, offset],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec,
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
def test_attention_prefill_no_compression():
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    attn = model.layers[0].attn
    init_weights(attn)

    x = torch.randn(1, 8, args.dim, dtype=torch.bfloat16)
    mesh = make_mesh()

    run_graph_test(
        attn,
        [x, 0],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_attn_shard_spec(attn),
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
def test_attention_decode_no_compression():
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    attn = model.layers[0].attn
    init_weights(attn)

    x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16)
    mesh = make_mesh()

    run_graph_test(
        attn,
        [x, 4],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_attn_shard_spec(attn),
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
def test_attention_prefill_with_compression():
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    attn = model.layers[1].attn
    init_weights(attn)

    x = torch.randn(1, 8, args.dim, dtype=torch.bfloat16)
    mesh = make_mesh()

    run_graph_test(
        attn,
        [x, 0],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_attn_shard_spec(attn),
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
def test_attention_decode_with_compression():
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    attn = model.layers[1].attn
    init_weights(attn)

    x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16)
    mesh = make_mesh()

    run_graph_test(
        attn,
        [x, 4],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_attn_shard_spec(attn),
        comparison_config=PCC_99,
    )
