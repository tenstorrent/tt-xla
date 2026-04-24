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

from third_party.tt_forge_models.deepseek_v4.modified_model.kernel import (
    hc_split_sinkhorn,
    sparse_attn,
)
from third_party.tt_forge_models.deepseek_v4.modified_model.model import (
    ModelArgs,
    Transformer,
)

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

MESH_SHAPE = (1, 2)
MESH_AXES = ("batch", "model")

PCC_99 = ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.99))


def make_mesh() -> Mesh:
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, MESH_SHAPE, MESH_AXES)


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
    # Do NOT call .to(bfloat16): Transformer.__init__ already sets default_dtype=bfloat16
    # so main weights are BF16, Compressor weights stay float32, and freqs_cis stays
    # complex (which .to(bfloat16) would destroy by discarding the imaginary part).
    return Transformer(args).eval()


# ---------------------------------------------------------------------------
# Wrappers for kernel functions (nn.Module so run_graph_test can move them)
# ---------------------------------------------------------------------------


class _SparseAttnModule(nn.Module):
    def __init__(self, softmax_scale: float):
        super().__init__()
        self.softmax_scale = softmax_scale

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        return sparse_attn(q, kv, attn_sink, topk_idxs, self.softmax_scale)


class _HcSplitSinkhornModule(nn.Module):
    def __init__(self, hc_mult: int, sinkhorn_iters: int, eps: float):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

    def forward(
        self,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        return hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.sinkhorn_iters, self.eps
        )


# ---------------------------------------------------------------------------
# Tests — kernel functions
# ---------------------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.dual_chip
def test_sparse_attn():
    xr.set_device_type("TT")

    b, s, h, d = 1, 4, 4, 32
    n, topk = 16, 6
    softmax_scale = d**-0.5

    torch.manual_seed(0)
    q = torch.randn(b, s, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, dtype=torch.bfloat16)
    attn_sink = torch.zeros(h, dtype=torch.bfloat16)
    # Mix of valid indices and -1 padding
    topk_idxs = torch.randint(0, n, (b, s, topk), dtype=torch.int32)
    topk_idxs[:, :, -1] = -1  # last slot always padding

    module = _SparseAttnModule(softmax_scale)
    mesh = make_mesh()

    run_graph_test(
        module,
        [q, kv, attn_sink, topk_idxs],
        framework=Framework.TORCH,
        mesh=mesh,
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
def test_hc_split_sinkhorn():
    xr.set_device_type("TT")

    args = small_args()
    hc = args.hc_mult
    mix_hc = (2 + hc) * hc
    b, s = 2, 4

    torch.manual_seed(0)
    mixes = torch.randn(b, s, mix_hc, dtype=torch.float32)
    hc_scale = torch.randn(3, dtype=torch.float32)
    hc_base = torch.randn(mix_hc, dtype=torch.float32)

    module = _HcSplitSinkhornModule(hc, args.hc_sinkhorn_iters, args.hc_eps)
    mesh = make_mesh()

    run_graph_test(
        module,
        [mixes, hc_scale, hc_base],
        framework=Framework.TORCH,
        mesh=mesh,
        comparison_config=PCC_99,
    )


# ---------------------------------------------------------------------------
# Tests — Compressor
# ---------------------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.dual_chip
# most parameterizations are failing due to PCC (TT output is all or partially zeroed)
# or compile error 'fused_0' object has no attribute 'xla_args'
@pytest.mark.parametrize("seqlen", [3, 4, 5, 8, 9])
@pytest.mark.parametrize(
    "compressor_layer", [1, 2], ids=["ratio4_overlap", "ratio8_no_overlap"]
)  # CSA uses overlapping compression, compression ratio = 4, on layer 1
def test_compressor_prefill(seqlen, compressor_layer):
    """Compressor prefill: seqlen divisible by compress_ratio triggers compression."""
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    # Layer 1 has compress_ratio=4
    compressor = model.layers[compressor_layer].attn.compressor
    with torch.no_grad():
        torch.nn.init.normal_(compressor.ape, mean=0.0, std=0.02)
        torch.nn.init.normal_(compressor.wkv.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(compressor.wgate.weight, mean=0.0, std=0.02)

    bsz = 1
    x = torch.randn(bsz, seqlen, args.dim, dtype=torch.bfloat16)
    mesh = make_mesh()

    run_graph_test(
        compressor,
        [x, 0],
        framework=Framework.TORCH,
        mesh=mesh,
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
# most parameterizations are failing due to PCC or
# compile error 'fused_0' object has no attribute 'xla_args'
@pytest.mark.parametrize("start_pos", [0, 1, 3, 4, 7, 8])
@pytest.mark.parametrize(
    "compressor_layer", [1, 2], ids=["ratio4_overlap", "ratio8_no_overlap"]
)  # CSA uses overlapping compression, compression ratio = 4, on layer 1
def test_compressor_decode(start_pos, compressor_layer):
    """Compressor decode at start_pos=3: (3+1)%4==0 triggers compression."""
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    compressor = model.layers[compressor_layer].attn.compressor
    with torch.no_grad():
        torch.nn.init.normal_(compressor.ape, mean=0.0, std=0.02)
        torch.nn.init.normal_(compressor.wkv.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(compressor.wgate.weight, mean=0.0, std=0.02)

    bsz = 1
    x = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)
    mesh = make_mesh()

    run_graph_test(
        compressor,
        [x, start_pos],
        framework=Framework.TORCH,
        mesh=mesh,
        comparison_config=PCC_99,
    )


# ---------------------------------------------------------------------------
# Tests — Indexer
# ---------------------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.dual_chip
def test_indexer_prefill():
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    attn1 = model.layers[1].attn
    indexer = attn1.indexer

    bsz, seqlen = 1, 8
    x = torch.randn(bsz, seqlen, args.dim, dtype=torch.bfloat16)
    qr = torch.randn(bsz, seqlen, args.q_lora_rank, dtype=torch.bfloat16)
    offset = seqlen  # compressed indices start after the window KV

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


# ---------------------------------------------------------------------------
# Tests — Attention (no compression, layer 0)
# ---------------------------------------------------------------------------


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
def test_attention_prefill_no_compression():
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    attn = model.layers[0].attn  # compress_ratio=0

    bsz, seqlen = 1, 8
    x = torch.randn(bsz, seqlen, args.dim, dtype=torch.bfloat16)
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
    attn = model.layers[0].attn  # compress_ratio=0

    bsz = 1
    x = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)
    mesh = make_mesh()

    run_graph_test(
        attn,
        [x, 4],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_attn_shard_spec(attn),
        comparison_config=PCC_99,
    )


# ---------------------------------------------------------------------------
# Tests — Attention (with compression, layer 1, compress_ratio=4)
# ---------------------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.dual_chip
def test_attention_prefill_with_compression():
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    attn = model.layers[1].attn  # compress_ratio=4, has Indexer

    bsz, seqlen = 1, 8
    x = torch.randn(bsz, seqlen, args.dim, dtype=torch.bfloat16)
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
    attn = model.layers[1].attn  # compress_ratio=4, has Indexer

    bsz = 1
    x = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)
    mesh = make_mesh()

    run_graph_test(
        attn,
        [x, 4],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=_attn_shard_spec(attn),
        comparison_config=PCC_99,
    )
