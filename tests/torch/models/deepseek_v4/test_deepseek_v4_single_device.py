# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch import nn

from third_party.tt_forge_models.deepseek_v4.modified_model.kernel import (
    hc_split_sinkhorn,
    sparse_attn,
)
from third_party.tt_forge_models.deepseek_v4.modified_model.model import (
    ModelArgs,
    RMSNorm,
    Transformer,
)

PCC_99 = ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.99))


def small_args(**overrides) -> ModelArgs:
    """Small ModelArgs that fit comfortably on a single device."""
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


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """Initialize all uninitialized parameters with a normal distribution."""
    with torch.no_grad():
        for sub in module.modules():
            if isinstance(sub, RMSNorm):
                continue
            for _, param in sub.named_parameters(recurse=False):
                torch.nn.init.normal_(param, mean=0.0, std=std)


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


@pytest.mark.nightly
@pytest.mark.single_device
def test_sparse_attn():
    xr.set_device_type("TT")

    b, s, h, d = 1, 4, 4, 32
    n, topk = 16, 6
    softmax_scale = d**-0.5

    torch.manual_seed(0)
    q = torch.randn(b, s, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, dtype=torch.bfloat16)
    attn_sink = torch.zeros(h, dtype=torch.bfloat16)
    topk_idxs = torch.randint(0, n, (b, s, topk), dtype=torch.int32)
    topk_idxs[:, :, -1] = -1

    run_graph_test(
        _SparseAttnModule(softmax_scale),
        [q, kv, attn_sink, topk_idxs],
        framework=Framework.TORCH,
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.single_device
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

    run_graph_test(
        _HcSplitSinkhornModule(hc, args.hc_sinkhorn_iters, args.hc_eps),
        [mixes, hc_scale, hc_base],
        framework=Framework.TORCH,
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("seqlen", [3, 4, 5, 8, 9])
@pytest.mark.parametrize(
    "compressor_layer", [1, 2], ids=["ratio4_overlap", "ratio8_no_overlap"]
)
def test_compressor_prefill(seqlen, compressor_layer):
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    compressor = model.layers[compressor_layer].attn.compressor
    with torch.no_grad():
        torch.nn.init.normal_(compressor.ape, mean=0.0, std=0.02)
        torch.nn.init.normal_(compressor.wkv.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(compressor.wgate.weight, mean=0.0, std=0.02)

    x = torch.randn(1, seqlen, args.dim, dtype=torch.bfloat16)

    run_graph_test(
        compressor,
        [x, 0],
        framework=Framework.TORCH,
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("start_pos", [0, 1, 3, 4, 7, 8])
@pytest.mark.parametrize(
    "compressor_layer", [1, 2], ids=["ratio4_overlap", "ratio8_no_overlap"]
)
def test_compressor_decode(start_pos, compressor_layer):
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    compressor = model.layers[compressor_layer].attn.compressor
    with torch.no_grad():
        torch.nn.init.normal_(compressor.ape, mean=0.0, std=0.02)
        torch.nn.init.normal_(compressor.wkv.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(compressor.wgate.weight, mean=0.0, std=0.02)

    x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16)

    run_graph_test(
        compressor,
        [x, start_pos],
        framework=Framework.TORCH,
        comparison_config=PCC_99,
    )


class _CompressorPrefillThenDecode(nn.Module):
    """Wrapper that runs prefill to warm up state, then runs decode."""

    def __init__(self, compressor: nn.Module):
        super().__init__()
        self.compressor = compressor

    def forward(
        self, prefill_x: torch.Tensor, decode_x: torch.Tensor, decode_start_pos: int
    ):
        # Run prefill to populate kv_state and score_state
        self.compressor(prefill_x, 0)
        # print("KV state after prefill:")
        print(self.compressor.kv_state)
        # print(torch.sum(self.compressor.kv_state, dim=(0,2)))
        # Run decode with warmed-up state
        y = self.compressor(decode_x, decode_start_pos)
        # print("KV state after decode:")
        # print(self.compressor.kv_state)
        # print(torch.sum(self.compressor.kv_state, dim=(0,2)))
        return y


@pytest.mark.nightly
@pytest.mark.single_device
def test_compressor_prefill_then_decode():
    """Compressor prefill followed by decode: tests stateful KV compression."""
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    # Layer 1 has compress_ratio=4 with overlap
    compressor = model.layers[1].attn.compressor
    with torch.no_grad():
        torch.nn.init.normal_(compressor.ape, mean=0.0, std=0.02)
        torch.nn.init.normal_(compressor.wkv.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(compressor.wgate.weight, mean=0.0, std=0.02)

    bsz = 1
    prefill_seqlen = 4  # Exactly one compress_ratio worth of tokens
    decode_start_pos = 4  # Position after prefill

    torch.manual_seed(42)
    prefill_x = torch.randn(bsz, prefill_seqlen, args.dim, dtype=torch.bfloat16)
    decode_x = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)

    wrapper = _CompressorPrefillThenDecode(compressor)

    run_graph_test(
        wrapper,
        [prefill_x, decode_x, decode_start_pos],
        torch_options={"tt_legacy_compile": True},
        framework=Framework.TORCH,
        comparison_config=PCC_99,
    )
