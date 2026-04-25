# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch import nn

from tests.infra.testers.compiler_config import CompilerConfig
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


class _CircularKVCacheDecode(nn.Module):
    def __init__(self, window_size: int, head_dim: int, max_batch_size: int = 2):
        super().__init__()
        self.win = window_size
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, window_size, head_dim, dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(self, kv: torch.Tensor, start_pos: int):
        """Decode: update cache with single token at start_pos.

        Args:
            kv: [batch, 1, head_dim] - single decode token
            start_pos: position to write to (wraps around window)
        """
        bsz, seqlen, _ = kv.shape
        win = self.win
        # This line triggers the clamp bug for start_pos > 0 (?)
        self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
        print("kv cache", self.kv_cache.sum(dim=(0, 2)))
        return self.kv_cache[:bsz]


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(
    "start_pos", [0, 1, 4, 7], ids=["pos0", "pos1", "pos4", "pos7"]
)
def test_circular_kv_cache_decode(start_pos):
    """Minimal repro for HLO clamp bug in KV cache updates.

    Bug: clamp(batch_indices=[0,1], min=[0,0], max=[0,0]) → [0,0]
    Expected: Only pos0 passes, others fail with PCC~0.21
    """
    xr.set_device_type("TT")
    bsz = 2
    seqlen = 1  # Decode: single token
    head_dim = 32
    window_size = 8
    module = _CircularKVCacheDecode(window_size, head_dim, max_batch_size=2)

    torch.manual_seed(42)
    kv = torch.randn(bsz, seqlen, head_dim, dtype=torch.bfloat16)

    run_graph_test(
        module,
        [kv, start_pos],
        framework=Framework.TORCH,
        comparison_config=PCC_99,
    )


class _CircularKVCacheUpdate(nn.Module):
    """Standalone module that implements circular buffer KV cache update logic."""

    def __init__(self, window_size: int, head_dim: int, max_batch_size: int = 2):
        super().__init__()
        self.win = window_size
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, window_size, head_dim, dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(self, kv: torch.Tensor, start_pos: int):
        """
        Update circular buffer with new KV tokens.

        Args:
            kv: New KV tokens [batch, seqlen, head_dim]
            start_pos: Starting position in the sequence

        Returns:
            Updated kv_cache for verification
        """
        bsz, seqlen, _ = kv.shape
        win = self.win

        # Actual logic from Attention.forward (lines 491-502)
        if start_pos == 0:
            # Prefill path
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
        else:
            # Decode path
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            # self.kv_cache[:bsz, start_pos % win] = kv.view(bsz, -1)

        return self.kv_cache[:bsz]

    # index copy based
    def forward_index_copy(self, kv, start_pos):
        bsz, seqlen, _ = kv.shape
        win = self.win

        if start_pos == 0:
            if seqlen <= win:
                # Simple sequential write
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                # Circular write using computed indices
                write_indices = torch.arange(win, device=kv.device)
                write_indices = (start_pos + seqlen - win + write_indices) % win
                self.kv_cache.index_copy_(
                    dim=1, index=write_indices, source=kv[:bsz, -win:]
                )
        else:
            # Decode: single position
            pos = start_pos % win
            self.kv_cache[:bsz, pos : pos + 1] = kv

        return self.kv_cache[:bsz]


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(
    "seqlen,window_size,start_pos",
    [
        # Prefill tests (start_pos=0)
        (4, 8, 0),  # seqlen < window_size
        (8, 8, 0),  # seqlen == window_size
        (12, 8, 0),  # seqlen > window_size (cutoff=4)
        (16, 8, 0),  # seqlen > window_size (cutoff=0, full wrap)
        (20, 8, 0),  # seqlen > window_size (cutoff=4)
        # Decode tests (seqlen=1, various positions)
        (1, 8, 0),  # Decode at position 0
        (1, 8, 4),  # Decode at position 4
        (1, 8, 7),  # Decode at position 7 (last in window)
        (1, 8, 8),  # Decode at position 8 (wraps to 0)
        (1, 8, 15),  # Decode at position 15 (wraps to 7)
    ],
    ids=[
        "prefill_short",
        "prefill_exact",
        "prefill_wrap_partial",
        "prefill_wrap_full",
        "prefill_wrap_multi",
        "decode_pos0",
        "decode_pos4",
        "decode_pos7",
        "decode_wrap0",
        "decode_wrap7",
    ],
)
def test_circular_kv_cache_update(seqlen, window_size, start_pos):
    """Test circular buffer KV cache update logic for both prefill and decode."""
    xr.set_device_type("TT")

    head_dim = 32
    bsz = 1  # with bsz=2, all tests pass... seems like device memory corruption?

    module = _CircularKVCacheUpdate(window_size, head_dim)

    torch.manual_seed(42)
    kv = torch.randn(bsz, seqlen, head_dim, dtype=torch.bfloat16)

    run_graph_test(
        module,
        [kv, start_pos],
        # torch_options={"tt_legacy_compile": True},
        framework=Framework.TORCH,
        comparison_config=PCC_99,
    )


class _CircularKVCachePrefillThenDoubleDecode(nn.Module):
    """Simulates prefill followed by two decode steps with circular buffer updates."""

    def __init__(self, window_size: int, head_dim: int, max_batch_size: int = 2):
        super().__init__()
        self.kv_cache_module = _CircularKVCacheUpdate(
            window_size, head_dim, max_batch_size
        )

    def forward(
        self,
        prefill_kv: torch.Tensor,
        decode_kv1: torch.Tensor,
        decode_kv2: torch.Tensor,
    ):
        """
        Simulate prefill + decode + decode pattern.

        Args:
            prefill_kv: Prefill KV tokens [batch, prefill_seqlen, head_dim]
            decode_kv1: First decode token [batch, 1, head_dim]
            decode_kv2: Second decode token [batch, 1, head_dim]

        Returns:
            Final kv_cache state after all updates
        """
        prefill_seqlen = prefill_kv.shape[1]

        # Prefill: populate cache with initial tokens (start_pos=0)
        self.kv_cache_module(prefill_kv, 0)
        print("prefill", self.kv_cache_module.kv_cache.sum(dim=(0, 2)))

        # First decode: add one token at position prefill_seqlen
        self.kv_cache_module(decode_kv1, prefill_seqlen)
        print("decode1", self.kv_cache_module.kv_cache.sum(dim=(0, 2)))

        # Second decode: add another token at position prefill_seqlen + 1
        final_cache = self.kv_cache_module(decode_kv2, prefill_seqlen + 1)
        print("decode2", self.kv_cache_module.kv_cache.sum(dim=(0, 2)))

        return final_cache


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(
    "prefill_seqlen,window_size",
    [
        (4, 8),  # Prefill doesn't fill window, decode adds to end
        (8, 8),  # Prefill fills window exactly, decode wraps around
        (10, 8),  # Prefill overfills, decode continues wrapping
    ],
    ids=["prefill_partial", "prefill_exact", "prefill_overflow"],
)
def test_circular_kv_cache_prefill_then_double_decode(prefill_seqlen, window_size):
    """Test circular buffer across prefill + decode + decode steps."""
    xr.set_device_type("TT")

    head_dim = 32
    bsz = 1

    module = _CircularKVCachePrefillThenDoubleDecode(window_size, head_dim)

    torch.manual_seed(42)
    prefill_kv = torch.randn(bsz, prefill_seqlen, head_dim, dtype=torch.bfloat16)
    decode_kv1 = torch.randn(bsz, 1, head_dim, dtype=torch.bfloat16)
    decode_kv2 = torch.randn(bsz, 1, head_dim, dtype=torch.bfloat16)

    run_graph_test(
        module,
        [prefill_kv, decode_kv1, decode_kv2],
        # torch_options={"tt_legacy_compile": True},
        framework=Framework.TORCH,
        comparison_config=PCC_99,
    )


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


class _AttentionPrefillThenDoubleDecode(nn.Module):
    """Wrapper that runs prefill, then two decode steps."""

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn

    def forward(
        self, prefill_x: torch.Tensor, decode_x1: torch.Tensor, decode_x2: torch.Tensor
    ):
        # Run prefill to populate KV caches
        x = self.attn(prefill_x, 0)
        # Run first decode step
        # print("kv cache after prefill:")
        # print(self.attn.kv_cache.sum(dim=(0,2)))
        # torch_xla.sync()
        y1 = self.attn(decode_x1, prefill_x.shape[1])
        # Run second decode step
        # print("kv cache after first decode:")
        # print(self.attn.kv_cache.sum(dim=(0,2)))
        # torch_xla.sync()

        y2 = self.attn(decode_x2, prefill_x.shape[1] + 1)
        torch_xla.sync()
        print("kv cache after second decode:")
        print(self.attn.kv_cache)
        # print(self.attn.kv_cache.sum(dim=(0,2)))

        return y2


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("layer_id", [0, 1, 2], ids=["SWA", "CSA", "HSA"])
def test_attention_prefill_then_double_decode(layer_id):
    """Attention prefill followed by two decode steps: tests stateful attention."""
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    attn = model.layers[layer_id].attn
    init_weights(attn)

    bsz = 2
    prefill_seqlen = 4

    torch.manual_seed(42)
    prefill_x = torch.randn(bsz, prefill_seqlen, args.dim, dtype=torch.bfloat16)
    decode_x1 = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)
    decode_x2 = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)

    wrapper = _AttentionPrefillThenDoubleDecode(attn)

    run_graph_test(
        wrapper,
        [prefill_x, decode_x1, decode_x2],
        # torch_options={"tt_legacy_compile": True},
        framework=Framework.TORCH,
        comparison_config=PCC_99,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("layer_id", [0, 1, 2], ids=["SWA", "CSA", "HSA"])
@pytest.mark.parametrize("prefill_seqlen", [4, 7, 8, 9, 15, 16])
@pytest.mark.parametrize("bsz", [1, 2], ids=["partial_batch", "full_batch"])
def test_attention_prefill(layer_id, prefill_seqlen, bsz):
    """Attention prefill followed by two decode steps: tests stateful attention."""
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    attn = model.layers[layer_id].attn
    init_weights(attn)

    torch.manual_seed(42)
    prefill_x = torch.randn(bsz, prefill_seqlen, args.dim, dtype=torch.bfloat16)

    run_graph_test(
        attn,
        [prefill_x, 0],
        # torch_options={"tt_legacy_compile": True},
        framework=Framework.TORCH,
        comparison_config=PCC_99,
    )
    # print (attn.kv_cache)


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("layer_id", [0, 1, 2], ids=["SWA", "CSA", "HSA"])
@pytest.mark.parametrize("start_pos", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("bsz", [1, 2], ids=["partial_batch", "full_batch"])
def test_attention_decode(layer_id, start_pos, bsz):
    """Attention prefill followed by two decode steps: tests stateful attention."""
    xr.set_device_type("TT")

    args = small_args()
    model = make_model(args)
    attn = model.layers[layer_id].attn
    init_weights(attn)

    torch.manual_seed(42)
    prefill_x = torch.randn(bsz, 1, args.dim, dtype=torch.bfloat16)

    run_graph_test(
        attn,
        [prefill_x, start_pos],
        # torch_options={"tt_legacy_compile": True},
        framework=Framework.TORCH,
        comparison_config=PCC_99,
    )
