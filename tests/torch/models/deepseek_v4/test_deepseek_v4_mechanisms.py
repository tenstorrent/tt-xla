# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# A smaller study of deepseek v4 mechanics that fail in weird ways.

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch import nn

PCC_99 = ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.99))


class _CircularKVCacheUpdate(nn.Module):
    """Standalone module that implements circular buffer KV cache update logic."""

    def __init__(self, window_size: int, head_dim: int, max_batch_size: int = 2):
        super().__init__()
        self.win = window_size
        self.register_buffer(
            "kv_cache",
            torch.ones(max_batch_size, window_size, head_dim, dtype=torch.bfloat16),
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

        return self.kv_cache[:bsz]

    # index copy based -- might be better?
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


# Decode tests will pass if:
# 1. run with forked
# 2. run before prefill tests
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(
    "seqlen,window_size,start_pos",
    [
        # Decode tests (seqlen=1, various positions)
        # (1, 8, 0),  # Decode at position 0
        (1, 8, 4),  # Decode at position 4
        (1, 8, 7),  # Decode at position 7 (last in window)
        (1, 8, 8),  # Decode at position 8 (wraps to 0)
        (1, 8, 15),  # Decode at position 15 (wraps to 7)
        # Prefill tests (start_pos=0)
        (4, 8, 0),  # seqlen < window_size
        (8, 8, 0),  # seqlen == window_size
        (12, 8, 0),  # seqlen > window_size (cutoff=4)
        (16, 8, 0),  # seqlen > window_size (cutoff=0, full wrap)
        (20, 8, 0),  # seqlen > window_size (cutoff=4)
    ],
    ids=[
        # "decode_pos0", # this is a mini prefill.
        "decode_pos4",
        "decode_pos7",
        "decode_wrap0",
        "decode_wrap7",
        "prefill_short",
        "prefill_exact",
        "prefill_wrap_partial",
        "prefill_wrap_full",
        "prefill_wrap_multi",
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
        # print("prefill", self.kv_cache_module.kv_cache.sum(dim=(0, 2)))

        # First decode: add one token at position prefill_seqlen
        self.kv_cache_module(decode_kv1, prefill_seqlen)
        # print("decode1", self.kv_cache_module.kv_cache.sum(dim=(0, 2)))

        # Second decode: add another token at position prefill_seqlen + 1
        final_cache = self.kv_cache_module(decode_kv2, prefill_seqlen + 1)
        # print("decode2", self.kv_cache_module.kv_cache.sum(dim=(0, 2)))

        return final_cache


# All fail
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


@pytest.mark.parametrize("start_pos", [0, 4, 7], ids=["pos0", "pos4", "pos7"])
def test_kv_cache_update_decode_only(start_pos):
    """Test decode operations at various positions.

    Each position gets its own independent test run via pytest parametrization
    to avoid graph caching contamination between tests.

    This is the simplest evidence of crosstalk between the prefill and decode tests.
    It can be avoided by running the pos0 test not first, or with --forked
    """
    xr.set_device_type("TT")
    device = xm.xla_device()

    window_size = 8
    head_dim = 32
    bsz = 1
    max_batch_size = 2

    # Use same seed for all positions
    torch.manual_seed(42)
    kv_input = (
        torch.arange(bsz * head_dim, dtype=torch.bfloat16)
        .view(bsz, 1, head_dim)
        .to(device)
        + 1
    )

    print(f"\nInput KV (1 token):")
    print(kv_input)

    # Fresh module
    module = _CircularKVCacheUpdate(window_size, head_dim, max_batch_size).to(device)
    compiled_module = torch.compile(module, backend="tt")

    # Run on device
    result = compiled_module(kv_input, start_pos)
    torch_xla.sync()

    print(f"\nKV cache after decode (position {start_pos}):")
    print(result)

    # Verify the written position matches input
    assert torch.allclose(
        result[:, start_pos % window_size, :], kv_input
    ), f"Position {start_pos % window_size} should contain input KV"
