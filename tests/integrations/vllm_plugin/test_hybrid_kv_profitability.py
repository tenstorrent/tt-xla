# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the hybrid-KV profitability gate.

``TTPlatform._maybe_disable_unprofitable_hybrid_kv_cache`` opts out of vLLM's
hybrid KV cache manager in the regime where a per-user sliding ring costs more
than plain full attention (see that method and ``sliding_ring_is_profitable``);
these tests pin the decision boundary and the cases it must leave alone.

Pure config manipulation -- no device, no engine.
"""

import types

import pytest
from vllm_tt.platform import TTPlatform

pytestmark = [pytest.mark.push, pytest.mark.cpu]


def _gate(
    max_model_len,
    sliding_window,
    block_size=32,
    disable_hybrid=None,
    original_max_model_len=None,
):
    """Run the gate over a stand-in for the slice of VllmConfig it reads.

    Returns whether it disabled the hybrid KV cache manager.
    """
    cfg = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            max_model_len=max_model_len,
            original_max_model_len=original_max_model_len or max_model_len,
            get_sliding_window=lambda: sliding_window,
        ),
        scheduler_config=types.SimpleNamespace(
            disable_hybrid_kv_cache_manager=disable_hybrid
        ),
        cache_config=types.SimpleNamespace(block_size=block_size),
    )
    TTPlatform._maybe_disable_unprofitable_hybrid_kv_cache(cfg)
    return bool(cfg.scheduler_config.disable_hybrid_kv_cache_manager)


# gemma-4-31B geometry: sliding_window=1024, block_size=32. A ring needs
# align8(cdiv(min(window, max_model_len), 32) + 1) blocks per user per layer
# against cdiv(max_model_len, 32) for full attention, so the ring only wins
# once max_model_len exceeds the window.
@pytest.mark.parametrize(
    "max_model_len,expect_disabled",
    [
        (128, True),  # ring 8 vs full 4 -- the reported regression
        (512, True),  # ring 24 vs full 16
        (1024, True),  # ring 40 vs full 32 -- window == max_model_len
        (2048, False),  # ring 40 vs full 64 -- window finally clips
        (131072, False),  # ring 40 vs full 4096 -- what the ring exists for
    ],
)
def test_gate_follows_the_ring_vs_full_crossover(max_model_len, expect_disabled):
    assert (
        _gate(max_model_len=max_model_len, sliding_window=1024) is expect_disabled
    ), f"wrong decision at max_model_len={max_model_len}"


def test_no_sliding_layers_leaves_hybrid_alone():
    """Nothing to trade off -- vLLM already emits a single full-attention group."""
    assert _gate(max_model_len=128, sliding_window=None) is False


@pytest.mark.parametrize("explicit", [True, False])
def test_explicit_user_choice_is_respected(explicit):
    """An explicit --[no-]disable-hybrid-kv-cache-manager wins in both
    directions, even where the gate would decide the other way."""
    # max_model_len=128 is where the gate would otherwise disable hybrid.
    assert _gate(max_model_len=128, sliding_window=1024, disable_hybrid=explicit) is (
        explicit
    )


def test_auto_fit_max_model_len_keeps_the_ring():
    """max_model_len == -1 means vLLM auto-fits it later from the KV budget, so
    the value present now is a placeholder and must not drive the decision."""
    assert (
        _gate(max_model_len=128, sliding_window=1024, original_max_model_len=-1)
        is False
    )


def test_larger_block_size_shifts_the_boundary():
    """The comparison is in blocks, so block_size moves the crossover: at
    block_size=256 a 1024-token window is 8 blocks (4+1 rounded to 8) while
    full attention at max_model_len=2048 is 8 blocks -- not yet a win."""
    assert _gate(max_model_len=2048, sliding_window=1024, block_size=256) is True
    assert _gate(max_model_len=4096, sliding_window=1024, block_size=256) is False
