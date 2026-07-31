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


class TestSlidingWindowAdmissionBound:
    """``install_tt_sliding_window_admission`` clamps a sliding-window layer's
    per-request KV bound to the per-request prefill chunk.

    vLLM reads ``max_num_batched_tokens`` as the largest chunk one request can be
    scheduled; under TT chunked prefill it is the batch-wide budget
    (``chunk * max_num_seqs``) while AscendScheduler caps each request at
    ``chunk``, so the unclamped bound over-charges by up to ``max_num_seqs`` x.
    """

    @staticmethod
    def _spec():
        import torch
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        return SlidingWindowSpec(
            block_size=32,
            num_kv_heads=16,
            head_size=256,
            dtype=torch.bfloat16,
            sliding_window=1024,
        )

    @staticmethod
    def _publish(chunk, enabled=True):
        from vllm_tt.platform import publish_tt_per_request_prefill_chunk

        publish_tt_per_request_prefill_chunk(
            types.SimpleNamespace(
                tt_chunked_prefill_enabled=enabled, tt_prefill_chunk_size=chunk
            )
        )

    def test_clamps_to_the_per_request_chunk(self):
        """gemma-4-31B at max_model_len=65536, chunk 1024, batch 32."""
        import vllm_tt.platform as platform_mod

        spec = self._spec()
        platform_mod.install_tt_sliding_window_admission()
        try:
            self._publish(1024)
            clamped = spec.max_admission_blocks_per_request(
                max_num_batched_tokens=1024 * 32, max_model_len=65536
            )
            # cdiv(1023 + 1024, 32) + 1, not cdiv(1023 + 32768, 32) + 1 = 1057
            assert clamped == 65
        finally:
            self._publish(0, enabled=False)

    def test_is_a_noop_when_tt_chunked_prefill_is_off(self):
        import vllm_tt.platform as platform_mod

        spec = self._spec()
        platform_mod.install_tt_sliding_window_admission()
        self._publish(0, enabled=False)
        assert (
            spec.max_admission_blocks_per_request(
                max_num_batched_tokens=1024 * 32, max_model_len=65536
            )
            == 1057
        )

    def test_never_raises_the_bound(self):
        """Clamping is a min(): a chunk larger than the batch budget is inert."""
        import vllm_tt.platform as platform_mod

        spec = self._spec()
        platform_mod.install_tt_sliding_window_admission()
        try:
            self._publish(0, enabled=False)
            unclamped = spec.max_admission_blocks_per_request(
                max_num_batched_tokens=4096, max_model_len=65536
            )
            self._publish(1 << 20)  # chunk far above the budget
            assert (
                spec.max_admission_blocks_per_request(
                    max_num_batched_tokens=4096, max_model_len=65536
                )
                == unclamped
            )
        finally:
            self._publish(0, enabled=False)

    def test_install_is_idempotent(self):
        """Re-installing must not stack wrappers and shrink the bound twice."""
        import vllm_tt.platform as platform_mod

        spec = self._spec()
        platform_mod.install_tt_sliding_window_admission()
        platform_mod.install_tt_sliding_window_admission()
        try:
            self._publish(1024)
            assert (
                spec.max_admission_blocks_per_request(
                    max_num_batched_tokens=1024 * 32, max_model_len=65536
                )
                == 65
            )
        finally:
            self._publish(0, enabled=False)
