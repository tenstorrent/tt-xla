# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the worker's sliding-window ring reservation.

``sliding_ring_reserve_for_spec`` must reserve exactly the rings the model runner
later allocates, and the runner keys off spec type. The trap these tests pin:
unification early-returns on a *uniform* spec dict, so an all-sliding model keeps
its SlidingWindowSpecs and its rings are real even with the hybrid manager
disabled -- skipping on the flag alone would under-reserve it, while reserving
unconditionally strands the bytes for a mixed model.

Pure spec manipulation -- no device, no engine.
"""

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm_tt.swa_cache_utils import sliding_ring_reserve_bytes, sliding_window_blocks
from vllm_tt.worker import sliding_ring_reserve_for_spec

pytestmark = [pytest.mark.push, pytest.mark.cpu]

MAX_NUM_REQS = 32
MAX_MODEL_LEN = 128
SLIDING_WINDOW = 1024


def _sliding(**kwargs):
    # gemma-4-31B sliding-layer geometry: 16 kv heads x head_size 256.
    return SlidingWindowSpec(
        block_size=32,
        num_kv_heads=16,
        head_size=256,
        dtype=torch.bfloat16,
        sliding_window=SLIDING_WINDOW,
        **kwargs,
    )


def _full():
    # gemma-4-31B full-attention geometry: 4 kv heads x head_size 512.
    return FullAttentionSpec(
        block_size=32, num_kv_heads=4, head_size=512, dtype=torch.bfloat16
    )


def _reserve(spec, hybrid_disabled):
    return sliding_ring_reserve_for_spec(
        spec, MAX_NUM_REQS, MAX_MODEL_LEN, hybrid_disabled
    )


def _expected_bytes(n_sliding):
    window_blocks = sliding_window_blocks(SLIDING_WINDOW, 32, MAX_MODEL_LEN)
    return n_sliding * sliding_ring_reserve_bytes(
        window_blocks, MAX_NUM_REQS, _sliding().page_size_bytes
    )


def _mixed_spec():
    """gemma-4-31B shape in miniature: sliding layers interleaved with full."""
    return {f"sliding.{i}": _sliding() for i in range(5)} | {
        f"full.{i}": _full() for i in range(2)
    }


def _all_sliding_spec():
    """Mistral shape: every layer sliding, so the dict is uniform."""
    return {f"sliding.{i}": _sliding() for i in range(5)}


def test_mixed_model_reserves_rings_when_hybrid_is_on():
    """The hybrid path is live, the runner allocates rings, so reserve them."""
    reserve, num_sliding = _reserve(_mixed_spec(), hybrid_disabled=False)
    assert num_sliding == 5
    assert reserve == _expected_bytes(5)


def test_mixed_model_reserves_nothing_when_hybrid_is_disabled():
    """Unification rewrites every SlidingWindowSpec to FullAttentionSpec, so the
    runner allocates no rings -- reserving for them would strand the bytes."""
    assert _reserve(_mixed_spec(), hybrid_disabled=True) == (0, 0)


def test_all_sliding_model_reserves_rings_when_hybrid_is_on():
    reserve, num_sliding = _reserve(_all_sliding_spec(), hybrid_disabled=False)
    assert num_sliding == 5
    assert reserve == _expected_bytes(5)


def test_all_sliding_model_still_reserves_when_hybrid_is_disabled():
    """The trap: unification early-returns on a uniform spec dict, so these
    specs survive and the runner really does allocate rings. Keying the skip on
    the flag instead of the post-unification spec would under-reserve here."""
    reserve, num_sliding = _reserve(_all_sliding_spec(), hybrid_disabled=True)
    assert num_sliding == 5, (
        "an all-sliding model keeps its SlidingWindowSpecs through unification; "
        "its rings are real and must still be reserved"
    )
    assert reserve == _expected_bytes(5)


def test_no_sliding_layers_reserves_nothing():
    full_only = {f"full.{i}": _full() for i in range(3)}
    assert _reserve(full_only, hybrid_disabled=False) == (0, 0)
    assert _reserve(full_only, hybrid_disabled=True) == (0, 0)


def test_caller_spec_is_not_mutated():
    """Unification runs over a copy: the worker is handed the runner's own spec
    dict and must not rewrite it underneath."""
    spec = _mixed_spec()
    _reserve(spec, hybrid_disabled=True)
    assert all(isinstance(spec[f"sliding.{i}"], SlidingWindowSpec) for i in range(5))
