# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the narrowed paged_fill_cache page table.

The fill page table is allocated at the width ``paged_fill_cache`` actually
indexes rather than the read table's full ``max_model_len`` width, which makes
"the fill table always spans the step's tokens" a load-bearing invariant. These
tests pin that bound, the 32B stick alignment, the read-width clamp, and both
host-side guards (at width computation and per step), all without a device.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm_tt.model_runner import TTModelRunner

# Device-free, but the vLLM push matrix only runs device-marked jobs; tag so the
# single_device job collects these (they need no device and run in milliseconds).
pytestmark = [pytest.mark.push, pytest.mark.single_device]

# Wide enough that the read-table clamp is never the binding constraint.
_UNCLAMPED = 1 << 20

# (max_num_tokens, block_size). Token buckets are 1 (decode_only) or a power of
# two >= 32 (_get_token_paddings); block sizes are what the platform and the
# tests that override it actually use.
_BOUND_CASES = [
    (1, 32),  # decode_only=True -> num_tokens_paddings == [1]
    (32, 8),  # test_model_runner_buffer_keying's shrunken block size
    (32, 32),
    (128, 32),
    (2048, 32),
    (2048, 64),
    (2048, 128),
    (4096, 32),
    (131072, 32),  # single-shot prefill of a full 128K context
    (131072, 64),
]


def _runner(max_num_tokens, block_size, group_block_sizes=None):
    """The only runner state ``_fill_page_table_width`` reads."""
    return SimpleNamespace(
        max_num_tokens=max_num_tokens,
        block_size=block_size,
        _group_block_sizes=(
            [block_size] if group_block_sizes is None else group_block_sizes
        ),
    )


def _width(runner, group_idx=0, read_width=_UNCLAMPED):
    return TTModelRunner._fill_page_table_width(runner, group_idx, read_width)


@pytest.mark.parametrize("max_num_tokens,block_size", _BOUND_CASES)
def test_width_covers_the_largest_possible_step(max_num_tokens, block_size):
    """The bound paged_fill_cache's validator enforces on the device."""
    width = _width(_runner(max_num_tokens, block_size))
    assert width * block_size >= max_num_tokens


@pytest.mark.parametrize("max_num_tokens,block_size", _BOUND_CASES)
def test_width_is_stick_aligned_when_the_read_table_does_not_clamp(
    max_num_tokens, block_size
):
    """A ttnn page-table stick is 32B aligned: 8 int32 block ids."""
    width = _width(_runner(max_num_tokens, block_size))
    assert width > 0
    assert width % 8 == 0


@pytest.mark.parametrize("read_width", [64, 65, 4096, _UNCLAMPED])
def test_never_wider_than_the_read_table(read_width):
    """Widening past the read table would break the ``[:, :fill_width]`` copy."""
    assert _width(_runner(2048, 32), read_width=read_width) <= read_width


def test_clamped_to_the_read_width_when_the_read_table_is_narrower():
    """Sliding groups only address their window ring, which can be the narrower
    of the two -- then the clamp wins and the 8-alignment comes from the ring
    (``sliding_window_blocks`` already rounds to 8)."""
    assert _width(_runner(128, 32), read_width=4) == 4


@pytest.mark.parametrize("read_width", [1, 7, 8, 63])
def test_rejects_a_read_table_too_short_for_the_largest_step(read_width):
    """A ring shorter than the chunk must fail at startup, not on the device."""
    with pytest.raises(AssertionError, match="cannot cover"):
        _width(_runner(2048, 32), read_width=read_width)


def test_uses_each_groups_own_block_size():
    """Hybrid KV caches can assign a different block_size per group; a larger
    block means fewer page-table entries for the same token count."""
    runner = _runner(2048, 32, group_block_sizes=[32, 128])
    assert _width(runner, group_idx=0) == 64  # cdiv(2048, 32)
    assert _width(runner, group_idx=1) == 16  # cdiv(2048, 128)


def test_falls_back_to_the_base_block_size_for_an_unknown_group():
    """The base block_size is the smallest, so the fallback errs wide (safe)."""
    runner = _runner(2048, 32, group_block_sizes=[32])
    assert _width(runner, group_idx=5) == _width(runner, group_idx=0)


def test_long_context_fill_table_is_far_narrower_than_the_read_table():
    """The case this narrowing exists for: at max_model_len=131072 / block_size=32
    the read table is 4096 entries, of which a 2048-token chunk indexes 64."""
    read_width = 131072 // 32
    assert _width(_runner(2048, 32), read_width=read_width) == 64


def test_copy_narrow_stages_the_leading_columns():
    """``_prepare_inputs`` rolls the chunk's blocks to the front, so the leading
    columns are exactly the ones paged_fill_cache wants."""
    host = torch.arange(2 * 16, dtype=torch.int32).reshape(2, 16)
    dev_buf = torch.zeros((2, 8), dtype=torch.int32)
    TTModelRunner._copy_narrow_fill_page_table(
        None, dev_buf, host, block_size=32, step_tokens=8 * 32, group_idx=0
    )
    assert torch.equal(dev_buf, host[:, :8])


def test_copy_narrow_rejects_a_table_too_short_for_the_step():
    """Fail on the host, where the page table is, instead of inside tt-metal."""
    host = torch.zeros((2, 16), dtype=torch.int32)
    dev_buf = torch.zeros((2, 8), dtype=torch.int32)
    with pytest.raises(AssertionError, match="cannot cover this step"):
        TTModelRunner._copy_narrow_fill_page_table(
            None, dev_buf, host, block_size=32, step_tokens=8 * 32 + 1, group_idx=0
        )
