# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from benchmark.utils import compute_pcc
from infra.utilities.types import Framework
from torch_xla.distributed.spmd import Mesh

from tests.infra.testers.single_chip.op.op_tester import OpTester, run_op_test

# TODO: Record superset properties for these tests.


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("num_heads", [12, 16])
@pytest.mark.parametrize("max_seq_len", [64, 128])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("seq_len_to_fill", [32, 64])
def test_fill_cache(num_heads, max_seq_len, head_size, seq_len_to_fill):

    cache = torch.zeros(1, num_heads, max_seq_len, head_size, dtype=torch.bfloat16)
    fill_value = torch.randn(
        1, num_heads, seq_len_to_fill, head_size, dtype=torch.bfloat16
    )

    run_op_test(
        torch.ops.tt.fill_cache, [cache, fill_value, 0], framework=Framework.TORCH
    )


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("num_heads", [12, 16])
@pytest.mark.parametrize("max_seq_len", [64, 128])
@pytest.mark.parametrize("head_size", [64, 128])
def test_update_cache(num_heads, max_seq_len, head_size):

    cache = torch.zeros(1, num_heads, max_seq_len, head_size, dtype=torch.bfloat16)
    fill_value = torch.randn(1, num_heads, 1, head_size, dtype=torch.bfloat16)

    cache_position = torch.tensor([10], dtype=torch.int32)

    run_op_test(
        torch.ops.tt.update_cache,
        [cache, fill_value, cache_position, 0],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_size, num_kv_heads, max_seq_len, is_causal, scale, sliding_window_size",
    [
        (1, 12, 32, 128, 12, 32, True, 1.0, 16),
        (1, 12, 32, 128, 12, 128, False, 1.0, 16),
        (8, 12, 32, 128, 12, 32, True, 1.0, 32),
        (8, 12, 32, 128, 12, 128, False, 1.0, 32),
        (1, 12, 32, 128, 4, 32, True, 1.0, 16),
        (1, 12, 32, 128, 4, 128, False, 1.0, 16),
        (8, 12, 32, 128, 4, 32, True, 1.0, 32),
        (8, 12, 32, 128, 4, 128, False, 1.0, 32),
    ],
)
def test_scaled_dot_product_attention(
    batch_size,
    num_heads,
    seq_len,
    head_size,
    num_kv_heads,
    max_seq_len,
    is_causal,
    scale,
    sliding_window_size,
):

    query = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=torch.bfloat16)
    key = torch.randn(
        batch_size, num_kv_heads, max_seq_len, head_size, dtype=torch.bfloat16
    )
    value = torch.randn(
        batch_size, num_kv_heads, max_seq_len, head_size, dtype=torch.bfloat16
    )
    attn_mask = (
        torch.randn(batch_size, 1, seq_len, max_seq_len, dtype=torch.bfloat16)
        if not is_causal
        else None
    )

    run_op_test(
        torch.ops.tt.scaled_dot_product_attention,
        [query, key, value, attn_mask, is_causal, scale, sliding_window_size],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "batch_size, num_heads, head_size, num_kv_heads, max_seq_len, is_causal, scale",
    [
        (1, 12, 128, 12, 32, True, 1.0),
        (1, 12, 128, 12, 32, False, 1.0),
        (1, 12, 128, 12, 128, False, 1.0),
        (8, 12, 128, 12, 32, True, 1.0),
        (8, 12, 128, 12, 128, False, 1.0),
        (1, 12, 128, 4, 32, True, 1.0),
        (1, 12, 128, 4, 32, False, 1.0),
        (1, 12, 128, 4, 128, False, 1.0),
        (8, 12, 128, 4, 32, True, 1.0),
        (8, 12, 128, 4, 128, False, 1.0),
    ],
)
def test_scaled_dot_product_attention_decode(
    batch_size, num_heads, head_size, num_kv_heads, max_seq_len, is_causal, scale
):

    query = torch.randn(1, batch_size, num_heads, head_size, dtype=torch.bfloat16)
    key = torch.randn(
        batch_size, num_kv_heads, max_seq_len, head_size, dtype=torch.bfloat16
    )
    value = torch.randn(
        batch_size, num_kv_heads, max_seq_len, head_size, dtype=torch.bfloat16
    )
    cur_pos_tensor = torch.arange(batch_size, dtype=torch.int32)
    attn_mask = (
        torch.randn(batch_size, 1, num_heads, max_seq_len, dtype=torch.bfloat16)
        if not is_causal
        else None
    )

    run_op_test(
        torch.ops.tt.scaled_dot_product_attention_decode,
        [query, key, value, cur_pos_tensor, attn_mask, None, is_causal, scale],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("num_users", [8, 16, 24, 32])
@pytest.mark.parametrize("max_num_blocks_per_seq", [16, 32])
@pytest.mark.parametrize("num_heads", [1, 8, 32])
@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_paged_update_cache(
    num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim
):
    max_num_blocks = max_num_blocks_per_seq * num_users
    max_seq_len = max_num_blocks_per_seq * block_size

    cache = torch.zeros(
        max_num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    fill_value = torch.randn(1, num_users, num_heads, head_dim, dtype=torch.bfloat16)

    # Create arbitrary update indices
    cache_idxs = torch.randperm(max_seq_len)[:num_users]
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(num_users, max_num_blocks_per_seq).to(
        torch.int32
    )

    run_op_test(
        torch.ops.tt.paged_update_cache,
        [cache, fill_value, cache_idxs, page_table, False],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("num_users", [8, 16])
@pytest.mark.parametrize("max_num_blocks_per_seq", [16, 32])
@pytest.mark.parametrize("num_heads", [1, 8, 32])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("seq_len_to_fill", [10, 20, 32, 50, 70])
def test_paged_fill_cache(
    num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim, seq_len_to_fill
):
    max_num_blocks = max_num_blocks_per_seq * num_users

    cache = torch.zeros(
        max_num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    fill_value = torch.randn(
        1, num_heads, seq_len_to_fill, head_dim, dtype=torch.bfloat16
    )

    # Create arbitrary page table
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(num_users, max_num_blocks_per_seq).to(
        torch.int32
    )

    batch_idx = torch.randint(0, num_users, (1,), dtype=torch.int32)

    run_op_test(
        torch.ops.tt.paged_fill_cache,
        [cache, fill_value, page_table, batch_idx],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("num_users", [8])
@pytest.mark.parametrize("max_num_blocks_per_seq", [16, 32])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("head_dim", [128])
def test_paged_scaled_dot_product_attention_decode(
    num_users, max_num_blocks_per_seq, num_heads, block_size, head_dim
):
    max_num_blocks = max_num_blocks_per_seq * num_users

    query = torch.randn(1, num_users, num_heads, head_dim, dtype=torch.bfloat16)
    key = torch.randn(
        max_num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    value = torch.randn(
        max_num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    page_table = torch.ones(num_users, max_num_blocks_per_seq).to(torch.int32)
    cur_pos_tensor = torch.ones(num_users).to(torch.int32)

    run_op_test(
        torch.ops.tt.paged_scaled_dot_product_attention_decode,
        [query, key, value, page_table, True, None, cur_pos_tensor],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_size, num_kv_heads, head_dim_v, has_value, is_causal, scale",
    [
        # MLA-from-latent (value=None, head_dim_v < head_size, d_rope=64)
        (1, 16, 64, 192, 1, 128, False, True, 1.0),
        (1, 16, 64, 192, 1, 128, False, False, 1.0),
        (2, 16, 128, 192, 1, 128, False, True, 1.0),
        (2, 16, 128, 192, 1, 128, False, False, 1.0),
        # MLA-from-latent with d_rope=0 (head_dim_v == head_size)
        (1, 32, 64, 128, 1, 128, False, True, 1.0),
        (1, 32, 64, 128, 1, 128, False, False, 1.0),
        # Explicit value tensor path
        (1, 16, 64, 128, 1, 64, True, True, 1.0),
        (2, 16, 128, 128, 1, 64, True, False, 1.0),
    ],
)
def test_flash_mla_prefill(
    batch_size,
    num_heads,
    seq_len,
    head_size,
    num_kv_heads,
    head_dim_v,
    has_value,
    is_causal,
    scale,
):

    query = torch.randn(batch_size, num_heads, seq_len, head_size, dtype=torch.bfloat16)
    key = torch.randn(
        batch_size, num_kv_heads, seq_len, head_size, dtype=torch.bfloat16
    )
    value = (
        torch.randn(batch_size, num_kv_heads, seq_len, head_dim_v, dtype=torch.bfloat16)
        if has_value
        else None
    )
    attn_mask = (
        torch.randn(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)
        if not is_causal
        else None
    )

    run_op_test(
        torch.ops.tt.flash_mla_prefill,
        [query, key, head_dim_v, value, attn_mask, is_causal, scale],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "num_users, num_heads, num_kv_heads, head_size, head_dim_v, has_value, block_size, max_num_blocks_per_seq, is_causal, scale",
    [
        # MLA-from-latent (value=None): single shared latent KV head, the Q/K
        # head dim carries the rope tail (dh_qk = head_dim_v + d_rope = 128 + 64).
        (8, 16, 1, 192, 128, False, 32, 16, True, 1.0),
        (8, 16, 1, 192, 128, False, 64, 16, True, 1.0),
        # MLA-from-latent with d_rope=0 (dh_qk == head_dim_v).
        (8, 32, 1, 128, 128, False, 32, 16, True, 1.0),
        # Explicit value tensor path (separate V cache) with a single latent KV
        # head broadcast across the query heads.
        (8, 16, 1, 128, 64, True, 32, 16, True, 1.0),
        # GQA latent: a few KV heads, each shared by a group of query heads.
        (8, 16, 4, 192, 128, False, 32, 16, True, 1.0),
        # MHA sanity (num_kv_heads == num_heads), explicit value.
        (8, 8, 8, 128, 128, True, 32, 16, True, 1.0),
    ],
)
def test_paged_flash_mla_decode(
    num_users,
    num_heads,
    num_kv_heads,
    head_size,
    head_dim_v,
    has_value,
    block_size,
    max_num_blocks_per_seq,
    is_causal,
    scale,
):
    max_num_blocks = max_num_blocks_per_seq * num_users
    max_seq_len = max_num_blocks_per_seq * block_size

    query = torch.randn(1, num_users, num_heads, head_size, dtype=torch.bfloat16)
    key = torch.randn(
        max_num_blocks, num_kv_heads, block_size, head_size, dtype=torch.bfloat16
    )
    value = (
        torch.randn(
            max_num_blocks, num_kv_heads, block_size, head_dim_v, dtype=torch.bfloat16
        )
        if has_value
        else None
    )

    # A valid page table: distinct, non-overlapping physical blocks per user.
    page_table = (
        torch.randperm(max_num_blocks)
        .reshape(num_users, max_num_blocks_per_seq)
        .to(torch.int32)
    )
    # Current decode position per user, kept within the cached range.
    cur_pos_tensor = torch.randint(0, max_seq_len, (num_users,), dtype=torch.int32)

    # args = [query, key, head_dim_v, page_table, value, is_causal,
    #         attn_mask, cur_pos_tensor, attention_sink, scale]
    run_op_test(
        torch.ops.tt.paged_flash_mla_decode,
        [
            query,
            key,
            head_dim_v,
            page_table,
            value,
            is_causal,
            None,
            cur_pos_tensor,
            None,
            scale,
        ],
        framework=Framework.TORCH,
    )


def _indexer_score_comparator(device_output, golden_output, args, kwargs):
    """Compare tt.indexer_score outputs, accounting for the causal -inf mask.

    Masked (future) key positions are ``-inf`` in the golden. Comparing them
    directly would poison PCC (the mean of a tensor containing -inf is NaN), so
    we (1) compute PCC over the visible (finite) positions only and (2) verify
    the device drives the masked positions to something top-k could never select
    (non-finite, or strictly below the smallest visible score).
    """
    device_output = device_output.cpu().to(torch.float32)
    golden_output = golden_output.to(torch.float32)

    masked = torch.isneginf(golden_output)
    visible = ~masked

    pcc = compute_pcc(golden_output[visible], device_output[visible])
    assert pcc >= 0.99, f"indexer_score visible-position PCC too low: {pcc}"

    if masked.any():
        # Masked (future) positions must never surface as competitive scores: the
        # device has to drive them non-finite or strictly below every visible
        # score so a downstream top-k can never select them. (An all-reduce over
        # the -inf mask on the sharded path may yield NaN on hardware; a NaN at a
        # masked position is still "not selectable", so it is tolerated.)
        visible_min = device_output[visible].min()
        masked_vals = device_output[masked]
        masked_out = (
            torch.isneginf(masked_vals)
            | torch.isnan(masked_vals)
            | (masked_vals < visible_min)
        )
        assert torch.all(
            masked_out
        ), "indexer_score failed to mask future key positions on device."


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "num_heads, query_seq_len, key_seq_len, head_dim, chunk_start_idx",
    # Every case uses a distinct (num_heads, Sq, T, D) shape. Two programs that
    # share operand shapes but differ only in chunk_start_idx (a custom-call
    # frontend attribute) alias in torch_xla's compiled-executable cache, so
    # keeping shapes unique keeps each parametrization independent.
    [
        # Causal square (chunk_start_idx=0): strict lower-triangular visibility.
        (4, 32, 32, 64, 0),
        (8, 32, 32, 128, 0),
        # More keys than queries with a chunk offset (prefill of a later chunk).
        (4, 32, 64, 64, 16),
        (8, 64, 128, 128, 32),
        # Fully visible: chunk_start_idx >= key_seq_len leaves no masking.
        (12, 32, 32, 96, 100),
        # Larger indexer-head count / sequence.
        (16, 64, 64, 128, 0),
    ],
)
def test_indexer_score(
    num_heads, query_seq_len, key_seq_len, head_dim, chunk_start_idx
):
    # ttnn.experimental.indexer_score requires batch == 1.
    batch = 1

    query = torch.randn(batch, num_heads, query_seq_len, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch, 1, key_seq_len, head_dim, dtype=torch.bfloat16)
    weights = torch.randn(batch, num_heads, query_seq_len, 1, dtype=torch.bfloat16)

    run_op_test(
        torch.ops.tt.indexer_score,
        [query, key, weights, chunk_start_idx],
        framework=Framework.TORCH,
        custom_comparator=_indexer_score_comparator,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
@pytest.mark.parametrize(
    "num_heads, query_seq_len, key_seq_len, head_dim, chunk_start_idx",
    # num_heads is divisible by 8 so the head split is valid for 2-, 4- and
    # 8-device meshes. Each case has a distinct (num_heads, Sq, T, D) shape (see
    # the note on test_indexer_score) so the executable cache never aliases two
    # different chunk_start_idx values onto one program.
    [
        # Causal square (chunk_start_idx=0): masking is over Sq/T, both of which
        # stay unsharded, so the head split does not perturb it.
        (8, 32, 32, 64, 0),
        # Prefill of a later chunk: more keys than queries, nonzero offset.
        (8, 32, 64, 128, 16),
        # More keys, larger head count, nonzero offset.
        (16, 64, 128, 128, 32),
        # Fully visible (no masking): chunk_start_idx >= key_seq_len.
        (16, 32, 32, 128, 1000),
        # Larger query/key sequence, causal square.
        (8, 64, 64, 64, 0),
    ],
)
def test_indexer_score_tensor_parallel(
    num_heads, query_seq_len, key_seq_len, head_dim, chunk_start_idx
):
    """Tensor-parallel indexer_score, mirroring DeepSeek-V3.2 DSA sharding.

    In production the lightning indexer's many heads are split across the
    tensor-parallel ("model") axis while the single shared indexer key is
    replicated (see the DeepSeek-V3.2-exp indexer sharding: ``wq_b``/
    ``weights_proj`` are ``model``-sharded, ``wk``/``k_cache`` are not). The op
    reduces over the head dim, so sharding it turns the head-sum into a
    cross-device all-reduce.

    Layout (batch is 1 as the ttnn op requires, so only the head axis is split):
        query   [1, Hi, Sq, D] -> heads sharded on "model"
        weights [1, Hi, Sq, 1] -> heads sharded on "model"
        key     [1, 1,  T,  D] -> replicated (single shared kv-head)
        output  [1, 1,  Sq, T] -> head dim reduced -> all-reduce -> replicated
    """
    num_devices = xr.global_runtime_device_count()
    assert (
        num_heads % num_devices == 0
    ), f"num_heads ({num_heads}) must be divisible by device count ({num_devices})."

    # ttnn.experimental.indexer_score requires batch == 1.
    batch = 1

    query = torch.randn(batch, num_heads, query_seq_len, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch, 1, key_seq_len, head_dim, dtype=torch.bfloat16)
    weights = torch.randn(batch, num_heads, query_seq_len, 1, dtype=torch.bfloat16)

    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(args, kwargs):
        q, k, w = args[0], args[1], args[2]
        return {
            # Split the indexer heads across the tensor-parallel axis.
            q: (None, "model", None, None),
            w: (None, "model", None, None),
            # The single shared kv-head is replicated across the heads.
            k: (None, None, None, None),
        }

    run_op_test(
        torch.ops.tt.indexer_score,
        [query, key, weights, chunk_start_idx],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        custom_comparator=_indexer_score_comparator,
    )


def _topk_large_indices_comparator(device_output, golden_output, args, kwargs):
    """Compare tt.topk_large_indices outputs by the *values* their indices select.

    The op returns only indices. When several elements tie (common in bf16, which
    has 8 mantissa bits) the order — and even the choice of representative — among
    the tied indices is unspecified, so positionally comparing the returned index
    labels against the torch golden is not a meaningful correctness metric (same
    rationale as the tt-mlir golden test, which skips PCC for this op). Instead we
    (1) check every returned index is a valid position along the row, and (2)
    gather the input values at the device and golden indices and require the
    sorted top-k value sets to match. The k-largest value multiset is a property
    of the input alone, so it is invariant to how ties are broken.
    """
    input_tensor = args[0].cpu().to(torch.float32)
    n = input_tensor.shape[-1]

    device_idx = device_output.cpu().to(torch.int64)
    golden_idx = golden_output.cpu().to(torch.int64)

    assert device_idx.shape == golden_idx.shape, (
        f"index shape mismatch: device {tuple(device_idx.shape)} vs "
        f"golden {tuple(golden_idx.shape)}."
    )
    assert torch.all(
        (device_idx >= 0) & (device_idx < n)
    ), "topk_large_indices returned an out-of-range index."

    # Gather the selected values and compare the sorted top-k value sets. A
    # different tie-break (order or representative) leaves the value set unchanged.
    device_vals = torch.gather(input_tensor, -1, device_idx).sort(dim=-1).values
    golden_vals = torch.gather(input_tensor, -1, golden_idx).sort(dim=-1).values
    assert torch.allclose(
        device_vals, golden_vals
    ), "topk_large_indices selected a different set of top-k values than the golden."


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "shape, k",
    [
        # N > k exercises a real top-k selection (not a full-row sort).
        ((2, 64), 16),
        ((4, 256), 64),
        ((8, 128), 32),
        ((1, 512), 128),
        # N == k selects (sorts) the whole row.
        ((2, 32), 32),
        # 3D input: leading dims are preserved, top-k is over the last dim.
        ((2, 4, 128), 48),
    ],
)
def test_topk_large_indices(shape, k):
    # ttnn.experimental.topk_large_indices requires a row-major bfloat16 input.
    input = torch.randn(shape, dtype=torch.bfloat16)

    run_op_test(
        torch.ops.tt.topk_large_indices,
        [input, k],
        framework=Framework.TORCH,
        custom_comparator=_topk_large_indices_comparator,
    )


@pytest.mark.nightly
@pytest.mark.dual_chip
@pytest.mark.parametrize(
    "num_rows, n, k",
    # num_rows is divisible by 8 so the row split is valid for 2-, 4- and
    # 8-device meshes.
    [
        (8, 64, 16),
        (8, 128, 32),
        (8, 256, 64),
        (16, 512, 128),
    ],
)
def test_topk_large_indices_data_parallel(num_rows, n, k):
    """Data-parallel topk_large_indices: split the rows across devices.

    topk_large_indices needs the whole row on one device (it selects over the last
    dimension N), so the natural parallelism is data parallelism over the leading
    (row) dimension: each device runs the op on its own row-shard with no
    collective (see tt-mlir's registered sharding rule for tt.topk_large_indices,
    which keeps N replicated and inserts no all-gather/all-reduce).

    Layout:
        input   [num_rows, N] -> rows sharded on "batch", N replicated
        output  [num_rows, k] -> rows sharded on "batch" (no reduction)
    """
    num_devices = xr.global_runtime_device_count()
    assert (
        num_rows % num_devices == 0
    ), f"num_rows ({num_rows}) must be divisible by device count ({num_devices})."

    input = torch.randn(num_rows, n, dtype=torch.bfloat16)

    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(args, kwargs):
        input_tensor = args[0]
        return {
            # Split the rows across the data-parallel axis; N stays replicated.
            input_tensor: ("batch", None),
        }

    run_op_test(
        torch.ops.tt.topk_large_indices,
        [input, k],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        custom_comparator=_topk_large_indices_comparator,
    )
