# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for _partition_spec_to_sdy_sharding (no hardware required).

Regression guard for the 1xN-mesh mesh_idx bug: tt-mlir's
replaceMeshIdxPlaceholders (ShardyUtils.cpp) resolves mesh_idx_i positionally
over ALL mesh axes (size-1 axes are kept), so the placeholder index must equal
the axis's RAW index. A previous "non-degenerate-only" numbering pinned
vocab-on-"model" to the size-1 leading axis on a 1xN mesh, producing an
unlowerable reshard (Shardy CollectiveInserter assertion) across multi-device
TP sampling.
"""

import types

import pytest

from tt_torch.sharding import _partition_spec_to_sdy_sharding


def _mesh(shape):
    # Mirrors xs.Mesh's attribute surface used by the function under test.
    return types.SimpleNamespace(axis_names=("batch", "model"), mesh_shape=shape)


# axis_names = ("batch", "model"): batch is axis 0, model is axis 1.
TP_MESHES = {
    "n300": (1, 2),
    "qb2": (1, 4),
    "galaxy": (1, 32),
    "dp_tp_2d": (2, 4),
}


@pytest.mark.parametrize("shape", TP_MESHES.values(), ids=TP_MESHES.keys())
@pytest.mark.parametrize(
    "spec",
    [
        (None, "model"),
        (None, "model", None, None),
        (None, None, "model"),
    ],
)
def test_model_axis_maps_to_raw_index_1(shape, spec):
    """'model' (raw axis index 1) must emit mesh_idx_1, never mesh_idx_0.

    mesh_idx_1 is what tt-mlir resolves to the size>1 model axis; mesh_idx_0
    would resolve to the (size-1 on 1xN) leading axis -- the original bug.
    """
    out = _partition_spec_to_sdy_sharding(_mesh(shape), spec)
    assert '"mesh_idx_1"' in out
    assert '"mesh_idx_0"' not in out


@pytest.mark.parametrize("shape", TP_MESHES.values(), ids=TP_MESHES.keys())
def test_fully_replicated_emits_no_placeholder(shape):
    out = _partition_spec_to_sdy_sharding(_mesh(shape), (None, None))
    assert "mesh_idx" not in out
    assert out == "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"


def test_size1_axis_is_replicated_not_sharded():
    # On a 1xN mesh, sharding the degenerate "batch" axis must replicate.
    out = _partition_spec_to_sdy_sharding(_mesh((1, 4)), ("batch", None))
    assert "mesh_idx" not in out


def test_genuine_2d_batch_shards_on_raw_index_0():
    # On a real 2D mesh, batch IS shardable -> raw axis index 0 -> mesh_idx_0.
    out = _partition_spec_to_sdy_sharding(_mesh((2, 4)), ("batch", "model"))
    assert '"mesh_idx_0"' in out  # batch
    assert '"mesh_idx_1"' in out  # model
