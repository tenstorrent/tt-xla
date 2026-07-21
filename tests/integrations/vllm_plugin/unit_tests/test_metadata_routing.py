# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm_tt.metadata_routing import (
    build_layer_to_kv_cache_group_idx,
    build_per_layer_attn_metadata,
)


@pytest.mark.push
@pytest.mark.cpu
def test_build_layer_to_kv_cache_group_idx_maps_all_group_layers():
    groups = [
        SimpleNamespace(layer_names=["l0", "l1"]),
        SimpleNamespace(layer_names=["l2"]),
    ]

    routed = build_layer_to_kv_cache_group_idx(groups)

    assert routed == {"l0": 0, "l1": 0, "l2": 1}


@pytest.mark.push
@pytest.mark.cpu
def test_build_layer_to_kv_cache_group_idx_includes_shared_layer_entries():
    # Represents the post-kv-sharing group layout where an extra layer was
    # appended to a target group.
    groups = [
        SimpleNamespace(layer_names=["target", "shared_layer"]),
        SimpleNamespace(layer_names=["other"]),
    ]

    routed = build_layer_to_kv_cache_group_idx(groups)

    assert routed["target"] == 0
    assert routed["shared_layer"] == 0
    assert routed["other"] == 1


@pytest.mark.push
@pytest.mark.cpu
def test_build_per_layer_attn_metadata_routes_layers_to_groups():
    meta0 = SimpleNamespace(name="g0")
    meta1 = SimpleNamespace(name="g1")
    routed = build_per_layer_attn_metadata(
        attention_layer_names=["l0", "l1", "l2"],
        layer_to_group_idx={"l0": 0, "l1": 1},
        group_attn_metadata={0: meta0, 1: meta1},
    )

    assert routed["l0"] is meta0
    assert routed["l1"] is meta1
    assert routed["l2"] is meta0


@pytest.mark.push
@pytest.mark.cpu
def test_multi_group_metadata_routing_uses_default_group_for_unmapped_layers():
    meta0 = SimpleNamespace(name="group0")
    meta1 = SimpleNamespace(name="group1")

    routed = build_per_layer_attn_metadata(
        attention_layer_names=["layer_a", "layer_b", "layer_c"],
        layer_to_group_idx={"layer_a": 0, "layer_b": 1},
        group_attn_metadata={0: meta0, 1: meta1},
    )

    assert routed["layer_a"] is meta0
    assert routed["layer_b"] is meta1
    assert routed["layer_c"] is meta0


@pytest.mark.push
@pytest.mark.cpu
def test_multi_group_metadata_routing_falls_back_for_unknown_group_indices():
    meta2 = SimpleNamespace(name="group2")
    meta4 = SimpleNamespace(name="group4")

    routed = build_per_layer_attn_metadata(
        attention_layer_names=["layer_x", "layer_y"],
        layer_to_group_idx={"layer_x": 9, "layer_y": 4},
        group_attn_metadata={2: meta2, 4: meta4},
    )

    assert routed["layer_x"] is meta2
    assert routed["layer_y"] is meta4


@pytest.mark.push
@pytest.mark.cpu
def test_multi_group_metadata_routing_returns_empty_when_no_group_metadata():
    routed = build_per_layer_attn_metadata(
        attention_layer_names=["layer0", "layer1"],
        layer_to_group_idx={"layer0": 0, "layer1": 1},
        group_attn_metadata={},
    )

    assert routed == {}
