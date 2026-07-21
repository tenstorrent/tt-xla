# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from integrations.vllm_plugin.vllm_tt.metadata_routing import (
    build_per_layer_attn_metadata,
)


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
