# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    from .attention_impls.attention import TTMetadata


def build_layer_to_kv_cache_group_idx(
    kv_cache_groups: Sequence["KVCacheGroupSpec"],
) -> dict[str, int]:
    """Build a layer -> KV-cache-group index mapping from configured groups."""
    layer_to_group_idx: dict[str, int] = {}
    for group_idx, kv_cache_group in enumerate(kv_cache_groups):
        for layer_name in kv_cache_group.layer_names:
            layer_to_group_idx[layer_name] = group_idx
    return layer_to_group_idx


def build_per_layer_attn_metadata(
    attention_layer_names: Sequence[str],
    layer_to_group_idx: dict[str, int],
    group_attn_metadata: dict[int, "TTMetadata"],
) -> dict[str, "TTMetadata"]:
    """Route attention metadata to each layer using its KV-cache group mapping."""
    if not group_attn_metadata:
        return {}

    default_metadata = group_attn_metadata[min(group_attn_metadata)]
    return {
        layer_name: group_attn_metadata.get(
            layer_to_group_idx.get(layer_name, 0), default_metadata
        )
        for layer_name in attention_layer_names
    }
