# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .attention_impls.attention import TTMetadata


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
