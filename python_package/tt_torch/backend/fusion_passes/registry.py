# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from torch.fx import GraphModule

from . import providers
from .utils import apply_fusion_pattern, get_registered_providers


def run_fusion_passes(gm: GraphModule) -> GraphModule:
    """
    Run all registered fusion passes on a GraphModule.

    Args:
        gm: The GraphModule to transform

    Returns:
        The transformed GraphModule
    """
    total_replacements = 0
    providers = get_registered_providers()

    for provider_cls in providers:
        provider = provider_cls()
        for fusion_pattern in provider.get_patterns():
            num_replaced = apply_fusion_pattern(gm, fusion_pattern)
            status = f"{num_replaced} match(es)" if num_replaced > 0 else "no match"
            print(f"[Fusion] {fusion_pattern.name}: {status}")
            total_replacements += num_replaced

    print(f"[Fusion] Total replacements: {total_replacements}")

    if total_replacements > 0:
        gm.graph.lint()
        gm.recompile()

    return gm
