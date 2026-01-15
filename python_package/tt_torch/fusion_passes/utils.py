# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Callable

from torch.fx import GraphModule
from torch.fx.subgraph_rewriter import replace_pattern


@dataclass
class FusionPattern:
    """
    Represents a fusion pattern for subgraph matching and replacement.

    Attributes:
        name: Unique identifier for the pattern
        pattern: Function defining the subgraph pattern to match
        replacement: Function defining the replacement subgraph
    """

    name: str
    pattern: Callable
    replacement: Callable


def apply_fusion_pattern(gm: GraphModule, fusion_pattern: FusionPattern) -> int:
    """
    Apply a single fusion pattern to a GraphModule.

    Args:
        gm: The GraphModule to transform
        fusion_pattern: The fusion pattern to apply

    Returns:
        Number of replacements made
    """
    replaced = replace_pattern(
        gm,
        fusion_pattern.pattern,
        fusion_pattern.replacement,
    )
    return len(replaced)
