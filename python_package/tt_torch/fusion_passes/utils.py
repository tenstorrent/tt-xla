# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from torch.fx import GraphModule
from torch.fx.subgraph_rewriter import replace_pattern_with_filters


@dataclass
class FusionPattern:
    """
    Represents a fusion pattern for subgraph matching and replacement.

    Attributes:
        name: Unique identifier for the pattern
        pattern: Function defining the subgraph pattern to match
        replacement: Function defining the replacement subgraph
        match_filters: Optional list of filter functions to validate matches
    """

    name: str
    pattern: Callable
    replacement: Callable
    match_filters: Optional[List[Callable]] = field(default_factory=list)


def apply_fusion_pattern(gm: GraphModule, fusion_pattern: FusionPattern) -> int:
    """
    Apply a single fusion pattern to a GraphModule.

    Args:
        gm: The GraphModule to transform
        fusion_pattern: The fusion pattern to apply

    Returns:
        Number of replacements made
    """
    filters = fusion_pattern.match_filters or []
    replaced = replace_pattern_with_filters(
        gm,
        fusion_pattern.pattern,
        fusion_pattern.replacement,
        match_filters=filters,
    )
    return len(replaced)
