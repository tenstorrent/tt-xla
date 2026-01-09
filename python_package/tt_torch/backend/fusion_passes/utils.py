# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Type

import torch
from torch.fx import GraphModule
from torch.fx.subgraph_rewriter import replace_pattern_with_filters

# Default dtypes for pattern generation
DEFAULT_DTYPES = [torch.bfloat16, torch.float32]

# Global registry populated by @register_fusion_provider decorator
_REGISTERED_PROVIDERS: List[Type] = []


def register_fusion_provider(cls: Type) -> Type:
    """
    Decorator to auto-register a fusion pattern provider.

    The provider will be automatically discovered by run_fusion_passes().
    """
    _REGISTERED_PROVIDERS.append(cls)
    return cls


def get_registered_providers() -> List[Type]:
    """Return all auto-registered fusion pattern providers."""
    return _REGISTERED_PROVIDERS.copy()


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


def make_dtype_patterns(
    name_prefix: str,
    pattern_fn: Callable[[torch.dtype], Callable],
    replacement_fn: Callable,
    dtypes: Optional[List[torch.dtype]] = None,
    match_filters: Optional[List[Callable]] = None,
) -> List[FusionPattern]:
    """
    Generate FusionPattern instances for multiple dtypes.

    This helper eliminates boilerplate when creating patterns that need
    to match different dtype variants (e.g., bfloat16, float32).

    Args:
        name_prefix: Base name for the patterns
        pattern_fn: Factory function that takes a dtype and returns a pattern callable
        replacement_fn: The replacement function (shared across all dtypes)
        dtypes: List of dtypes to generate patterns for (default: DEFAULT_DTYPES)
        match_filters: Optional list of filter functions for all patterns

    Returns:
        List of FusionPattern instances, one per dtype

    Example:
        def get_patterns(self) -> List[FusionPattern]:
            return make_dtype_patterns(
                name_prefix="rms_norm",
                pattern_fn=self._make_pattern,
                replacement_fn=self._replacement,
            )

        @staticmethod
        def _make_pattern(dtype: torch.dtype) -> Callable:
            def pattern(x, weight, eps):
                # ... pattern using dtype for casts
                return result.to(dtype)
            return pattern
    """
    if dtypes is None:
        dtypes = DEFAULT_DTYPES

    patterns = []
    for dtype in dtypes:
        dtype_name = str(dtype).replace("torch.", "")
        patterns.append(
            FusionPattern(
                name=f"{name_prefix}_{dtype_name}",
                pattern=pattern_fn(dtype),
                replacement=replacement_fn,
                match_filters=match_filters or [],
            )
        )
    return patterns
