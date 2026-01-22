# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Central registry of all fusion pattern providers.

All fusion provider classes are defined in this file.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Type

import torch
from torch import Tensor
from torch.fx.subgraph_rewriter import replace_pattern_with_filters

InternalMatch = Any


class FusionProvider(ABC):
    """Base class for all fusion pattern providers.

    Subclasses are automatically registered via __init_subclass__.

    To create a new fusion provider:
    1. Inherit from FusionProvider
    2. Implement the `name` property
    3. Implement the `pattern` static method (the pattern to match)
    4. Implement the `replacement` static method (the replacement function)

    Optional:
    5. Implement the `match_filter` method (single match filter to apply to the pattern)
    or alternatively,
    Implement the `get_match_filters` method (list of match filters to apply to the pattern)
    """

    _registered_providers: List[Type["FusionProvider"]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        FusionProvider._registered_providers.append(cls)

    @classmethod
    def get_registered_providers(cls) -> List[Type["FusionProvider"]]:
        """Return all registered provider classes."""
        return cls._registered_providers.copy()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this provider."""
        pass

    @staticmethod
    @abstractmethod
    def pattern(*args, **kwargs) -> Tensor:
        """The pattern to match in the graph."""
        pass

    @staticmethod
    @abstractmethod
    def replacement(*args, **kwargs) -> Tensor:
        """The replacement function."""
        pass

    @staticmethod
    def match_filter(*args, **kwargs) -> bool:
        """The match filter to apply to the pattern."""
        return True

    def get_match_filters(self) -> List[Callable]:
        """Return the match filters for the provider."""
        return [self.match_filter]

    def replace_pattern(self, gm: torch.fx.GraphModule) -> int:
        """
        Replace a pattern in the graph.

        Args:
            gm: The GraphModule to transform

        Returns:
            Number of replacements made
        """
        replaced = replace_pattern_with_filters(
            gm,
            self.pattern,
            self.replacement,
            match_filters=self.get_match_filters(),
        )
        return len(replaced)


# ================================ Fusion Providers ================================


class RMSNormFusionProvider(FusionProvider):
    """
    Provides fusion patterns for RMS Normalization operations.

    Matches patterns like LlamaRMSNorm:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states.to(input_dtype)

    And replaces with torch.nn.functional.rms_norm
    """

    @property
    def name(self) -> str:
        return "rms_norm_fusion"

    @staticmethod
    def pattern(hidden_states: Tensor, weight: Tensor, eps: float, dtype) -> Tensor:
        """
        Pattern function for RMS normalization.

        Note:
            Uses method calls (.add(), .mul()) instead of operators (+, *)
            because dynamo traces tensor operations as call_method, not call_function.

            The dtype parameter allows matching any dtype variant, it becomes a
            wildcard in the pattern graph that matches any value.
        """
        hidden_fp32 = hidden_states.to(torch.float32)
        variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
        variance_eps = variance.add(eps)  # Use .add() instead of +
        rsqrt_var = torch.rsqrt(variance_eps)
        hidden_normalized = hidden_fp32.mul(rsqrt_var)  # Use .mul() instead of *
        hidden_cast = hidden_normalized.to(dtype)  # dtype is a wildcard
        return weight.mul(hidden_cast)  # Use .mul() instead of *

    @staticmethod
    def replacement(hidden_states: Tensor, weight: Tensor, eps: float, dtype) -> Tensor:
        """Replacement function for RMS norm pattern."""
        return torch.nn.functional.rms_norm(
            hidden_states, normalized_shape=weight.shape, weight=weight, eps=eps
        )

    @staticmethod
    def match_filter(match, gm: torch.fx.Graph, subgraph: torch.fx.Graph) -> bool:
        # TODO: Remove this once https://github.com/tenstorrent/tt-metal/issues/36094 is fixed
        upper_bound = 3968
        for pn, gn in match.nodes_map.items():
            if pn.target != "weight":
                continue
            if (value := gn.meta.get("example_value", None)) is None:
                raise ValueError("Example value is not set for weight node")
            if value.size()[-1] > upper_bound:
                print(
                    f"[Fusion] Skipping RMSNorm fusion for weight node with size {value.size()[-1]} because it is greater than the upper bound of {upper_bound}"
                )
                return False

        return True
