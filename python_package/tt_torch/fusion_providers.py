# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Central registry of all fusion pattern providers.

All fusion provider classes are defined in this file.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Type

import torch
from torch import Tensor
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
from ttxla_tools.logging import logger


class FusionProvider(ABC):
    """Base class for all fusion pattern providers.

    Subclasses are automatically registered via __init_subclass__.

    To create a new fusion provider:
    1. Inherit from FusionProvider
    2. Implement the `name` property
    3. Implement the `pattern` static method (the pattern to match)
    4. Implement the `replacement` static method (the replacement function)

    For providers with multiple pattern variants, override `get_patterns`
    to return a list of (pattern, replacement) tuples instead.

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

    def get_patterns(self) -> List[tuple]:
        """Return (pattern, replacement) pairs. Override for multi-pattern providers."""
        return [(self.pattern, self.replacement)]

    def replace_pattern(self, gm: torch.fx.GraphModule) -> int:
        """
        Replace patterns in the graph.

        Iterates over all (pattern, replacement) pairs from get_patterns().

        Args:
            gm: The GraphModule to transform

        Returns:
            Number of replacements made
        """
        total = 0
        for pattern, replacement in self.get_patterns():
            replaced = replace_pattern_with_filters(
                gm,
                pattern,
                replacement,
                match_filters=self.get_match_filters(),
            )
            total += len(replaced)
        return total


# ================================ Fusion Providers ================================


class RMSNormFusionProvider(FusionProvider):
    """
    Provides fusion patterns for RMS Normalization operations.

    Matches patterns like LlamaRMSNorm (common case):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states.to(input_dtype)

    There is also a GPT-OSS variant where the cast happens after multiply with weight:
        return (weight * hidden_states).to(input_dtype)

    Both are replaced with torch.nn.functional.rms_norm.
    """

    @property
    def name(self) -> str:
        return "rms_norm_fusion"

    @staticmethod
    def pattern(hidden_states: Tensor, weight: Tensor, eps: float, dtype) -> Tensor:
        """
        Llama variant: cast happens before multiply with weight.

        Matches: weight * hidden_states.to(input_dtype)

        Note:
            Uses method calls (.add(), .mul()) instead of operators (+, *)
            because dynamo traces tensor operations as call_method, not call_function.

            The dtype parameter allows matching any dtype variant, it becomes a
            wildcard in the pattern graph that matches any value.
        """
        hidden_fp32 = hidden_states.to(torch.float32)
        variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
        variance_eps = variance.add(eps)
        rsqrt_var = torch.rsqrt(variance_eps)
        hidden_normalized = hidden_fp32.mul(rsqrt_var)
        hidden_cast = hidden_normalized.to(dtype)
        return weight.mul(hidden_cast)

    @staticmethod
    def pattern_cast_after_mul(
        hidden_states: Tensor, weight: Tensor, eps: float, dtype
    ) -> Tensor:
        """
        GPT-OSS variant: cast happens after multiply with weight.

        Matches: (weight * hidden_states).to(input_dtype)
        """
        hidden_fp32 = hidden_states.to(torch.float32)
        variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
        variance_eps = variance.add(eps)
        rsqrt_var = torch.rsqrt(variance_eps)
        hidden_normalized = hidden_fp32.mul(rsqrt_var)
        result = weight.mul(hidden_normalized)
        return result.to(dtype)

    @staticmethod
    def replacement(hidden_states: Tensor, weight: Tensor, eps: float, dtype) -> Tensor:
        """Shared replacement for both RMS norm pattern variants."""
        return torch.nn.functional.rms_norm(
            hidden_states, normalized_shape=weight.shape, weight=weight, eps=eps
        )

    @staticmethod
    def pattern_pow_variant(
        hidden_states: Tensor, weight: Tensor, eps: float
    ) -> Tensor:
        """
        Gemma4 variant: uses torch.pow(x, -0.5) instead of torch.rsqrt(x),
        adds eps inline with mean, and multiplies normed * weight (not weight * normed).
        Cast via .float() and .type_as() instead of .to(dtype).

        Matches Gemma4RMSNorm._norm + forward:
            hidden_fp32 = hidden_states.float()
            mean_squared = hidden_fp32.pow(2).mean(-1, keepdim=True) + self.eps
            normed = hidden_fp32 * torch.pow(mean_squared, -0.5)
            normed_output = normed * self.weight.float()
            return normed_output.type_as(hidden_states)
        """
        hidden_fp32 = hidden_states.float()
        mean_squared = hidden_fp32.pow(2).mean(-1, keepdim=True)
        mean_squared_eps = mean_squared.add(eps)
        inv_rms = torch.pow(mean_squared_eps, -0.5)
        hidden_normalized = hidden_fp32.mul(inv_rms)
        weight_fp32 = weight.float()
        result = hidden_normalized.mul(weight_fp32)
        return result.type_as(hidden_states)

    @staticmethod
    def replacement_pow_variant(
        hidden_states: Tensor, weight: Tensor, eps: float
    ) -> Tensor:
        """Replacement for the pow variant — no dtype param since pattern uses type_as."""
        return torch.nn.functional.rms_norm(
            hidden_states, normalized_shape=weight.shape, weight=weight, eps=eps
        )

    @staticmethod
    def pattern_pow_no_weight(hidden_states: Tensor, eps: float) -> Tensor:
        """
        Gemma4 variant without weight (with_scale=False):
            hidden_fp32 = hidden_states.float()
            mean_squared = hidden_fp32.pow(2).mean(-1, keepdim=True) + self.eps
            normed = hidden_fp32 * torch.pow(mean_squared, -0.5)
            return normed.type_as(hidden_states)
        """
        hidden_fp32 = hidden_states.float()
        mean_squared = hidden_fp32.pow(2).mean(-1, keepdim=True)
        mean_squared_eps = mean_squared.add(eps)
        inv_rms = torch.pow(mean_squared_eps, -0.5)
        hidden_normalized = hidden_fp32.mul(inv_rms)
        return hidden_normalized.type_as(hidden_states)

    @staticmethod
    def replacement_pow_no_weight(hidden_states: Tensor, eps: float) -> Tensor:
        """Replacement for no-weight pow variant."""
        # rms_norm with weight=None just does the normalization without scaling
        normalized_shape = [hidden_states.shape[-1]]
        return torch.nn.functional.rms_norm(
            hidden_states, normalized_shape=normalized_shape, weight=None, eps=eps
        )

    def get_patterns(self) -> List[tuple]:
        return [
            (self.pattern, self.replacement),
            (self.pattern_cast_after_mul, self.replacement),
            (self.pattern_pow_variant, self.replacement_pow_variant),
            (self.pattern_pow_no_weight, self.replacement_pow_no_weight),
        ]
