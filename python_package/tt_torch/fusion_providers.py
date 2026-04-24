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
import torch.nn.functional as F
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
    default_enabled: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        FusionProvider._registered_providers.append(cls)

    @classmethod
    def get_registered_providers(
        cls,
        provider_names: List[str] | None = None,
        include_default_disabled: bool = False,
    ) -> List[Type["FusionProvider"]]:
        """Return registered providers filtered by name and default-enabled state."""
        selected_names = set(provider_names) if provider_names is not None else None
        selected: List[Type["FusionProvider"]] = []

        for provider_cls in cls._registered_providers:
            provider = provider_cls()
            if not include_default_disabled and not provider_cls.default_enabled:
                continue
            if selected_names is not None and provider.name not in selected_names:
                continue
            selected.append(provider_cls)

        return selected

    @classmethod
    def get_registered_provider_names(
        cls, include_default_disabled: bool = False
    ) -> List[str]:
        return [
            provider_cls().name
            for provider_cls in cls.get_registered_providers(
                include_default_disabled=include_default_disabled
            )
        ]

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

    def get_patterns(self) -> List[tuple]:
        return [
            (self.pattern, self.replacement),
            (self.pattern_cast_after_mul, self.replacement),
        ]


class QuetzalFuseGELUProvider(FusionProvider):
    """Collapse tanh-GELU decompositions back to torch.nn.functional.gelu."""

    default_enabled = False

    @property
    def name(self) -> str:
        return "fuse_gelu"

    @staticmethod
    def pattern_method(x: Tensor) -> Tensor:
        half_x = x.mul(0.5)
        x_pow_3 = x.pow(3.0)
        cubic_term = x_pow_3.mul(0.044715)
        inner = x.add(cubic_term)
        tanh_input = inner.mul(0.7978845608028654)
        tanh_output = torch.tanh(tanh_input)
        return half_x.mul(tanh_output.add(1.0))

    @staticmethod
    def pattern_operator(x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))

    @staticmethod
    def replacement(x: Tensor) -> Tensor:
        return F.gelu(x, approximate="tanh")

    def get_patterns(self) -> List[tuple]:
        return [
            (self.pattern_method, self.replacement),
            (self.pattern_operator, self.replacement),
        ]


class QuetzalReconstructSDPAProvider(FusionProvider):
    """Reconstruct SDPA from manual matmul/softmax/matmul attention."""

    default_enabled = False

    @property
    def name(self) -> str:
        return "reconstruct_sdpa"

    @staticmethod
    def pattern_scaled_method(
        query: Tensor, key: Tensor, value: Tensor, scale: float
    ) -> Tensor:
        key_t = key.transpose(-2, -1)
        scores = torch.matmul(query, key_t)
        scaled_scores = scores.mul(scale)
        weights = torch.softmax(scaled_scores, dim=-1)
        return torch.matmul(weights, value)

    @staticmethod
    def pattern_scaled_operator(
        query: Tensor, key: Tensor, value: Tensor, scale: float
    ) -> Tensor:
        return torch.softmax((query @ key.transpose(-2, -1)) * scale, dim=-1) @ value

    @staticmethod
    def pattern_unscaled_method(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        key_t = key.transpose(-2, -1)
        scores = torch.matmul(query, key_t)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)

    @staticmethod
    def pattern_unscaled_operator(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return torch.softmax(query @ key.transpose(-2, -1), dim=-1) @ value

    @staticmethod
    def replacement_scaled(
        query: Tensor, key: Tensor, value: Tensor, scale: float
    ) -> Tensor:
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, scale=scale
        )

    @staticmethod
    def replacement_unscaled(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

    def get_patterns(self) -> List[tuple]:
        return [
            (self.pattern_scaled_method, self.replacement_scaled),
            (self.pattern_scaled_operator, self.replacement_scaled),
            (self.pattern_unscaled_method, self.replacement_unscaled),
            (self.pattern_unscaled_operator, self.replacement_unscaled),
        ]
