# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Central registry of all pattern providers.

All pattern provider classes are defined in this file, including:
- FusionProvider: For replacing patterns with different operations
- CompositeWrapProvider: For wrapping patterns in StableHLO composites
"""

import inspect
from abc import ABC, abstractmethod
from typing import Callable, List, Type

import torch
from torch import Tensor
from torch.fx import GraphModule
from torch.fx.subgraph_rewriter import replace_pattern

from .utils import create_composite_wrap_replacement

# ============================= Pattern Providers =============================


class PatternProvider(ABC):
    """Base class for all pattern-based graph transformations.

    Subclasses are automatically registered via __init_subclass__.
    Each subclass must implement:
    - name: Human-readable identifier
    - pattern(): The pattern to match
    - get_replacement(): Returns the replacement function (polymorphic)

    To create a new pattern provider:
    1. Inherit from FusionProvider or CompositeWrapProvider
    2. Implement the required abstract methods
    """

    _registered_providers: List[Type["PatternProvider"]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            PatternProvider._registered_providers.append(cls)

    @classmethod
    def get_registered_providers(cls) -> List[Type["PatternProvider"]]:
        """Return all registered providers that are subclasses of the calling class.

        Examples:
            PatternProvider.get_registered_providers() -> all providers
            FusionProvider.get_registered_providers() -> only FusionProvider subclasses
            CompositeWrapProvider.get_registered_providers() -> only CompositeWrapProvider subclasses
        """
        return [p for p in PatternProvider._registered_providers if issubclass(p, cls)]

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

    @abstractmethod
    def get_replacement(self) -> Callable:
        """Return the replacement function. The replacement function is used to replace the matched pattern with."""
        pass

    def replace_pattern(self, gm: GraphModule) -> int:
        """Apply this pattern to the GraphModule.

        Returns the number of replacements made.
        """
        replaced = replace_pattern(gm, self.pattern, self.get_replacement())
        return len(replaced)


class FusionProvider(PatternProvider):
    """Provider for replacing patterns with different operations.

    Subclasses must implement replacement() which defines
    what to replace the matched pattern with.

    To create a new fusion provider:
    1. Inherit from FusionProvider
    2. Implement the `name` property
    3. Implement the `pattern` static method (the pattern to match)
    4. Implement the `replacement` static method (the replacement function)
    """

    @staticmethod
    @abstractmethod
    def replacement(*args, **kwargs) -> Tensor:
        """The replacement function (different from pattern)."""
        pass

    def get_replacement(self) -> Callable:
        """Returns the user-defined replacement function."""
        return self.replacement


class CompositeWrapProvider(PatternProvider):
    """Provider for wrapping patterns in StableHLO composites.

    Unlike FusionProvider, only pattern() is required.
    The replacement is auto-generated to wrap pattern into a StableHLO composite.

    To create a new composite wrap provider:
    1. Inherit from CompositeWrapProvider
    2. Implement the `name` property
    3. Implement the `composite_name` property (e.g., 'tenstorrent.op, must match the name in tt-mlir')
    4. Implement the `pattern` static method (the pattern to match)
    """

    @property
    @abstractmethod
    def composite_name(self) -> str:
        """StableHLO composite name (e.g., 'tenstorrent.rope')."""
        pass

    @property
    def composite_attr(self) -> dict | None:
        """Optional attributes for the composite."""
        return None

    def get_replacement(self) -> Callable:
        """Auto-generates replacement that wraps pattern in composite."""
        return create_composite_wrap_replacement(
            self.pattern, self.composite_name, self.composite_attr
        )


# ============================= Fusion Providers =============================


class RMSNormFusionProvider(FusionProvider):
    """Provider for replacing RMS norm pattern with torch.nn.functional.rms_norm."""

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
