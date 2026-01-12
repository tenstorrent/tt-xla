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

from .utils import FusionPattern, make_dtype_patterns


class FusionProvider(ABC):
    """Base class for all fusion pattern providers.

    Subclasses are automatically registered via __init_subclass__.
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

    @abstractmethod
    def get_patterns(self) -> List[FusionPattern]:
        """Return list of fusion patterns provided by this class."""
        pass

    @abstractmethod
    def _make_pattern(self, dtype: torch.dtype) -> Callable:
        """Create a pattern function."""
        pass

    @abstractmethod
    def _replacement_fn(self, *args, **kwargs) -> Tensor:
        """Create a replacement function for the pattern."""
        pass


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
        return "RMSNorm Fusion"

    def get_patterns(self) -> List[FusionPattern]:
        """Generate fusion patterns for all supported dtypes."""
        return make_dtype_patterns(
            name_prefix="rms_norm",
            pattern_fn=self._make_pattern,
            replacement_fn=self._replacement_fn,
        )

    @staticmethod
    def _make_pattern(dtype: torch.dtype) -> Callable:
        """
        Create a pattern function for a specific dtype.

        Args:
            dtype: The dtype for the final cast operation (e.g., torch.bfloat16)

        Returns:
            Pattern function that matches LlamaRMSNorm with the specified dtype

        Note:
            Uses method calls (.add(), .mul()) instead of operators (+, *)
            because dynamo traces tensor operations as call_method, not call_function.
        """

        def pattern(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
            hidden_fp32 = hidden_states.to(torch.float32)
            variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
            variance_eps = variance.add(eps)  # Use .add() instead of +
            rsqrt_var = torch.rsqrt(variance_eps)
            hidden_normalized = hidden_fp32.mul(rsqrt_var)  # Use .mul() instead of *
            hidden_cast = hidden_normalized.to(dtype)
            return weight.mul(hidden_cast)  # Use .mul() instead of *

        return pattern

    @staticmethod
    def _replacement_fn(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
        """
        Replacement function for RMS norm pattern.
        """
        return torch.nn.functional.rms_norm(
            hidden_states, normalized_shape=weight.shape, weight=weight, eps=eps
        )
