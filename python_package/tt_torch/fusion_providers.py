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

    Each variant has two flavors:
    - method-call form, emitted by dynamo when tracing without AOTAutograd
      (.to/.pow/.mean/etc as call_method nodes).
    - aten-overload form, emitted when AOTAutograd is enabled (e.g. via
      tt_use_aot_autograd) which lowers tensor ops to torch.ops.aten.* call_function
      nodes. Without the aten-form pattern, vLLM serving paths (which run under
      inference_mode + AOTAutograd) leave the decomposed pattern unfused.

    Gemma / Gemma2 / Gemma3 use a slightly different shape:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)
    The weight is upcast to fp32 and shifted by +1 before multiply. The trailing
    cast uses .type_as() which AOTAutograd lowers to aten._to_copy with extra
    layout/device kwargs that vary by platform; we therefore truncate the Gemma
    patterns at the weighted-multiply and let the platform-dependent cast remain
    in the graph (it becomes a cheap no-op when downstream produces the same
    dtype).

    Gemma4 is structurally Gemma without the +1 shift and uses torch.pow(x, -0.5)
    in place of torch.rsqrt.
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
    def pattern_aot(hidden_states: Tensor, weight: Tensor, eps: float, dtype) -> Tensor:
        """
        Llama variant as emitted by AOTAutograd: tensor ops appear as direct
        torch.ops.aten.* overloads rather than method calls.
        """
        h_fp32 = torch.ops.aten._to_copy.default(hidden_states, dtype=torch.float32)
        squared = torch.ops.aten.pow.Tensor_Scalar(h_fp32, 2)
        variance = torch.ops.aten.mean.dim(squared, [-1], True)
        variance_eps = torch.ops.aten.add.Tensor(variance, eps)
        rsqrt_var = torch.ops.aten.rsqrt.default(variance_eps)
        hidden_normalized = torch.ops.aten.mul.Tensor(h_fp32, rsqrt_var)
        hidden_cast = torch.ops.aten._to_copy.default(hidden_normalized, dtype=dtype)
        return torch.ops.aten.mul.Tensor(weight, hidden_cast)

    @staticmethod
    def pattern_aot_cast_after_mul(
        hidden_states: Tensor, weight: Tensor, eps: float, dtype
    ) -> Tensor:
        """GPT-OSS variant as emitted by AOTAutograd."""
        h_fp32 = torch.ops.aten._to_copy.default(hidden_states, dtype=torch.float32)
        squared = torch.ops.aten.pow.Tensor_Scalar(h_fp32, 2)
        variance = torch.ops.aten.mean.dim(squared, [-1], True)
        variance_eps = torch.ops.aten.add.Tensor(variance, eps)
        rsqrt_var = torch.ops.aten.rsqrt.default(variance_eps)
        hidden_normalized = torch.ops.aten.mul.Tensor(h_fp32, rsqrt_var)
        weighted = torch.ops.aten.mul.Tensor(weight, hidden_normalized)
        return torch.ops.aten._to_copy.default(weighted, dtype=dtype)

    @staticmethod
    def replacement(hidden_states: Tensor, weight: Tensor, eps: float, dtype) -> Tensor:
        """Shared replacement for the Llama/GPT-OSS pattern variants."""
        return torch.nn.functional.rms_norm(
            hidden_states, normalized_shape=weight.shape, weight=weight, eps=eps
        )

    # ----- Gemma family (Gemma / Gemma2 / Gemma3) -----

    @staticmethod
    def pattern_gemma(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
        """
        Gemma dynamo: weight is upcast and shifted by +1 before multiply, then
        type_as cast back. Pattern matches up to the weighted-multiply only; the
        trailing type_as remains in the graph after fusion.

        Uses tensor method calls (.add, .mul) instead of operators because
        tt_torch's dynamo compatibility patches funnel +/* through call_method.
        For the same reason, `1.0 + weight_fp32` is written as
        `weight_fp32.add(1.0)` (the patched __radd__ swaps arg order).
        """
        x_fp32 = hidden_states.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        variance_eps = variance.add(eps)
        rsqrt_var = torch.rsqrt(variance_eps)
        normed = x_fp32.mul(rsqrt_var)
        weight_fp32 = weight.float()
        shifted_w = weight_fp32.add(1.0)
        return normed.mul(shifted_w)

    @staticmethod
    def pattern_gemma_aot(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
        """Gemma AOT: same shape as pattern_gemma but in aten op form."""
        x_fp32 = torch.ops.aten._to_copy.default(hidden_states, dtype=torch.float32)
        squared = torch.ops.aten.pow.Tensor_Scalar(x_fp32, 2)
        variance = torch.ops.aten.mean.dim(squared, [-1], True)
        variance_eps = torch.ops.aten.add.Tensor(variance, eps)
        rsqrt_var = torch.ops.aten.rsqrt.default(variance_eps)
        normed = torch.ops.aten.mul.Tensor(x_fp32, rsqrt_var)
        weight_fp32 = torch.ops.aten._to_copy.default(weight, dtype=torch.float32)
        shifted_w = torch.ops.aten.add.Tensor(weight_fp32, 1.0)
        return torch.ops.aten.mul.Tensor(normed, shifted_w)

    @staticmethod
    def replacement_gemma(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
        """Replacement: rms_norm with effective weight = 1 + weight.float()."""
        return torch.nn.functional.rms_norm(
            hidden_states,
            normalized_shape=weight.shape,
            weight=1.0 + weight.float(),
            eps=eps,
        )

    # ----- Gemma4 (no +1, pow(-0.5) instead of rsqrt) -----

    @staticmethod
    def pattern_gemma4(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
        """
        Gemma4 dynamo: weight upcast (no +1), torch.pow(ms, -0.5) for inv-sqrt.
        Uses tensor method calls per the patched-dynamo convention (see
        pattern_gemma docstring).
        """
        x_fp32 = hidden_states.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        mean_squared = variance.add(eps)
        inv_std = torch.pow(mean_squared, -0.5)
        normed = x_fp32.mul(inv_std)
        weight_fp32 = weight.float()
        return normed.mul(weight_fp32)

    @staticmethod
    def pattern_gemma4_aot(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
        """Gemma4 AOT."""
        x_fp32 = torch.ops.aten._to_copy.default(hidden_states, dtype=torch.float32)
        squared = torch.ops.aten.pow.Tensor_Scalar(x_fp32, 2)
        variance = torch.ops.aten.mean.dim(squared, [-1], True)
        mean_squared = torch.ops.aten.add.Tensor(variance, eps)
        inv_std = torch.ops.aten.pow.Tensor_Scalar(mean_squared, -0.5)
        normed = torch.ops.aten.mul.Tensor(x_fp32, inv_std)
        weight_fp32 = torch.ops.aten._to_copy.default(weight, dtype=torch.float32)
        return torch.ops.aten.mul.Tensor(normed, weight_fp32)

    @staticmethod
    def replacement_gemma4(hidden_states: Tensor, weight: Tensor, eps: float) -> Tensor:
        """Replacement: rms_norm with weight upcast to fp32."""
        return torch.nn.functional.rms_norm(
            hidden_states,
            normalized_shape=weight.shape,
            weight=weight.float(),
            eps=eps,
        )

    def get_patterns(self) -> List[tuple]:
        return [
            (self.pattern, self.replacement),
            (self.pattern_cast_after_mul, self.replacement),
            (self.pattern_aot, self.replacement),
            (self.pattern_aot_cast_after_mul, self.replacement),
            (self.pattern_gemma, self.replacement_gemma),
            (self.pattern_gemma_aot, self.replacement_gemma),
            (self.pattern_gemma4, self.replacement_gemma4),
            (self.pattern_gemma4_aot, self.replacement_gemma4),
        ]
