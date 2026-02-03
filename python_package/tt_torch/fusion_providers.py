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
from ttxla_tools.logging import logger


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
        # Print what's actually in the graph before matching
        print(f"\n=== {self.name} === Attempting pattern match ===")
        print("Graph nodes:")
        for node in gm.graph.nodes:
            print(
                f"  {node.op:20s} {node.name:40s} target={node.target} args={node.args} kwargs={node.kwargs}"
            )

        # Print what the pattern traces to
        pattern_gm = torch.fx.symbolic_trace(self.pattern)

        print("\nPattern nodes:")
        for node in pattern_gm.graph.nodes:
            print(
                f"  {node.op:20s} {node.name:40s} target={node.target} args={node.args} kwargs={node.kwargs}"
            )

        replaced = replace_pattern_with_filters(
            gm,
            self.pattern,
            self.replacement,
            match_filters=self.get_match_filters(),
        )
        print(f"\n=== {self.name} === Replacements: {len(replaced)} ===\n")

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
        # TODO: This filter should be removed once tt-metal starts supporting splitting work
        # across multiple cores on column axis (for now it works on row axis only).
        # Check https://github.com/tenstorrent/tt-metal/issues/36094 for more details.

        # From testing, this was the last multiple of 32 that worked.
        UPPER_BOUND = 3968

        for pn, gn in match.nodes_map.items():
            if pn.target != "weight":
                continue
            if (value := gn.meta.get("example_value", None)) is None:
                raise ValueError(
                    f"Weight node is missing required metadata 'example_value'. "
                    f"Available meta keys: {list(gn.meta.keys())}"
                )
            if value.size()[-1] > UPPER_BOUND:
                logger.debug(
                    f"[Fusion] Skipping RMSNorm fusion for weight node with size {value.size()[-1]} because it is greater than the upper bound of {UPPER_BOUND}"
                )
                return False

        return True


class RotaryEmbFusionProvider(FusionProvider):

    @property
    def name(self) -> str:
        return "rotary_emb_fusion"

    @staticmethod
    def pattern(
        x: Tensor,
        freqs_cis: Tensor,
        # x.view() shape args (5 dims)
        x_d0,
        x_d1,
        x_d2,
        x_d3,
        x_d4,
        # freqs_cis.view() shape args (4 dims)
        f_d0,
        f_d1,
        f_d2,
        f_d3,
        # output dtype
        dtype,
    ) -> Tensor:
        x_float = x.float()
        x_pairs = x_float.view(x_d0, x_d1, x_d2, x_d3, x_d4)

        freqs_view = freqs_cis.view(f_d0, f_d1, f_d2, f_d3)
        freqs_ri = torch.view_as_real(freqs_view)
        cos = torch.ops.aten.select.int(freqs_ri, -1, 0)
        sin = torch.ops.aten.select.int(freqs_ri, -1, 1)

        x1 = torch.ops.aten.select.int(x_pairs, -1, 0)
        x2 = torch.ops.aten.select.int(x_pairs, -1, 1)

        x1_cos = torch.mul(x1, cos)
        x2_sin = torch.mul(x2, sin)
        out_real = torch.sub(x1_cos, x2_sin)

        x1_sin = torch.mul(x1, sin)
        x2_cos = torch.mul(x2, cos)
        out_imag = torch.add(x1_sin, x2_cos)

        out = torch.stack([out_real, out_imag], dim=-1)
        flat = out.flatten(3)
        return flat.to(dtype)

    @staticmethod
    def replacement(
        x: Tensor,
        freqs_cis: Tensor,
        x_d0,
        x_d1,
        x_d2,
        x_d3,
        x_d4,
        f_d0,
        f_d1,
        f_d2,
        f_d3,
        dtype,
    ) -> Tensor:
        # This is where you'd put your fused kernel call.
        # For now it's a 1:1 functional replacement.
        x_float = x.float()
        x_pairs = x_float.view(x_d0, x_d1, x_d2, x_d3, x_d4)

        freqs_view = freqs_cis.view(f_d0, f_d1, f_d2, f_d3)
        freqs_ri = torch.view_as_real(freqs_view)
        cos = torch.ops.aten.select.int(freqs_ri, -1, 0)
        sin = torch.ops.aten.select.int(freqs_ri, -1, 1)

        x1 = torch.ops.aten.select.int(x_pairs, -1, 0)
        x2 = torch.ops.aten.select.int(x_pairs, -1, 1)

        out_real = x1 * cos - x2 * sin
        out_imag = x1 * sin + x2 * cos

        out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
        return out.to(dtype)

    @staticmethod
    def match_filter(match, gm: torch.fx.Graph, subgraph: torch.fx.Graph) -> bool:
        for pn, gn in match.nodes_map.items():
            if pn.target == "view" and pn.name == "view":
                shape = gn.args[1:]
                if shape[-1] != 2:
                    return False
        return True
