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
from torch.fx import GraphModule
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder
from ttxla_tools.logging import logger


def create_composite_wrap_replacement(
    replacement_fn: Callable,
    example_inputs: tuple,
) -> GraphModule:
    """Creates a pre-traced replacement GraphModule that wraps pattern in composite.

    This function is needed because StableHLOCompositeBuilder.mark_inputs() doesn't
    work with torch.fx symbolic tracing (it receives Proxy objects instead of tensors).
    By pre-tracing with make_fx using real tensors, we produce a GraphModule that
    can be used directly with replace_pattern_with_filters.

    Args:
        replacement_fn: The replacement function that uses StableHLOCompositeBuilder
        example_inputs: Tuple of example inputs for tracing (must match replacement signature)

    Returns:
        A traced GraphModule
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    return make_fx(replacement_fn)(*example_inputs)


def reorder_placeholders(
    graph: torch.fx.Graph, expected_order: List[str]
) -> torch.fx.Graph:
    """Reorder placeholder nodes in a graph to match expected order.

    This ensures that when replace_pattern_with_filters maps pattern placeholders
    to replacement placeholders by position, they are correctly aligned.

    Args:
        graph: The graph to reorder placeholders in.
        expected_order: List of placeholder name prefixes in desired order.
                       e.g., ["hidden_states", "weight", "eps", "dtype"]

    Returns:
        The graph with placeholders reordered.
    """
    # Collect all placeholder nodes
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]

    # Create mapping from name prefix to placeholder node
    prefix_to_node = {}
    for ph in placeholders:
        # Handle names like "hidden_states_1" -> "hidden_states"
        name = ph.name
        for prefix in expected_order:
            if name == prefix or name.startswith(prefix + "_"):
                prefix_to_node[prefix] = ph
                break

    # Check if reordering is needed
    current_order = [ph.name for ph in placeholders]
    needs_reorder = False
    for i, prefix in enumerate(expected_order):
        if prefix in prefix_to_node:
            node = prefix_to_node[prefix]
            if placeholders.index(node) != i:
                needs_reorder = True
                break

    if not needs_reorder:
        return graph

    # Reorder by moving nodes to the beginning in reverse order
    # This preserves the relative order of non-placeholder nodes
    first_non_placeholder = None
    for node in graph.nodes:
        if node.op != "placeholder":
            first_non_placeholder = node
            break

    if first_non_placeholder is None:
        return graph

    # Move placeholders in correct order
    for prefix in reversed(expected_order):
        if prefix in prefix_to_node:
            node = prefix_to_node[prefix]
            node.prepend(first_non_placeholder)

    return graph


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
        """Replacement function that wraps the RMS norm pattern in a StableHLO composite."""
        # IMPORTANT: Access hidden_states BEFORE weight to ensure correct placeholder
        # ordering when traced with make_fx. Placeholders are created in access order.
        hidden_fp32 = hidden_states.to(torch.float32)

        # Get normalized_shape from weight dimensions (RMS norm normalizes over last dim)
        normalized_shape = list(weight.shape)

        builder = StableHLOCompositeBuilder(
            name="tenstorrent.rms_norm",
            attr={"epsilon": eps, "normalized_shape": normalized_shape},
        )

        # Mark float32 tensors as inputs
        marked_hidden, marked_weight = builder.mark_inputs(hidden_fp32, weight)

        # Execute the same operations as the pattern (input is already f32)
        variance = marked_hidden.pow(2).mean(-1, keepdim=True)
        variance_eps = variance.add(eps)
        rsqrt_var = torch.rsqrt(variance_eps)
        hidden_normalized = marked_hidden.mul(rsqrt_var)
        hidden_cast = hidden_normalized.to(dtype)
        result = marked_weight.mul(hidden_cast)

        return builder.mark_outputs(result)

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

    def _create_replacement_for_shape(self, weight_shape: list, hidden_shape: list):
        """Create a traced replacement graph with the correct weight shape.

        Args:
            weight_shape: Shape of the weight tensor from the matched pattern.
            hidden_shape: Shape of the hidden_states tensor from the matched pattern.

        Returns:
            A traced GraphModule with the correct normalized_shape attribute.
        """
        example_inputs = (
            torch.randn(*hidden_shape, dtype=torch.bfloat16),  # hidden_states
            torch.randn(*weight_shape),  # weight with actual shape
            1e-6,  # eps
            torch.bfloat16,  # dtype
        )
        return create_composite_wrap_replacement(self.replacement, example_inputs)

    def replace_pattern(self, gm: torch.fx.GraphModule) -> int:
        """Replace the RMS norm pattern with composite-wrapped version.

        Uses replacement_callback to extract actual weight shape from each match
        and generate a correctly-shaped replacement graph.
        """

        def replacement_callback(match, original_graph, pattern_graph):
            # Extract weight shape from matched nodes
            weight_shape = None
            hidden_shape = None
            for pn, gn in match.nodes_map.items():
                if pn.target == "weight":
                    if (value := gn.meta.get("example_value", None)) is not None:
                        weight_shape = list(value.shape)
                if pn.target == "hidden_states":
                    if (value := gn.meta.get("example_value", None)) is not None:
                        hidden_shape = list(value.shape)

            # Fallback to defaults if metadata not available
            if weight_shape is None:
                weight_shape = [32]
            if hidden_shape is None:
                hidden_shape = [1, 32, 32]

            # Create replacement with correct shapes
            traced_replacement = self._create_replacement_for_shape(
                weight_shape, hidden_shape
            )

            # Ensure placeholder order matches pattern order for correct mapping
            # Pattern order is: hidden_states, weight, eps, dtype
            reorder_placeholders(
                traced_replacement.graph,
                ["hidden_states", "weight", "eps", "dtype"],
            )

            return traced_replacement.graph

        replaced = replace_pattern_with_filters(
            gm,
            self.pattern,
            replacement=None,  # Use callback instead
            match_filters=self.get_match_filters(),
            replacement_callback=replacement_callback,
        )
        return len(replaced)
