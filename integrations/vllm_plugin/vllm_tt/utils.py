# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.distributed.spmd as xs
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import MergedColumnParallelLinear

from .logger import tt_init_logger

logger = tt_init_logger(__name__)


class XlaMergedColumnParallelLinear(nn.Module):

    def __init__(self, merged_column_parallel_linear: nn.Module, mesh: "xs.Mesh"):
        super().__init__()
        assert isinstance(merged_column_parallel_linear, MergedColumnParallelLinear)
        self.skip_bias_add = merged_column_parallel_linear.skip_bias_add
        self.return_bias = merged_column_parallel_linear.return_bias
        self.num_outputs = len(merged_column_parallel_linear.output_sizes)
        self.output_sizes = merged_column_parallel_linear.output_sizes
        self.layer_prefix = merged_column_parallel_linear.prefix.replace(".", "_")

        self.weights: List[Parameter] = []
        self.biases: List[Parameter] = []
        self._load_weights_from_merged_column_parallel_linear(
            merged_column_parallel_linear
        )
        if mesh is not None:
            self._shard_weight(mesh)

    def _shard_weight(self, mesh: "xs.Mesh"):
        for i in range(self.num_outputs):
            self.weights[i] = Parameter(self.weights[i].to("xla"), requires_grad=False)
            xs.mark_sharding(self.weights[i], mesh, ("x", None))

            if self.biases[i] is not None:
                self.biases[i] = Parameter(
                    self.biases[i].to("xla"), requires_grad=False
                )
                xs.mark_sharding(self.biases[i], mesh, ("x",))

    def _load_weights_from_merged_column_parallel_linear(
        self, merged_column_parallel_linear: nn.Module
    ):
        # The weight is a concatenation of all output weights along the output dimension
        merged_column_parallel_weight = merged_column_parallel_linear.weight.data.cpu()

        start_idx = 0
        for i, output_size in enumerate(self.output_sizes):
            end_idx = start_idx + output_size
            weight = Parameter(
                merged_column_parallel_weight[start_idx:end_idx], requires_grad=False
            )
            self.weights.append(weight)
            logger.info(f"weight{i}: {self.layer_prefix}_weight_{i}")
            self.register_parameter(f"{self.layer_prefix}_weight_{i}", weight)
            logger.info(f"weight_{i}: {weight}")
            start_idx = end_idx

        if merged_column_parallel_linear.bias is not None:
            start_idx = 0
            for i, output_size in enumerate(self.output_sizes):
                end_idx = start_idx + output_size
                bias = Parameter(
                    merged_column_parallel_linear.bias[start_idx:end_idx],
                    requires_grad=False,
                )
                self.biases.append(bias)
                self.register_parameter(f"{self.layer_prefix}_bias_{i}", bias)
                start_idx = end_idx
        else:
            for i in range(self.num_outputs):
                self.biases.append(None)
                self.register_parameter(f"{self.layer_prefix}_bias_{i}", None)

    def forward(self, input):
        projections = []
        output_biases = []

        for i in range(self.num_outputs):
            bias = self.biases[i] if not self.skip_bias_add else None
            proj = F.linear(input, self.weights[i], bias)
            projections.append(proj)

            if self.skip_bias_add and self.biases[i] is not None:
                output_biases.append(self.biases[i])

        # Concatenate all projections to match the original output shape.
        merged_proj = torch.cat(projections, dim=-1)

        output_bias = (
            torch.cat(output_biases, dim=-1)
            if (self.skip_bias_add and output_biases)
            else None
        )

        if not self.return_bias:
            return merged_proj
        return merged_proj, output_bias


def partition_merged_column_parallel_linear(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    assert isinstance(layer, MergedColumnParallelLinear)
    xla_layer = XlaMergedColumnParallelLinear(layer, mesh)
    logger.info("Applied parallel sharding to %s", layer)
    return xla_layer


MODULE_TYPE_TO_WRAPPING_FUNC = OrderedDict(
    [("MergedColumnParallelLinear", partition_merged_column_parallel_linear)]
)


def get_fqn(module):
    # Get the fully qualified name of the module
    return module.__class__.__qualname__


def shard_model(model: torch.nn.Module, mesh: "xs.Mesh") -> None:
    """
    Recursively check a PyTorch model and apply appropriate sharding based on
    the MODULE_TYPE_TO_WRAPPING_FUNC mapping.

    Args:
        model: torch.nn.Module to process
        mesh: An XLA SPMD mesh object used for sharding
    """
    logger.info("my sharding invoked")

    def _process_module(module, name=None, parent=None):
        for module_type, wrapping_func in MODULE_TYPE_TO_WRAPPING_FUNC.items():
            if get_fqn(module) == module_type:
                wrapped_module = wrapping_func(module, mesh)

                assert (
                    parent is not None and name is not None
                ), "Top Level module is not expected to be wrapped."
                if wrapped_module is not module:
                    # Wrapped module and module are different py object.
                    # The original module should be replaced by the
                    # wrapped_module.
                    logger.debug("replace %s with %s", module, wrapped_module)
                    setattr(parent, name, wrapped_module)

                module = wrapped_module
                break

        for child_name, child_module in list(module.named_children()):
            _process_module(child_module, child_name, module)

    _process_module(model)
