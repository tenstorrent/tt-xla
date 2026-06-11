# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from collections import OrderedDict
from enum import Enum
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.distributed.spmd as xs
from torch.nn.parameter import Parameter
from tt_torch.sharding import sharding_constraint_hook
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)

from .logger import tt_init_logger

logger = tt_init_logger(__name__)


def safe_mark_sharding(tensor, mesh, partition_spec, strict=False):
    """xs.mark_sharding that replicates any dim whose size is not divisible
    by the mesh axis it would be sharded on. When ``strict`` is True, raise
    instead of falling back to replication — use this from call sites where
    silent replication would mask a perf regression."""
    if not hasattr(mesh, "shape"):
        axis_sizes = dict(zip(mesh.axis_names, mesh.mesh_shape))
    else:
        axis_sizes = mesh.shape()

    safe_spec = []
    for i, axis in enumerate(partition_spec):
        if axis is None or i >= tensor.ndim:
            safe_spec.append(None)
            continue
        mesh_axis_size = axis_sizes.get(axis)
        if mesh_axis_size is None or tensor.shape[i] % mesh_axis_size != 0:
            msg = (
                f"safe_mark_sharding: dim {i} (size {tensor.shape[i]}) "
                f"not divisible by mesh axis {axis!r} (size {mesh_axis_size})"
            )
            if strict:
                raise ValueError(msg + "; strict=True")
            logger.warning("%s; replicating instead", msg)
            safe_spec.append(None)
        else:
            safe_spec.append(axis)
    xs.mark_sharding(tensor, mesh, tuple(safe_spec))


class ParallelismMode(Enum):
    DISABLED = "disabled"
    DATA_PARALLEL_ONLY = "data_parallel_only"
    TENSOR_PARALLEL_ONLY_1D = "tensor_parallel_only_1D"
    TENSOR_PARALLEL_ONLY_2D = "tensor_parallel_only_2d"
    DATA_TENSOR_PARALLEL = "data_tensor_parallel"


class XlaMergedColumnParallelLinear(nn.Module):

    def __init__(
        self,
        merged_column_parallel_linear: nn.Module,
        mesh: "xs.Mesh",
        shard_weights_on_batch_axis: bool = True,
    ):
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
            self._shard_weight(mesh, shard_weights_on_batch_axis)

    def _shard_weight(self, mesh: "xs.Mesh", shard_weights_on_batch_axis: bool):
        batch_axis = "batch" if shard_weights_on_batch_axis else None
        for i in range(self.num_outputs):
            self.weights[i] = Parameter(self.weights[i].to("xla"), requires_grad=False)
            safe_mark_sharding(self.weights[i], mesh, ("model", batch_axis))

            if self.biases[i] is not None:
                self.biases[i] = Parameter(
                    self.biases[i].to("xla"), requires_grad=False
                )
                safe_mark_sharding(self.biases[i], mesh, ("model",))

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
                start_idx = end_idx
        else:
            for i in range(self.num_outputs):
                self.biases.append(None)

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


class XlaQKVParallelLinear(nn.Module):

    def __init__(
        self,
        qkv_linear: nn.Module,
        mesh: "xs.Mesh",
        shard_weights_on_batch_axis: bool = True,
    ):
        super().__init__()
        assert isinstance(qkv_linear, QKVParallelLinear)
        self.skip_bias_add = qkv_linear.skip_bias_add
        self.return_bias = qkv_linear.return_bias
        assert qkv_linear.tp_size == 1, "TP > 1 is only supported under SPMD."

        self.q_weight: Parameter
        self.k_weight: Parameter
        self.v_weight: Parameter
        self.q_bias: Optional[Parameter]
        self.k_bias: Optional[Parameter]
        self.v_bias: Optional[Parameter]
        self._load_weights_from_qkv_linear(qkv_linear)
        self._shard_weight(mesh, shard_weights_on_batch_axis)

    def _shard_weight(self, mesh: "xs.Mesh", shard_weights_on_batch_axis: bool):
        batch_axis = "batch" if shard_weights_on_batch_axis else None
        self.q_weight = Parameter(self.q_weight.to("xla"), requires_grad=False)
        self.k_weight = Parameter(self.k_weight.to("xla"), requires_grad=False)
        self.v_weight = Parameter(self.v_weight.to("xla"), requires_grad=False)
        safe_mark_sharding(self.q_weight, mesh, ("model", batch_axis))
        safe_mark_sharding(self.k_weight, mesh, ("model", batch_axis))
        safe_mark_sharding(self.v_weight, mesh, ("model", batch_axis))
        if self.q_bias is not None:
            assert (
                self.k_bias is not None and self.v_bias is not None
            ), "QKVParallelLinear should have q, k, and v biases together."
            self.q_bias = Parameter(self.q_bias.to("xla"), requires_grad=False)
            safe_mark_sharding(self.q_bias, mesh, ("model",))
            self.k_bias = Parameter(self.k_bias.to("xla"), requires_grad=False)
            safe_mark_sharding(self.k_bias, mesh, ("model",))
            self.v_bias = Parameter(self.v_bias.to("xla"), requires_grad=False)
            safe_mark_sharding(self.v_bias, mesh, ("model",))

    def _load_weights_from_qkv_linear(self, qkv_linear: nn.Module):
        q_proj_size, k_proj_size, _ = qkv_linear.output_sizes
        # The weight of qkv linear is a concatenation of q, k, and v weights
        # along the output dimension.
        qkv_weight = qkv_linear.weight.data.cpu()
        q_weight = Parameter(qkv_weight[:q_proj_size], requires_grad=False)
        k_weight = Parameter(
            qkv_weight[q_proj_size : q_proj_size + k_proj_size], requires_grad=False
        )
        v_weight = Parameter(
            qkv_weight[q_proj_size + k_proj_size :], requires_grad=False
        )
        self.register_parameter("q_weight", q_weight)
        self.register_parameter("k_weight", k_weight)
        self.register_parameter("v_weight", v_weight)

        if qkv_linear.bias is not None:
            q_bias = Parameter(qkv_linear.bias[:q_proj_size], requires_grad=False)
            k_bias = Parameter(
                qkv_linear.bias[q_proj_size : q_proj_size + k_proj_size],
                requires_grad=False,
            )
            v_bias = Parameter(
                qkv_linear.bias[q_proj_size + k_proj_size :], requires_grad=False
            )
            self.register_parameter("q_bias", q_bias)
            self.register_parameter("k_bias", k_bias)
            self.register_parameter("v_bias", v_bias)
        else:
            self.register_parameter("q_bias", None)
            self.register_parameter("k_bias", None)
            self.register_parameter("v_bias", None)

    def forward(self, input):
        # Same forward functionality as QKVParallelLinear, but doing qkv porj
        # separately.
        q_bias = self.q_bias if not self.skip_bias_add else None
        k_bias = self.k_bias if not self.skip_bias_add else None
        v_bias = self.v_bias if not self.skip_bias_add else None
        q_proj = F.linear(input, self.q_weight, q_bias)
        k_proj = F.linear(input, self.k_weight, k_bias)
        v_proj = F.linear(input, self.v_weight, v_bias)
        # The q/k/v projections will be split outside of the QKVParallelLinear.
        # Because we are replacing XlaQKVParallelLinear with the
        # QKVParallelLinear, we need to concatenate q, k, and v projections to
        # match the output shape of the QKVParallelLinear implementation even if
        # it seems to be redundant.
        # The concat and the following split will be noop, and should be
        # optimized away by the compiler.
        qkv_proj = torch.cat([q_proj, k_proj, v_proj], dim=-1)
        output_bias = (
            torch.cat([q_bias, k_bias, v_bias], dim=-1) if self.skip_bias_add else None
        )
        if not self.return_bias:
            return qkv_proj
        return qkv_proj, output_bias


def partition_merged_column_parallel_linear(
    layer: torch.nn.Module, mesh: xs.Mesh, shard_weights_on_batch_axis: bool = True
) -> torch.nn.Module:
    assert isinstance(layer, MergedColumnParallelLinear)
    xla_layer = XlaMergedColumnParallelLinear(layer, mesh, shard_weights_on_batch_axis)
    logger.debug("Applied parallel sharding to %s", layer)
    return xla_layer


def partition_qkv_parallel_linear(
    layer: torch.nn.Module, mesh: xs.Mesh, shard_weights_on_batch_axis: bool = True
) -> torch.nn.Module:
    assert isinstance(layer, QKVParallelLinear)
    xla_layer = XlaQKVParallelLinear(layer, mesh, shard_weights_on_batch_axis)
    logger.debug("Applied parallel sharding to %s", layer)
    return xla_layer


def partition_column_parallel_linear(
    layer: torch.nn.Module, mesh: xs.Mesh, shard_weights_on_batch_axis: bool = True
) -> torch.nn.Module:
    assert isinstance(layer, ColumnParallelLinear)
    safe_mark_sharding(layer.weight, mesh, ("model", None))
    logger.debug("Applied parallel sharding to %s", layer)
    return layer


def partition_row_parallel_linear(
    layer: torch.nn.Module, mesh: xs.Mesh, shard_weights_on_batch_axis: bool = True
) -> torch.nn.Module:
    assert isinstance(layer, RowParallelLinear)
    batch_axis = "batch" if shard_weights_on_batch_axis else None
    safe_mark_sharding(layer.weight, mesh, (batch_axis, "model"))
    logger.debug("Applied parallel sharding to %s", layer)
    return layer


def partition_parallel_lm_head(
    layer: torch.nn.Module, mesh: xs.Mesh, shard_weights_on_batch_axis: bool = True
) -> torch.nn.Module:
    assert isinstance(layer, ParallelLMHead)
    logger.debug("Applied parallel sharding to %s", layer)
    xs.mark_sharding(layer.weight, mesh, ("model", None))
    return layer


def partition_vocab_parallel_embedding(
    layer: torch.nn.Module, mesh: xs.Mesh, shard_weights_on_batch_axis: bool = True
) -> torch.nn.Module:
    assert isinstance(layer, VocabParallelEmbedding)
    safe_mark_sharding(layer.weight, mesh, (None, "model"))
    # Apply sharding constraint to the output of the layer.
    hook_forward = sharding_constraint_hook(layer, mesh, (None, None, None))
    layer.register_forward_hook(hook_forward)
    logger.debug("Applied parallel sharding to %s", layer)
    return layer


MODULE_TYPE_TO_WRAPPING_FUNC = OrderedDict(
    [
        ("MergedColumnParallelLinear", partition_merged_column_parallel_linear),
        ("QKVParallelLinear", partition_qkv_parallel_linear),
        ("ColumnParallelLinear", partition_column_parallel_linear),
        ("RowParallelLinear", partition_row_parallel_linear),
        ("ParallelLMHead", partition_parallel_lm_head),
        ("VocabParallelEmbedding", partition_vocab_parallel_embedding),
    ]
)


def get_fqn(module):
    # Get the fully qualified name of the module
    return module.__class__.__qualname__


def shard_model(
    model: torch.nn.Module,
    mesh: "xs.Mesh",
    shard_weights_on_batch_axis: bool = True,
) -> None:
    """
    Recursively check a PyTorch model and apply appropriate sharding based on
    the MODULE_TYPE_TO_WRAPPING_FUNC mapping.

    Args:
        model: torch.nn.Module to process
        mesh: An XLA SPMD mesh object used for sharding
        shard_weights_on_batch_axis: When True, weight partition specs include
            the "batch" (DP) axis (FSDP-style). When False, weights are only
            sharded on "model" (TP) axis and replicated across DP replicas.
    """
    logger.info("Applying parallel sharding to the model...")

    def _process_module(module, name=None, parent=None):
        for module_type, wrapping_func in MODULE_TYPE_TO_WRAPPING_FUNC.items():
            if get_fqn(module) == module_type:
                wrapped_module = wrapping_func(
                    module, mesh, shard_weights_on_batch_axis
                )

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
