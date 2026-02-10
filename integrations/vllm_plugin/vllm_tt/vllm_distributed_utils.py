# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from collections import OrderedDict
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

    def __init__(self, qkv_linear: nn.Module, mesh: "xs.Mesh"):
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
        self._shard_weight(mesh)

    def _shard_weight(self, mesh: "xs.Mesh"):
        self.q_weight = Parameter(self.q_weight.to("xla"), requires_grad=False)
        self.k_weight = Parameter(self.k_weight.to("xla"), requires_grad=False)
        self.v_weight = Parameter(self.v_weight.to("xla"), requires_grad=False)
        xs.mark_sharding(self.q_weight, mesh, ("x", None))
        xs.mark_sharding(self.k_weight, mesh, ("x", None))
        xs.mark_sharding(self.v_weight, mesh, ("x", None))
        if self.q_bias is not None:
            assert (
                self.k_bias is not None and self.v_bias is not None
            ), "QKVParallelLinear should have q, k, and v biases together."
            self.q_bias = Parameter(self.q_bias.to("xla"), requires_grad=False)
            xs.mark_sharding(self.q_bias, mesh, ("x",))
            self.k_bias = Parameter(self.k_bias.to("xla"), requires_grad=False)
            xs.mark_sharding(self.k_bias, mesh, ("x",))
            self.v_bias = Parameter(self.v_bias.to("xla"), requires_grad=False)
            xs.mark_sharding(self.v_bias, mesh, ("x",))

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
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    assert isinstance(layer, MergedColumnParallelLinear)
    xla_layer = XlaMergedColumnParallelLinear(layer, mesh)
    logger.debug("Applied parallel sharding to %s", layer)
    return xla_layer


def partition_qkv_parallel_linear(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    assert isinstance(layer, QKVParallelLinear)
    xla_layer = XlaQKVParallelLinear(layer, mesh)
    logger.debug("Applied parallel sharding to %s", layer)
    return xla_layer


def partition_column_parallel_linear(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    assert isinstance(layer, ColumnParallelLinear)
    xs.mark_sharding(layer.weight, mesh, ("x", None))
    logger.debug("Applied parallel sharding to %s", layer)
    return layer


def partition_row_parallel_linear(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    assert isinstance(layer, RowParallelLinear)
    xs.mark_sharding(layer.weight, mesh, (None, "x"))
    logger.debug("Applied parallel sharding to %s", layer)
    return layer


def partition_parallel_lm_head(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    assert isinstance(layer, ParallelLMHead)
    logger.debug("Applied parallel sharding to %s", layer)
    xs.mark_sharding(layer.weight, mesh, ("x", None))
    return layer


def partition_vocab_parallel_embedding(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    assert isinstance(layer, VocabParallelEmbedding)
    logger.debug("Applied parallel sharding to %s", layer)
    xs.mark_sharding(layer.weight, mesh, (None, "x"))
    # Apply sharding constraint to the output of the layer.
    hook_forward = sharding_constraint_hook(layer, mesh, (None, None, None))
    layer.register_forward_hook(hook_forward)
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


def shard_model(model: torch.nn.Module, mesh: "xs.Mesh") -> None:
    """
    Recursively check a PyTorch model and apply appropriate sharding based on
    the MODULE_TYPE_TO_WRAPPING_FUNC mapping.

    Args:
        model: torch.nn.Module to process
        mesh: An XLA SPMD mesh object used for sharding
    """
    logger.info("Applying parallel sharding to the model...")

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


class TTRMSNorm(nn.Module):
    """TT-compatible RMSNorm replacement for vLLM's RMSNorm.

    vLLM's RMSNorm.forward_native accesses `self.weight.data`, which causes an
    AssertionError during torch.compile/torch.export tracing with FakeTensors.
    Accessing `.data` on a FakeTensor lifts it out of the fake tensor context,
    resulting in: "cannot call `.data` on a Tensor, the Tensor is a FakeTensor".

    This class reimplements the RMSNorm forward pass using `self.weight` directly
    (without `.data`), making it compatible with TT tracing and compilation.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        assert isinstance(layer, RMSNorm)
        self.hidden_size = layer.hidden_size
        self.variance_epsilon = layer.variance_epsilon
        self.variance_size_override = layer.variance_size_override
        self.has_weight = layer.has_weight
        self.weight = layer.weight

        if hasattr(layer, "rocm_norm_func") and hasattr(
            layer, "rocm_norm_func_with_add"
        ):
            self.rocm_norm_func = layer.rocm_norm_func
            self.rocm_norm_func_with_add = layer.rocm_norm_func_with_add

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            # residual promoted f16->f32 automatically,
            # otherwise Inductor eliminates the casts to and from f16,
            # increasing memory usage (and complicating pattern matching)
            x = x + residual
            residual = x.to(orig_dtype)

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size to be {self.hidden_size}, but found: {x.shape[-1]}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if self.hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {self.hidden_size}"
                )

            x_var = x[:, :, : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        if self.has_weight and self.weight is not None:
            x = x * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def tt_rmsnorm_module(layer: torch.nn.Module) -> torch.nn.Module:
    assert isinstance(layer, RMSNorm)
    tt_layer = TTRMSNorm(layer)
    # logger.info("Wrapped RMSNorm module %s with XLA-compatible version", layer)
    return tt_layer


def replace_rmsnorm_modules(model: torch.nn.Module) -> None:
    def _process_module(module, name=None, parent=None):
        if get_fqn(module) == "RMSNorm":
            wrapped_module = tt_rmsnorm_module(module)

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

        for child_name, child_module in list(module.named_children()):
            _process_module(child_module, child_name, module)

    _process_module(model)
