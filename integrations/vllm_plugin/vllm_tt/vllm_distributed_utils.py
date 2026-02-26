# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.distributed.spmd as xs
from torch.nn.parameter import Parameter
from tt_torch.sharding import sharding_constraint_hook, sharding_constraint_tensor
from tt_torch.sparse_mlp import A2aSparseMLP
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

from .attention import TTAttentionBackendImpl
from .logger import tt_init_logger
from .moe_integration import VllmA2aSparseMLP
from .overrides import TTRMSNorm

logger = tt_init_logger(__name__)


def create_default_mesh() -> "xs.Mesh":
    """Create the default mesh configuration for TT sharding.

    Returns:
        xs.Mesh: Default mesh with shape (2, 4) and axes ("batch", "model")
    """
    mesh_shape = (2, 4)
    device_ids = np.array(range(8))
    mesh = xs.Mesh(device_ids, mesh_shape, ("batch", "model"))
    return mesh


def create_custom_mesh(mesh_shape: tuple, axes: tuple) -> "xs.Mesh":
    """Create a custom mesh configuration for specialized sharding needs.

    Args:
        mesh_shape: Shape of the mesh (e.g., (2, 4) or (1, 8))
        axes: Axis names (e.g., ("batch", "model") or ("data", "fsdp"))

    Returns:
        xs.Mesh: Custom mesh configuration
    """
    num_devices = np.prod(mesh_shape)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, axes)
    return mesh


def partition_with_config(
    layer: torch.nn.Module, mesh: xs.Mesh, sharding_config: dict
) -> torch.nn.Module:
    """Apply sharding based on a configuration dictionary.

    Args:
        layer: Layer to shard
        mesh: Mesh for sharding
        sharding_config: Dictionary mapping attribute names to sharding specs

    Example:
        config = {
            "weight": ("batch", "model"),
            "bias": ("batch",),
            "experts.gate_proj": (("batch", "model"), None, None)
        }
    """
    logger.debug("Applied configurable sharding to %s", layer)

    for attr_path, sharding_spec in sharding_config.items():
        # Navigate nested attributes (e.g., "experts.gate_proj")
        obj = layer
        for attr in attr_path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                logger.warning(f"Attribute {attr_path} not found in layer")
                break
        else:
            # Only shard if we successfully navigated to the attribute
            if hasattr(obj, "to"):  # Check if it's a tensor
                if not obj.device.type == "xla":
                    obj = obj.to("xla")
                xs.mark_sharding(obj, mesh, sharding_spec)

    return layer


def shard_sinks_tensor(
    sinks: torch.Tensor, mesh: Optional["xs.Mesh"] = None
) -> torch.Tensor:
    """Apply sharding to sinks tensor.

    Args:
        sinks: Tensor to shard
        mesh: Optional mesh, creates default if None

    Returns:
        Sharded tensor
    """
    if mesh is None:
        mesh = create_default_mesh()

    if not sinks.device.type == "xla":
        sinks = sinks.to("xla")
    xs.mark_sharding(sinks, mesh, ("batch",))
    return sinks


def shard_weight_tensor(
    weight: torch.Tensor,
    mesh: Optional["xs.Mesh"] = None,
    sharding_spec: tuple = ("batch", "model"),
) -> torch.Tensor:
    """Apply sharding to weight tensors with flexible sharding specification.

    Args:
        weight: Weight tensor to shard
        mesh: Optional mesh, creates default if None
        sharding_spec: Sharding specification tuple

    Returns:
        Sharded tensor
    """
    if mesh is None:
        mesh = create_default_mesh()

    if not weight.device.type == "xla":
        weight = weight.to("xla")
    xs.mark_sharding(weight, mesh, sharding_spec)
    return weight


def shard_expert_tensors(
    layer: torch.nn.Module, mesh: Optional["xs.Mesh"] = None
) -> torch.nn.Module:
    """Apply expert-specific sharding to MoE-style layers.

    Args:
        layer: Layer containing expert tensors
        mesh: Optional mesh, creates default if None

    Returns:
        Layer with sharded expert tensors
    """
    if mesh is None:
        mesh = create_default_mesh()

    # Check if layer has experts attribute and apply sharding
    if hasattr(layer, "experts"):
        experts = layer.experts
        if hasattr(experts, "gate_up_proj"):
            xs.mark_sharding(
                experts.gate_up_proj, mesh, (("batch", "model"), None, None)
            )
        if hasattr(experts, "gate_up_proj_bias"):
            xs.mark_sharding(
                experts.gate_up_proj_bias, mesh, (("batch", "model"), None)
            )
        if hasattr(experts, "down_proj"):
            xs.mark_sharding(experts.down_proj, mesh, (("batch", "model"), None, None))
        if hasattr(experts, "down_proj_bias"):
            xs.mark_sharding(experts.down_proj_bias, mesh, (("batch", "model"), None))

    return layer


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
            xs.mark_sharding(self.weights[i], mesh, ("batch", "model"))

            if self.biases[i] is not None:
                self.biases[i] = Parameter(
                    self.biases[i].to("xla"), requires_grad=False
                )
                xs.mark_sharding(self.biases[i], mesh, ("batch",))

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
        xs.mark_sharding(self.q_weight, mesh, ("batch", "model"))
        xs.mark_sharding(self.k_weight, mesh, ("batch", "model"))
        xs.mark_sharding(self.v_weight, mesh, ("batch", "model"))
        if self.q_bias is not None:
            assert (
                self.k_bias is not None and self.v_bias is not None
            ), "QKVParallelLinear should have q, k, and v biases together."
            self.q_bias = Parameter(self.q_bias.to("xla"), requires_grad=False)
            xs.mark_sharding(self.q_bias, mesh, ("batch",))
            self.k_bias = Parameter(self.k_bias.to("xla"), requires_grad=False)
            xs.mark_sharding(self.k_bias, mesh, ("batch",))
            self.v_bias = Parameter(self.v_bias.to("xla"), requires_grad=False)
            xs.mark_sharding(self.v_bias, mesh, ("batch",))

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
    xs.mark_sharding(layer.weight, mesh, ("batch", None))
    logger.debug("Applied parallel sharding to %s", layer)
    return layer


def partition_row_parallel_linear(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    assert isinstance(layer, RowParallelLinear)
    xs.mark_sharding(layer.weight, mesh, ("model", "batch"))
    logger.debug("Applied parallel sharding to %s", layer)
    return layer


def partition_parallel_lm_head(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    assert isinstance(layer, ParallelLMHead)
    logger.debug("Applied parallel sharding to %s", layer)
    xs.mark_sharding(layer.weight, mesh, ("batch", None))
    return layer


def partition_vocab_parallel_embedding(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    assert isinstance(layer, VocabParallelEmbedding)
    logger.debug("Applied parallel sharding to %s", layer)
    xs.mark_sharding(layer.weight, mesh, (None, "batch"))
    # Apply sharding constraint to the output of the layer.
    hook_forward = sharding_constraint_hook(layer, mesh, (None, None, None))
    layer.register_forward_hook(hook_forward)
    return layer


def partition_vllm_a2a_sparse_mlp(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    assert isinstance(layer, A2aSparseMLP)
    logger.debug("Applied parallel sharding to %s", layer)
    # logger.info(f"layer: {layer}")
    # logger.info(f"layer.experts: {layer.experts}")
    # logger.info(f"layer.experts.weights: {layer.experts.weights}")
    # logger.info(f"Sharding expert dimensions across mesh {mesh}")
    # Shard expert dimension (first dim) across 'model' axis
    xs.mark_sharding(layer.experts.gate_up_proj, mesh, (("batch", "model"), None, None))
    xs.mark_sharding(layer.experts.gate_up_proj_bias, mesh, (("batch", "model"), None))
    xs.mark_sharding(layer.experts.down_proj, mesh, (("batch", "model"), None, None))
    xs.mark_sharding(layer.experts.down_proj_bias, mesh, (("batch", "model"), None))

    return layer


def partition_rmsnorm_layer(layer: torch.nn.Module, mesh: xs.Mesh) -> torch.nn.Module:
    assert isinstance(layer, TTRMSNorm)
    logger.debug("Applied parallel sharding to %s", layer)
    logger.info(f"layer.weight.shape={layer.weight.shape}")
    xs.mark_sharding(layer.weight, mesh, ("model",))
    # Apply sharding constraint to the output of the layer.
    # hook_forward = sharding_constraint_hook(layer, mesh, (None, None, None))
    # layer.register_forward_hook(hook_forward)
    return layer


def partition_attention_backend_impl(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    # assert isinstance(layer, TTAttentionBackendImpl)
    logger.debug("Applied parallel sharding to %s", layer)
    logger.info(f"hasattr(layer, 'sinks')={hasattr(layer, 'sinks')}")
    if not hasattr(layer, "sinks"):
        return layer
    logger.info(f"layer.attn: {layer.attn if hasattr(layer, 'attn') else 'N/A'}")
    logger.info(
        f"layer.sinks: {layer.attn.sinks if hasattr(layer, 'attn') and hasattr(layer.attn, 'sinks') else 'N/A'}"
    )
    sinks = getattr(layer, "sinks")
    logger.info(
        f"Attention backend impl sinks shape: {sinks.shape if sinks is not None else 'None'}"
    )
    if sinks is None:
        return layer
    logger.info(
        f"Sharding attention backend impl sinks across mesh {mesh} -- {sinks.device}"
    )
    if not sinks.device.type == "xla":
        sinks = sinks.to("xla")
    # Apply the same sharding constraints that were working in attention.py
    # Use sharding_constraint_tensor to apply proper batch sharding
    # sharded_sinks = sharding_constraint_tensor(sinks, mesh, ("batch",))
    # layer.sinks = sharded_sinks
    xs.mark_sharding(sinks, mesh, ("batch",))
    if "sinks" in dict(layer.named_buffers()):
        logger.info(f"Asif::0")
        layer.register_buffer("sinks", sinks)
    else:
        logger.info(f"Asif::1")
        setattr(layer, "sinks", sinks)
    return layer


def partition_custom_attention_layer(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    """Custom sharding for attention layers with specific requirements."""
    logger.debug("Applied parallel sharding to %s", layer)

    # Example: Shard attention weights
    if hasattr(layer, "attention_weights"):
        xs.mark_sharding(layer.attention_weights, mesh, ("batch", "model"))

    # Example: Shard bias terms
    if hasattr(layer, "attention_bias"):
        xs.mark_sharding(layer.attention_bias, mesh, ("batch",))

    return layer


def partition_custom_mlp_layer(
    layer: torch.nn.Module, mesh: xs.Mesh
) -> torch.nn.Module:
    """Custom sharding for MLP layers."""
    logger.debug("Applied parallel sharding to %s", layer)

    # Shard common MLP components
    if hasattr(layer, "fc1"):
        xs.mark_sharding(layer.fc1.weight, mesh, ("batch", "model"))
    if hasattr(layer, "fc2"):
        xs.mark_sharding(layer.fc2.weight, mesh, ("model", "batch"))
    return layer


MODULE_TYPE_TO_WRAPPING_FUNC = OrderedDict(
    [
        ("MergedColumnParallelLinear", partition_merged_column_parallel_linear),
        ("QKVParallelLinear", partition_qkv_parallel_linear),
        ("ColumnParallelLinear", partition_column_parallel_linear),
        ("RowParallelLinear", partition_row_parallel_linear),
        ("ParallelLMHead", partition_parallel_lm_head),
        ("VocabParallelEmbedding", partition_vocab_parallel_embedding),
        ("A2aSparseMLP", partition_vllm_a2a_sparse_mlp),
        ("TTRMSNorm", partition_rmsnorm_layer),
        # Add your custom layer types here:
        # ("CustomAttentionLayer", partition_custom_attention_layer),
        # ("CustomMLPLayer", partition_custom_mlp_layer),
        # ("attn", partition_attention_backend_impl),
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
        # logger.info(f"Processing module: {module} (name: {name})")
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

        for module_type, wrapping_func in MODULE_TYPE_TO_WRAPPING_FUNC.items():
            logger.info(f"Checking module type: {name} against {module_type}")
            if name == module_type:
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
