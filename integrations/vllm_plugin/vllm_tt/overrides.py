# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

from typing import OrderedDict, cast

import torch
import torch.nn as nn
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.fused_moe import layer as fused_moe_layer
from vllm.model_executor.layers.fused_moe.router import fused_topk_router
from vllm.model_executor.layers.layernorm import RMSNorm

from .logger import tt_init_logger

logger = tt_init_logger(__name__)


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


def tt_get_layer_from_name(layer_name: str):
    """TT-compatible MoE layer lookup that handles layer reduction.

    When num_hidden_layers is reduced (e.g., from 24 to 1), the compiled graph
    still contains multiple MoE forward calls but fewer layers are registered.
    This function wraps the layer index to reuse available layers.
    """
    forward_context: ForwardContext = get_forward_context()
    if layer_name == "from_forward_context":
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        logger.info(
            f"moe_layer_index: {moe_layer_index}, all_moe_layers: {len(all_moe_layers)}"
        )

        # Handle layer reduction: wrap around if index exceeds available layers
        # This occurs when num_hidden_layers is reduced but compiled graph still has multiple MoE calls
        if moe_layer_index >= len(all_moe_layers):
            logger.warning(
                f"MoE layer index {moe_layer_index} exceeds available layers {len(all_moe_layers)}, wrapping to index {moe_layer_index % len(all_moe_layers)}"
            )
            moe_layer_index = moe_layer_index % len(all_moe_layers)

        layer_name = all_moe_layers[moe_layer_index]
        forward_context.moe_layer_index += 1

    # Import here to avoid circular import issues
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    self = cast(FusedMoE, forward_context.no_compile_layers[layer_name])
    return self


def tt_fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype = None,
    scoring_func: str = "softmax",
):
    """TT-compatible version of fused_topk that handles multi-dimensional tensors."""
    # Flatten hidden_states to 2D if it has more dimensions
    original_hidden_shape = hidden_states.shape
    if len(hidden_states.shape) > 2:
        # Reshape to [batch_size * seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    # Ensure gating_output matches the batch dimension
    original_gating_shape = gating_output.shape
    if len(gating_output.shape) > 2:
        gating_output = gating_output.view(-1, gating_output.shape[-1])

    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    # Pure PyTorch implementation without C++ extensions
    if scoring_func == "softmax":
        # Apply softmax to gating outputs
        routing_weights = torch.softmax(gating_output.float(), dim=-1)
    elif scoring_func == "sigmoid":
        # Apply sigmoid to gating outputs
        routing_weights = torch.sigmoid(gating_output.float())
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    # Get top-k weights and indices
    topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)

    if renormalize:
        # Renormalize the top-k weights to sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Ensure correct dtype for indices
    if indices_type is not None:
        topk_ids = topk_ids.to(indices_type)
    else:
        topk_ids = topk_ids.to(torch.int32)

    # Create token_expert_indices (flattened version of topk_ids for expert assignment)
    token_expert_indices = topk_ids.flatten()

    return topk_weights, topk_ids, token_expert_indices


def get_fqn(module):
    return module.__class__.__qualname__


def tt_rmsnorm_module(layer: torch.nn.Module) -> torch.nn.Module:
    assert isinstance(layer, RMSNorm)
    tt_layer = TTRMSNorm(layer)
    logger.debug("Wrapped RMSNorm module %s with TT-compatible version", layer)
    return tt_layer


MODULE_TYPE_TO_TT_OVERRIDE = OrderedDict(
    [
        ("RMSNorm", tt_rmsnorm_module),
    ]
)


def replace_modules(model: torch.nn.Module) -> None:
    logger.info(
        "Replacing vLLM modules with TT-compatible overrides where necessary..."
    )

    # Apply monkey-patch to fused_topk function for tensor shape compatibility
    if hasattr(fused_topk_router, "fused_topk"):
        logger.info("Monkey-patching fused_topk function for TT tensor compatibility")
        fused_topk_router.fused_topk = tt_fused_topk

    # Apply monkey-patch to get_layer_from_name function for MoE layer reduction compatibility
    if hasattr(fused_moe_layer, "get_layer_from_name"):
        logger.info(
            "Monkey-patching get_layer_from_name function for TT MoE layer reduction compatibility"
        )
        fused_moe_layer.get_layer_from_name = tt_get_layer_from_name

    def _process_module(module, name=None, parent=None):
        if get_fqn(module) in MODULE_TYPE_TO_TT_OVERRIDE:
            tt_override_cls = MODULE_TYPE_TO_TT_OVERRIDE[get_fqn(module)]
            replacement_module = tt_override_cls(module)

            assert (
                parent is not None and name is not None
            ), "Top Level module is not expected to be wrapped."
            logger.debug("replace %s with %s", module, replacement_module)
            setattr(parent, name, replacement_module)

            module = replacement_module

        for child_name, child_module in list(module.named_children()):
            _process_module(child_module, child_name, module)

    _process_module(model)
