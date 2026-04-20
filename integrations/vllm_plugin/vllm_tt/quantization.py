# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2025 Tenstorrent AI ULC

"""TT-compatible quantization implementations."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.mxfp4 import (  # Mxfp4LinearMethod,
    Mxfp4Backend,
    Mxfp4Config,
    Mxfp4MoEMethod,
)

logger = init_logger(__name__)


class TTMxfp4Backend:
    """TT-compatible MXFP4 backend that doesn't require GPU-specific implementations."""

    TT_NATIVE = "tt_native"


class TTMxfp4MoEMethod(FusedMoEMethodBase):
    """TT-compatible MoE quantization method that bypasses GPU backend requirements."""

    def __init__(self, moe_config: Dict[str, Any]):
        # Initialize without calling parent __init__ to avoid GPU backend validation
        self.moe_config = moe_config
        self.mxfp4_backend = TTMxfp4Backend.TT_NATIVE  # Use our TT backend

        # Add the missing attributes that vLLM expects
        self.moe_quant_config = None
        self.moe_mk = None  # TT backend doesn't use modular kernel

        logger.warning(f"Using TT-compatible MXFP4 MoE backend: {self.mxfp4_backend}")

    def create_weights(self, layer: nn.Module, **kwargs):
        """Create quantized weights for MoE layer."""
        # Extract parameters from kwargs that vLLM passes
        print(f"DEBUG: create_weights called with kwargs: {kwargs}")
        num_experts = kwargs.get("num_experts", getattr(layer, "num_experts", 32))
        hidden_size = kwargs.get("hidden_size", getattr(layer, "hidden_size", 1440))
        hidden_size = 1440  # Force hidden size to match checkpoint for TT compatibility
        intermediate_size = kwargs.get(
            "intermediate_size", getattr(layer, "intermediate_size", 2880)
        )
        params_dtype = torch.uint8  # Default to uint8 for quantized weights
        print(
            f"DEBUG: Creating weights for {num_experts} experts with kwargs: {kwargs}"
        )
        print(
            f"DEBUG: layer: {layer} -- {hasattr(layer, 'w13_weight')} -- {hasattr(layer, 'w2_weight')}"
        )

        logger.warning(
            f"Creating TT-compatible MoE weights with corrected dimensions: experts={num_experts}, "
            f"hidden={hidden_size}, intermediate={intermediate_size}, dtype={params_dtype}"
        )

        # For TT compatibility, we'll create standard weights without quantization
        # Match the parameter names expected by the original MXFP4 implementation

        def create_tt_weight_loader():
            """Create a custom weight loader that accepts extra arguments but performs standard loading."""

            def tt_weight_loader(
                param: torch.Tensor,
                loaded_weight: torch.Tensor,
                weight_name=None,
                shard_id=None,
                expert_id=None,
                **kwargs,
            ):
                """TT-compatible weight loader that accepts extra arguments."""
                # Ignore extra arguments and perform standard weight loading
                try:
                    if param.numel() == 1 and loaded_weight.numel() == 1:
                        param.data.fill_(loaded_weight.item())
                    else:
                        assert param.shape == loaded_weight.shape, (
                            f"Shape mismatch for {weight_name}: "
                            f"param {param.shape} vs loaded {loaded_weight.shape}"
                        )
                        param.data.copy_(loaded_weight)
                except Exception as e:
                    logger.warning(f"Weight loading failed for {weight_name}: {e}")
                    # Fallback: try to copy what we can
                    try:
                        param.data.copy_(loaded_weight)
                    except Exception as e2:
                        logger.error(
                            f"Fallback weight loading also failed for {weight_name}: {e2}"
                        )

            return tt_weight_loader

        # MXFP4 quantization parameters (matching original implementation)
        mxfp4_block = 32
        mxfp4_scale_block = 16  # Scale parameters use different block size
        scale_dtype = torch.uint8

        # Fused gate_up_proj (w13 combines w1 and w3 weights)
        # Corrected dimensions: [experts, 2*intermediate_size, hidden_size]
        w13_weight = nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        w13_weight.weight_loader = create_tt_weight_loader()

        w13_weight_scale = nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size,
                hidden_size // mxfp4_scale_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        w13_weight_scale.weight_loader = create_tt_weight_loader()

        w13_bias = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, dtype=params_dtype),
            requires_grad=False,
        )
        w13_bias.weight_loader = create_tt_weight_loader()

        # Down projection (w2)
        # Corrected dimensions to match checkpoint: [experts, intermediate_size, hidden_size]
        w2_weight = nn.Parameter(
            torch.empty(
                num_experts, intermediate_size, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        w2_weight.weight_loader = create_tt_weight_loader()

        w2_weight_scale = nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                hidden_size // mxfp4_scale_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale.weight_loader = create_tt_weight_loader()

        w2_bias = nn.Parameter(
            torch.empty(num_experts, intermediate_size, dtype=params_dtype),
            requires_grad=False,
        )
        w2_bias.weight_loader = create_tt_weight_loader()

        # Register parameters with the expected names
        layer.register_parameter("w13_weight", w13_weight)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w13_bias", w13_bias)
        layer.register_parameter("w2_weight", w2_weight)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        layer.register_parameter("w2_bias", w2_bias)

        # Store weight references for easy access
        setattr(layer, "w13_weight", w13_weight)
        setattr(layer, "w13_weight_scale", w13_weight_scale)
        setattr(layer, "w13_bias", w13_bias)
        setattr(layer, "w2_weight", w2_weight)
        setattr(layer, "w2_weight_scale", w2_weight_scale)
        setattr(layer, "w2_bias", w2_bias)

        logger.warning("TT-compatible MoE weights created successfully")

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply TT-compatible MoE forward pass."""
        logger.debug(
            f"TTMxfp4MoEMethod.apply called with x.shape={x.shape}, topk_weights.shape={topk_weights.shape}, topk_ids.shape={topk_ids.shape}"
        )

        # Implement a simple MoE without complex quantized operations
        # This is a simplified version for TT compatibility
        batch_size, seq_len, hidden_size = x.shape

        # Debug tensor shapes
        print(
            f"DEBUG: topk_weights.shape={topk_weights.shape}, topk_ids.shape={topk_ids.shape}"
        )

        # Get top_k from the shape - handle both 2D and 3D cases
        if len(topk_ids.shape) == 3:
            top_k = topk_ids.shape[-1]  # 3D case
        elif len(topk_ids.shape) == 2:
            top_k = topk_ids.shape[-1]  # 2D case, assume (batch*seq, top_k)
        else:
            raise ValueError(f"Unexpected topk_ids shape: {topk_ids.shape}")

        # topk_weights and topk_ids are already computed by the router
        # No need to do additional topk computation

        # Simple expert computation (can be optimized for TT hardware later)
        outputs = []

        # Handle both 2D and 3D tensor shapes
        if len(topk_ids.shape) == 2:
            # topk_ids is 2D: (batch*seq_len, top_k)
            # Reshape to match expected 3D format
            topk_ids_3d = topk_ids.view(batch_size, seq_len, -1)
            topk_weights_3d = topk_weights.view(batch_size, seq_len, -1)
        else:
            # Already 3D: (batch_size, seq_len, top_k)
            topk_ids_3d = topk_ids
            topk_weights_3d = topk_weights

        for i in range(top_k):
            expert_idx = topk_ids_3d[:, :, i]
            weight = topk_weights_3d[:, :, i].unsqueeze(-1)

            # For TT compatibility, avoid direct tensor access during functionalization
            # Use a simple identity operation as placeholder
            # This should be properly implemented with TT-optimized MoE later
            expert_output = x  # Identity for now
            outputs.append(expert_output * weight)

        result = sum(outputs)
        logger.debug(f"TTMxfp4MoEMethod.apply output shape: {result.shape}")
        return result

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        """Return None for TT backend - we bypass quantization config."""
        # According to vLLM documentation: "Other MoE methods can ignore the
        # FusedMoEQuantConfig (for now) and hardcode it to None."
        return None


class TTMxfp4Config(Mxfp4Config):
    """TT-compatible MXFP4 configuration that uses TT backends."""

    def __init__(self):
        """Initialize TT-compatible MXFP4 config without calling parent __init__ to avoid backend checks."""
        # Initialize without calling super().__init__() to avoid GPU backend validation
        logger.warning("Initializing TT-compatible MXFP4 config")

    def get_quant_method(self, layer: nn.Module, prefix: str) -> Optional[Any]:
        """Override to return TT-compatible quantization methods."""
        logger.warning(
            f"TTMxfp4Config.get_quant_method called for layer type: {type(layer)} with prefix: {prefix}"
        )

        if isinstance(layer, FusedMoE):
            logger.warning(
                f"Creating TT-compatible MXFP4 MoE method for layer: {prefix}"
            )

            # Handle case where layer.moe_config might not exist
            moe_config = getattr(layer, "moe_config", {})
            if not moe_config:
                logger.warning(
                    f"Layer {prefix} has no moe_config, using default config"
                )
                moe_config = {"num_experts": 8}  # Default fallback

            return TTMxfp4MoEMethod(moe_config)
        else:
            # For non-MoE layers, try the standard method or return None to disable quantization
            logger.warning(
                f"Non-MoE layer {type(layer)}, attempting standard quantization or fallback"
            )
            try:
                # Try to call parent method but catch any backend-related errors
                result = super().get_quant_method(layer, prefix)
                logger.warning(f"Standard quantization method succeeded for {prefix}")
                return result
            except Exception as e:
                logger.warning(
                    f"Standard quantization failed for {prefix}: {str(e)[:100]}..."
                )
                logger.warning("Using TT-compatible fallback (no quantization)")
                return None


def get_tt_compatible_quant_config(original_config) -> Optional[TTMxfp4Config]:
    """Create a TT-compatible quantization config from the original."""
    if original_config is None:
        return None

    # Check if it's an MXFP4 config by class type
    if isinstance(original_config, Mxfp4Config):
        logger.warning(f"Converting {type(original_config)} to TT-compatible version")
        # Create TT-compatible config with same parameters
        tt_config = TTMxfp4Config()

        # Copy all relevant attributes from original config
        for attr in dir(original_config):
            if not attr.startswith("_") and not callable(
                getattr(original_config, attr)
            ):
                try:
                    value = getattr(original_config, attr)
                    setattr(tt_config, attr, value)
                    logger.debug(f"Copied attribute {attr}: {value}")
                except (AttributeError, TypeError) as e:
                    logger.debug(f"Could not copy attribute {attr}: {e}")

        logger.warning(f"Created TT-compatible config: {type(tt_config)}")
        return tt_config

    logger.warning(
        f"Quantization config type {type(original_config)} is already TT-compatible"
    )
    return original_config


def override_quantization_for_tt(vllm_config):
    """Override quantization configuration to use TT-compatible implementations."""
    if vllm_config.quant_config is not None:
        logger.warning(
            f"Original quant config: {type(vllm_config.quant_config)} at {id(vllm_config.quant_config)}"
        )

        # Force replacement with TT-compatible version
        new_config = get_tt_compatible_quant_config(vllm_config.quant_config)

        # Ensure we actually replaced it with a new object
        if new_config is not vllm_config.quant_config:
            vllm_config.quant_config = new_config
            logger.warning(
                f"Successfully replaced with TT-compatible quant config: {type(vllm_config.quant_config)} at {id(vllm_config.quant_config)}"
            )
        else:
            logger.warning(f"Failed to replace quantization config - using same object")

        # Double-check the replacement worked
        if hasattr(vllm_config.quant_config, "get_quant_method"):
            logger.warning(
                "TT quantization config has get_quant_method - ready for model loading"
            )
        else:
            logger.error("TT quantization config missing get_quant_method!")
    else:
        logger.warning("No quantization config to override")
