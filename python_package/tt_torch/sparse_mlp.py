# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sparse MLP module for MoE (Mixture of Experts) models.

This module provides utilities to replace dense MLP layers with sparse MLP
implementations that use sparse_matmul for efficient expert computation.

Usage:
    import tt_torch
    from tt_torch.sparse_mlp import enable_sparse_mlp

    model = load_your_moe_model()
    model = enable_sparse_mlp(model)  # Replace MLP layers with SparseMLP
"""

import os
from typing import List, Optional, Type

import torch
import torch.nn as nn


class SparseMLP(nn.Module):
    """
    Sparse MLP implementation that uses sparse_matmul for MoE computation.

    This module wraps an existing MLP and replaces dense expert computation
    with sparse_matmul operations that skip inactive experts.

    Uses INTERLEAVED gate_up_proj layout directly from original model:
    - Weights stored as [g0, u0, g1, u1, ...] (interleaved)
    - Split with [::2]/[1::2] strided slices
    - TP sharding works because UpdateGlobalToLocalShapes pass handles
      strided slices where stride == shard_factor
    """

    def __init__(self, original_mlp, num_experts: int, num_experts_per_tok: int):
        super().__init__()

        self.original_mlp = original_mlp
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Copy references to original module's components
        # Keep same structure as original MLP: router + experts
        self.router = original_mlp.router
        self.experts = original_mlp.experts  # Keep same structure for sharding

        # Use INTERLEAVED gate_up_proj directly (no conversion needed)
        # The UpdateGlobalToLocalShapes pass now handles strided slices
        # where stride == shard_factor, making [::2]/[1::2] TP-compatible.
        if hasattr(self.experts, "gate_up_proj"):
            # intermediate_size is half of the last dimension (interleaved)
            self.intermediate_size = self.experts.gate_up_proj.shape[-1] // 2

            print(f"[SparseMLP] Using interleaved gate_up_proj layout:")
            print(
                f"  gate_up_proj shape: {self.experts.gate_up_proj.shape} (interleaved)"
            )
            print(f"  down_proj shape: {self.experts.down_proj.shape}")
            print(f"  intermediate_size: {self.intermediate_size}")
        else:
            raise ValueError("Expected fused gate_up_proj in experts module")

        # GPT-OSS specific activation parameters
        self.alpha = getattr(self.experts, "alpha", 1.702)
        self.limit = getattr(self.experts, "limit", 7.0)

    def forward(self, hidden_states):
        """
        Forward pass using sparse_matmul for MoE computation.

        Uses FUSED gate+up projection with INTERLEAVED layout:
        - Single sparse_matmul for gate+up (fused, efficient)
        - Split with [::2]/[1::2] strided slices (interleaved layout)
        - TP sharding works because stride == shard_factor

        The flow:
        1. Router computes expert scores
        2. Top-k experts are selected -> sparsity mask
        3. Fused gate+up projection via single sparse_matmul
        4. Split with [::2]/[1::2] (interleaved layout)
        5. Apply activation (GPT-OSS style)
        6. Down projection via sparse_matmul
        7. Weight by router scores and sum
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Get router scores
        router_logits = torch.nn.functional.linear(
            hidden_states, self.router.weight, self.router.bias
        )
        # Convert to 4D for TTNN softmax compatibility (TTNN expects 4D tensors)
        # [batch, seq, num_experts] -> [batch, 1, seq, num_experts]
        router_logits_4d = router_logits.unsqueeze(1)
        router_probs_4d = torch.softmax(router_logits_4d, dim=-1)
        router_probs = router_probs_4d.squeeze(1)  # Back to 3D

        # Get top-k experts
        topk_weights, topk_indices = torch.topk(
            router_probs, k=self.num_experts_per_tok, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Create sparsity mask [batch, seq, 1, num_experts]
        sparsity = torch.zeros(
            batch_size,
            seq_len,
            1,
            self.num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Scatter 1s at selected expert positions
        sparsity.scatter_(
            dim=-1,
            index=topk_indices.unsqueeze(2),
            src=torch.ones_like(topk_indices.unsqueeze(2), dtype=hidden_states.dtype),
        )

        # Reshape input for sparse_matmul
        # input: [batch, seq, hidden] -> [batch, seq, 1, hidden]
        hidden_4d = hidden_states.unsqueeze(2)

        # nnz = 0 means runtime will calculate from sparsity tensor
        # This is required for EP/TP sharding where nnz changes per device
        nnz = 0

        # === FUSED GATE+UP PROJECTION (single sparse_matmul, interleaved layout) ===
        # gate_up_proj: [E, hidden, inter*2] (interleaved: [g0, u0, g1, u1, ...])
        gate_up_weight = self.experts.gate_up_proj.unsqueeze(
            0
        )  # [1, E, hidden, inter*2]
        gate_up_out = torch.ops.tt.sparse_matmul(
            hidden_4d,
            gate_up_weight,
            sparsity,
            nnz=nnz,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )
        # gate_up_out shape: [batch, seq, 1, E, 1, inter*2] -> [batch, seq, 1, E, inter*2]
        gate_up_out = gate_up_out.squeeze(4)

        # === SPLIT INTERLEAVED OUTPUT (before bias add to reduce memory) ===
        # [::2] and [1::2] work with TP sharding when stride == shard_factor
        # UpdateGlobalToLocalShapes pass handles this case specially
        gate_out = gate_up_out[..., ::2]  # [batch, seq, 1, E, inter] - even indices
        up_out = gate_up_out[..., 1::2]  # [batch, seq, 1, E, inter] - odd indices

        # Add bias AFTER split to reduce const_eval memory pressure
        # Split bias first: [E, inter*2] -> [E, inter] each
        # This avoids materializing [1, seq, 1, E, inter*2] in const_eval
        gate_bias = self.experts.gate_up_proj_bias[..., ::2]  # [E, inter]
        up_bias = self.experts.gate_up_proj_bias[..., 1::2]  # [E, inter]
        # Broadcast: [E, inter] -> [1, 1, 1, E, inter] (half the size!)
        gate_out = gate_out + gate_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        up_out = up_out + up_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # === GPT-OSS ACTIVATION (NOT SwiGLU) ===
        # gate = gate.clamp(max=limit)
        # up = up.clamp(-limit, limit)
        # glu = gate * sigmoid(gate * alpha)
        # activated = (up + 1) * glu
        gate_out = gate_out.clamp(max=self.limit)
        up_out = up_out.clamp(-self.limit, self.limit)
        glu = gate_out * torch.sigmoid(gate_out * self.alpha)
        activated = (up_out + 1) * glu

        # === DOWN PROJECTION ===
        # activated: [batch, seq, 1, E, inter] -> reshape to [batch*seq, E, 1, inter]
        activated_reshaped = activated.view(
            batch_size * seq_len, self.num_experts, 1, self.intermediate_size
        )

        # down_proj: [E, inter, hidden] -> [1, E, inter, hidden]
        down_weight = self.experts.down_proj.unsqueeze(0)

        # Create sparsity for down projection [1, 1, batch*seq, E]
        sparsity_down = sparsity.view(
            batch_size * seq_len, 1, 1, self.num_experts
        ).permute(1, 2, 0, 3)

        down_out = torch.ops.tt.sparse_matmul(
            activated_reshaped,
            down_weight,
            sparsity_down,
            nnz=0,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
        )

        # down_out: [batch*seq, E, 1, hidden] -> [batch*seq, E, hidden]
        down_out = down_out.squeeze(2)
        # bias: [E, hidden] -> [1, E, hidden]
        down_out = down_out + self.experts.down_proj_bias.unsqueeze(0)

        # Weight by router scores and sum across experts
        down_out = down_out.view(batch_size, seq_len, self.num_experts, hidden_size)

        # Create full weight tensor [batch, seq, num_experts]
        full_weights = torch.zeros(
            batch_size,
            seq_len,
            self.num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        full_weights.scatter_(dim=-1, index=topk_indices, src=topk_weights)

        # Weighted sum
        output = (down_out * full_weights.unsqueeze(-1)).sum(dim=2)

        # Return tuple (output, router_scores) to match original MLP interface
        # router_scores shape: [batch, seq, num_experts_per_tok]
        return output, topk_weights


def _is_moe_mlp(module: nn.Module) -> bool:
    """Check if a module is an MoE MLP that can be replaced with SparseMLP."""
    # Check for common MoE MLP patterns
    module_name = type(module).__name__.lower()

    # Known MoE MLP class names
    moe_patterns = ["gptossmlp", "mixtralmlp", "qwen2moemlp", "deepseekmlp"]

    if any(pattern in module_name for pattern in moe_patterns):
        return True

    # Check if module has router and experts attributes (common MoE pattern)
    has_router = hasattr(module, "router")
    has_experts = hasattr(module, "experts")

    return has_router and has_experts


def _get_moe_config(module: nn.Module) -> Optional[tuple]:
    """Extract MoE configuration from a module."""
    try:
        num_experts = None
        # Try to get from experts
        if hasattr(module, "experts"):
            experts = module.experts
            num_experts = getattr(experts, "num_experts", None)
            # Try fused gate_up_proj (expected input format)
            if num_experts is None and hasattr(experts, "gate_up_proj"):
                num_experts = experts.gate_up_proj.shape[0]

        # Try to get num_experts_per_tok from router
        if hasattr(module, "router"):
            router = module.router
            num_experts_per_tok = getattr(router, "top_k", None)
            if num_experts_per_tok is None:
                num_experts_per_tok = getattr(router, "num_experts_per_tok", 2)
        else:
            num_experts_per_tok = 2  # Default

        if num_experts is not None:
            return (num_experts, num_experts_per_tok)
    except Exception:
        pass

    return None


def enable_sparse_mlp(
    model: nn.Module, target_classes: Optional[List[Type]] = None, verbose: bool = False
) -> nn.Module:
    """
    Replace MoE MLP layers in a model with SparseMLP implementations.

    This function traverses the model and replaces compatible MLP layers
    with SparseMLP, which uses sparse_matmul for efficient expert computation.

    Args:
        model: The model to transform
        target_classes: Optional list of specific MLP classes to replace.
                       If None, auto-detects MoE MLPs.
        verbose: If True, print information about replaced layers

    Returns:
        The transformed model with SparseMLP layers

    Example:
        >>> import tt_torch
        >>> from tt_torch.sparse_mlp import enable_sparse_mlp
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
        >>> model = enable_sparse_mlp(model, verbose=True)
    """
    replaced_count = 0

    def replace_mlp(parent: nn.Module, name: str, module: nn.Module):
        nonlocal replaced_count

        # Check if this is a target MLP
        should_replace = False
        if target_classes:
            should_replace = any(isinstance(module, cls) for cls in target_classes)
        else:
            should_replace = _is_moe_mlp(module)

        if not should_replace:
            return False

        # Get MoE configuration
        config = _get_moe_config(module)
        if config is None:
            if verbose:
                print(f"[SparseMLP] Skipping {name}: could not determine MoE config")
            return False

        num_experts, num_experts_per_tok = config

        # Create SparseMLP wrapper
        sparse_mlp = SparseMLP(module, num_experts, num_experts_per_tok)

        # Replace the module
        setattr(parent, name, sparse_mlp)
        replaced_count += 1

        if verbose:
            print(
                f"[SparseMLP] Replaced {name}: {type(module).__name__} -> SparseMLP "
                f"(experts={num_experts}, top_k={num_experts_per_tok})"
            )

        return True

    # Traverse and replace
    for name, module in list(model.named_modules()):
        if "." in name:
            # Get parent module
            parts = name.rsplit(".", 1)
            parent_name, child_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name

        replace_mlp(parent, child_name, module)

    if verbose:
        print(f"[SparseMLP] Total layers replaced: {replaced_count}")

    return model


# Auto-enable based on environment variable
_AUTO_SPARSE_MLP = os.environ.get("TT_ENABLE_SPARSE_MLP", "0").lower() in (
    "1",
    "true",
    "yes",
)

if _AUTO_SPARSE_MLP:
    import warnings

    warnings.warn(
        "TT_ENABLE_SPARSE_MLP is set. SparseMLP will be automatically enabled for MoE models "
        "when loading via AutoModelForCausalLM.from_pretrained(). "
        "Use tt_torch.sparse_mlp.enable_sparse_mlp(model) for explicit control.",
        UserWarning,
    )

    # Patch HuggingFace AutoModelForCausalLM.from_pretrained to auto-enable SparseMLP
    try:
        from transformers import AutoModelForCausalLM

        _original_from_pretrained = AutoModelForCausalLM.from_pretrained.__func__

        @classmethod
        def _patched_from_pretrained(cls, *args, **kwargs):
            """Patched from_pretrained that auto-enables SparseMLP for MoE models."""
            model = _original_from_pretrained(cls, *args, **kwargs)

            # Check if this is an MoE model by looking for router/experts pattern
            has_moe = any(_is_moe_mlp(m) for _, m in model.named_modules())

            if has_moe:
                print("[SparseMLP] MoE model detected, auto-enabling SparseMLP...")
                model = enable_sparse_mlp(model, verbose=True)

            return model

        AutoModelForCausalLM.from_pretrained = _patched_from_pretrained
        print(
            "[SparseMLP] Patched AutoModelForCausalLM.from_pretrained for auto SparseMLP"
        )

    except ImportError:
        warnings.warn(
            "transformers not installed. Auto SparseMLP patching skipped.", UserWarning
        )
    except Exception as e:
        warnings.warn(f"Failed to patch AutoModelForCausalLM: {e}", UserWarning)
