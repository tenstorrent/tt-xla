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

from typing import List, Optional, Type

import torch
import torch.nn as nn
from torch.nn import functional as F


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

    def __init__(
        self,
        original_mlp,
        num_experts: int,
        num_experts_per_tok: int,
        config: Optional[object] = None,
    ):
        super().__init__()

        # Note: We intentionally do NOT store original_mlp to avoid memory duplication.
        # Only store references to the components we actually need.
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
        else:
            raise ValueError("Expected fused gate_up_proj in experts module")

        # Get hidden_size from config or infer from down_proj shape
        if config is not None and hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            # Infer from down_proj shape: [E, inter, hidden]
            hidden_size = self.experts.down_proj.shape[-1]

        # GPT-OSS specific activation parameters
        self.alpha = getattr(self.experts, "alpha", 1.702)
        self.limit = getattr(self.experts, "limit", 7.0)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 1. Router Execution
        # router_scores: [batch*seq, num_experts] (scattered probabilities)
        # router_indices: [batch*seq, top_k]
        router_scores, router_indices = self.router(hidden_states)

        # 2. Create Sparsity Mask [batch, seq, 1, num_experts]
        sparsity = torch.zeros(
            batch_size,
            seq_len,
            1,
            self.num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Reshape indices for scatter: [batch, seq, 1, top_k]
        topk_indices_unsqueezed = router_indices.view(
            batch_size, seq_len, 1, self.num_experts_per_tok
        )

        sparsity.scatter_(
            dim=-1,
            index=topk_indices_unsqueezed,
            src=torch.ones_like(topk_indices_unsqueezed, dtype=hidden_states.dtype),
        )

        # 3. Reshape Input for sparse_matmul [batch, seq, 1, hidden]
        hidden_4d = hidden_states.view(batch_size, seq_len, 1, hidden_size)

        # 4. Fused Gate+Up Projection
        # gate_up_weight: [1, E, hidden, inter*2]
        gate_up_proj = self.experts.gate_up_proj.unsqueeze(0)
        gate_up_out = torch.ops.tt.sparse_matmul(
            hidden_4d,
            gate_up_proj,
            sparsity,
            nnz=0,  # Let runtime calculate
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )

        # Output: [batch, seq, 1, E, M, inter*2] where M=1
        # Reshape to [batch, seq, E, inter*2]
        gate_up_out = gate_up_out.view(
            batch_size, seq_len, self.num_experts, self.intermediate_size * 2
        )
        gate_up_out = gate_up_out + self.experts.gate_up_proj_bias

        # 5. Split & Activation (Interleaved Layout)
        # Slicing works for TP because stride (2) matches shard_factor
        gate_out = gate_up_out[..., ::2]  # Even indices
        up_out = gate_up_out[..., 1::2]  # Odd indices

        gate_out = gate_out.clamp(max=self.limit)
        up_out = up_out.clamp(-self.limit, self.limit)
        glu = gate_out * torch.sigmoid(gate_out * self.alpha)
        activated = (up_out + 1) * glu

        # 6. Down Projection setup
        # activated: [batch*seq, E, 1, inter]
        activated_reshaped = activated.view(
            batch_size * seq_len, self.num_experts, 1, self.intermediate_size
        )

        # sparsity_down: [1, 1, batch*seq, E]
        sparsity_down = sparsity.view(1, 1, batch_size * seq_len, self.num_experts)

        # Reshape down_proj for sparse_matmul
        # down_proj: [1, E, inter, hidden]
        down_proj = self.experts.down_proj.view(
            1, self.num_experts, self.intermediate_size, hidden_size
        )

        # 7. Down Projection (sparse_matmul)
        # down_weight: [1, E, inter, hidden]
        # Output: [batch*seq, E, M, hidden] where M=1
        down_out = torch.ops.tt.sparse_matmul(
            activated_reshaped,
            down_proj,
            sparsity_down,
            nnz=0,
            is_input_a_sparse=True,  # Activations are sparse (only TopK are valid)
            is_input_b_sparse=False,
        )

        # Squeeze M dimension: [batch*seq, E, 1, hidden] -> [batch*seq, E, hidden]
        down_out = down_out.squeeze(2)
        down_out = down_out + self.experts.down_proj_bias

        # 8. Weighted Sum & Final Output
        # down_out: [BS, E, H]
        # router_scores: [BS, E] -> [BS, E, 1]
        # output: [BS, H] (Sum over Experts dim=1)
        output = (down_out * router_scores.unsqueeze(-1)).sum(dim=1)

        # Reshape back to [batch, seq, hidden]
        output = output.view(batch_size, seq_len, hidden_size)

        return output, router_scores


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
    model: nn.Module,
    target_classes: Optional[List[Type]] = None,
    verbose: bool = False,
    config: Optional[object] = None,
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
        config: Optional model config for extracting MoE parameters

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

    # Try to get config from model if not provided
    if config is None:
        config = getattr(model, "config", None)

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

        # Get MoE configuration from module first, then fall back to config
        moe_config = _get_moe_config(module)
        if moe_config is None and config is not None:
            # Try to get from model config
            num_experts = getattr(config, "num_local_experts", None)
            num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
            if num_experts is not None:
                moe_config = (num_experts, num_experts_per_tok)

        if moe_config is None:
            if verbose:
                print(f"[SparseMLP] Skipping {name}: could not determine MoE config")
            return False

        num_experts, num_experts_per_tok = moe_config

        # Create SparseMLP wrapper
        sparse_mlp = SparseMLP(module, num_experts, num_experts_per_tok, config)

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
