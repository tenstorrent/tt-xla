# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2025 Tenstorrent AI ULC

"""TT MoE integration - complete pipeline from FusedMoE replacement to vLLM compatibility."""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Import TT sparse MLP implementation
from tt_torch.sparse_mlp import (
    ACTIVATION_DEEPSEEK,
    ACTIVATION_GPT_OSS,
    A2aSparseMLP,
    A2aSparseMLPWithSharedExperts,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

logger = init_logger(__name__)


class FusedMoEAdapter(nn.Module):
    """
    Minimal adapter to make vLLM FusedMoE compatible with A2aSparseMLP.

    This provides the minimal interface that A2aSparseMLP expects (router + experts)
    while delegating actual computation to the original FusedMoE via CPU fallback.
    """

    def __init__(self, fused_moe: FusedMoE):
        super().__init__()
        self.fused_moe = fused_moe
        logger.warning(
            f"Initializing FusedMoEAdapter for {type(fused_moe)} with logical_num_experts={fused_moe.logical_num_experts}, top_k={fused_moe.top_k}"
        )

        # Create minimal router interface
        self.router = self._create_router_stub()

        # Create minimal experts interface
        self.experts = self._create_experts_stub()

    def _create_router_stub(self):
        """Router stub that triggers CPU fallback in A2aSparseMLP."""

        class RouterStub(nn.Module):
            def __init__(self, fused_moe):
                super().__init__()
                self.fused_moe = fused_moe

            def forward(self, hidden_states):
                # logger.warning(f"[RouterStub] Called with hidden_states.shape={hidden_states.shape}")
                # A2aSparseMLP checks if hidden_states.device.type == "cpu" for CPU fallback
                # Since we want to delegate to FusedMoE, we'll create outputs that work
                # but let A2aSparseMLP handle the device logic

                # Handle both 2D and 3D input shapes
                if hidden_states.dim() == 3:
                    batch_size, seq_len, _ = hidden_states.shape
                    batch_seq = batch_size * seq_len
                elif hidden_states.dim() == 2:
                    batch_seq, _ = hidden_states.shape
                else:
                    raise ValueError(
                        f"Expected 2D or 3D hidden_states, got {hidden_states.dim()}D with shape {hidden_states.shape}"
                    )
                # logger.warning(f"RouterStub: batch_seq={batch_seq}")

                num_experts = self.fused_moe.logical_num_experts
                top_k = self.fused_moe.top_k
                device = hidden_states.device
                dtype = hidden_states.dtype

                # Create router outputs on the same device as input
                scores = (
                    torch.ones(batch_seq, num_experts, device=device, dtype=dtype)
                    / num_experts
                )
                indices = (
                    torch.arange(top_k, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(batch_seq, -1)
                )

                # Create router_logits from scores (A2aSparseMLP expects 3 values: logits, scores, indices)
                router_logits = torch.log(
                    scores + 1e-8
                )  # Add epsilon for numerical stability

                # logger.warning(f"[RouterStub] Returning router_logits.shape={router_logits.shape}, scores.shape={scores.shape}, indices.shape={indices.shape}")
                return router_logits, scores, indices

        return RouterStub(self.fused_moe)

    def _create_experts_stub(self):
        """Experts stub that provides required attributes and delegates to FusedMoE."""

        class ExpertsStub(nn.Module):
            def __init__(self, fused_moe):
                super().__init__()
                self.fused_moe = fused_moe

                # A2aSparseMLP expects these attributes - create minimal stubs
                # Get dimensions from the fused_moe config
                moe_config = fused_moe.moe_config
                num_experts = fused_moe.logical_num_experts
                hidden_dim = moe_config.hidden_dim
                intermediate_size = moe_config.intermediate_size_per_partition
                device = next(fused_moe.parameters()).device
                dtype = next(fused_moe.parameters()).dtype

                # Store as attributes that A2aSparseMLP expects
                self.intermediate_size = intermediate_size
                self.num_experts = num_experts
                self.hidden_dim = hidden_dim

                # Create dummy tensors with the expected shapes for A2aSparseMLP
                # These will not be used in computation, only for attribute access
                self.gate_up_proj = nn.Parameter(
                    torch.zeros(
                        num_experts,
                        hidden_dim,
                        intermediate_size * 2,
                        device=device,
                        dtype=dtype,
                    )
                )

                self.down_proj = nn.Parameter(
                    torch.zeros(
                        num_experts,
                        intermediate_size,
                        hidden_dim,
                        device=device,
                        dtype=dtype,
                    )
                )

                # Add bias attributes that A2aSparseMLP expects to access
                self.gate_up_proj_bias = nn.Parameter(
                    torch.zeros(
                        num_experts,
                        intermediate_size * 2,
                        device=device,
                        dtype=dtype,
                    )
                )

                self.down_proj_bias = nn.Parameter(
                    torch.zeros(
                        num_experts,
                        hidden_dim,
                        device=device,
                        dtype=dtype,
                    )
                )

                # GPT-OSS activation parameters
                self.alpha = 1.702
                self.limit = 7.0

                # Add attributes expected by sparse_forward
                self.w1 = self.gate_up_proj  # Alias for compatibility
                self.w2 = self.down_proj  # Alias for compatibility
                self.w3 = None  # No separate up projection in fused case
                self.w1_bias = self.gate_up_proj_bias  # Alias for compatibility
                self.w3_bias = None  # No separate up bias in fused case

            def sparse_forward(
                self,
                dispatched,
                sparsity_remap,
                activation_type,
                alpha=1.702,
                limit=7.0,
                output_shape=None,
            ):
                """
                Sparse forward implementation that delegates to FusedMoE.
                This is a compatibility shim for A2aSparseMLP.
                """
                # Handle different tensor formats - dispatched can be 4D or 5D
                # logger.warning(f"[ExpertsStub] Called sparse_forward with dispatched.shape={dispatched.shape}")

                if dispatched.dim() == 4:
                    # 4D format: [batch_seq, num_experts, tokens_per_expert, hidden_dim] or similar
                    batch_seq, num_experts, tokens_per_expert, hidden_dim = (
                        dispatched.shape
                    )
                    # FusedMoE expects [total_tokens, hidden_dim]
                    reshaped_input = dispatched.view(
                        -1, hidden_dim
                    )  # [batch_seq * num_experts * tokens_per_expert, hidden_dim]
                    batch_dim = batch_seq  # For compatibility with 5D case
                elif dispatched.dim() == 5:
                    # 5D format: [A, B, E, M, H] - original assumption
                    batch_dim = (
                        dispatched.shape[0] * dispatched.shape[1] * dispatched.shape[3]
                    )
                    hidden_dim = dispatched.shape[-1]
                    num_experts = dispatched.shape[2]
                    reshaped_input = dispatched.view(batch_dim, hidden_dim)
                    batch_seq = batch_dim  # For compatibility with 4D case
                    tokens_per_expert = 1  # Default for 5D case
                else:
                    raise ValueError(
                        f"Unexpected dispatched tensor dimensions: {dispatched.shape}"
                    )

                # logger.warning(f"[ExpertsStub] Processed: batch_seq={batch_seq}, num_experts={num_experts}, hidden_dim={hidden_dim}")
                # if dispatched.dim() == 4:
                #   logger.warning(f"[ExpertsStub] tokens_per_expert={tokens_per_expert}")
                # logger.warning(f"[ExpertsStub] Reshaped input: {reshaped_input.shape}")

                # Create dummy router logits for FusedMoE
                dummy_router_logits = torch.zeros(
                    reshaped_input.shape[0],  # Use reshaped input's batch dimension
                    num_experts,
                    device=dispatched.device,
                    dtype=dispatched.dtype,
                )

                # Call FusedMoE
                fused_output = self.fused_moe(reshaped_input, dummy_router_logits)
                # logger.warning(f"[ExpertsStub] FusedMoE output shape: {fused_output.shape}")

                # Reshape output to match expected sparse_forward format
                if dispatched.dim() == 4:
                    # For 4D input [batch_seq, num_experts, tokens_per_expert, hidden_dim], return same format
                    result = fused_output.view(
                        batch_seq, num_experts, tokens_per_expert, hidden_dim
                    )
                elif dispatched.dim() == 5:
                    # For 5D input, return expected [E, 1, BD*S, H] format
                    result = fused_output.view(num_experts, 1, batch_dim, hidden_dim)

                # logger.warning(f"[ExpertsStub] Returning output shape: {result.shape}")
                return result

            def forward(self, hidden_states, router_indices=None, routing_weights=None):
                # This shouldn't be called when we have router_logits available
                # as A2aSparseMLP will bypass this and call fused_moe directly
                # But if it is called, create dummy router_logits for FusedMoE
                # logger.warning(f"[ExpertsStub] Called with hidden_states.shape={hidden_states.shape}, router_indices.shape={router_indices.shape if router_indices is not None else None}, routing_weights.shape={routing_weights.shape if routing_weights is not None else None}")

                # Handle both 2D and 3D input shapes
                if hidden_states.dim() == 3:
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    batch_seq = batch_size * seq_len
                elif hidden_states.dim() == 2:
                    batch_seq, hidden_dim = hidden_states.shape
                else:
                    raise ValueError(
                        f"Expected 2D or 3D hidden_states, got {hidden_states.dim()}D with shape {hidden_states.shape}"
                    )

                num_experts = self.fused_moe.logical_num_experts

                dummy_router_logits = torch.zeros(
                    batch_seq,
                    num_experts,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

                return self.fused_moe(hidden_states, dummy_router_logits)

        return ExpertsStub(self.fused_moe)

    def forward(self, *args, **kwargs):
        """Forward call delegates to original FusedMoE."""
        return self.fused_moe(*args, **kwargs)


class VllmA2aSparseMLP(nn.Module):
    """
    vLLM-compatible wrapper for A2aSparseMLP that handles router_logits parameter
    and provides correct return types for different usage patterns.
    """

    def __init__(self, core_a2a_mlp: A2aSparseMLP):
        super().__init__()
        self.core_mlp = core_a2a_mlp

        # Expose important attributes from the core MLP
        self.num_experts = core_a2a_mlp.num_experts
        self.num_experts_per_tok = core_a2a_mlp.num_experts_per_tok
        self.cluster_axis = core_a2a_mlp.cluster_axis
        self.num_devices = core_a2a_mlp.num_devices
        self.dispatch_devices = core_a2a_mlp.dispatch_devices

        # Store expert_mapping for compatibility
        self.expert_mapping = core_a2a_mlp.expert_mapping

    def forward(
        self, hidden_states: torch.Tensor, router_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        vLLM-compatible forward that accepts router_logits and returns just the output tensor.

        Args:
            hidden_states: Input tensor
            router_logits: Optional pre-computed router logits from vLLM

        Returns:
            Output tensor (not tuple) - compatible with vLLM's GPT-OSS expectations
        """
        # Guard against batch of tensors that cannot be handled by TT
        # This prevents dynamo compilation issues with unsupported tensor formats
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(hidden_states)}")

        batch_size, seq_len, hidden_size = hidden_states.shape
        K = self.num_experts_per_tok

        # 1. Router logic
        if router_logits is not None:
            # Use provided router logits instead of computing them
            router_probs = torch.softmax(
                router_logits.view(-1, router_logits.shape[-1]), dim=-1
            )
            router_scores, router_indices = torch.topk(router_probs, K, dim=-1)
            # Ensure router indices are on the same device as hidden_states
            router_indices = router_indices.to(hidden_states.device)
            router_scores = router_scores.to(hidden_states.device)
        else:
            # Compute router scores using internal router
            router_scores, router_indices = self.core_mlp.router(hidden_states)
            # Ensure outputs are on the same device as hidden_states
            router_indices = router_indices.to(hidden_states.device)
            router_scores = router_scores.to(hidden_states.device)

        # 2. Check if we should use CPU fallback (only for actual CPU tensors)
        use_cpu_fallback = hidden_states.device.type == "cpu"

        if use_cpu_fallback:
            # Standard CPU fallback case - use adapted experts
            routed_out = self.core_mlp.experts(
                hidden_states,
                router_indices=router_indices,
                routing_weights=router_scores,
            )
            # Extract just the output from the tuple
            if isinstance(routed_out, tuple):
                return routed_out[0]
            return routed_out

        # 3. Use torch.compiler.allow_in_graph for TT custom operations
        # Store router_logits for potential CPU fallback in small tensor cases
        if router_logits is not None:
            # Use setattr to avoid dynamo tracking issues
            setattr(self.core_mlp, "_last_router_logits", router_logits)

        # Call core MLP directly - ensure we get clean tensor output
        try:
            # logger.warning(f"MLP: {hidden_states.shape}, router_logits: {router_logits.shape if router_logits is not None else None}")
            # logger.warning(f"self.core_mlp: {self.core_mlp}")
            result = self.core_mlp(hidden_states)
        except Exception as e:
            # Fallback to CPU if TT operations fail
            logger.warning(f"TT MLP failed ({e}), falling back to CPU")
            routed_out = self.core_mlp.experts(
                hidden_states,
                router_indices=router_indices,
                routing_weights=router_scores,
            )
            if isinstance(routed_out, tuple):
                return routed_out[0]
            return routed_out

        # Core A2aSparseMLP returns (output, router_scores) tuple, but vLLM expects just output
        if isinstance(result, tuple):
            return result[0]
        return result


class VllmA2aSparseMLPWithSharedExperts(nn.Module):
    """
    vLLM-compatible wrapper for A2aSparseMLPWithSharedExperts.
    """

    def __init__(
        self, core_a2a_mlp: A2aSparseMLP, shared_experts: Optional[nn.Module] = None
    ):
        super().__init__()
        # Wrap the core MLP with vLLM compatibility
        self.mlp = VllmA2aSparseMLP(core_a2a_mlp)
        self.shared_experts = shared_experts

    def forward(
        self, hidden_states: torch.Tensor, router_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass that handles both sparse and shared experts.

        Args:
            hidden_states: Input tensor
            router_logits: Optional pre-computed router logits from vLLM

        Returns:
            Output tensor with combined sparse + shared expert outputs
        """
        # Get sparse expert output
        sparse_out = self.mlp(hidden_states, router_logits=router_logits)

        # Add shared expert output if available
        if self.shared_experts is not None:
            shared_out = self.shared_experts(hidden_states)
            return sparse_out + shared_out

        return sparse_out


def create_vllm_a2a_sparse_mlp(
    original_mlp,
    num_experts: int,
    num_experts_per_tok: int,
    num_devices: int = 1,
    cluster_axis: int = 0,
    config: Optional[object] = None,
    activation_type: str = "gpt_oss",
    dispatch_devices: Optional[int] = None,
    mesh_shape: Optional[tuple] = None,
    shared_experts: Optional[nn.Module] = None,
) -> Union[VllmA2aSparseMLP, VllmA2aSparseMLPWithSharedExperts]:
    """
    Create a vLLM-compatible A2aSparseMLP wrapper.

    Args:
        original_mlp: Original MLP module (e.g., FusedMoEAdapter)
        num_experts: Number of experts
        num_experts_per_tok: Number of experts per token
        num_devices: Total number of devices
        cluster_axis: Axis for all-to-all operations
        config: Optional configuration object
        activation_type: Activation function type
        dispatch_devices: Number of dispatch devices
        mesh_shape: Optional mesh configuration (rows, cols)
        shared_experts: Optional shared experts module

    Returns:
        vLLM-compatible sparse MLP wrapper
    """
    # Create core A2aSparseMLP
    logger.info(
        f"num_experts={num_experts}, num_experts_per_tok={num_experts_per_tok}, num_devices={num_devices}, cluster_axis={cluster_axis}, activation_type={activation_type}, dispatch_devices={dispatch_devices}, mesh_shape={mesh_shape}"
    )
    core_mlp = A2aSparseMLP(
        original_mlp=original_mlp,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        num_devices=num_devices,
        cluster_axis=cluster_axis,
        config=config,
        activation_type=activation_type,
        dispatch_devices=dispatch_devices,
        # mesh_shape=mesh_shape,
    )

    # Wrap with vLLM compatibility
    if shared_experts is not None:
        return VllmA2aSparseMLPWithSharedExperts(core_mlp, shared_experts)
    else:
        return VllmA2aSparseMLP(core_mlp)


def replace_fusedmoe_with_sparse_mlp(
    model: nn.Module,
    mesh_shape: tuple = (4, 2),
    cluster_axis: int = 0,
    activation_type: str = ACTIVATION_GPT_OSS,
) -> nn.Module:
    """
    Replace all FusedMoE layers in the model with A2aSparseMLP.

    Args:
        model: The model containing FusedMoE layers
        mesh_shape: Mesh configuration (rows, cols)
        cluster_axis: Axis for all-to-all dispatch (0=rows, 1=cols)
        activation_type: Activation function type for experts

    Returns:
        Model with FusedMoE layers replaced by A2aSparseMLP
    """
    mesh_rows, mesh_cols = mesh_shape
    num_devices = mesh_rows * mesh_cols
    dispatch_devices = mesh_shape[cluster_axis]

    replacements = {}

    def find_and_replace_moe(module, name_prefix=""):
        logger.info(f"Searching for FusedMoE in module: {name_prefix} ({type(module)})")
        logger.info(
            f"Children of {name_prefix}: {[name for name, _ in module.named_children()]}"
        )

        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            if isinstance(child, FusedMoE):
                logger.info(f"Replacing FusedMoE at {full_name} with A2aSparseMLP")

                # Create adapter to make FusedMoE compatible with A2aSparseMLP
                adapted_moe = FusedMoEAdapter(child)

                # Extract configuration from original FusedMoE
                num_experts = child.logical_num_experts
                num_experts_per_tok = child.top_k
                config = getattr(child, "config", None)
                # logger.warning(f"Original FusedMoE config for {full_name}: num_experts={num_experts}, num_experts_per_tok={num_experts_per_tok}, config={config}")

                # Create vLLM-compatible A2aSparseMLP replacement
                sparse_mlp = create_vllm_a2a_sparse_mlp(
                    original_mlp=adapted_moe,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    num_devices=num_devices,
                    cluster_axis=cluster_axis,
                    config=config,
                    activation_type=activation_type,
                    dispatch_devices=dispatch_devices,
                    mesh_shape=mesh_shape,
                )

                replacements[full_name] = sparse_mlp
                logger.warning(
                    f"Created A2aSparseMLP: experts={num_experts}, "
                    f"experts_per_tok={num_experts_per_tok}, "
                    f"mesh_shape={mesh_shape}, cluster_axis={cluster_axis}, "
                    f"num_devices={num_devices}, dispatch_devices={dispatch_devices}"
                )
            else:
                # Recursively search in child modules
                find_and_replace_moe(child, full_name)

    # Find all FusedMoE layers
    find_and_replace_moe(model)

    # Apply replacements
    for full_name, new_module in replacements.items():
        # Navigate to the parent module and replace the child
        parts = full_name.split(".")
        parent = model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        setattr(parent, parts[-1], new_module)
        logger.info(f"Successfully replaced {full_name}")

    if len(replacements) > 0:
        logger.warning(
            f"Completed MLP override: replaced {len(replacements)} FusedMoE layers"
        )
    else:
        logger.warning("No FusedMoE layers found to replace (this may be expected)")
    return model


def get_activation_type_from_config(config) -> str:
    """
    Determine activation type from model configuration.

    Args:
        config: Model configuration object

    Returns:
        Activation type string (ACTIVATION_GPT_OSS or ACTIVATION_DEEPSEEK)
    """
    # Check if it's a DeepSeek model (uses SiLU/Swish)
    model_type = getattr(config, "model_type", "").lower()
    architectures = getattr(config, "architectures", [])

    if "deepseek" in model_type or any(
        "deepseek" in arch.lower() for arch in architectures
    ):
        return ACTIVATION_DEEPSEEK

    # Default to GPT-OSS activation (clamp, sigmoid, alpha, glu)
    return ACTIVATION_GPT_OSS


def override_moe_for_tt(
    model: nn.Module,
    vllm_config,
    mesh_shape: tuple = (4, 2),
    cluster_axis: int = 0,
) -> nn.Module:
    """
    Main override function to replace vLLM FusedMoE with TT A2aSparseMLP.

    Args:
        model: The vLLM model
        vllm_config: vLLM configuration object
        mesh_shape: TT mesh configuration
        cluster_axis: Dispatch axis for all-to-all operations

    Returns:
        Model with MoE layers replaced
    """
    logger.warning(
        f"Starting MoE override for TT with mesh_shape={mesh_shape}, cluster_axis={cluster_axis}"
    )

    # Determine activation type from model config
    activation_type = ACTIVATION_GPT_OSS  # Default
    if hasattr(vllm_config, "model_config") and hasattr(
        vllm_config.model_config, "hf_config"
    ):
        logger.warning("Determining activation type from model configuration")
        activation_type = get_activation_type_from_config(
            vllm_config.model_config.hf_config
        )
        logger.warning(f"Determined activation type: {activation_type}")

    logger.warning(f"Using activation type: {activation_type}")

    # Replace FusedMoE layers with A2aSparseMLP
    model = replace_fusedmoe_with_sparse_mlp(
        model=model,
        mesh_shape=mesh_shape,
        cluster_axis=cluster_axis,
        activation_type=activation_type,
    )

    logger.info("MoE override for TT completed successfully")
    return model


def count_moe_layers(model: nn.Module) -> tuple[int, int]:
    """
    Count FusedMoE and A2aSparseMLP layers in the model.

    Returns:
        Tuple of (fusedmoe_count, sparse_mlp_count)
    """
    fusedmoe_count = 0
    sparse_mlp_count = 0

    def count_recursive(module, inside_sparse_mlp=False):
        nonlocal fusedmoe_count, sparse_mlp_count

        if isinstance(module, (VllmA2aSparseMLP, A2aSparseMLP)):
            # Only count if we're not already inside another sparse MLP wrapper
            if not inside_sparse_mlp:
                sparse_mlp_count += 1
            # Don't count nested instances (e.g., core_mlp inside VllmA2aSparseMLP)
            inside_sparse_mlp = True
        elif isinstance(module, FusedMoE) and not inside_sparse_mlp:
            # Only count FusedMoE if it's not inside an A2aSparseMLP
            fusedmoe_count += 1

        for child in module.children():
            count_recursive(child, inside_sparse_mlp)

    count_recursive(model)
    return fusedmoe_count, sparse_mlp_count
