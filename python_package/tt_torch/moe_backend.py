# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent MoE experts backend for HuggingFace transformers.

HuggingFace exposes `ExpertsInterface` (`transformers/integrations/moe.py`) as
a pluggable registry of expert-layer forward functions. Built-in backends are
"eager", "batched_mm" and "grouped_mm". Any MoE model whose Experts class is
decorated with `@use_experts_implementation` (e.g. AfmoeExperts in Trinity
Nano, OlmoeExperts in OLMoE, GptOssExperts, Qwen3-MoE, ...) will dispatch to
whichever backend is selected via `experts_implementation="..."` at
`from_pretrained` time.

This module adds a backend named "tt_moe" whose forward is built from
`torch.ops.tt.sparse_matmul` — the block-sparse batched GEMM that backs MoE
expert computation on Tenstorrent devices. See
`python_package/tt_torch/custom_ops.py` for the op definition and
`python_package/tt_torch/sparse_mlp.py` for the canonical calling convention.

Usage:

    from tt_torch.moe_backend import register_tt_moe_backend, TT_MOE_BACKEND_NAME

    register_tt_moe_backend()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, experts_implementation=TT_MOE_BACKEND_NAME
    )

Shape constraint: the sparse_matmul MoE fast-path tiles the token dimension
by `REDUCTION_SIZE=32`, so the total token count `T` must be a multiple of 32
(or `T < 32`, which collapses to `M=1`). Pad inputs accordingly.

The Experts module is expected to follow the canonical transformers layout
(same as AfmoeExperts/OlmoeExperts/GptOssExperts):

    self.num_experts     : int
    self.gate_up_proj    : Parameter [num_experts, 2*intermediate_dim, hidden_dim]
    self.down_proj       : Parameter [num_experts,     hidden_dim,     intermediate_dim]
    self._apply_gate     : callable installed by @use_experts_implementation
"""

from __future__ import annotations

from typing import Callable

import torch
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS, ExpertsInterface
from transformers.modeling_utils import PreTrainedModel

# Ensure torch.ops.tt.* are registered.
from . import custom_ops  # noqa: F401

TT_MOE_BACKEND_NAME = "tt_moe"
REDUCTION_SIZE = 32


def tt_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Experts forward implemented with `tt::sparse_matmul`.

    Signature matches the other backends in
    `transformers/integrations/moe.py`:

        fn(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor

    where `self` is the decorated experts module and

        hidden_states : (T, H)
        top_k_index   : (T, K) selected expert ids per token
        top_k_weights : (T, K) router scores for the selected experts
    """
    T, H = hidden_states.shape
    K = top_k_index.shape[-1]
    E = self.num_experts
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Build (a) a per-token routing-weight map [1, 1, T, E] and
    #       (b) a tile-reduced binary sparsity mask [1, 1, ceil(T/32), E].
    #
    # `tt::moe_expert_token_remap` does this in a single custom op, but it
    # returns a tuple and the tt-mlir Shardy-propagation pass does not support
    # tuple-typed custom_calls, so under torch.compile(backend="tt") we build
    # the tensors with regular torch ops instead. sparse_matmul below is the
    # op that actually matters for the MoE GEMMs.
    routing_map = torch.zeros(1, 1, T, E, dtype=dtype, device=device)
    routing_map.view(T, E).scatter_(-1, top_k_index, top_k_weights.to(dtype))

    reduced = (T + REDUCTION_SIZE - 1) // REDUCTION_SIZE
    if T % REDUCTION_SIZE == 0:
        sparsity = (
            routing_map.view(1, 1, reduced, REDUCTION_SIZE, E)
            .ne(0)
            .any(dim=3)
            .to(dtype)
        )
    else:
        pad = reduced * REDUCTION_SIZE - T
        padded = torch.nn.functional.pad(routing_map, (0, 0, 0, pad))
        sparsity = (
            padded.view(1, 1, reduced, REDUCTION_SIZE, E).ne(0).any(dim=3).to(dtype)
        )

    # --- Fused gate/up projection via block-sparse batched GEMM. ----------
    # sparse_matmul's MoE auto-shape expects the "dispatch" layout
    #   input_a : [1, BD, S, H] with BD=1, S=T
    #   input_b : [1, E, H, 2*I]  (note: K is the contraction dim)
    # HF Experts modules store gate_up_proj as [E, 2*I, H], so we transpose.
    input_a = hidden_states.view(1, 1, T, H)
    gate_up_w = self.gate_up_proj.transpose(-2, -1).unsqueeze(0).contiguous()

    # nnz is a hint consumed only by the sparsity-based short-circuits. We
    # pass 0 explicitly because the op's pybind dispatcher cannot cast the
    # Python default `None` to `SymInt` (see sparse_mlp.py).
    gate_up_out = torch.ops.tt.sparse_matmul(
        input_a,
        gate_up_w,
        sparsity,
        nnz=0,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
    )  # 5D tiled on XLA: [A=1, B=S//M, E, M, 2*I]  (4D on CPU auto-unpack)

    # SwiGLU / ACT(gate) * up supplied by the experts decorator. Works on
    # both 5D and 4D — _apply_gate chunks along the last dim.
    activated = self._apply_gate(gate_up_out)  # [A, B, E, M, I]  or  [1, T, E, I]

    # --- Down projection via block-sparse batched GEMM (sparse_a path). ---
    # Flatten to canonical 4D [A*B, E, M, I] so the op does NOT re-enter the
    # MoE auto-shape path (avoids a spurious re-tiling). This matches
    # sparse_mlp.py's pattern.
    if activated.dim() == 5:
        A, B, _, M_tile, _ = activated.shape
        activated = activated.reshape(A * B, E, M_tile, activated.shape[-1])
    down_w = self.down_proj.transpose(-2, -1).unsqueeze(0).contiguous()  # [1, E, I, H]
    down_out = torch.ops.tt.sparse_matmul(
        activated,
        down_w,
        sparsity,
        nnz=0,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
    )  # Canonical output: [A*B, E, M, H]  (or [1, T, E, H] on CPU auto-unpack)

    # Untile back to [1, T, E, H] for routing-weight combination.
    if down_out.dim() == 4 and down_out.shape[1] == E:
        AB, _, M_tile, Hout = down_out.shape
        # [A*B, E, M, H] -> [1, A*B, M, E, H] -> [1, T, E, H]
        down_out = down_out.permute(0, 2, 1, 3).reshape(1, AB * M_tile, E, Hout)

    # Weight by router scores (routing_map already carries top-k weights at
    # the selected expert positions and zeros elsewhere) and sum over experts.
    routing = routing_map.view(1, T, E, 1).to(down_out.dtype)
    combined = (down_out * routing).sum(dim=2).view(T, H)
    return combined.to(dtype)


_original_validator: Callable | None = None


def register_tt_moe_backend() -> None:
    """Register the `tt_moe` experts backend globally.

    Idempotent. Also patches
    `PreTrainedModel.get_correct_experts_implementation` — HF hard-codes the
    accepted backend names there, so a custom key needs an additional escape
    hatch. `ExpertsInterface` itself is already extensible via `register()`.
    """
    global _original_validator

    ExpertsInterface.register(TT_MOE_BACKEND_NAME, tt_experts_forward)
    assert TT_MOE_BACKEND_NAME in ALL_EXPERTS_FUNCTIONS, "registration did not stick"

    if _original_validator is not None:
        return  # already patched

    _original_validator = PreTrainedModel.get_correct_experts_implementation

    def patched_validator(self, requested_experts):
        if (
            requested_experts in ALL_EXPERTS_FUNCTIONS.valid_keys()
            and requested_experts
            not in (
                "eager",
                "grouped_mm",
                "batched_mm",
                "deepgemm",
            )
        ):
            return requested_experts
        return _original_validator(self, requested_experts)

    PreTrainedModel.get_correct_experts_implementation = patched_validator
