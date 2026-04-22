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

This module adds a backend named "tt_moe" that lowers HF MoE expert compute
to Tenstorrent custom ops:

  - Single-device / data-parallel: a pair of `torch.ops.tt.sparse_matmul`
    block-sparse batched GEMMs, with sparsity derived from the router's
    top-k indices.

  - Expert-parallel (multi-device SPMD): adds
    `torch.ops.tt.all_to_all_dispatch` before the GEMMs and
    `torch.ops.tt.all_to_all_combine` after them. The sparsity mask for the
    post-dispatch token layout is built by
    `torch.ops.tt.moe_expert_token_remap`.

Which path runs is decided at trace time from the global torch_xla SPMD mesh
(`torch_xla.distributed.spmd.get_global_mesh()`). When no mesh is set or the
mesh axis chosen at registration has size 1, the EP collectives collapse away
and the backend behaves like the data-parallel path. No model-specific code
is required: any Experts module following the canonical HF layout

    self.num_experts   : int
    self.gate_up_proj  : Parameter [num_experts, 2*intermediate_dim, hidden_dim]
    self.down_proj     : Parameter [num_experts,     hidden_dim,     intermediate_dim]
    self._apply_gate   : callable installed by @use_experts_implementation

routes through the same code path.

Usage:

    from tt_torch.moe_backend import register_tt_moe_backend, TT_MOE_BACKEND_NAME

    register_tt_moe_backend(cluster_axis=0)   # mesh axis along which experts are sharded
    model = AutoModelForCausalLM.from_pretrained(
        model_id, experts_implementation=TT_MOE_BACKEND_NAME,
    )

Shape constraint: `sparse_matmul`'s MoE fast-path tiles the token dimension
by `REDUCTION_SIZE=32`. On the DP path the total token count `T` must be a
multiple of 32 (`T < 32` collapses to `M=1`). On the EP path the same
constraint applies to `dispatch_devices * T`. Pad inputs accordingly.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS, ExpertsInterface
from transformers.modeling_utils import PreTrainedModel

# Ensure torch.ops.tt.* are registered.
from . import custom_ops  # noqa: F401

TT_MOE_BACKEND_NAME = "tt_moe"
REDUCTION_SIZE = 32

# Populated by `register_tt_moe_backend`. Module-level (rather than stashed on
# `self`) because cluster_axis is a property of the mesh, not of a specific
# Experts instance, and HF gives no hook to thread it through from_pretrained.
_config: dict = {"cluster_axis": 0}


def _mesh_info() -> Tuple[int, int, Tuple[int, ...]]:
    """Return (total_devices, dispatch_devices_on_cluster_axis, mesh_shape).

    Reads the currently-set torch_xla global SPMD mesh. Returns (1, 1, (1,))
    when no mesh is registered or torch_xla is unavailable — in that case the
    backend falls back to the single-device sparse_matmul path.
    """
    try:
        from torch_xla.distributed.spmd import get_global_mesh
    except ImportError:
        return 1, 1, (1,)
    mesh = get_global_mesh()
    if mesh is None:
        return 1, 1, (1,)
    mesh_shape = tuple(int(d) for d in mesh.mesh_shape)
    total = 1
    for d in mesh_shape:
        total *= d
    ax = _config["cluster_axis"]
    dispatch = mesh_shape[ax] if 0 <= ax < len(mesh_shape) else 1
    return total, dispatch, mesh_shape


def _expert_mapping(
    num_experts: int,
    num_devices: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the `[1, 1, E, D]` one-hot expert-to-device mapping.

    Experts are sharded only along the selected dispatch axis. Keep the mapping
    aligned with that single-axis placement by assigning contiguous expert
    ranges to each device on the axis.
    """
    assert (
        num_experts % num_devices == 0
    ), f"num_experts ({num_experts}) must be divisible by num_devices ({num_devices})"

    experts_per_device = num_experts // num_devices
    device_ids_list = [i // experts_per_device for i in range(num_experts)]

    device_ids = torch.tensor(device_ids_list, device=device, dtype=torch.int64)

    mapping = (
        device_ids.unsqueeze(-1)
        == torch.arange(num_devices, device=device, dtype=torch.int64)
    ).to(torch.int64)
    return mapping.view(1, 1, num_experts, num_devices)


def _build_routing_scores(
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Expand top-k weights into a full `[T, E]` sparse router-scores tensor.

    Use a pure functional one_hot + einsum construction here instead of
    `scatter_`. AOTAutograd's functionalization step under XLA trips over the
    in-place scatter path during `run_decompositions`, which makes export of
    the EP backend fail before we ever reach lowering.
    """
    one_hot = (
        top_k_index.unsqueeze(-1) == torch.arange(num_experts, device=device)
    ).to(dtype)
    return torch.einsum("tk,tke->te", top_k_weights.to(dtype), one_hot)


def _tt_experts_forward_dp(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Single-device expert compute: two sparse_matmul GEMMs with a scattered
    sparsity mask. No collectives.
    """
    T, H = hidden_states.shape
    E = self.num_experts
    dtype = hidden_states.dtype
    device = hidden_states.device

    routing_map = torch.zeros(1, 1, T, E, dtype=dtype, device=device)
    routing_map.view(T, E).scatter_(-1, top_k_index, top_k_weights.to(dtype))

    # Reduced binary sparsity mask [1, 1, ceil(T/32), E]: 1 wherever any token
    # in a 32-token tile selects expert e. `moe_expert_token_remap` would
    # produce this in a single op, but it returns a tuple and the single-device
    # path is simpler to reason about (and trace) with plain torch ops.
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

    # Fused gate/up projection via block-sparse batched GEMM.
    # HF stores gate_up_proj as [E, 2*I, H]; the op expects [1, E, H, 2*I].
    input_a = hidden_states.view(1, 1, T, H)
    gate_up_w = self.gate_up_proj.transpose(-2, -1).unsqueeze(0).contiguous()
    gate_up_out = torch.ops.tt.sparse_matmul(
        input_a,
        gate_up_w,
        sparsity,
        nnz=0,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
    )  # XLA: 5D tiled [A=1, B=T/M, E, M, 2I]; CPU: unpacked to [1, T, E, 2I]

    activated = self._apply_gate(gate_up_out)  # same rank, last dim -> I

    # Flatten tiled output to canonical 4D so the down GEMM isn't re-detected
    # as MoE-shape (which would re-tile).
    if activated.dim() == 5:
        A, B, _, M_tile, _ = activated.shape
        activated = activated.reshape(A * B, E, M_tile, activated.shape[-1])
    down_w = self.down_proj.transpose(-2, -1).unsqueeze(0).contiguous()
    down_out = torch.ops.tt.sparse_matmul(
        activated,
        down_w,
        sparsity,
        nnz=0,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
    )  # Canonical: [A*B, E, M, H]; CPU auto-unpack: [1, T, E, H]

    # Untile back to [1, T, E, H] for routing-weight combination.
    if down_out.dim() == 4 and down_out.shape[1] == E:
        AB, _, M_tile, Hout = down_out.shape
        down_out = down_out.permute(0, 2, 1, 3).reshape(1, AB * M_tile, E, Hout)

    routing = routing_map.view(1, T, E, 1).to(down_out.dtype)
    combined = (down_out * routing).sum(dim=2).view(T, H)
    return combined.to(dtype)


def _tt_experts_forward_ep(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    routing_scores: torch.Tensor,
    dispatch_devices: int,
) -> torch.Tensor:
    """Expert-parallel compute: dispatch tokens to the devices holding their
    selected experts, run the sparse_matmul chain on the dispatched layout,
    then combine per-expert outputs back to original token positions.
    """
    T, H = hidden_states.shape
    K = top_k_index.shape[-1]
    E = self.num_experts
    dtype = hidden_states.dtype
    device = hidden_states.device
    cluster_axis = _config["cluster_axis"]

    # `[1, 1, E, D]` one-hot mapping lifted into the graph as a pure tensor op
    # sequence so AOTAutograd/XLA can trace it without CPU copies or in-place
    # Python-side writes.
    expert_mapping = _expert_mapping(E, dispatch_devices, device)

    # Dispatch tokens along cluster_axis. Output is a full [1, B*D, T, H]
    # tensor where each of the D slices carries only the tokens its experts
    # will consume; the rest are zeros. `metadata` all-gathers top_k_index so
    # each device knows which expert slot produced which token.
    hidden_3d = hidden_states.view(1, T, H)
    dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
        hidden_3d,
        top_k_index,
        expert_mapping,
        num_devices=dispatch_devices,
        cluster_axis=cluster_axis,
    )  # dispatched: [1, BD, T, H];  metadata: [1, BD, T, K]
    BD = dispatched.shape[1]
    # Flatten BD and T so sparse_matmul sees a single token axis and the
    # sparsity mask is consistent with the tiled output.
    metadata = metadata.reshape(1, 1, BD * T, K)

    # Sparsity over the dispatched token layout. Must come from the
    # all-gathered metadata, not the local router scores, because after
    # dispatch the relevant experts-per-tile set is the union across devices.
    _, sparsity = torch.ops.tt.moe_expert_token_remap(
        routing_scores,
        expert_mapping,
        metadata,
        num_devices=dispatch_devices,
    )  # sparsity: [1, 1, ceil(BD*T/32), E]

    # Fused gate/up projection.
    gate_up_w = self.gate_up_proj.transpose(-2, -1).unsqueeze(0).contiguous()
    gate_up_out = torch.ops.tt.sparse_matmul(
        dispatched,
        gate_up_w,
        sparsity,
        nnz=0,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
    )  # 5D tiled: [BD, T/M, E, M, 2I]  (decode tiles on BD instead of T)

    activated = self._apply_gate(gate_up_out)

    # Flatten to canonical 4D before down GEMM.
    if activated.dim() == 5:
        A, B, _, M_tile, _ = activated.shape
        activated = activated.reshape(A * B, E, M_tile, activated.shape[-1])
    down_w = self.down_proj.transpose(-2, -1).unsqueeze(0).contiguous()
    down_out = torch.ops.tt.sparse_matmul(
        activated,
        down_w,
        sparsity,
        nnz=0,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
    )  # [A*B, E, M, H]

    # Reshape to the layout combine expects: experts on dim 0, all tokens on
    # one axis. A*B and M are adjacent so this is a permute + reshape, no copy.
    AB, E_, M_, Hout = down_out.shape
    down_out = down_out.permute(1, 0, 2, 3).reshape(E_, 1, AB * M_, Hout)

    combined = torch.ops.tt.all_to_all_combine(
        down_out,
        metadata,
        expert_mapping,
        num_devices=dispatch_devices,
        cluster_axis=cluster_axis,
        num_experts_per_tok=K,
        output_shard_dim=2,
    )  # [K, 1, T, H]

    # Per-slot weighted sum with already-normalized top-K weights.
    weights_k = top_k_weights.permute(1, 0).view(K, 1, T, 1).to(combined.dtype)
    output = (combined * weights_k).sum(dim=0).view(T, H)
    return output.to(dtype)


def tt_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """HF `ExpertsInterface` forward for `tt_moe`.

    Signature matches the other backends in
    `transformers/integrations/moe.py`:

        fn(self, hidden_states, top_k_index, top_k_weights) -> torch.Tensor

    where

        hidden_states : (T, H)
        top_k_index   : (T, K)  selected expert ids per token
        top_k_weights : (T, K)  router scores for the selected experts

    Dispatches to the single-device `sparse_matmul`-only path or the
    expert-parallel path (adds `all_to_all_dispatch` + `all_to_all_combine`)
    based on the global torch_xla SPMD mesh and the `cluster_axis` chosen at
    registration time.
    """
    _, dispatch_devices, _ = _mesh_info()

    if dispatch_devices <= 1:
        return _tt_experts_forward_dp(self, hidden_states, top_k_index, top_k_weights)

    routing_scores = _build_routing_scores(
        top_k_index,
        top_k_weights,
        self.num_experts,
        hidden_states.dtype,
        hidden_states.device,
    )

    return _tt_experts_forward_ep(
        self,
        hidden_states,
        top_k_index,
        top_k_weights,
        routing_scores,
        dispatch_devices,
    )


_original_validator: Optional[Callable] = None


def register_tt_moe_backend(cluster_axis: int = 0) -> None:
    """Register the `tt_moe` experts backend globally.

    Idempotent. Also patches
    `PreTrainedModel.get_correct_experts_implementation` — HF hard-codes the
    accepted backend names there, so a custom key needs an additional escape
    hatch. `ExpertsInterface` itself is already extensible via `register()`.

    Args:
        cluster_axis: Mesh axis along which experts are sharded. Read at
            trace time from the global torch_xla SPMD mesh; ignored when no
            mesh is set. Calling this function again updates the value.
    """
    global _original_validator

    _config["cluster_axis"] = cluster_axis
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
