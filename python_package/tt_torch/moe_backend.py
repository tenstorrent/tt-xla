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

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS, ExpertsInterface
from transformers.modeling_utils import PreTrainedModel

# Ensure torch.ops.tt.* are registered.
from . import custom_ops  # noqa: F401

TT_MOE_BACKEND_NAME = "tt_moe"
REDUCTION_SIZE = 32

# This is pure tech debt.
#
# When True, the DP path emits a per-token sparsity mask `[1, 1, T, E]`
# instead of the default tile-reduced `[1, 1, ⌈T/32⌉, E]`. Reproduces the
# legacy `sparse_mlp.SparseMLP` shape so the two paths can be A/B'd.
#
# The matmul systolic array runs 32 rows in lock-step; the per-token bit
# can't trigger any compute skipping below the 32-row tile — it can only
# act as a multiplicative post-gate, which is redundant with the
# downstream `down_out * top_k_weights` zero-multiply in the combine. On
# TT the kernel does lower this path (the stablehlo_custom_call accepts
# `M=1` single-token tile mode), but it runs ≈2.3× slower than the
# tile-reduced default across T ∈ {32..512}, because each "tile" holds
# one real token row padded with 31 zero rows and the tile-level skip
# set is larger (per-token tiles each carry only K active experts, so
# total `(tile, expert)` computed cells is T·K vs (T/32)·|active_set|).
# On CPU the difference is ≤5% either way — CPU BLAS doesn't tile-round.
# See `tmp/sparsity_bench/` for the numbers. Delete when nobody's asking.
USE_LEGACY_PER_TOKEN_SPARSITY = False

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


def _as_sparse_matmul_weight(weight: torch.Tensor, is_transposed: bool) -> torch.Tensor:
    """Return weight in the `[1, E, in_features, out_features]` layout that
    `sparse_matmul` expects, respecting HF's `is_transposed` convention.

    `is_transposed=True`  → weight stored as [E, in, out] (e.g. GptOss).
    `is_transposed=False` → weight stored as [E, out, in] (e.g. Olmoe,
                                                           Qwen3-MoE).
    """
    if is_transposed:
        return weight.unsqueeze(0).contiguous()
    return weight.transpose(-2, -1).unsqueeze(0).contiguous()


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
    `scatter_`. `torch.export.export` functionalizes the in-place scatter
    without error, but the resulting graph fails to lower on the TT PJRT
    device — see `tmp/moe_scatter/scatter_repro.py` for evidence. Going
    functional here and in the DP path sidesteps the lowering failure.
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
    """Single-device expert compute: two sparse_matmul GEMMs with a binary
    sparsity mask. No collectives, no in-place scatter — fully functional so
    the graph survives AOTAutograd functionalization under XLA export.
    """
    T, H = hidden_states.shape
    E = self.num_experts
    dtype = hidden_states.dtype
    device = hidden_states.device

    # Binary per-token mask [T, E]: `one_hot + any` over the top-K axis —
    # functional (no scatter_, AOT-safe) and ones-valued (weight-valued masks
    # would silently drop experts whose router weight is exactly zero).
    one_hot = top_k_index.unsqueeze(-1) == torch.arange(
        E, device=device
    )  # [T, K, E] bool
    per_token = one_hot.any(dim=1)  # [T, E] bool — expert picked by any of K

    if USE_LEGACY_PER_TOKEN_SPARSITY:
        # Pure tech debt — see toggle definition above. Per-token mask
        # `[1, 1, T, E]` matching the deleted sparse_mlp.SparseMLP shape.
        # On TT it's rejected at MLIR lowering (the kernel requires M=32 tile
        # mode, not M=1); on CPU it runs at roughly the same speed as the
        # tile-reduced default. Kept only for A/B regression checks against
        # history.
        sparsity = per_token.view(1, 1, T, E).to(dtype)
    else:
        # Tile-reduced binary sparsity mask [1, 1, ceil(T/32), E]: 1 wherever
        # any token in a 32-token tile selects expert e. This is the only
        # shape the on-device sparse_matmul kernel accepts for the M dim.
        reduced = (T + REDUCTION_SIZE - 1) // REDUCTION_SIZE
        if T % REDUCTION_SIZE != 0:
            pad = reduced * REDUCTION_SIZE - T
            per_token = torch.nn.functional.pad(per_token, (0, 0, 0, pad))
        sparsity = (
            per_token.view(1, 1, reduced, REDUCTION_SIZE, E).any(dim=3).to(dtype)
        )  # [1, 1, ceil(T/32), E]

    # Fused gate/up projection via block-sparse batched GEMM.
    # sparse_matmul expects input_b as [1, E, in_features, out_features].
    # HF stores weights one of two ways (@use_experts_implementation metadata):
    #   is_transposed=True  → [E, in, out] = [E, H, 2I], ready as-is.
    #   is_transposed=False → [E, out, in] = [E, 2I, H], needs a transpose.
    input_a = hidden_states.view(1, 1, T, H)
    gate_up_w = _as_sparse_matmul_weight(self.gate_up_proj, self.is_transposed)
    gate_up_out = torch.ops.tt.sparse_matmul(
        input_a,
        gate_up_w,
        sparsity,
        nnz=0,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
    )  # 5D tiled [A=1, B=T/M, E, M, 2I] on both XLA and CPU.

    gate_up_bias = getattr(self, "gate_up_proj_bias", None)
    if gate_up_bias is not None:
        gate_up_out = gate_up_out + gate_up_bias.view(1, 1, E, 1, -1)

    activated = self._apply_gate(gate_up_out)  # same rank as gate_up_out

    # Flatten tiled output to canonical 4D so the down GEMM isn't re-detected
    # as MoE-shape (which would re-tile). On the default tile-reduced path
    # `activated` is 5D `[A, B, E, M, I]` with M=32. With the legacy per-token
    # toggle ON, M=1 and sparse_matmul's `squeeze(-2)` collapses the M dim,
    # yielding 4D `[A, B, E, I]`; restore an explicit M=1 before reshape so
    # the down GEMM sees canonical 4D input.
    if activated.dim() == 5:
        A, B, _, M_tile, _ = activated.shape
    else:  # 4D, M=1 collapsed (legacy per-token)
        A, B, _, _ = activated.shape
        M_tile = 1
        activated = activated.unsqueeze(3)
    activated = activated.reshape(A * B, E, M_tile, activated.shape[-1])
    down_w = _as_sparse_matmul_weight(self.down_proj, self.is_transposed)
    down_out = torch.ops.tt.sparse_matmul(
        activated,
        down_w,
        sparsity,
        nnz=0,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
    )  # [A*B, E, M, H]

    down_bias = getattr(self, "down_proj_bias", None)
    if down_bias is not None:
        down_out = down_out + down_bias.view(1, E, 1, -1)

    # Untile to [1, T, E, H] for per-expert gather. A*B and M are adjacent
    # so this is a permute + reshape, no copy.
    AB, _, M_tile, Hout = down_out.shape
    down_out = down_out.permute(0, 2, 1, 3).reshape(1, AB * M_tile, E, Hout)

    # Combine: gather the K expert outputs per token and weight-sum.
    # down_out: [1, T, E, H] → gather on E with top_k_index → [T, K, H].
    # This is O(T*K*H) vs the naive dense sum-over-E O(T*E*H), and needs no
    # dense routing_map (and thus no scatter_).
    Hout = down_out.shape[-1]
    idx = top_k_index.unsqueeze(-1).expand(-1, -1, Hout)  # [T, K, H]
    gathered = down_out.view(T, E, Hout).gather(1, idx)  # [T, K, H]
    weights = top_k_weights.to(gathered.dtype).unsqueeze(-1)  # [T, K, 1]
    output = (gathered * weights).sum(dim=1)  # [T, H]
    return output.to(dtype)


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
    gate_up_w = _as_sparse_matmul_weight(self.gate_up_proj, self.is_transposed)
    gate_up_out = torch.ops.tt.sparse_matmul(
        dispatched,
        gate_up_w,
        sparsity,
        nnz=0,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
    )  # 5D tiled: [BD, T/M, E, M, 2I]  (decode tiles on BD instead of T)

    gate_up_bias = getattr(self, "gate_up_proj_bias", None)
    if gate_up_bias is not None:
        if gate_up_out.dim() == 5:
            gate_up_out = gate_up_out + gate_up_bias.view(1, 1, E, 1, -1)
        else:
            gate_up_out = gate_up_out + gate_up_bias.view(1, 1, E, -1)

    activated = self._apply_gate(gate_up_out)

    # Flatten to canonical 4D before down GEMM.
    if activated.dim() == 5:
        A, B, _, M_tile, _ = activated.shape
        activated = activated.reshape(A * B, E, M_tile, activated.shape[-1])
    down_w = _as_sparse_matmul_weight(self.down_proj, self.is_transposed)
    down_out = torch.ops.tt.sparse_matmul(
        activated,
        down_w,
        sparsity,
        nnz=0,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
    )  # [A*B, E, M, H]

    # Per-expert bias [E, H] on canonical [A*B, E, M, H].
    down_bias = getattr(self, "down_proj_bias", None)
    if down_bias is not None:
        down_out = down_out + down_bias.view(1, E, 1, -1)

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
    # Fail fast with a clear error rather than a cryptic AttributeError mid-
    # forward when the Experts module doesn't follow the HF canonical layout.
    for _attr in (
        "num_experts",
        "gate_up_proj",
        "down_proj",
        "_apply_gate",
        "is_transposed",
    ):
        if not hasattr(self, _attr):
            raise RuntimeError(
                f"tt_moe backend requires the Experts module to expose "
                f"{_attr!r}. Got {type(self).__name__}. Is "
                f"@use_experts_implementation applied to this class?"
            )

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


def get_tt_moe_shard_specs(
    model: nn.Module,
    original_spec_fn: Callable[[nn.Module], Dict[Any, Any]],
    mesh_names: Tuple[str, str],
) -> Dict[Any, Any]:
    """Compound-shard expert weights across both mesh axes.

    Starts from `original_spec_fn(model)`, then for every transformer layer
    whose `mlp.experts` follows the HF canonical layout, shards `gate_up_proj`
    and `down_proj` (and their biases when present) with compound sharding
    `(mesh_names[0], mesh_names[1])` on the expert (first) dimension.

    The expected expert weight shapes are the HF canonical ones:
        gate_up_proj: [E, 2*I, H]
        down_proj:    [E, H, I]
    and biases:
        gate_up_proj_bias: [E, 2*I]
        down_proj_bias:    [E, H]
    """
    shard_specs = original_spec_fn(model)
    compound = (mesh_names[0], mesh_names[1])

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return shard_specs

    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        experts = getattr(mlp, "experts", None)
        if experts is None or not (
            hasattr(experts, "num_experts")
            and hasattr(experts, "gate_up_proj")
            and hasattr(experts, "down_proj")
        ):
            continue

        shard_specs[experts.gate_up_proj] = (compound, None, None)
        shard_specs[experts.down_proj] = (compound, None, None)

        gate_up_bias = getattr(experts, "gate_up_proj_bias", None)
        if gate_up_bias is not None:
            shard_specs[gate_up_bias] = (compound, None)
        down_bias = getattr(experts, "down_proj_bias", None)
        if down_bias is not None:
            shard_specs[down_bias] = (compound, None)

    return shard_specs
