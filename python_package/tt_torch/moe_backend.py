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
by `REDUCTION_SIZE=32`. Inputs are padded to a multiple of 32 tokens
internally (`_pad_moe_inputs`) and sliced back after the combine. The legacy
per-token toggle skips padding and runs at `M=1`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, Tuple, cast

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


class _ExpertsModule(Protocol):
    num_experts: int
    gate_up_proj: torch.Tensor
    down_proj: torch.Tensor
    _apply_gate: Callable[[torch.Tensor], torch.Tensor]
    is_transposed: bool


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


def _pad_moe_inputs(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad token-axis inputs so the per-shard token axis is 32-tile aligned.

    The current `sparse_matmul` MoE fast-path only tiles cleanly when the
    token axis presented to the kernel is a multiple of `REDUCTION_SIZE`.
    That is true for the DP path directly, and for the EP path because each
    shard still carries a length-`T` token axis after dispatch. Pad with
    zero-valued dummy tokens and slice them back off after combine.
    """
    token_count, hidden_dim = hidden_states.shape
    if token_count == 0:
        return hidden_states, top_k_index, top_k_weights, token_count

    token_multiple = REDUCTION_SIZE
    padded_token_count = (
        (token_count + token_multiple - 1) // token_multiple
    ) * token_multiple
    if padded_token_count == token_count:
        return hidden_states, top_k_index, top_k_weights, token_count

    pad_tokens = padded_token_count - token_count
    hidden_pad = torch.zeros(
        pad_tokens,
        hidden_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    index_pad = torch.zeros(
        pad_tokens,
        top_k_index.shape[1],
        dtype=top_k_index.dtype,
        device=top_k_index.device,
    )
    weight_pad = torch.zeros(
        pad_tokens,
        top_k_weights.shape[1],
        dtype=top_k_weights.dtype,
        device=top_k_weights.device,
    )
    return (
        torch.cat((hidden_states, hidden_pad), dim=0),
        torch.cat((top_k_index, index_pad), dim=0),
        torch.cat((top_k_weights, weight_pad), dim=0),
        token_count,
    )


def _selected_mesh_axis_name(mesh_names: Tuple[str, ...]) -> str:
    """Return the mesh axis that owns expert sharding for this backend."""
    cluster_axis = _config["cluster_axis"]
    if not 0 <= cluster_axis < len(mesh_names):
        raise ValueError(
            f"cluster_axis={cluster_axis} is out of range for mesh_names={mesh_names}"
        )
    return mesh_names[cluster_axis]


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
    if USE_LEGACY_PER_TOKEN_SPARSITY:
        # Legacy per-token mask needs no 32-tile alignment: M=1 always.
        original_token_count = hidden_states.shape[0]
    else:
        hidden_states, top_k_index, top_k_weights, original_token_count = (
            _pad_moe_inputs(hidden_states, top_k_index, top_k_weights)
        )

    experts = cast(_ExpertsModule, self)
    T, H = hidden_states.shape
    E = experts.num_experts
    dtype = hidden_states.dtype
    device = hidden_states.device

    # Binary per-token mask [T, E]: `one_hot + any` over the top-K axis —
    # functional (no scatter_, AOT-safe) and ones-valued (weight-valued masks
    # would silently drop experts whose router weight is exactly zero).
    one_hot = top_k_index.unsqueeze(-1) == torch.arange(E, device=device)
    per_token = one_hot.any(dim=1)  # [T, E] bool

    # Bucket the per-token mask into [1, 1, T/tile, E] where a bit is 1 iff
    # any of its `tile` tokens selects the expert. Default uses 32-wide tiles
    # aligned to sparse_matmul's M=32 fast path; legacy uses `tile=1` straight
    # through — pure tech debt, see the module-level toggle comment. T is
    # pre-padded to a multiple of REDUCTION_SIZE on the default path, so the
    # view divides cleanly.
    tile = 1 if USE_LEGACY_PER_TOKEN_SPARSITY else REDUCTION_SIZE
    sparsity = per_token.view(1, 1, T // tile, tile, E).any(dim=3).to(dtype)

    # Fused gate/up projection via block-sparse batched GEMM.
    # sparse_matmul expects input_b as [1, E, in_features, out_features].
    # HF stores weights one of two ways (@use_experts_implementation metadata):
    #   is_transposed=True  → [E, in, out] = [E, H, 2I], ready as-is.
    #   is_transposed=False → [E, out, in] = [E, 2I, H], needs a transpose.
    input_a = hidden_states.view(1, 1, T, H)
    gate_up_w = _as_sparse_matmul_weight(experts.gate_up_proj, experts.is_transposed)
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

    activated = experts._apply_gate(gate_up_out)

    # Restore M if sparse_matmul squeezed it away. `sparse_matmul` emits a
    # `.squeeze(-2)` whenever M=1, which only happens on the legacy per-token
    # path (toggle above): the default path pads T to a multiple of 32
    # upstream and always reaches M=32.
    if activated.dim() == 4:
        activated = activated.unsqueeze(-2)
    A, B, _, M_tile, I_dim = activated.shape
    activated = activated.reshape(A * B, E, M_tile, I_dim)

    down_w = _as_sparse_matmul_weight(experts.down_proj, experts.is_transposed)
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

    # Merge tile axis into token axis: [A*B, E, M, H] → [T, E, H].
    # A*B and M are adjacent so this is a permute + reshape, no copy.
    down_out = down_out.permute(0, 2, 1, 3).reshape(T, E, H)

    # Gather the K expert outputs per token and weight-sum — O(T*K*H) vs
    # the naive dense sum-over-E O(T*E*H), and no dense routing_map.
    idx = top_k_index.unsqueeze(-1).expand(-1, -1, H)  # [T, K, H]
    gathered = down_out.gather(1, idx)  # [T, K, H]
    output = (gathered * top_k_weights.to(gathered.dtype).unsqueeze(-1)).sum(dim=1)
    output = output[:original_token_count]
    return output.to(dtype)


def _tt_experts_forward_ep(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    dispatch_devices: int,
) -> torch.Tensor:
    """Expert-parallel compute: dispatch tokens to the devices holding their
    selected experts, run the sparse_matmul chain on the dispatched layout,
    then combine per-expert outputs back to original token positions.
    """
    hidden_states, top_k_index, top_k_weights, original_token_count = _pad_moe_inputs(
        hidden_states, top_k_index, top_k_weights
    )

    experts = cast(_ExpertsModule, self)
    T, H = hidden_states.shape
    K = top_k_index.shape[-1]
    E = experts.num_experts
    dtype = hidden_states.dtype
    device = hidden_states.device
    cluster_axis = _config["cluster_axis"]
    routing_scores = _build_routing_scores(
        top_k_index,
        top_k_weights,
        E,
        dtype,
        device,
    )

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
    gate_up_w = _as_sparse_matmul_weight(experts.gate_up_proj, experts.is_transposed)
    gate_up_out = torch.ops.tt.sparse_matmul(
        dispatched,
        gate_up_w,
        sparsity,
        nnz=0,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
    )  # 5D tiled: [BD, T/M, E, M, 2I]  (decode tiles on BD instead of T).
    # T is a multiple of 32 post-pad, so M=32 is always reached; no
    # M=1-squeezed 4D case here (contrast the DP path).

    gate_up_bias = getattr(self, "gate_up_proj_bias", None)
    if gate_up_bias is not None:
        gate_up_out = gate_up_out + gate_up_bias.view(1, 1, E, 1, -1)

    activated = experts._apply_gate(gate_up_out)

    A, B, _, M_tile, I_dim = activated.shape
    activated = activated.reshape(A * B, E, M_tile, I_dim)

    down_w = _as_sparse_matmul_weight(experts.down_proj, experts.is_transposed)
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

    # Combine expects experts on dim 0 with all tokens flattened onto one
    # axis. A*B and M are adjacent so this is a permute + reshape, no copy.
    down_out = down_out.permute(1, 0, 2, 3).reshape(E, 1, -1, H)

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
    output = output[:original_token_count]
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

    return _tt_experts_forward_ep(
        self,
        hidden_states,
        top_k_index,
        top_k_weights,
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
    original_validator = _original_validator

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
        return original_validator(self, requested_experts)

    PreTrainedModel.get_correct_experts_implementation = patched_validator


def get_tt_moe_shard_specs(
    model: nn.Module,
    original_spec_fn: Callable[[nn.Module], Dict[Any, Any]],
    mesh_names: Tuple[str, ...],
) -> Dict[Any, Any]:
    """Shard expert weights only along the backend's selected cluster axis.

    Starts from `original_spec_fn(model)`, then for every transformer layer
    whose `mlp.experts` follows the HF canonical layout, shards `gate_up_proj`
    and `down_proj` (and their biases when present) on the expert (first)
    dimension using the same mesh axis that `_expert_mapping()` dispatches
    across. This keeps parameter sharding aligned with the expert-to-device
    mapping used by the EP collectives.

    The expected expert weight shapes are the HF canonical ones:
        gate_up_proj: [E, 2*I, H]
        down_proj:    [E, H, I]
    and biases:
        gate_up_proj_bias: [E, 2*I]
        down_proj_bias:    [E, H]
    """
    shard_specs = original_spec_fn(model)
    expert_axis = _selected_mesh_axis_name(mesh_names)

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

        shard_specs[experts.gate_up_proj] = (expert_axis, None, None)
        shard_specs[experts.down_proj] = (expert_axis, None, None)

        gate_up_bias = getattr(experts, "gate_up_proj_bias", None)
        if gate_up_bias is not None:
            shard_specs[gate_up_bias] = (expert_axis, None)
        down_bias = getattr(experts, "down_proj_bias", None)
        if down_bias is not None:
            shard_specs[down_bias] = (expert_axis, None)

    return shard_specs
