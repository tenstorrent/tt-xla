# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent MoE experts backend for HuggingFace transformers.

Registers two ``ExpertsInterface`` backends selectable via
``experts_implementation=`` at ``from_pretrained`` time:

  - ``tt_moe``   — multi-chip EP via all_to_all_dispatch / sparse_matmul / all_to_all_combine.
  - ``tt_dense`` — dense bmm across all experts, single-device-friendly.

Works with any ``@use_experts_implementation`` Experts module that exposes
``gate_up_proj``, ``down_proj``, and ``_apply_gate``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS, ExpertsInterface
from transformers.modeling_utils import PreTrainedModel

# Ensure torch.ops.tt.* are registered.
from . import custom_ops  # noqa: F401

TT_MOE_BACKEND_NAME = "tt_moe"
TT_DENSE_EXPERTS_BACKEND_NAME = "tt_dense"
TT_MOE_GPT_BACKEND_NAME = "tt_moe_gpt"
REDUCTION_SIZE = 32

# Global batch size used by the experts-level tt_moe_gpt forward to tell decode
# (S==1) from prefill (S>1). The HF ExpertsInterface forward only receives
# flattened [T, H] tokens (T=B*S), so it cannot see S directly; with S==1 the
# token count T == batch. It is stored as a MODULE-LEVEL GLOBAL (not an
# nn.Module instance attribute) because torch.compile/Dynamo does not resolve
# arbitrary instance attributes during tracing, but does read module globals as
# guarded constants — so ``T == _tt_moe_gpt_decode_batch`` is a clean
# compile-time-constant branch. Set once before compile by
# register_tt_moe_gpt_decode_hooks.
_tt_moe_gpt_decode_batch: Optional[int] = None
_TT_MOE_GPT_BATCH_ATTR = "_tt_moe_gpt_batch"

# Buffer names for the dispatch/moe_gpt expert mappings, registered on each
# experts module at setup so they enter the graph as replicated parameters
# (not inline const-eval constants). See register_tt_moe_gpt_decode_hooks.
_TT_DISPATCH_MAPPING_ATTR = "_tt_dispatch_mapping"
_TT_MOE_GPT_MAPPING_ATTR = "_tt_moe_gpt_mapping"

# Parameter names for the preprocessed 6D fused kernel weights, stamped on each
# experts module before device transfer (CPU preprocessing) and sharded with a
# 6D ("batch","model",...) spec. See preprocess_tt_moe_gpt_fused_weights.
_TT_FUSED_W0_W1_ATTR = "fused_w0_w1"
_TT_FUSED_W2_ATTR = "fused_w2"

# The SPMD mesh shape/cluster_axis, also stored as module-level globals set once
# before compile. ``get_global_mesh()`` returns None during the torch.compile
# trace of the experts forward, so the live mesh must be captured at setup time
# and read here as Dynamo-readable constants.
_tt_moe_gpt_mesh_shape: Optional[Tuple[int, ...]] = None
_tt_moe_gpt_cluster_axis: Optional[int] = None


def _tt_moe_gpt_mesh_info() -> Tuple[int, int, Tuple[int, ...], int]:
    """Mesh info for the tt_moe_gpt forward, preferring the globals stashed at
    setup over ``get_global_mesh()`` (which is None under torch.compile)."""
    mesh_shape = _tt_moe_gpt_mesh_shape
    if mesh_shape is not None and len(mesh_shape) >= 1:
        total = 1
        for d in mesh_shape:
            total *= int(d)
        axis = _tt_moe_gpt_cluster_axis
        if axis is None:
            axis = next(
                (i for i, d in enumerate(mesh_shape) if int(d) > 1),
                0,
            )
        dispatch = int(mesh_shape[axis]) if 0 <= axis < len(mesh_shape) else 1
        return total, dispatch, tuple(int(d) for d in mesh_shape), int(axis)
    return _mesh_info()


# HF built-in backend keys — patched validator falls through for these.
_HF_BUILTIN_EXPERTS_KEYS = frozenset({"eager", "grouped_mm", "batched_mm", "deepgemm"})

# Module-level EP config; set by register_tt_moe_backend().
_config: dict = {"cluster_axis": None}


def _resolve_cluster_axis(mesh: Any) -> int:
    configured_axis = _config["cluster_axis"]
    if configured_axis is not None:
        return int(configured_axis)

    for axis, size in enumerate(tuple(int(d) for d in mesh.mesh_shape)):
        if size > 1:
            return axis
    return 0


def _mesh_info() -> Tuple[int, int, Tuple[int, ...], int]:
    """Return (total_devices, dispatch_devices_on_cluster_axis, mesh_shape, axis).

    Reads the currently-set torch_xla global SPMD mesh. Returns (1, 1, (1,), 0)
    when no mesh is registered or torch_xla is unavailable.
    """
    try:
        from torch_xla.distributed.spmd import get_global_mesh
    except ImportError:
        return 1, 1, (1,), 0
    mesh = get_global_mesh()
    if mesh is None:
        return 1, 1, (1,), 0
    mesh_shape = tuple(int(d) for d in mesh.mesh_shape)
    total = 1
    for d in mesh_shape:
        total *= d
    ax = _resolve_cluster_axis(mesh)
    dispatch = mesh_shape[ax] if 0 <= ax < len(mesh_shape) else 1
    return total, dispatch, mesh_shape, ax


def _as_sparse_matmul_weight(weight: torch.Tensor, is_transposed: bool) -> torch.Tensor:
    """Reshape weight to `[1, E, in, out]` for `sparse_matmul`."""
    if is_transposed:
        return weight.unsqueeze(0).contiguous()
    return weight.transpose(-2, -1).unsqueeze(0).contiguous()


class _ExpertAdapter:
    """Normalize model-specific expert parameter names into semantic weights."""

    def __init__(self, module: nn.Module):
        self.module = module

    @property
    def num_experts(self) -> int:
        return int(getattr(self.module, "num_experts"))

    @property
    def has_fused_gate_up(self) -> bool:
        return False

    def gate_up_weight(self) -> torch.Tensor:
        raise RuntimeError("Adapter does not expose fused gate/up weights")

    def gate_weight(self) -> torch.Tensor:
        raise RuntimeError("Adapter does not expose separate gate weights")

    def up_weight(self) -> torch.Tensor:
        raise RuntimeError("Adapter does not expose separate up weights")

    def down_weight(self) -> torch.Tensor:
        raise NotImplementedError

    def gate_up_bias(self) -> Optional[torch.Tensor]:
        return None

    def gate_bias(self) -> Optional[torch.Tensor]:
        return None

    def up_bias(self) -> Optional[torch.Tensor]:
        return None

    def down_bias(self) -> Optional[torch.Tensor]:
        return None

    def apply_gate(
        self, gate_or_gate_up: torch.Tensor, up: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if up is None:
            raise RuntimeError("Separate gate/up adapter requires an up tensor")
        return F.silu(gate_or_gate_up) * up


class _FusedGateUpAdapter(_ExpertAdapter):
    @property
    def has_fused_gate_up(self) -> bool:
        return True

    def gate_up_weight(self) -> torch.Tensor:
        return _as_sparse_matmul_weight(
            self.module.gate_up_proj, bool(getattr(self.module, "is_transposed", False))
        )

    def down_weight(self) -> torch.Tensor:
        return _as_sparse_matmul_weight(
            self.module.down_proj, bool(getattr(self.module, "is_transposed", False))
        )

    def gate_up_bias(self) -> Optional[torch.Tensor]:
        return getattr(self.module, "gate_up_proj_bias", None)

    def down_bias(self) -> Optional[torch.Tensor]:
        return getattr(self.module, "down_proj_bias", None)

    def apply_gate(
        self, gate_or_gate_up: torch.Tensor, up: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.module._apply_gate(gate_or_gate_up)


class _SeparateGateUpAdapter(_ExpertAdapter):
    def _weight(self, name: str) -> torch.Tensor:
        return _as_sparse_matmul_weight(
            getattr(self.module, name),
            bool(getattr(self.module, "is_transposed", False)),
        )

    def gate_weight(self) -> torch.Tensor:
        return self._weight("gate_proj")

    def up_weight(self) -> torch.Tensor:
        return self._weight("up_proj")

    def down_weight(self) -> torch.Tensor:
        return self._weight("down_proj")

    def gate_bias(self) -> Optional[torch.Tensor]:
        return getattr(self.module, "gate_proj_bias", None)

    def up_bias(self) -> Optional[torch.Tensor]:
        return getattr(self.module, "up_proj_bias", None)

    def down_bias(self) -> Optional[torch.Tensor]:
        return getattr(self.module, "down_proj_bias", None)

    def apply_gate(
        self, gate_or_gate_up: torch.Tensor, up: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if hasattr(self.module, "_apply_gate"):
            return self.module._apply_gate(torch.cat((gate_or_gate_up, up), dim=-1))
        return super().apply_gate(gate_or_gate_up, up)


def _get_expert_adapter(module: nn.Module) -> _ExpertAdapter:
    if hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
        return _FusedGateUpAdapter(module)
    if (
        hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
    ):
        return _SeparateGateUpAdapter(module)
    raise RuntimeError(
        f"tt_moe backend could not adapt Experts module {type(module).__name__}. "
        "Expected fused gate_up/down or separate gate/up/down experts."
    )


def _expert_mapping(
    num_experts: int,
    num_devices: int,
    device: torch.device,
) -> torch.Tensor:
    """Build `[1, 1, E, D]` one-hot expert-to-device mapping."""
    if num_experts % num_devices != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by num_devices "
            f"({num_devices}) to build a one-hot expert-to-device mapping."
        )

    experts_per_device = num_experts // num_devices
    device_ids = (
        torch.arange(num_experts, device=device, dtype=torch.int64)
        // experts_per_device
    ).to(torch.uint16)

    mapping = (
        device_ids.unsqueeze(-1)
        == torch.arange(num_devices, device=device, dtype=torch.int64).to(torch.uint16)
    ).to(torch.uint16)
    return mapping.view(1, 1, num_experts, num_devices)


def _build_routing_scores(
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Expand top-k weights into a full `[T, E]` sparse router-scores tensor."""
    one_hot = (
        top_k_index.unsqueeze(-1) == torch.arange(num_experts, device=device)
    ).to(dtype)
    return torch.einsum("tk,tke->te", top_k_weights.to(dtype), one_hot)


def _pad_moe_inputs(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad token axis to a multiple of REDUCTION_SIZE (32) for sparse_matmul."""
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


def _tt_experts_forward_ep(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    total_devices: int,
    dispatch_devices: int,
    cluster_axis: int,
) -> torch.Tensor:
    """Expert-parallel compute: dispatch tokens to the devices holding their
    selected experts, run the sparse_matmul chain on the dispatched layout,
    then combine per-expert outputs back to original token positions.
    """
    hidden_states, top_k_index, top_k_weights, original_token_count = _pad_moe_inputs(
        hidden_states, top_k_index, top_k_weights
    )

    experts = _get_expert_adapter(self)
    T, H = hidden_states.shape
    K = top_k_index.shape[-1]
    E = experts.num_experts
    dtype = hidden_states.dtype
    device = hidden_states.device
    routing_scores = _build_routing_scores(
        top_k_index,
        top_k_weights,
        E,
        dtype,
        device,
    )

    expert_mapping = _expert_mapping(E, total_devices, device)  # [1, 1, E, D_total]

    # num_devices = dispatch_devices (cluster_axis size), not total.
    hidden_3d = hidden_states.view(1, T, H)
    dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
        hidden_3d,
        top_k_index,
        expert_mapping,
        num_devices=dispatch_devices,
        cluster_axis=cluster_axis,
    )  # dispatched: [1, BD, T, H];  metadata: [1, BD, T, K]
    BD = dispatched.shape[1]
    metadata = metadata.reshape(1, 1, BD * T, K)

    # Sparsity from all-gathered metadata (not local router scores).
    _, sparsity = torch.ops.tt.moe_expert_token_remap(
        routing_scores,
        expert_mapping,
        metadata,
        num_devices=dispatch_devices,
    )  # sparsity: [1, 1, ceil(BD*T/32), E]

    if experts.has_fused_gate_up:
        gate_up_out = torch.ops.tt.sparse_matmul(
            dispatched,
            experts.gate_up_weight(),
            sparsity,
            nnz=0,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )
        gate_up_bias = experts.gate_up_bias()
        if gate_up_bias is not None:
            gate_up_out = gate_up_out + gate_up_bias.view(1, 1, E, 1, -1)
        activated = experts.apply_gate(gate_up_out)
    else:
        gate_out = torch.ops.tt.sparse_matmul(
            dispatched,
            experts.gate_weight(),
            sparsity,
            nnz=0,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )
        up_out = torch.ops.tt.sparse_matmul(
            dispatched,
            experts.up_weight(),
            sparsity,
            nnz=0,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
        )
        gate_bias = experts.gate_bias()
        if gate_bias is not None:
            gate_out = gate_out + gate_bias.view(1, 1, E, 1, -1)
        up_bias = experts.up_bias()
        if up_bias is not None:
            up_out = up_out + up_bias.view(1, 1, E, 1, -1)
        activated = experts.apply_gate(gate_out, up_out)

    A, B, _, M_tile, I_dim = activated.shape  # 5D [A, B, E, M, N]
    activated = activated.reshape(A * B, E, M_tile, I_dim)

    down_out = torch.ops.tt.sparse_matmul(
        activated,
        experts.down_weight(),
        sparsity,
        nnz=0,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
    )  # [A*B, E, M, H]

    down_bias = experts.down_bias()
    if down_bias is not None:
        down_out = down_out + down_bias.view(1, E, 1, -1)

    down_out = down_out.permute(1, 0, 2, 3).reshape(E, 1, -1, H)  # [E, 1, BD*S, H]

    combined = torch.ops.tt.all_to_all_combine(
        down_out,
        metadata,
        expert_mapping,
        num_devices=dispatch_devices,
        cluster_axis=cluster_axis,
        num_experts_per_tok=K,
        output_shard_dim=2,
    )  # [K, 1, T, H]

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
    """Multi-chip EP forward. CPU tensors fall back to HF `batched_mm`."""
    if hidden_states.device.type == "cpu":
        return ALL_EXPERTS_FUNCTIONS["batched_mm"](
            self, hidden_states, top_k_index, top_k_weights
        )

    total_devices, dispatch_devices, mesh_shape, cluster_axis = _mesh_info()

    if total_devices <= 1 or dispatch_devices <= 1:
        raise RuntimeError(
            f"{TT_MOE_BACKEND_NAME} requires a multi-chip SPMD mesh with an EP "
            f"axis larger than 1, got mesh_shape={mesh_shape} and "
            f"cluster_axis={cluster_axis}. Use a built-in HF experts backend "
            "such as 'eager' or 'batched_mm' for single-device runs."
        )

    return _tt_experts_forward_ep(
        self,
        hidden_states,
        top_k_index,
        top_k_weights,
        total_devices,
        dispatch_devices,
        cluster_axis,
    )


def tt_dense_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Dense-bmm forward over all experts, masked by routing weights.
    Requires fused gate_up_proj / down_proj."""
    if not (hasattr(self, "gate_up_proj") and hasattr(self, "down_proj")):
        raise RuntimeError(
            f"{TT_DENSE_EXPERTS_BACKEND_NAME} requires fused gate_up_proj/down_proj "
            f"experts; got {type(self).__name__} which does not expose them. "
            "Use a built-in HF backend (batched_mm/grouped_mm) for separate "
            "gate/up experts."
        )

    T, H = hidden_states.shape
    E = int(self.num_experts)
    dtype = hidden_states.dtype
    device = hidden_states.device

    routing_weights = torch.zeros(T, E, dtype=dtype, device=device).scatter(
        1, top_k_index, top_k_weights.to(dtype)
    )

    is_transposed = bool(getattr(self, "is_transposed", False))

    h = hidden_states.repeat(E, 1).view(E, T, H)
    gate_up_w = (
        self.gate_up_proj if is_transposed else self.gate_up_proj.transpose(-1, -2)
    )
    gate_up = torch.bmm(h, gate_up_w)
    gate_up_bias = getattr(self, "gate_up_proj_bias", None)
    if gate_up_bias is not None:
        gate_up = gate_up + gate_up_bias.unsqueeze(1)

    activated = self._apply_gate(gate_up)

    down_w = self.down_proj if is_transposed else self.down_proj.transpose(-1, -2)
    down_out = torch.bmm(activated, down_w)
    down_bias = getattr(self, "down_proj_bias", None)
    if down_bias is not None:
        down_out = down_out + down_bias.unsqueeze(1)

    weighted = down_out * routing_weights.transpose(0, 1).unsqueeze(-1)
    return weighted.sum(dim=0).to(dtype)


# ---------------------------------------------------------------------------
# GPT-OSS fused decode backend (tt_moe_gpt).
#
# Emits the ``tenstorrent.moe_gpt_decode`` StableHLO composite — whose body is
# the all_to_all_dispatch_metadata -> moe_gpt -> selective_reduce_combine chain
# — which tt-MLIR legalizes to ``ttir.moe_gpt_decode``. Decode-only (S==1);
# prefill (S>1) falls back to the dense backend. The HF ExpertsInterface
# contract is preserved: the router still runs as plain HF ops, so this backend
# never uses tt.topk_router_gpt.
# ---------------------------------------------------------------------------


def build_expert_mapping_linearized(
    num_experts: int,
    num_mesh_devices: int,
    mesh_shape: Tuple[int, int],
    cluster_axis: int,
) -> torch.Tensor:
    """Build the ``[1, 1, D_total, E]`` linearized expert-to-device mapping.

    Matches tt-metal's ``gen_expert_mapping`` for the fused decode path: every
    mesh device holds an identical copy of the mapping (a broadcast of a single
    row), and ``mapping[0, 0, d, e]`` stores the linearized global device id
    (``row_id * mesh_cols + col_id``) that owns expert ``e``.

    ``cluster_axis`` selects which mesh axis is the dispatch ring:
      - 0: ring runs down rows; each column holds a distinct expert group.
      - 1: ring runs across columns; experts distributed as ``e // E_per_dev``.

    The last dim is ``num_experts`` and ``shape[2]`` (``D_total``) is the full
    mesh device count — tt-MLIR's ``ttir.moe_gpt_decode`` verifier requires
    ``D_total`` to be a positive multiple of ``num_devices`` (the ring size).
    """
    assert (
        num_experts % num_mesh_devices == 0
    ), f"num_experts ({num_experts}) must be divisible by num_mesh_devices ({num_mesh_devices})"
    rows, cols = mesh_shape
    assert (
        rows * cols == num_mesh_devices
    ), f"mesh_shape {mesh_shape} does not match num_mesh_devices {num_mesh_devices}"
    assert cluster_axis in (0, 1), f"cluster_axis must be 0 or 1, got {cluster_axis}"

    experts_per_device = num_experts // num_mesh_devices
    num_replicated_devices = cols if cluster_axis == 0 else rows
    experts_per_cluster = num_experts // num_replicated_devices

    # Build the per-expert device id as a plain python list so the tensor is a
    # single traced constant (no in-graph scatter).
    row = []
    for e in range(num_experts):
        if cluster_axis == 0:
            cluster_id = e // experts_per_cluster
            within = e % experts_per_cluster
            dev_in_cluster = within // experts_per_device
            linear = dev_in_cluster * num_replicated_devices + cluster_id
        else:
            linear = e // experts_per_device
        row.append(linear)

    # The dispatch / moe_gpt kernels read the mapping as uint16 (the tilize
    # reader casts the buffer to uint16_t*). Emit it as uint16 from the start so
    # no si32->u16 typecast is inserted on the way into the kernels: that
    # typecast would tilize the mapping and then host-untilize + to_device it,
    # reloading the (ROW_MAJOR) mapping as TILE and tripping the runtime layout
    # check. Device ids fit in uint16 (mesh size << 65536).
    mapping = torch.tensor(row, dtype=torch.int64).view(1, 1, 1, num_experts)
    return (
        mapping.expand(1, 1, num_mesh_devices, num_experts)
        .contiguous()
        .to(torch.uint16)
    )


def _moe_gpt_decode_fallback(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    dispatch_mapping: torch.Tensor,
    moe_gpt_mapping: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    num_devices: int,
    cluster_axis: int,
    num_experts_per_tok: int,
    fused_w0_w1: Optional[torch.Tensor] = None,
    fused_w2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode-only GPT-OSS decomposition expressed with the SHLO custom ops.

    Expects 4D inputs: hidden_states ``[B, 1, S, H]``, topk_indices/scores
    ``[B, 1, S, K]``. Returns the unweighted combine output ``[K, S, B, H]``
    (output_shard_dim=2); the weighted sum is applied by the caller.

    When ``fused_w0_w1`` / ``fused_w2`` (preprocessed 6D kernel weights) are
    supplied, ``moe_gpt`` consumes those directly; otherwise the tt-MLIR
    decomposition substitutes zero placeholders for the fused weights.
    """
    dispatched, metadata_indices, metadata_scores = (
        torch.ops.tt.all_to_all_dispatch_metadata(
            hidden_states,
            topk_indices,
            topk_scores,
            dispatch_mapping,
            num_devices=num_devices,
            cluster_axis=cluster_axis,
        )
    )

    # Mirror tt-metal's fused decode sequence:
    # dispatch_metadata -> moe_gpt(metadata bundle) -> selective_reduce_combine.
    moe_gpt_outputs = torch.ops.tt.moe_gpt(
        dispatched,
        metadata_indices,
        metadata_scores,
        moe_gpt_mapping,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        num_experts_per_tok=num_experts_per_tok,
        num_devices=num_devices,
        cluster_axis=cluster_axis,
        fused_w0_w1=fused_w0_w1,
        fused_w2=fused_w2,
    )

    combined = torch.ops.tt.selective_reduce_combine(
        moe_gpt_outputs[4],
        moe_gpt_outputs[1],
        moe_gpt_outputs[2],
        moe_gpt_outputs[0],
        num_devices=num_devices,
        cluster_axis=cluster_axis,
        num_experts_per_tok=num_experts_per_tok,
        output_shard_dim=2,
    )
    return combined


def _composite_moe_gpt_decode(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    dispatch_mapping: torch.Tensor,
    moe_gpt_mapping: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    num_devices: int,
    cluster_axis: int,
    num_experts: int,
    num_experts_per_tok: int,
    intermediate_size: int,
    alpha: float,
    limit: float,
    fused_w0_w1: torch.Tensor,
    fused_w2: torch.Tensor,
) -> torch.Tensor:
    """Wrap the GPT-OSS decode expert flow into a ``tenstorrent.moe_gpt_decode``
    StableHLO composite. tt-MLIR legalizes the composite to a placeholder
    ``ttir.moe_gpt_decode`` op, while the embedded decomposition exposes the
    constituent custom calls for sharding propagation.

    The preprocessed 6D ``fused_w0_w1`` / ``fused_w2`` kernel weights are marked
    as the trailing two composite operands (operands 9, 10) so the composite
    re-forms with 11 operands and ``ttir.moe_gpt_decode`` binds the real fused
    weights instead of zero placeholders.
    """
    from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder

    builder = StableHLOCompositeBuilder(
        name="tenstorrent.moe_gpt_decode",
        attr={
            "num_devices": num_devices,
            "cluster_axis": cluster_axis,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "intermediate_size": intermediate_size,
            "alpha": alpha,
            "limit": limit,
        },
    )

    (
        hidden_states,
        topk_indices,
        topk_scores,
        dispatch_mapping,
        moe_gpt_mapping,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        fused_w0_w1,
        fused_w2,
    ) = builder.mark_inputs(
        hidden_states,
        topk_indices,
        topk_scores,
        dispatch_mapping,
        moe_gpt_mapping,
        gate_up_proj,
        gate_up_proj_bias,
        down_proj,
        down_proj_bias,
        fused_w0_w1,
        fused_w2,
    )

    output = _moe_gpt_decode_fallback(
        hidden_states=hidden_states,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
        dispatch_mapping=dispatch_mapping,
        moe_gpt_mapping=moe_gpt_mapping,
        gate_up_proj=gate_up_proj,
        gate_up_proj_bias=gate_up_proj_bias,
        down_proj=down_proj,
        down_proj_bias=down_proj_bias,
        num_devices=num_devices,
        cluster_axis=cluster_axis,
        num_experts_per_tok=num_experts_per_tok,
        fused_w0_w1=fused_w0_w1,
        fused_w2=fused_w2,
    )
    return builder.mark_outputs(output)


def _tt_moe_gpt_decode_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    total_devices: int,
    dispatch_devices: int,
    mesh_shape: Tuple[int, ...],
    cluster_axis: int,
) -> torch.Tensor:
    """GPT-OSS fused decode (S==1) via the moe_gpt_decode composite.

    ``hidden_states`` arrives flattened as ``[T, H]`` with T == B (S==1), and
    ``top_k_index`` / ``top_k_weights`` as ``[T, K]``. We frame each token as a
    batch element with S=1, build the linearized expert mapping from the live
    SPMD mesh, emit the composite, then apply the top-k weighted sum.
    """
    if not (hasattr(self, "gate_up_proj") and hasattr(self, "down_proj")):
        raise RuntimeError(
            f"{TT_MOE_GPT_BACKEND_NAME} requires fused gate_up_proj/down_proj "
            f"experts; got {type(self).__name__}."
        )

    T, H = hidden_states.shape
    K = top_k_index.shape[-1]
    E = int(self.num_experts)
    dtype = hidden_states.dtype
    device = hidden_states.device

    if len(mesh_shape) != 2:
        raise RuntimeError(
            f"{TT_MOE_GPT_BACKEND_NAME} expects a 2D mesh (rows, cols), got "
            f"mesh_shape={mesh_shape}."
        )

    # Prefer the replicated mapping buffers stashed on the experts module at
    # setup. As registered buffers they enter the graph as parameters (runtime
    # inputs), so the moe_gpt op's mapping reshape/to_layout are not const-eval
    # candidates (inline-constant mappings get hoisted into L1 and fail
    # ConstEvalHoistTransform). They are also two DISTINCT operands, so
    # ReoutlineComposite keeps the full 9-operand composite (no dedup). Fall back
    # to inline builds only if the hooks were not registered.
    if hasattr(self, _TT_DISPATCH_MAPPING_ATTR) and hasattr(
        self, _TT_MOE_GPT_MAPPING_ATTR
    ):
        dispatch_mapping = getattr(self, _TT_DISPATCH_MAPPING_ATTR)
        moe_gpt_mapping = getattr(self, _TT_MOE_GPT_MAPPING_ATTR)
    else:
        dispatch_mapping = build_expert_mapping_linearized(
            E, total_devices, (int(mesh_shape[0]), int(mesh_shape[1])), cluster_axis
        ).to(device)
        moe_gpt_mapping = build_expert_mapping_linearized(
            E, total_devices, (int(mesh_shape[0]), int(mesh_shape[1])), cluster_axis
        ).to(device)

    # Preprocessed 6D fused kernel weights, stashed on the experts module at
    # setup. Without them ``ttir.moe_gpt_decode`` binds zero-filled placeholder
    # weights (see TTIRToTTIRDecomposition), so they are mandatory for decode.
    if not (hasattr(self, _TT_FUSED_W0_W1_ATTR) and hasattr(self, _TT_FUSED_W2_ATTR)):
        raise RuntimeError(
            f"{TT_MOE_GPT_BACKEND_NAME} decode requires preprocessed fused "
            f"weights ('{_TT_FUSED_W0_W1_ATTR}'/'{_TT_FUSED_W2_ATTR}') on the "
            f"experts module; call preprocess_tt_moe_gpt_fused_weights(model, "
            f"...) before torch.compile."
        )
    fused_w0_w1 = getattr(self, _TT_FUSED_W0_W1_ATTR)
    fused_w2 = getattr(self, _TT_FUSED_W2_ATTR)

    # Frame tokens as [B, 1, S, H] with B=T, S=1 (decode).
    hidden_4d = hidden_states.view(T, 1, 1, H)
    indices_4d = top_k_index.view(T, 1, 1, K)
    scores_4d = top_k_weights.to(dtype).view(T, 1, 1, K)

    intermediate_size = int(self.down_proj.shape[1])
    alpha = float(getattr(self, "alpha", 1.702))
    limit = float(getattr(self, "limit", 7.0))

    combined = _composite_moe_gpt_decode(
        hidden_states=hidden_4d,
        topk_indices=indices_4d,
        topk_scores=scores_4d,
        dispatch_mapping=dispatch_mapping,
        moe_gpt_mapping=moe_gpt_mapping,
        gate_up_proj=self.gate_up_proj,
        gate_up_proj_bias=self.gate_up_proj_bias,
        down_proj=self.down_proj,
        down_proj_bias=self.down_proj_bias,
        num_devices=dispatch_devices,
        cluster_axis=cluster_axis,
        num_experts=E,
        num_experts_per_tok=K,
        intermediate_size=intermediate_size,
        alpha=alpha,
        limit=limit,
        fused_w0_w1=fused_w0_w1,
        fused_w2=fused_w2,
    )  # combined: [K, S=1, B=T, H]

    # Top-k weighted sum (done outside the composite so batch-sharded weights
    # are not pulled into the dispatch-replicated composite body).
    weights_k = top_k_weights.permute(1, 0).view(K, T, 1).to(combined.dtype)
    output = (combined.squeeze(1) * weights_k).sum(dim=0)  # [T, H]
    return output.to(dtype)


def _tt_moe_gpt_is_decode(self: torch.nn.Module, hidden_states: torch.Tensor) -> bool:
    """Decode (S==1) ⇔ flattened token count equals the global batch size.

    Reads the batch size from the module-level global (Dynamo-readable) so the
    comparison is a compile-time-constant branch under torch.compile. Falls back
    to the per-module attribute for eager execution.
    """
    batch = _tt_moe_gpt_decode_batch
    if batch is None:
        batch = getattr(self, _TT_MOE_GPT_BATCH_ATTR, None)
    return batch is not None and int(hidden_states.shape[0]) == int(batch)


def tt_moe_gpt_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """GPT-OSS experts backend that fuses decode through ``moe_gpt_decode``.

    Routing decision:
      - CPU tensors -> HF ``batched_mm`` reference.
      - prefill (S>1) or no SPMD mesh -> dense ``tt_dense`` fallback.
      - decode (S==1) on a multi-chip mesh -> moe_gpt_decode composite.

    Decode is detected from the token count: with S==1 the flattened token axis
    T == batch size, which is stamped on the module before compile (see
    ``register_tt_moe_gpt_decode_hooks``). When the batch size is unknown we
    conservatively fall back to the dense path so the HF interface still works.
    """
    if hidden_states.device.type == "cpu":
        return ALL_EXPERTS_FUNCTIONS["batched_mm"](
            self, hidden_states, top_k_index, top_k_weights
        )

    total_devices, dispatch_devices, mesh_shape, cluster_axis = _tt_moe_gpt_mesh_info()
    is_decode = _tt_moe_gpt_is_decode(self, hidden_states)

    if (
        is_decode
        and total_devices > 1
        and dispatch_devices > 1
        and len(mesh_shape) == 2
    ):
        return _tt_moe_gpt_decode_forward(
            self,
            hidden_states,
            top_k_index,
            top_k_weights,
            total_devices,
            dispatch_devices,
            mesh_shape,
            cluster_axis,
        )

    # Prefill (S>1), single device, or unknown batch size: dense fallback.
    return tt_dense_experts_forward(self, hidden_states, top_k_index, top_k_weights)


def preprocess_tt_moe_gpt_fused_weights(
    model: nn.Module,
    mesh_shape: Optional[Tuple[int, ...]] = None,
    cluster_axis: Optional[int] = None,
) -> list:
    """Preprocess each experts module's weights into the 6D fused decode kernel
    layout and stamp them as ``fused_w0_w1`` / ``fused_w2`` parameters.

    ``ttir.moe_gpt_decode`` binds these preprocessed weights directly; without
    them the tt-MLIR decomposition substitutes zero-filled placeholders, so the
    kernel computes on zero weights.

    MUST be called:
      - BEFORE ``model.to(device)`` — the transform runs on the full (global)
        expert weights, so doing it on CPU avoids replicating them on-device.
      - BEFORE ``shard_spec_fn`` — so the new parameters exist and can be sharded
        with a 6D ``("batch","model",None,None,None,None)`` spec.

    Returns the list of experts modules that were stamped.
    """
    from .fused_moe_weights import preprocess_fused_moe_weights

    if mesh_shape is None or len(mesh_shape) != 2:
        return []
    mesh_shape = (int(mesh_shape[0]), int(mesh_shape[1]))
    num_devices = mesh_shape[0] * mesh_shape[1]
    if cluster_axis is None:
        cluster_axis = next((i for i, d in enumerate(mesh_shape) if int(d) > 1), 0)

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return []

    stamped = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None) if mlp is not None else None
        if experts is None or not hasattr(experts, "gate_up_proj"):
            continue
        E = int(getattr(experts, "num_experts", experts.gate_up_proj.shape[0]))
        fused_w0_w1, fused_w2 = preprocess_fused_moe_weights(
            gate_up_proj=experts.gate_up_proj.data,
            gate_up_proj_bias=experts.gate_up_proj_bias.data,
            down_proj=experts.down_proj.data,
            down_proj_bias=experts.down_proj_bias.data,
            num_experts=E,
            num_devices=num_devices,
            cluster_axis=int(cluster_axis),
            mesh_shape=mesh_shape,
        )
        setattr(
            experts,
            _TT_FUSED_W0_W1_ATTR,
            nn.Parameter(fused_w0_w1, requires_grad=False),
        )
        setattr(experts, _TT_FUSED_W2_ATTR, nn.Parameter(fused_w2, requires_grad=False))
        stamped.append(experts)
    return stamped


def register_tt_moe_gpt_decode_hooks(
    model: nn.Module,
    batch_size: Optional[int] = None,
    mesh_shape: Optional[Tuple[int, ...]] = None,
    cluster_axis: Optional[int] = None,
) -> list:
    """Prepare GPT-OSS MoE modules so ``tt_moe_gpt`` can detect + run the decode step.

    The HF ExpertsInterface forward only receives flattened ``[T, H]`` tokens
    (T=B*S), so it cannot tell decode (S==1) from prefill (S>1) on its own. We
    stamp the global ``batch_size`` as an immutable constant on each experts
    submodule: since S==1 makes T==batch, the experts forward detects decode
    with a compile-time-constant comparison that survives torch.compile/Dynamo.

    The ``mesh_shape``/``cluster_axis`` are captured too, because
    ``get_global_mesh()`` returns None during the torch.compile trace of the
    experts forward. All three are stored as module-level globals (read by the
    experts forward under torch.compile) and the batch is also stamped on each
    experts module (eager fallback). Set ONCE before compile (no per-forward
    mutation, no forward replacement), so the HF interface stays intact. Returns
    the list of experts modules that were stamped.
    """
    global _tt_moe_gpt_decode_batch, _tt_moe_gpt_mesh_shape, _tt_moe_gpt_cluster_axis
    if mesh_shape is not None:
        _tt_moe_gpt_mesh_shape = tuple(int(d) for d in mesh_shape)
    _tt_moe_gpt_cluster_axis = cluster_axis

    if batch_size is None:
        return []
    _tt_moe_gpt_decode_batch = int(batch_size)

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return []

    # Resolve the dispatch (cluster) axis + device count the same way the forward
    # does, so the stashed mapping buffers match what the moe_gpt op expects.
    have_mesh = mesh_shape is not None and len(mesh_shape) == 2
    resolved_axis = cluster_axis
    if resolved_axis is None and have_mesh:
        resolved_axis = next((i for i, d in enumerate(mesh_shape) if int(d) > 1), 0)
    total_devices = 1
    if have_mesh:
        for d in mesh_shape:
            total_devices *= int(d)

    stamped = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None) if mlp is not None else None
        if experts is None:
            continue
        setattr(experts, _TT_MOE_GPT_BATCH_ATTR, int(batch_size))
        # Stash the dispatch/moe_gpt expert mappings as two DISTINCT replicated
        # buffers (graph parameters), not inline constants. As parameters they
        # become runtime inputs, so the moe_gpt op's mapping reshape/to_layout
        # are NOT const-eval candidates (inline-constant mappings get hoisted by
        # ConstEvalHoistTransform, land in L1, and fail the const-eval check).
        # Two separate buffers also keep them as distinct composite operands so
        # ReoutlineComposite does not dedup them into an 8-operand composite.
        if have_mesh and hasattr(experts, "gate_up_proj"):
            E = int(getattr(experts, "num_experts", experts.gate_up_proj.shape[0]))
            dev = experts.gate_up_proj.device
            ms = (int(mesh_shape[0]), int(mesh_shape[1]))
            dispatch_map = build_expert_mapping_linearized(
                E, total_devices, ms, int(resolved_axis)
            ).to(dev)
            moe_gpt_map = build_expert_mapping_linearized(
                E, total_devices, ms, int(resolved_axis)
            ).to(dev)
            experts.register_buffer(
                _TT_DISPATCH_MAPPING_ATTR, dispatch_map, persistent=False
            )
            experts.register_buffer(
                _TT_MOE_GPT_MAPPING_ATTR, moe_gpt_map, persistent=False
            )
        stamped.append(experts)
    return stamped


_original_validator: Optional[Callable] = None


def register_tt_moe_backend(cluster_axis: Optional[int] = None) -> None:
    """Register tt_moe and tt_dense backends. Idempotent."""
    global _original_validator

    _config["cluster_axis"] = cluster_axis
    ExpertsInterface.register(TT_MOE_BACKEND_NAME, tt_experts_forward)
    if TT_MOE_BACKEND_NAME not in ALL_EXPERTS_FUNCTIONS:
        raise RuntimeError(f"{TT_MOE_BACKEND_NAME} registration failed")
    ExpertsInterface.register(TT_DENSE_EXPERTS_BACKEND_NAME, tt_dense_experts_forward)
    if TT_DENSE_EXPERTS_BACKEND_NAME not in ALL_EXPERTS_FUNCTIONS:
        raise RuntimeError(f"{TT_DENSE_EXPERTS_BACKEND_NAME} registration failed")
    ExpertsInterface.register(TT_MOE_GPT_BACKEND_NAME, tt_moe_gpt_experts_forward)
    if TT_MOE_GPT_BACKEND_NAME not in ALL_EXPERTS_FUNCTIONS:
        raise RuntimeError(f"{TT_MOE_GPT_BACKEND_NAME} registration failed")

    if _original_validator is not None:
        return  # already patched

    _original_validator = PreTrainedModel.get_correct_experts_implementation
    original_validator = _original_validator

    def patched_validator(self, requested_experts):
        if (
            requested_experts in ALL_EXPERTS_FUNCTIONS.valid_keys()
            and requested_experts not in _HF_BUILTIN_EXPERTS_KEYS
        ):
            return requested_experts
        return original_validator(self, requested_experts)

    PreTrainedModel.get_correct_experts_implementation = patched_validator


def get_tt_moe_shard_specs(
    model: nn.Module,
    original_spec_fn: Callable[[nn.Module], Dict[Any, Any]],
    mesh_names: Tuple[str, ...],
) -> Dict[Any, Any]:
    """Add expert-dimension sharding to the upstream shard spec for MoE layers."""
    shard_specs = original_spec_fn(model)
    expert_axis: Any = tuple(mesh_names) if len(mesh_names) > 1 else mesh_names[0]

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return shard_specs

    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        experts = getattr(mlp, "experts", None)
        if experts is None:
            continue

        # Shard only the expert dimension; compound axis for 2D meshes.
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            shard_specs[experts.gate_up_proj] = (expert_axis, None, None)
            gate_up_bias = getattr(experts, "gate_up_proj_bias", None)
            if gate_up_bias is not None:
                shard_specs[gate_up_bias] = (expert_axis, None)
        elif all(
            hasattr(experts, name) for name in ("gate_proj", "up_proj", "down_proj")
        ):
            shard_specs[experts.gate_proj] = (expert_axis, None, None)
            shard_specs[experts.up_proj] = (expert_axis, None, None)
            gate_bias = getattr(experts, "gate_proj_bias", None)
            if gate_bias is not None:
                shard_specs[gate_bias] = (expert_axis, None)
            up_bias = getattr(experts, "up_proj_bias", None)
            if up_bias is not None:
                shard_specs[up_bias] = (expert_axis, None)
        elif all(hasattr(experts, name) for name in ("w1", "w2", "w3")):
            shard_specs[experts.w1] = (expert_axis, None, None)
            shard_specs[experts.w3] = (expert_axis, None, None)
        else:
            continue

        down = getattr(experts, "down_proj", getattr(experts, "w2", None))
        if down is not None:
            shard_specs[down] = (expert_axis, None, None)
        down_bias = getattr(experts, "down_proj_bias", None)
        if down_bias is not None:
            shard_specs[down_bias] = (expert_axis, None)

    return shard_specs


register_tt_moe_backend()
