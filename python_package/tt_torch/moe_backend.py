# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent MoE experts backend for HuggingFace transformers.

Registers three ``ExpertsInterface`` backends selectable via
``experts_implementation=`` at ``from_pretrained`` time:

  - ``tt_moe``        — multi-chip EP via all_to_all_dispatch / sparse_matmul / all_to_all_combine.
  - ``tt_dense``      — dense bmm across all experts, single-device-friendly.
  - ``tt_moe_stream`` — low-batch: stream only the top-k selected experts (b1-style),
                        computing k of E instead of all E (memory-bound decode win).

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
TT_MOE_STREAM_BACKEND_NAME = "tt_moe_stream"
REDUCTION_SIZE = 32

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
    )

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


def tt_stream_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Low-batch MoE forward: stream only the top-k selected experts (b1-style)
    instead of computing all E. Requires fused gate_up_proj / down_proj.

    Mathematically identical to ``tt_dense_experts_forward`` (same weighted sum over
    the selected experts), but only ``k`` of ``E`` expert weights are read — the
    memory-bound win at batch=1 decode. Each ``stream_experts_matmul`` lowers to b1's
    DRAM-streaming generic_op (in-kernel ``base + expert_idx`` gather).
    """
    if not (hasattr(self, "gate_up_proj") and hasattr(self, "down_proj")):
        raise RuntimeError(
            f"{TT_MOE_STREAM_BACKEND_NAME} requires fused gate_up_proj/down_proj "
            f"experts; got {type(self).__name__} which does not expose them. "
            "Use a built-in HF backend (batched_mm/grouped_mm) for separate "
            "gate/up experts."
        )

    k = top_k_index.shape[-1]
    dtype = hidden_states.dtype
    is_transposed = bool(getattr(self, "is_transposed", False))

    # Normalize fused weights to [E, K, N] with K the contraction dim.
    gate_up_w = (
        self.gate_up_proj if is_transposed else self.gate_up_proj.transpose(-1, -2)
    )
    down_w = self.down_proj if is_transposed else self.down_proj.transpose(-1, -2)

    # gate_up: one shared activation per token, b1 DRAM-streamed over the k selected
    # experts (reads k of E expert weights -> the batch=1 memory-bound win). Lowers to
    # b1's DRAM streaming-experts generic_op via the tt_lang_op path.
    gate_up = torch.ops.tt.stream_experts_matmul(
        hidden_states, gate_up_w, top_k_index, k, per_expert_input=False
    )  # [T, k, 2*inter]
    gate_up_bias = getattr(self, "gate_up_proj_bias", None)
    if gate_up_bias is not None:
        gate_up = gate_up + gate_up_bias[top_k_index]

    activated = self._apply_gate(gate_up)  # [T, k, inter]

    # down: per-expert activation, streamed over the same k selected experts.
    down_out = torch.ops.tt.stream_experts_matmul(
        activated, down_w, top_k_index, k, per_expert_input=True
    )  # [T, k, H]
    down_bias = getattr(self, "down_proj_bias", None)
    if down_bias is not None:
        down_out = down_out + down_bias[top_k_index]

    # Combine: weighted sum over the k selected experts.
    weighted = down_out * top_k_weights.to(down_out.dtype).unsqueeze(-1)
    return weighted.sum(dim=1).to(dtype)


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
    ExpertsInterface.register(TT_MOE_STREAM_BACKEND_NAME, tt_stream_experts_forward)
    if TT_MOE_STREAM_BACKEND_NAME not in ALL_EXPERTS_FUNCTIONS:
        raise RuntimeError(f"{TT_MOE_STREAM_BACKEND_NAME} registration failed")

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
