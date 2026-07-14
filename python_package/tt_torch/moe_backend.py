# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent MoE experts backend for HuggingFace transformers.

Registers three ``ExpertsInterface`` backends selectable via
``experts_implementation=`` at ``from_pretrained`` time:

  - ``tt_moe``       — multi-chip EP via all_to_all_dispatch / sparse_matmul / all_to_all_combine.
  - ``tt_dense``     — dense bmm across all experts, single-device-friendly.
  - ``tt_moe_fused`` — dense bmm during prefill, tt.moe_decode composite during decode.

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
TT_MOE_FUSED_BACKEND_NAME = "tt_moe_fused"
REDUCTION_SIZE = 32

# Default flattened-token count at/below which tt_moe_fused treats a call as
# decode (emit tt.moe_decode) rather than prefill (tt_dense_experts_forward).
# One tile: decode runs with one token per sequence, so the token count equals
# the batch size; configurable via register_tt_moe_backend().
DEFAULT_MOE_DECODE_TOKEN_THRESHOLD = 32

# Default output_height_shard_dim for the emitted tt.moe_decode op. tt-mlir's
# moe_compute requires this to be positive (it drives the data-parallel
# tilize-drain core layout); 4 matches the tt-mlir op default. Configurable via
# register_tt_moe_backend().
DEFAULT_MOE_OUTPUT_HEIGHT_SHARD_DIM = 4

# HF built-in backend keys — patched validator falls through for these.
_HF_BUILTIN_EXPERTS_KEYS = frozenset({"eager", "grouped_mm", "batched_mm", "deepgemm"})

# Module-level EP config; set by register_tt_moe_backend().
_config: dict = {
    "cluster_axis": None,
    "moe_decode_activation": "silu",
    "moe_decode_token_threshold": DEFAULT_MOE_DECODE_TOKEN_THRESHOLD,
    "moe_use_interleaved_gate_up": False,
    "moe_output_height_shard_dim": DEFAULT_MOE_OUTPUT_HEIGHT_SHARD_DIM,
}


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

    # Build the [T, E] router-scores tensor via one_hot + einsum rather than
    # scatter: scatter lowers to a costly, TT-incompatible per-chunk decomposition
    # (the rest of this backend avoids scatter for the same reason; cf. tt-xla2's
    # _build_routing_scores). top-k indices are distinct per token, so this is
    # numerically equivalent.
    one_hot = (top_k_index.unsqueeze(-1) == torch.arange(E, device=device)).to(dtype)
    routing_weights = torch.einsum("tk,tke->te", top_k_weights.to(dtype), one_hot)

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


def _moe_decode_params(
    experts: _ExpertAdapter,
    use_interleaved: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Build the tt.moe_decode weights/biases from an expert adapter.

    Returns ``(w0, w1, w2, bias0, bias1, bias2)`` where:
        w0 (gate) [1, E, H, N]   w1 (up) [1, E, H, N]   w2 (down) [1, E, N, H]
        bias0 [1, E, N]  bias1 [1, E, N]  bias2 [1, E, H]  (or all None)

    The adapters already return weights in sparse_matmul ``[1, E, in, out]``
    orientation, which is exactly what the composite expects. A fused ``gate_up``
    weight packs gate and up into the trailing ``2N`` dimension; ``use_interleaved``
    selects how it is de-packed into the separate ``w0``/``w1`` the op expects:

      - ``False`` (default): concat packing ``[gate | up]`` — gate is the first
        ``N`` columns, up the second (the ``chunk(2)`` convention used by most
        models, e.g. Llama4 / DeepSeek / GLM).
      - ``True``: interleaved packing ``[g0, u0, g1, u1, ...]`` — gate is the even
        columns, up the odd (e.g. GPT-OSS, which also wants
        ``activation_function="swiglu"``).

    Separate gate/up weights are already de-packed, so the flag is a no-op there.
    The caller is responsible for setting ``use_interleaved`` to match the model.

    tt.moe_decode is all-or-none on bias: if any of gate/up/down bias is present,
    the missing ones are zero-filled to satisfy the 9-operand contract.
    """
    if experts.has_fused_gate_up:
        gate_up = experts.gate_up_weight()  # [1, E, H, 2N]
        gate_up_bias = experts.gate_up_bias()  # [E, 2N] | None
        if use_interleaved:
            # Gate = even output columns, up = odd. The strided gathers are
            # non-contiguous, so materialize before handing them to the op.
            w0 = gate_up[..., 0::2].contiguous()  # [1, E, H, N]
            w1 = gate_up[..., 1::2].contiguous()
            gate_bias = None if gate_up_bias is None else gate_up_bias[..., 0::2]
            up_bias = None if gate_up_bias is None else gate_up_bias[..., 1::2]
        else:
            # Gate = first half, up = second half.
            n = gate_up.shape[-1] // 2
            w0 = gate_up[..., :n].contiguous()  # [1, E, H, N]
            w1 = gate_up[..., n:].contiguous()
            gate_bias = None if gate_up_bias is None else gate_up_bias[..., :n]
            up_bias = None if gate_up_bias is None else gate_up_bias[..., n:]
    else:
        w0 = experts.gate_weight()  # [1, E, H, N]
        w1 = experts.up_weight()
        gate_bias = experts.gate_bias()  # [E, N] | None
        up_bias = experts.up_bias()
    w2 = experts.down_weight()  # [1, E, N, H]
    down_bias = experts.down_bias()  # [E, H] | None

    _, E, H, N = w0.shape

    if gate_bias is None and up_bias is None and down_bias is None:
        return w0, w1, w2, None, None, None

    def _bias_or_zeros(bias: Optional[torch.Tensor], feat: int) -> torch.Tensor:
        if bias is None:
            return torch.zeros(1, E, feat, dtype=w0.dtype, device=w0.device)
        return bias.reshape(1, E, feat)

    return (
        w0,
        w1,
        w2,
        _bias_or_zeros(gate_bias, N),
        _bias_or_zeros(up_bias, N),
        _bias_or_zeros(down_bias, H),
    )


# Attribute names for the stacked (all-layer) fused-decode weights + per-layer
# index, stamped by preprocess_tt_moe_compute_stacked_weights. When present, the
# decode forward passes ONE shared [L,E,...] weight (the runtime prepare packs it
# into a single DRAM-resident buffer, num_layers=L) indexed by layer_id, instead
# of preparing a fresh 1-layer buffer per layer (layer_id=0) which collapses decode
# PCC past a single layer.
_TT_MOE_STACKED_W0_ATTR = "_tt_moe_stacked_w0"
_TT_MOE_STACKED_W1_ATTR = "_tt_moe_stacked_w1"
_TT_MOE_STACKED_W2_ATTR = "_tt_moe_stacked_w2"
_TT_MOE_STACKED_B0_ATTR = "_tt_moe_stacked_b0"
_TT_MOE_STACKED_B1_ATTR = "_tt_moe_stacked_b1"
_TT_MOE_STACKED_B2_ATTR = "_tt_moe_stacked_b2"
_TT_MOE_LAYER_IDX_ATTR = "layer_idx"


def _tt_moe_decode_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Decode forward: emit tt.moe_decode, then routing-weight and sum its
    per-top-k outputs back into ``[T, H]``."""
    hidden_states, top_k_index, top_k_weights, original_token_count = _pad_moe_inputs(
        hidden_states, top_k_index, top_k_weights
    )

    experts = _get_expert_adapter(self)
    M, H = hidden_states.shape
    K = top_k_index.shape[-1]
    dtype = hidden_states.dtype

    # Prefer the stacked all-layer weights (preprocess_tt_moe_compute_stacked_weights):
    # ONE shared [L,E,...] weight that the runtime prepare packs into a single
    # DRAM-resident buffer (num_layers=L), indexed per layer by layer_id. Falls back
    # to this layer's own weights (L=1, layer_id=0) when not stacked.
    stacked_w0 = getattr(self, _TT_MOE_STACKED_W0_ATTR, None)
    if stacked_w0 is not None:
        w0 = stacked_w0
        w1 = getattr(self, _TT_MOE_STACKED_W1_ATTR)
        w2 = getattr(self, _TT_MOE_STACKED_W2_ATTR)
        bias0 = getattr(self, _TT_MOE_STACKED_B0_ATTR, None)
        bias1 = getattr(self, _TT_MOE_STACKED_B1_ATTR, None)
        bias2 = getattr(self, _TT_MOE_STACKED_B2_ATTR, None)
    else:
        w0, w1, w2, bias0, bias1, bias2 = _moe_decode_params(
            experts, use_interleaved=_config["moe_use_interleaved_gate_up"]
        )
    intermediate_size = w0.shape[-1]

    tokens = hidden_states.view(1, 1, M, H)
    indices = top_k_index.view(1, 1, M, K)
    scores = top_k_weights.to(dtype).view(1, 1, M, K)

    _, _, _, cluster_axis = _mesh_info()
    # layer_id selects this layer's block inside the packed multi-layer weight
    # buffer (dm0.cpp offset = layer_id * layer_pages_per_ring_core). Real per-layer
    # index with stacked weights; 0 otherwise.
    layer_id = int(getattr(self, _TT_MOE_LAYER_IDX_ATTR, 0) or 0)

    combined = torch.ops.tt.moe_decode(
        tokens,
        indices,
        scores,
        w0,
        w1,
        w2,
        bias0=bias0,
        bias1=bias1,
        bias2=bias2,
        cluster_axis=cluster_axis,
        layer_id=layer_id,
        output_height_shard_dim=_config["moe_output_height_shard_dim"],
        intermediate_size=intermediate_size,
        activation_function=_config["moe_decode_activation"],
    )  # [K, M, H]

    weights_k = top_k_weights.permute(1, 0).view(K, M, 1).to(combined.dtype)
    output = (combined * weights_k).sum(dim=0).view(M, H)  # [M, H]
    output = output[:original_token_count]
    return output.to(dtype)


def preprocess_tt_moe_compute_stacked_weights(model: nn.Module) -> list:
    """Stack every MoE layer's expert weights into shared ``[L, E, ...]`` parameters
    for the fused-decode packed-prepare design.

    Each ``tt.moe_decode`` call then passes the SAME stacked weight plus a unique
    ``layer_id``, so the runtime ``prepare_moe_compute_*`` op packs ALL layers into
    ONE DRAM-resident weight buffer (``num_layers = L``, derived from
    ``w0.logical_shape()[0]``) and each ``moe_compute`` reads its own block via
    ``layer_id`` (``dm0.cpp`` offset ``= layer_id * layer_pages_per_ring_core``).
    Without this each layer prepares its own 1-layer buffer with ``layer_id = 0`` and
    decode PCC collapses past a single layer (2 layers ~0.97 -> 24 layers ~0.28).

    MUST run on CPU BEFORE ``model.to(device)`` (so the ``cat`` is a host op rather
    than an on-device replicate) and BEFORE ``shard_spec_fn`` (so the stacked params
    exist to be sharded). The unfused per-layer expert weights are left in place for
    the dense prefill path. Returns the list of experts modules stamped.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return []

    experts_list = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None) if mlp is not None else None
        if experts is None:
            continue
        experts_list.append(experts)
    if not experts_list:
        return []

    use_interleaved = _config["moe_use_interleaved_gate_up"]
    per_layer = [
        _moe_decode_params(_get_expert_adapter(e), use_interleaved=use_interleaved)
        for e in experts_list
    ]

    def _stack(idx: int):
        parts = [p[idx] for p in per_layer]
        if any(x is None for x in parts):
            return None
        # Each part is [1, E, ...]; cat over the leading (layer) dim -> [L, E, ...].
        return nn.Parameter(torch.cat(parts, dim=0).contiguous(), requires_grad=False)

    stacked_w0 = _stack(0)
    stacked_w1 = _stack(1)
    stacked_w2 = _stack(2)
    stacked_b0 = _stack(3)
    stacked_b1 = _stack(4)
    stacked_b2 = _stack(5)

    for i, experts in enumerate(experts_list):
        # Same Parameter objects on every layer -> identical operands -> the L
        # prepare ops CSE into a single packed-buffer prepare in tt-mlir.
        setattr(experts, _TT_MOE_STACKED_W0_ATTR, stacked_w0)
        setattr(experts, _TT_MOE_STACKED_W1_ATTR, stacked_w1)
        setattr(experts, _TT_MOE_STACKED_W2_ATTR, stacked_w2)
        setattr(experts, _TT_MOE_STACKED_B0_ATTR, stacked_b0)
        setattr(experts, _TT_MOE_STACKED_B1_ATTR, stacked_b1)
        setattr(experts, _TT_MOE_STACKED_B2_ATTR, stacked_b2)
        setattr(experts, _TT_MOE_LAYER_IDX_ATTR, i)

    return experts_list


def tt_moe_fused_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Fused MoE forward: dense bmm during prefill, tt.moe_decode during decode.

    The call is treated as decode when the flattened token count is at/below
    ``_config["moe_decode_token_threshold"]`` (decode runs one token per
    sequence, so the token count is the batch size); otherwise it is prefill and
    routes to ``tt_dense_experts_forward``. CPU tensors fall back to HF
    ``batched_mm``.

    tt.moe_decode is an expert-parallel decode kernel: tt-mlir synthesizes its
    expert_mapping from the module mesh and shards experts along the cluster
    axis, so it only lowers on a multi-chip mesh. On a single chip (or a
    degenerate mesh whose resolved cluster axis is size 1) the decode path falls
    back to the dense bmm as well, since there is no usable EP mesh to lower onto.
    """
    if hidden_states.device.type == "cpu":
        return ALL_EXPERTS_FUNCTIONS["batched_mm"](
            self, hidden_states, top_k_index, top_k_weights
        )

    _, dispatch_devices, _, _ = _mesh_info()
    is_decode = (
        dispatch_devices > 1
        and hidden_states.shape[0] <= _config["moe_decode_token_threshold"]
    )
    if not is_decode:
        return tt_dense_experts_forward(self, hidden_states, top_k_index, top_k_weights)

    return _tt_moe_decode_forward(self, hidden_states, top_k_index, top_k_weights)


_original_validator: Optional[Callable] = None


def register_tt_moe_backend(
    cluster_axis: Optional[int] = None,
    moe_decode_activation: str = "silu",
    moe_decode_token_threshold: int = DEFAULT_MOE_DECODE_TOKEN_THRESHOLD,
    use_interleaved: bool = False,
    moe_output_height_shard_dim: int = DEFAULT_MOE_OUTPUT_HEIGHT_SHARD_DIM,
) -> None:
    """Register tt_moe, tt_dense and tt_moe_fused backends. Idempotent and
    re-entrant: re-resolves transformers each call so it survives a version swap.

    Args:
        cluster_axis: EP/dispatch mesh axis; ``None`` auto-resolves the first
            mesh axis larger than 1.
        moe_decode_activation: GLU activation tt_moe_fused stamps onto the
            emitted tt.moe_decode op ("silu" or "swiglu").
        moe_decode_token_threshold: flattened-token count at/below which
            tt_moe_fused treats a call as decode.
        use_interleaved: how tt_moe_fused de-packs a fused ``gate_up`` weight for
            tt.moe_decode — ``False`` for concat ``[gate | up]`` packing
            (default), ``True`` for interleaved ``[g0, u0, g1, u1, ...]`` packing
            (e.g. GPT-OSS). The caller sets this to match the model under test.
        moe_output_height_shard_dim: ``output_height_shard_dim`` stamped onto the
            emitted tt.moe_decode op; must be positive (tt-mlir's moe_compute
            rejects 0). Defaults to 4 (the tt-mlir op default).
    """
    global _original_validator, ALL_EXPERTS_FUNCTIONS, ExpertsInterface
    global PreTrainedModel

    import transformers.integrations.moe as _moe
    import transformers.modeling_utils as _mu

    ALL_EXPERTS_FUNCTIONS = _moe.ALL_EXPERTS_FUNCTIONS
    ExpertsInterface = _moe.ExpertsInterface
    PreTrainedModel = _mu.PreTrainedModel

    if moe_decode_activation not in ("silu", "swiglu"):
        raise ValueError(
            f"moe_decode_activation must be 'silu' or 'swiglu', got "
            f"{moe_decode_activation!r}"
        )
    if moe_output_height_shard_dim <= 0:
        raise ValueError(
            f"moe_output_height_shard_dim must be positive, got "
            f"{moe_output_height_shard_dim}"
        )

    _config["cluster_axis"] = cluster_axis
    _config["moe_decode_activation"] = moe_decode_activation
    _config["moe_decode_token_threshold"] = moe_decode_token_threshold
    _config["moe_use_interleaved_gate_up"] = use_interleaved
    _config["moe_output_height_shard_dim"] = moe_output_height_shard_dim
    ExpertsInterface.register(TT_MOE_BACKEND_NAME, tt_experts_forward)
    if TT_MOE_BACKEND_NAME not in ALL_EXPERTS_FUNCTIONS:
        raise RuntimeError(f"{TT_MOE_BACKEND_NAME} registration failed")
    ExpertsInterface.register(TT_DENSE_EXPERTS_BACKEND_NAME, tt_dense_experts_forward)
    if TT_DENSE_EXPERTS_BACKEND_NAME not in ALL_EXPERTS_FUNCTIONS:
        raise RuntimeError(f"{TT_DENSE_EXPERTS_BACKEND_NAME} registration failed")
    ExpertsInterface.register(TT_MOE_FUSED_BACKEND_NAME, tt_moe_fused_forward)
    if TT_MOE_FUSED_BACKEND_NAME not in ALL_EXPERTS_FUNCTIONS:
        raise RuntimeError(f"{TT_MOE_FUSED_BACKEND_NAME} registration failed")

    # Re-patch the live PreTrainedModel; a version swap brings a fresh class.
    if getattr(
        PreTrainedModel.get_correct_experts_implementation, "_tt_patched", False
    ):
        return
    original_validator = PreTrainedModel.get_correct_experts_implementation
    _original_validator = original_validator

    def patched_validator(self, requested_experts):
        if (
            requested_experts in ALL_EXPERTS_FUNCTIONS.valid_keys()
            and requested_experts not in _HF_BUILTIN_EXPERTS_KEYS
        ):
            return requested_experts
        return original_validator(self, requested_experts)

    patched_validator._tt_patched = True
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
