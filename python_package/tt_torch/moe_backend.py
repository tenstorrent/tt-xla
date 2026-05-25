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

This module adds two Tenstorrent-targeted backends:

  - `tt_moe`   — multi-chip expert parallelism. Lowers to
                 `torch.ops.tt.all_to_all_dispatch`, `sparse_matmul`,
                 and `all_to_all_combine`. Requires a multi-chip global
                 torch_xla SPMD mesh.
  - `tt_dense` — dense bmm across all experts followed by a routing-weight
                 mask + sum. `O(num_experts * tokens)` compute, but tt-metal
                 bmm kernels make this the GPT-OSS perf path today.
                 Single-device-friendly; previously installed via a
                 `GptOssExperts.forward` monkey patch in
                 `torch_overrides.py`.

The sparsity mask for the post-dispatch `tt_moe` token layout is built by
`torch.ops.tt.moe_expert_token_remap`. No model-specific code is required:
any Experts module following the canonical HF layout

    self.num_experts   : int
    self.gate_up_proj  : Parameter [num_experts, 2*intermediate_dim, hidden_dim]
    self.down_proj     : Parameter [num_experts,     hidden_dim,     intermediate_dim]
    self._apply_gate   : callable installed by @use_experts_implementation

routes through the same code path.

Usage:

    from tt_torch.moe_backend import (
        TT_MOE_BACKEND_NAME,
        TT_DENSE_EXPERTS_BACKEND_NAME,
    )

    # Multi-chip expert parallelism.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, experts_implementation=TT_MOE_BACKEND_NAME,
    )

    # Single-device dense bmm path (replaces the legacy 4.57.1 monkey patch).
    model = AutoModelForCausalLM.from_pretrained(
        model_id, experts_implementation=TT_DENSE_EXPERTS_BACKEND_NAME,
    )

Shape constraint: `sparse_matmul`'s MoE fast-path tiles the token dimension
by `REDUCTION_SIZE=32`. Inputs are padded to a multiple of 32 tokens
internally (`_pad_moe_inputs`) and sliced back after the combine.
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
REDUCTION_SIZE = 32

# HF's built-in experts backends that `get_correct_experts_implementation`
# already accepts. The patched validator falls through to the original for
# these so the upstream behavior is preserved; non-built-in names that match
# anything in `ALL_EXPERTS_FUNCTIONS.valid_keys()` are short-circuited and
# returned directly. Update this when transformers ships a new built-in.
_HF_BUILTIN_EXPERTS_KEYS = frozenset({"eager", "grouped_mm", "batched_mm", "deepgemm"})

# Populated by `register_tt_moe_backend`. Module-level (rather than stashed on
# `self`) because these axes are properties of the mesh, not of a specific
# Experts instance, and HF gives no hook to thread them through from_pretrained.
#
# `cluster_axis`: mesh axis used for EP collectives. When unset, use the first
#                 non-singleton mesh axis.
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
    """Return weight in the `[1, E, in_features, out_features]` layout that
    `sparse_matmul` expects, respecting HF's `is_transposed` convention.

    `is_transposed=True`  → weight stored as [E, in, out] (e.g. GptOss).
    `is_transposed=False` → weight stored as [E, out, in] (e.g. Olmoe,
                                                           Qwen3-MoE).
    """
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


class _W123Adapter(_ExpertAdapter):
    def _weight(self, name: str) -> torch.Tensor:
        weight = getattr(self.module, name)
        if isinstance(weight, nn.Linear):
            weight = weight.weight
        return _as_sparse_matmul_weight(weight, True)

    @property
    def num_experts(self) -> int:
        return int(getattr(self.module, "num_experts", self.module.w1.shape[0]))

    def gate_weight(self) -> torch.Tensor:
        return self._weight("w1")

    def up_weight(self) -> torch.Tensor:
        return self._weight("w3")

    def down_weight(self) -> torch.Tensor:
        return self._weight("w2")


def _get_expert_adapter(module: nn.Module) -> _ExpertAdapter:
    if hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
        return _FusedGateUpAdapter(module)
    if (
        hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
    ):
        return _SeparateGateUpAdapter(module)
    if hasattr(module, "w1") and hasattr(module, "w2") and hasattr(module, "w3"):
        return _W123Adapter(module)
    raise RuntimeError(
        f"tt_moe backend could not adapt Experts module {type(module).__name__}. "
        "Expected fused gate_up/down, separate gate/up/down, or w1/w2/w3 experts."
    )


def _expert_mapping(
    num_experts: int,
    num_devices: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the `[1, 1, E, D]` one-hot expert-to-device mapping.

    Experts are sharded across the full mesh. For a 2D mesh this corresponds
    to a compound expert-axis sharding over all devices, while the CCL op still
    uses `cluster_axis` to choose the physical all-to-all direction. Keep the
    legacy sparse_mlp mapping contract: contiguous expert ranges map to the
    logical device id in `[0, D)`.
    """
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
    """Expand top-k weights into a full `[T, E]` sparse router-scores tensor.

    Use a pure functional one_hot + einsum construction here instead of
    `scatter_`. `torch.export.export` functionalizes the in-place scatter
    without error, but the resulting graph fails to lower on the TT PJRT
    device.
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
    In the EP path each shard still carries a length-`T` token axis after
    dispatch. Pad with zero-valued dummy tokens and slice them back off after
    combine.
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

    # `[1, 1, E, D]` one-hot mapping lifted into the graph as a pure tensor op
    # sequence so AOTAutograd/XLA can trace it without CPU copies or in-place
    # Python-side writes. D is the full mesh device count: expert weights are
    # compound-sharded across all mesh axes even though dispatch itself uses a
    # single cluster_axis.
    expert_mapping = _expert_mapping(E, total_devices, device)

    # Dispatch tokens along cluster_axis. Output is a full [1, B*D, T, H]
    # tensor where each of the D slices carries only the tokens its experts
    # will consume; the rest are zeros. `metadata` all-gathers top_k_index so
    # each device knows which expert slot produced which token.
    # `num_devices` here is `dispatch_devices` (cluster_axis size), not the
    # full mesh: the a2a/remap ops only fan out along the dispatch axis, and
    # the total-device info already lives in `expert_mapping`'s D dimension.
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

    # sparse_matmul's `is_input_b_sparse=True` MoE path returns a 5D tile layout
    # [A, B, E, M, N]; if that contract ever changes this unpack fails fast.
    A, B, _, M_tile, I_dim = activated.shape
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

    # Combine expects experts on dim 0 with all tokens flattened onto one
    # axis. Permute makes A*B and M adjacent, then reshape collapses them.
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

    Requires a multi-chip global torch_xla SPMD mesh and dispatches through
    `all_to_all_dispatch` + `all_to_all_combine`.
    """
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
    """HF `ExpertsInterface` forward for `tt_dense`.

    Dense-bmm experts: every token is run through every expert, then the
    per-expert outputs are masked by a full `[T, E]` routing-weights tensor
    and summed. Compute scales as `O(num_experts * tokens)`, but Tenstorrent
    bmm kernels make this faster than the scatter/gather of `batched_mm` on
    today's stack, which is why GPT-OSS bench used this shape via a
    monkey-patched `GptOssExperts.forward` previously.

    Requires a fused `gate_up_proj` / `down_proj` layout (GPT-OSS, Olmoe,
    GLM4-MoE, Qwen3-MoE, ...). Separate gate/up or w1/w2/w3 experts are not
    supported here — use `batched_mm` / `grouped_mm` for those. The per-expert
    activation is delegated to `self._apply_gate`, which
    `@use_experts_implementation` installs on the Experts module (GPT-OSS's
    interleaved clamp+sigmoid GLU, default `chunk(2)` silu*up, ...).

    No tile-padding is applied here (unlike `tt_experts_forward`); `bmm` does
    not require 32-aligned token counts so any `T` works.
    """
    if not (hasattr(self, "gate_up_proj") and hasattr(self, "down_proj")):
        raise RuntimeError(
            f"{TT_DENSE_EXPERTS_BACKEND_NAME} requires fused gate_up_proj/down_proj "
            f"experts; got {type(self).__name__} which does not expose them. "
            "Use a built-in HF backend (batched_mm/grouped_mm) for separate "
            "gate/up or w1/w2/w3 experts."
        )

    T, H = hidden_states.shape
    E = int(self.num_experts)
    dtype = hidden_states.dtype
    device = hidden_states.device

    # Expand compact `[T, K]` router output to full `[T, E]` routing weights.
    # Functional scatter (not in-place) so `torch.export` functionalizes
    # cleanly. The legacy 4.57.1 router did the same expansion server-side.
    routing_weights = torch.zeros(T, E, dtype=dtype, device=device).scatter(
        1, top_k_index, top_k_weights.to(dtype)
    )

    is_transposed = bool(getattr(self, "is_transposed", False))

    # Replicate hidden across experts so every token meets every expert.
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


_original_validator: Optional[Callable] = None


def register_tt_moe_backend(cluster_axis: Optional[int] = None) -> None:
    """Register the `tt_moe` and `tt_dense` experts backends globally.

    Idempotent. Also patches
    `PreTrainedModel.get_correct_experts_implementation` — HF hard-codes the
    accepted backend names there, so a custom key needs an additional escape
    hatch. `ExpertsInterface` itself is already extensible via `register()`.

    Args:
        cluster_axis: Mesh axis for EP all-to-all collectives. When None, the
            backend resolves it at trace time from the global mesh by using the
            first non-singleton axis. Calling this function again updates the
            value.
    """
    global _original_validator

    _config["cluster_axis"] = cluster_axis
    ExpertsInterface.register(TT_MOE_BACKEND_NAME, tt_experts_forward)
    assert TT_MOE_BACKEND_NAME in ALL_EXPERTS_FUNCTIONS, "registration did not stick"
    ExpertsInterface.register(TT_DENSE_EXPERTS_BACKEND_NAME, tt_dense_experts_forward)
    assert (
        TT_DENSE_EXPERTS_BACKEND_NAME in ALL_EXPERTS_FUNCTIONS
    ), "tt_dense registration did not stick"

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
    """Shard expert weights across the full expert-parallel mesh.

    Starts from `original_spec_fn(model)`, then for every transformer layer
    whose `mlp.experts` follows the HF canonical layout, shards
    `gate_up_proj` / `down_proj` (and their biases when present) on the single
    expert dimension. On 2D meshes this is compound sharding: one tensor axis
    is sharded by multiple mesh axes at once, matching the legacy
    `sparse_mlp.get_moe_shard_specs` behavior.

    Weight layout conventions:
        is_transposed=False (Olmoe, Qwen3-MoE, Afmoe, ...):
            gate_up_proj: [E, 2*I, H]   -> (expert, None, None)
            down_proj:    [E, H, I]     -> (expert, None, None)
        is_transposed=True (GptOss):
            gate_up_proj: [E, H, 2*I]   -> (expert, None, None)
            down_proj:    [E, I, H]     -> (expert, None, None)
    and biases:
            gate_up_proj_bias: [E, 2*I] -> (expert, None)
            down_proj_bias:    [E, H]   -> (expert, None)
    """
    shard_specs = original_spec_fn(model)
    expert_axis: Any = tuple(mesh_names) if len(mesh_names) > 1 else mesh_names[0]

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        # Either the model doesn't follow the HF `.model.layers` convention or
        # it has no transformer layers — return the upstream shard spec as-is.
        # Surface this rather than silently no-op'ing so callers can spot a
        # missing-MoE-sharding regression.
        import logging

        logging.getLogger(__name__).warning(
            "get_tt_moe_shard_specs: %s has no .model.layers; returning "
            "unchanged shard specs.",
            type(model).__name__,
        )
        return shard_specs

    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        experts = getattr(mlp, "experts", None)
        if experts is None:
            continue

        # Match the legacy sparse_mlp contract: only the expert dimension is
        # sharded. For 2D meshes `expert_axis` is a tuple, meaning one tensor
        # dimension is compound-sharded across all mesh axes.
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


# Make `experts_implementation="tt_moe"` available as soon as this module is
# imported. Callers may still call `register_tt_moe_backend(...)` to override
# mesh-axis configuration for unusual meshes.
register_tt_moe_backend()
