# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT FusedMoE integration for vLLM 0.25.1.

Replaces the upstream fused-expert kernel (which doesn't lower cleanly through
the TT compiler) with expert dispatch delegated to ``tt_torch.moe_backend``:
expert-parallel ``tt_experts_forward`` on a genuine 2D mesh, else dense-bmm
``tt_dense_experts_forward``.

0.25.1 dropped the subclassable ``FusedMoE`` / ``op_registry_oot`` hook the old
``TTFusedMoE`` used: ``FusedMoE`` is now a factory returning a ``MoERunner``, and
the weights + routing state live on ``RoutedExperts``. So we monkey-patch the
factory and post-process each ``MoERunner`` (Option 3): swap its quant method to
force the monolithic path into a TT ``forward_native``, and bypass the opaque
``vllm.moe_forward`` custom op the compiler can't trace.

TODO (needs tt_torch.moe_backend owner review) before MoE models go back to
supported: (a) confirm the TT platform never reshuffles w13/w2 out of the plain
[E, out, in] layout at load time; (b) confirm this seam is preferred over
``runner_cls`` / ``routed_experts_cls`` injection.
"""

from __future__ import annotations

import functools
import types

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)


class _TTExpertView:
    """Presents a ``RoutedExperts`` as the fused-expert module moe_backend expects.

    moe_backend reads only these names (weights read live, never snapshotted).
    """

    # vLLM stores w13/w2 as [E, out, in]; moe_backend transposes before the bmm.
    is_transposed = False

    def __init__(self, routed_experts: RoutedExperts):
        self._re = routed_experts

    @property
    def num_experts(self) -> int:
        return self._re.global_num_experts

    @property
    def gate_up_proj(self) -> torch.Tensor:
        return self._re.w13_weight

    @property
    def down_proj(self) -> torch.Tensor:
        return self._re.w2_weight

    @property
    def gate_up_proj_bias(self):
        return getattr(self._re, "w13_bias", None)

    @property
    def down_proj_bias(self):
        return getattr(self._re, "w2_bias", None)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        activation = self._re.activation
        if activation == MoEActivation.SILU:
            return F.silu(gate) * up
        if activation in (MoEActivation.GELU, MoEActivation.GELU_TANH):
            return F.gelu(gate, approximate="tanh") * up
        raise NotImplementedError(f"TT FusedMoE: activation {activation} not supported")


def _tt_moe_forward_native(
    routed_experts: RoutedExperts,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Routing + TT expert dispatch, bound onto a ``RoutedExperts`` as forward_native."""
    from tt_torch import tt_dense_experts_forward, tt_experts_forward
    from tt_torch.moe_backend import _mesh_info

    orig_shape = hidden_states.shape
    h_flat = hidden_states.view(-1, orig_shape[-1])
    logits_flat = router_logits.view(-1, router_logits.shape[-1])

    if routed_experts.custom_routing_function is not None:
        topk_weights, topk_ids = routed_experts.custom_routing_function(
            h_flat, logits_flat, routed_experts.top_k, routed_experts.renormalize
        )
    else:
        scores = F.softmax(logits_flat.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, routed_experts.top_k, dim=-1)
        if routed_experts.renormalize:
            renorm = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            topk_weights = topk_weights / renorm

    topk_weights = topk_weights.to(h_flat.dtype)
    experts = _TTExpertView(routed_experts)

    # EP is only valid on a genuine 2D mesh; otherwise fall back to dense bmm.
    _, _, mesh_shape, _ = _mesh_info()
    is_2d_mesh = len(mesh_shape) == 2 and all(d > 1 for d in mesh_shape)
    if is_2d_mesh:
        out_flat = tt_experts_forward(experts, h_flat, topk_ids, topk_weights)
    else:
        out_flat = tt_dense_experts_forward(experts, h_flat, topk_ids, topk_weights)
    return out_flat.view(orig_shape)


class TTUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Forces the monolithic path onto the TT-scoped forward_native fallback."""

    @property
    def is_monolithic(self) -> bool:
        return True

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        if (
            self.unquantized_backend != UnquantizedMoeBackend.CPU
            and self.moe_kernel is None
            and hasattr(layer, "forward_native")
        ):
            return layer.forward_native(x, router_logits)
        return super().apply_monolithic(layer, x, router_logits, input_ids)


def _patch_moe_runner(runner: MoERunner) -> MoERunner:
    """In-place TT adaptation of a freshly-built MoERunner."""
    routed_experts = runner.routed_experts
    quant_method = routed_experts.quant_method

    if isinstance(quant_method, UnquantizedFusedMoEMethod) and not isinstance(
        quant_method, TTUnquantizedFusedMoEMethod
    ):
        tt_method = TTUnquantizedFusedMoEMethod(routed_experts.moe_config)
        for attr in (
            "moe_kernel",
            "moe_quant_config",
            "cpu_fused_moe",
            "unquantized_backend",
            "experts_cls",
        ):
            if hasattr(quant_method, attr):
                setattr(tt_method, attr, getattr(quant_method, attr))
        runner._replace_quant_method(tt_method)

    routed_experts.forward_native = types.MethodType(
        _tt_moe_forward_native, routed_experts
    )

    # Skip the opaque vllm.moe_forward custom op; call _forward_impl directly so
    # the compiler traces the real expert ops.
    def _forward_entry_direct(
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids,
        _layer_name,
        _hidden_dim_unpadded,
    ):
        return runner._forward_impl(
            hidden_states, router_logits, shared_experts_input, input_ids
        )

    runner._forward_entry = _forward_entry_direct
    return runner


def install_tt_fused_moe() -> None:
    """Patch the FusedMoE factory to TT-adapt every MoERunner it builds.

    Fired from register_oot_layers before models import; patches both the
    package symbol (what models import) and its defining module. Idempotent.
    """
    import vllm.model_executor.layers.fused_moe as _moe_pkg
    import vllm.model_executor.layers.fused_moe.layer as _moe_layer

    orig_factory = _moe_layer.FusedMoE
    if getattr(orig_factory, "_tt_patched", False):
        return

    @functools.wraps(orig_factory)
    def _tt_fused_moe(*args, **kwargs):
        return _patch_moe_runner(orig_factory(*args, **kwargs))

    _tt_fused_moe._tt_patched = True
    _tt_fused_moe._tt_orig = orig_factory
    _moe_layer.FusedMoE = _tt_fused_moe
    _moe_pkg.FusedMoE = _tt_fused_moe
