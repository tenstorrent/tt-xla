# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT FusedMoE integration for vLLM 0.25.1.

Replaces the upstream fused-expert kernel (which doesn't lower cleanly through
the TT compiler) with expert dispatch delegated to ``tt_torch.moe_backend``:
expert-parallel ``tt_experts_forward`` on a genuine 2D mesh, else dense-bmm
``tt_dense_experts_forward``.

0.25.1 turned ``FusedMoE`` into a factory function returning a ``MoERunner``
(the weights + routing state live on a ``RoutedExperts`` submodule) and dropped
the subclassable ``FusedMoE`` / ``op_registry_oot`` hook the old ``TTFusedMoE``
relied on. Instead it exposes ``runner_cls`` / ``routed_experts_cls`` factory
params for out-of-tree backends. We use those seams:

- ``TTRoutedExperts`` (``routed_experts_cls``) owns the TT expert dispatch as a
  real ``forward_native`` method and installs the TT quant method via the
  supported ``_get_quant_method`` override -- no post-hoc attribute rebinding.
- ``TTMoERunner`` (``runner_cls``) bypasses the opaque ``vllm.moe_forward``
  custom op (untraceable by the TT compiler) by selecting the direct
  ``_forward_impl`` path, exactly as vLLM already does for CPU/TPU.
- ``TTUnquantizedFusedMoEMethod`` pins the ``OOT`` backend so vLLM never
  reshuffles/repacks/pads w13/w2 out of the plain ``[E, out, in]`` layout
  ``tt_torch.moe_backend`` expects (see ``_assert_plain_expert_layout``).

``install_tt_fused_moe`` wraps the factory only to inject the two ``*_cls``
kwargs (models call ``FusedMoE(...)`` without them); all TT behaviour lives in
the subclasses above.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _moe_forward,
    _moe_forward_shared,
)
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


def _assert_plain_expert_layout(re: RoutedExperts) -> None:
    """Guarantee w13/w2 are still the plain ``[E, out, in]`` params moe_backend reads.

    vLLM's ``process_weights_after_loading`` reshuffles (AITER), repacks (CPU),
    or pads (ROCm) the expert weights for every unquantized backend EXCEPT
    ``OOT``/``TPU``, which early-return untouched. ``TTUnquantizedFusedMoEMethod``
    pins ``OOT`` so the plain layout survives; this asserts that invariant so a
    future backend-selection change fails loudly instead of silently feeding
    moe_backend a shuffled/padded tensor and computing wrong results.

    Checks are on per-expert (trailing) dims only, so they hold under both dense
    and expert-parallel (sharded ``num_local_experts``) placement.
    """
    w13, w2 = re.w13_weight, re.w2_weight
    if w13.dim() != 3 or w2.dim() != 3:
        raise AssertionError(
            f"TT FusedMoE expects 3D [E, out, in] expert weights, got "
            f"w13={tuple(w13.shape)} w2={tuple(w2.shape)}"
        )
    if getattr(w13, "is_shuffled", False) or getattr(w2, "is_shuffled", False):
        raise AssertionError(
            "TT FusedMoE: expert weights are marked is_shuffled; the OOT backend "
            "should never reshuffle. Backend selection likely regressed."
        )
    h = re.hidden_size
    i = re.intermediate_size_per_partition
    gate_up_dim = (2 if re.moe_config.is_act_and_mul else 1) * i
    expected_w13 = (w13.shape[0], gate_up_dim, h)
    expected_w2 = (w2.shape[0], h, i)
    if tuple(w13.shape) != expected_w13 or tuple(w2.shape) != expected_w2:
        raise AssertionError(
            "TT FusedMoE: expert-weight layout changed from plain [E, out, in]. "
            f"Expected w13={expected_w13} w2={expected_w2}, got "
            f"w13={tuple(w13.shape)} w2={tuple(w2.shape)}. Weights were likely "
            "padded/repacked by a non-OOT backend."
        )


class TTUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Unquantized MoE method that dispatches experts through ``tt_torch``.

    Forces the monolithic path onto the RoutedExperts ``forward_native`` fallback
    and pins the ``OOT`` backend so ``process_weights_after_loading`` never
    reshuffles the [E, out, in] expert weights (TODO 1).
    """

    def __init__(self, moe):
        # Skip UnquantizedFusedMoEMethod.__init__'s call to
        # select_unquantized_moe_backend: that picks TRITON/AITER/etc. based on
        # live platform detection and would reshuffle/pad w13/w2. Pin OOT
        # explicitly -- process_weights_after_loading early-returns for OOT, so
        # the plain [E, out, in] layout survives regardless of when/where the
        # method is constructed. moe_kernel stays None (set by base __init__),
        # which routes apply_monolithic to forward_native below.
        FusedMoEMethodBase.__init__(self, moe)
        self.unquantized_backend = UnquantizedMoeBackend.OOT
        self.experts_cls = None

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


class TTRoutedExperts(RoutedExperts):
    """RoutedExperts whose experts run through ``tt_torch.moe_backend``.

    Injected via the ``routed_experts_cls`` factory param. Overrides
    ``_get_quant_method`` to install the TT quant method (before create_weights,
    so the plain-layout weights are created and never reshuffled) and provides
    the TT expert dispatch as a real ``forward_native``.
    """

    def _get_quant_method(self, prefix, quant_config, moe_config):
        quant_method = super()._get_quant_method(prefix, quant_config, moe_config)
        # Only the unquantized path lowers through moe_backend; leave quantized
        # methods (quant_config != None) untouched.
        if type(quant_method) is UnquantizedFusedMoEMethod:
            return TTUnquantizedFusedMoEMethod(moe_config)
        return quant_method

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Routing + TT expert dispatch (the monolithic fallback for TT)."""
        from tt_torch import tt_dense_experts_forward, tt_experts_forward
        from tt_torch.moe_backend import _mesh_info

        _assert_plain_expert_layout(self)

        orig_shape = hidden_states.shape
        h_flat = hidden_states.view(-1, orig_shape[-1])
        logits_flat = router_logits.view(-1, router_logits.shape[-1])

        if self.custom_routing_function is not None:
            topk_weights, topk_ids = self.custom_routing_function(
                h_flat, logits_flat, self.top_k, self.renormalize
            )
        else:
            # The plain softmax top-k below does not implement the extra router
            # features vLLM 0.25.1 exposes on RoutedExperts (real multi-group
            # top-k, non-softmax scoring, e_score correction bias,
            # router-weight-on-input). A single-group config (num_expert_group<=1
            # and topk_group<=1) is exempted: vLLM's grouped_topk degenerates to
            # plain top-k over all experts in that case (one group spanning every
            # expert, always selected), which is exactly what DeepSeek-V2-Lite
            # uses (n_group=1, topk_group=1). Models with real multi-group
            # routing (e.g. DeepSeek-V3, Kimi, GLM) must run via the HF
            # moe_backend; fail loudly rather than silently mis-routing. See #5610.
            uses_real_grouped_topk = self.use_grouped_topk and (
                (self.num_expert_group or 1) > 1 or (self.topk_group or 1) > 1
            )
            if (
                uses_real_grouped_topk
                or self.scoring_func != "softmax"
                or self.e_score_correction_bias is not None
                or self.apply_router_weight_on_input
            ):
                raise NotImplementedError(
                    "TT vLLM FusedMoE forward_native only supports plain softmax "
                    "top-k. (use_grouped_topk="
                    f"{self.use_grouped_topk}, num_expert_group={self.num_expert_group}, "
                    f"topk_group={self.topk_group}, scoring_func={self.scoring_func!r}, "
                    f"e_score_correction_bias={self.e_score_correction_bias is not None}, "
                    f"apply_router_weight_on_input={self.apply_router_weight_on_input}). "
                    "For models using these router features, please run with the HF "
                    "moe_backend (experts_implementation)."
                )
            scores = F.softmax(logits_flat.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
            if self.renormalize:
                renorm = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                topk_weights = topk_weights / renorm

        topk_weights = topk_weights.to(h_flat.dtype)
        experts = _TTExpertView(self)

        # EP is only valid on a genuine 2D mesh; otherwise fall back to dense bmm.
        _, _, mesh_shape, _ = _mesh_info()
        is_2d_mesh = len(mesh_shape) == 2 and all(d > 1 for d in mesh_shape)
        if is_2d_mesh:
            out_flat = tt_experts_forward(experts, h_flat, topk_ids, topk_weights)
        else:
            out_flat = tt_dense_experts_forward(experts, h_flat, topk_ids, topk_weights)
        return out_flat.view(orig_shape)


class TTMoERunner(MoERunner):
    """MoERunner that runs the MoE forward without the opaque custom op.

    Injected via the ``runner_cls`` factory param. The default ``_select_forward``
    returns ``torch.ops.vllm.moe_forward`` (an opaque custom op the TT compiler
    can't trace into). Selecting the module-level ``_moe_forward`` /
    ``_moe_forward_shared`` instead calls ``_forward_impl`` directly -- the same
    path vLLM uses for CPU/TPU -- so the compiler sees the real expert ops.
    """

    def _select_forward(self):
        return _moe_forward if self._shared_experts is None else _moe_forward_shared


def install_tt_fused_moe() -> None:
    """Patch the FusedMoE factory to inject the TT runner/experts classes.

    Fired from register_oot_layers before models import. Models call
    ``FusedMoE(...)`` without ``runner_cls`` / ``routed_experts_cls``, so this
    thin wrapper supplies them; all TT behaviour lives in the injected
    subclasses. Patches both the package symbol (what models import) and its
    defining module. Idempotent.
    """
    import vllm.model_executor.layers.fused_moe as _moe_pkg
    import vllm.model_executor.layers.fused_moe.layer as _moe_layer

    orig_factory = _moe_layer.FusedMoE
    if getattr(orig_factory, "_tt_patched", False):
        return

    @functools.wraps(orig_factory)
    def _tt_fused_moe(*args, **kwargs):
        kwargs.setdefault("routed_experts_cls", TTRoutedExperts)
        kwargs.setdefault("runner_cls", TTMoERunner)
        return orig_factory(*args, **kwargs)

    _tt_fused_moe._tt_patched = True
    _tt_fused_moe._tt_orig = orig_factory
    _moe_layer.FusedMoE = _tt_fused_moe
    _moe_pkg.FusedMoE = _tt_fused_moe
