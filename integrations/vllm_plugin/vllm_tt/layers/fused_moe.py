# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT FusedMoE OOT layer for vLLM.

``TTFusedMoE`` (OOT-registered for ``FusedMoE``) replaces the upstream fused
expert kernel — which doesn't lower cleanly through the TT compiler — with a
TT-friendly expert dispatch delegated to ``tt_torch.moe_backend``:

* genuine 2D mesh (both axes > 1): expert-parallel ``tt_experts_forward``
  (experts are sharded across the mesh by ``partition_fused_moe``);
* otherwise (1D / degenerate / single chip): dense-bmm
  ``tt_dense_experts_forward`` (route every token through every expert, then
  mask by the routing weights).

Routing uses the model's own ``custom_routing_function`` when present, else
standard softmax / top_k / renormalize. Registered at import time via
``@CustomOp.register_oot``; the import is fired from ``register_oot_layer``
(the ``vllm.general_plugins`` entry point), mirroring the MLA backend.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)


class TTUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """TT-scoped override for unquantized FusedMoE method behavior.

    Keeps the fallback logic local to TT FusedMoE instances instead of
    monkey-patching vLLM globally.
    """

    @property
    def is_monolithic(self) -> bool:
        # TT uses a compile-time constant monolithic path so the runner can
        # stay on the TT-specific apply_monolithic fallback without tracing a
        # runtime branch on moe_kernel state.
        return True

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        if (
            self.unquantized_backend != UnquantizedMoeBackend.CPU
            and self.moe_kernel is None
            and hasattr(layer, "forward_native")
        ):
            # TTFusedMoE exposes forward_native(hidden_states, router_logits)
            # that does routing + experts in a TT-friendly way without
            # relying on vLLM's internal moe kernel object.
            return layer.forward_native(x, router_logits)
        return super().apply_monolithic(layer, x, router_logits, input_ids)


@CustomOp.register_oot(name="FusedMoE")
class TTFusedMoE(FusedMoE):
    """OOT FusedMoE specialised for the TT compile pipeline (see module docstring)."""

    def __init__(self, *args, **kwargs):
        # Captured BEFORE super().__init__(): FusedMoE.__init__ zeroes
        # self.routed_scaling_factor to 1.0 whenever the caller passes
        # apply_routed_scale_to_output=True (DeepSeek's own model code always
        # does -- deepseek_v2.py sets it to `not is_rocm_aiter_moe_enabled`,
        # i.e. True for TT), on the assumption that vLLM's own MoERunner will
        # apply the real scale to the combined output afterward
        # (MoERunner._maybe_apply_routed_scale_to_output). That call only
        # exists on MoERunner.forward()'s code path; the path TT's monolithic
        # override actually uses (_forward_impl -> _apply_quant_method ->
        # apply_monolithic) never reaches it. So for TT the real factor has
        # to be applied here instead, inside forward_native -- see there.
        self._tt_routed_scaling_factor = float(kwargs.get("routed_scaling_factor", 1.0))
        super().__init__(*args, **kwargs)

        # Use a TT-scoped quant method override rather than monkey-patching
        # UnquantizedFusedMoEMethod globally.
        if isinstance(self.quant_method, UnquantizedFusedMoEMethod) and not isinstance(
            self.quant_method, TTUnquantizedFusedMoEMethod
        ):
            tt_method = TTUnquantizedFusedMoEMethod(self.moe_config)

            # Preserve runtime state initialized by vLLM on the original method.
            for attr in (
                "moe_kernel",
                "moe_quant_config",
                "cpu_fused_moe",
                "unquantized_backend",
                "experts_cls",
            ):
                if hasattr(self.quant_method, attr):
                    setattr(tt_method, attr, getattr(self.quant_method, attr))

            self.quant_method = tt_method
            self.base_quant_method = tt_method
            self.runner._replace_quant_method(tt_method)

        # vLLM::MoERunner calling sequence:
        # forward
        # └── _forward_entry
        #     └── vllm.moe_forward or vllm.moe_forward_shared (custom op)
        #         └── _forward_impl
        #
        # Override the runner's custom-op entry point to dispatch directly to the
        # TT MoE implementation.
        def _forward_entry_direct(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            _layer_name,
            _hidden_dim_unpadded,
        ):
            return self.runner._forward_impl(
                self,
                hidden_states,
                router_logits,
                shared_experts_input,
                input_ids,
            )

        self.runner._forward_entry = _forward_entry_direct

    # vLLM FusedMoE → tt_torch.moe_backend expert-module interface adapter.
    @property
    def num_experts(self):
        return self.global_num_experts

    @property
    def gate_up_proj(self):
        return self.w13_weight

    @property
    def down_proj(self):
        return self.w2_weight

    @property
    def is_transposed(self):
        # vLLM stores w13 / w2 already in row-major [E, out, in] orientation,
        # matching tt_dense_experts_forward's "not transposed" expectation
        # (it will transpose(-1, -2) before the bmm).
        return False

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        if self.activation == MoEActivation.SILU:
            return F.silu(gate) * up
        if self.activation == MoEActivation.GELU:
            return F.gelu(gate, approximate="tanh") * up
        if self.activation == MoEActivation.GELU_TANH:
            return F.gelu(gate, approximate="tanh") * up
        raise NotImplementedError(
            f"TTFusedMoE: activation {self.activation} not supported"
        )

    def forward_native(self, hidden_states, router_logits):
        # Lazy import keeps tt_torch.moe_backend out of the import-time cycle.
        from tt_torch import tt_dense_experts_forward, tt_experts_forward
        from tt_torch.moe_backend import _mesh_info

        orig_shape = hidden_states.shape
        h_flat = hidden_states.view(-1, orig_shape[-1])
        # Routing operates on [T, E]; flatten any leading dims of the logits.
        logits_flat = router_logits.view(-1, router_logits.shape[-1])

        if self.custom_routing_function is not None:
            # Model supplied its own routing (e.g. Gemma-4 folds
            # per_expert_scale into the top-k weights here).
            topk_weights, topk_ids = self.custom_routing_function(
                h_flat, logits_flat, self.top_k, self.renormalize
            )
        elif self.use_grouped_topk:
            # DeepSeek-V2/V3-style routing: sigmoid (not softmax) scores,
            # expert-group-limited top-k, and (for topk_method="noaux_tc")
            # a learned bias added for expert SELECTION only -- routing
            # WEIGHTS still come from the unbiased scores. Mirrors
            # vllm.model_executor.layers.fused_moe.router.grouped_topk_router
            # .grouped_topk exactly. Not called directly: that function
            # carries its own @torch.compile targeting
            # current_platform.simple_compile_backend (not the TT backend),
            # and self.router (vLLM's own correct implementation of this
            # same math) is never invoked for the monolithic path TT always
            # takes -- forward_native has to do it itself.
            #
            # Before this branch existed, forward_native always fell through
            # to the plain-softmax-top-k `else` below regardless of
            # use_grouped_topk, since DeepSeek never sets
            # custom_routing_function (grouped-topk is wired through
            # use_grouped_topk/self.router instead, a separate vLLM
            # mechanism forward_native didn't participate in). That silently
            # selected experts by an entirely different, untrained-for
            # scoring rule at every MoE layer -- not a precision difference.
            if self.scoring_func == "sigmoid":
                scores = logits_flat.float().sigmoid()
            elif self.scoring_func == "softmax":
                scores = F.softmax(logits_flat.float(), dim=-1)
            else:
                raise NotImplementedError(
                    f"TTFusedMoE: scoring_func {self.scoring_func!r} not supported"
                )

            num_tokens = scores.shape[0]
            bias = self.e_score_correction_bias
            if bias is not None:
                # Selection uses biased scores; routing weights use the
                # original (unbiased) ones.
                original_scores = scores
                scores = scores + bias.unsqueeze(0)
                group_scores = (
                    scores.view(num_tokens, self.num_expert_group, -1)
                    .topk(2, dim=-1)[0]
                    .sum(dim=-1)
                )
            else:
                group_scores = (
                    scores.view(num_tokens, self.num_expert_group, -1)
                    .max(dim=-1)
                    .values
                )

            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    num_tokens,
                    self.num_expert_group,
                    scores.shape[-1] // self.num_expert_group,
                )
                .reshape(num_tokens, -1)
            )
            tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

            if bias is not None:
                topk_ids = torch.topk(tmp_scores, k=self.top_k, dim=-1)[1]
                topk_weights = original_scores.gather(1, topk_ids)
            else:
                topk_weights, topk_ids = torch.topk(tmp_scores, k=self.top_k, dim=-1)

            if self.renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            # self.routed_scaling_factor is neutered to 1.0 here (see
            # __init__) -- use the real value captured before that happened.
            if self._tt_routed_scaling_factor != 1.0:
                topk_weights = topk_weights * self._tt_routed_scaling_factor
        else:
            scores = F.softmax(logits_flat.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(scores, self.top_k, dim=-1)
            if self.renormalize:
                renorm = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                topk_weights = topk_weights / renorm

        topk_weights = topk_weights.to(h_flat.dtype)
        # Expert-parallel tt-moe is only valid on a genuine 2D mesh (both axes
        # > 1), where partition_fused_moe shards the experts across the mesh.
        # On 1D / degenerate (1, N) / single-chip meshes, use dense bmm.
        _, _, mesh_shape, _ = _mesh_info()
        is_2d_mesh = len(mesh_shape) == 2 and all(d > 1 for d in mesh_shape)
        if is_2d_mesh:
            out_flat = tt_experts_forward(self, h_flat, topk_ids, topk_weights)
        else:
            out_flat = tt_dense_experts_forward(self, h_flat, topk_ids, topk_weights)
        return out_flat.view(orig_shape)
