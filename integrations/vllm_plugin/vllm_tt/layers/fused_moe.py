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
        # Escape hatch for CPU, which stays on the old monolithic path.
        if self.unquantized_backend == UnquantizedMoeBackend.CPU:
            return True
        # If the modular kernel was not initialized, force monolithic mode and
        # let apply_monolithic route to the TT layer fallback.
        if self.moe_kernel is None:
            return True
        return self.moe_kernel.is_monolithic

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
            self.runner.quant_method = tt_method

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
            # HF Gemma-4 uses tanh-approximated GELU; vLLM's "gelu" maps here.
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
