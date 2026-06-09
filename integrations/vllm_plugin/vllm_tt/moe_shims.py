# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT MoE plug-in for vLLM.

``TTFusedMoE`` (OOT-registered for ``FusedMoE``) runs a dense-bmm expert
dispatch via ``tt_torch.moe_backend.tt_dense_experts_forward``. Routing
uses the model's own ``custom_routing_function`` when present, else
standard softmax / top_k / renormalize.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


@CustomOp.register_oot(name="FusedMoE")
class TTFusedMoE(FusedMoE):
    """OOT FusedMoE specialised for the TT compile pipeline.

    The upstream fused kernel path doesn't lower cleanly through the TT
    compiler, so we run a dense-bmm expert dispatch instead: route every
    token through every expert, then mask by routing weights. The actual
    FFN is delegated to ``tt_torch.moe_backend.tt_dense_experts_forward``.
    """

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
        # Multi-chip expert-parallel path when an EP mesh is present; the
        # expert weights must be sharded along the expert dim (see
        # partition_fused_moe). Fall back to dense bmm on single chip.
        total_devices, dispatch_devices, _, _ = _mesh_info()
        if total_devices > 1 and dispatch_devices > 1:
            out_flat = tt_experts_forward(self, h_flat, topk_ids, topk_weights)
        else:
            out_flat = tt_dense_experts_forward(self, h_flat, topk_ids, topk_weights)
        return out_flat.view(orig_shape)


def _move_stale_closure_tensors_to_device(model: torch.nn.Module) -> None:
    """Move CPU tensors captured by MoE ``custom_routing_function`` closures
    onto the model's device.

    Models that build ``custom_routing_function`` as a closure over a
    parameter (e.g. Gemma-4's ``per_expert_scale``) keep the original CPU
    tensor in the closure cell after ``model.to(device)``, so routing later
    mixes cpu/xla tensors and fails to trace. Rewrite any such cell to the
    device tensor — name- and model-agnostic."""
    # Infer the model's device from any parameter (all weights already moved).
    device = next((p.device for p in model.parameters()), None)
    if device is None or device.type == "cpu":
        return

    for module in model.modules():
        fn = getattr(module, "custom_routing_function", None)
        closure = getattr(fn, "__closure__", None)
        if not closure:
            continue
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:
                continue  # empty cell
            if isinstance(val, torch.Tensor) and val.device != device:
                cell.cell_contents = val.to(device)


def install_moe_shims(model: torch.nn.Module) -> None:
    """Apply TT MoE workarounds after the model is on device.

    The ``FusedMoE`` substitution itself happens via ``TTFusedMoE`` (OOT
    registered at import time). Here we only repair routing closures that
    captured[118;1:3u stale CPU tensors before ``model.to(device)``.
    """
    _move_stale_closure_tensors_to_device(model)
