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
``@CustomOp.register_oot``; the import is fired from ``register_moe_oot_layer``
(the ``vllm.general_plugins`` entry point), mirroring the MLA backend.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE


@CustomOp.register_oot(name="FusedMoE")
class TTFusedMoE(FusedMoE):
    """OOT FusedMoE specialised for the TT compile pipeline (see module docstring)."""

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
        from tt_torch.moe_backend import moe_expert_parallel_devices

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
        # Expert-parallel tt-moe runs whenever the experts are sharded across
        # the mesh (any mesh with an EP axis > 1, including a 1D (1, N) mesh).
        # partition_fused_moe shards the expert weights under the *same*
        # decision, so the kernel always matches the on-device weight layout.
        # Only a true single-chip mesh falls back to the dense (replicated) bmm.
        ep_devices = moe_expert_parallel_devices()
        use_ep = ep_devices > 1 and int(self.global_num_experts) % ep_devices == 0
        if use_ep:
            out_flat = tt_experts_forward(self, h_flat, topk_ids, topk_weights)
        else:
            out_flat = tt_dense_experts_forward(self, h_flat, topk_ids, topk_weights)
        return out_flat.view(orig_shape)


@CustomOp.register_oot(name="SharedFusedMoE")
class TTSharedFusedMoE(SharedFusedMoE, TTFusedMoE):
    """OOT ``SharedFusedMoE`` (shared + routed experts) for the TT pipeline.

    DeepSeek-style models use ``SharedFusedMoE``. Unlike the plain ``FusedMoE``
    that ``TTFusedMoE`` already replaces, it defaults to the *overlapped* runner
    path: the routed experts run inside vLLM's ``moe_forward_shared`` custom op
    with the router gate called internally. That op is ``auto_functionalized``
    during the TT torch->StableHLO trace, and its internal gate ``F.linear``
    then trips PyTorch's "composite op functionalization fallback expects its
    inputs all not to be functional tensors" assert, killing compilation.

    We force ``use_overlapped = False`` so the layer matches the plain-FusedMoE
    contract ``TTFusedMoE`` already supports:

    * ``is_internal_router`` flips off, so ``DeepseekV2MoE.forward`` runs the
      gate itself and hands us real ``[tokens, num_experts]`` router logits
      (exactly what ``TTFusedMoE.forward_native`` expects);
    * the inherited non-overlapped ``SharedFusedMoE.forward`` then computes the
      shared experts eagerly as an ordinary sub-module and routes the experts
      through ``super().forward() -> TTFusedMoE.forward_native`` (the TT dense /
      expert-parallel kernels via the MRO), returning ``(shared, routed)``.

    No call ever reaches the ``moe_forward_shared`` custom op. The MRO
    ``TTSharedFusedMoE -> SharedFusedMoE -> TTFusedMoE -> FusedMoE`` provides
    ``forward`` from ``SharedFusedMoE`` and ``forward_native`` (plus the expert
    adapter properties) from ``TTFusedMoE``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_overlapped = False
