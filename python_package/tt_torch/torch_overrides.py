# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.overrides import TorchFunctionMode


class TorchFunctionOverride(TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        if (
            func.__name__ == "matmul" or func.__name__ == "linear"
        ) and not torch.compiler.is_compiling():
            kwargs = kwargs or {}
            if func.__name__ == "linear":
                inp = args[0] if len(args) > 0 else kwargs.get("input")
                weight = args[1] if len(args) > 1 else kwargs.get("weight")
                bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
            else:
                inp = args[0] if len(args) > 0 else kwargs.get("input")
                weight = args[1] if len(args) > 1 else kwargs.get("other")
                bias = None
            if (
                inp is not None
                and weight is not None
                and (len(inp.shape) >= 4 or len(weight.shape) >= 4)
            ):
                if func.__name__ == "linear":
                    res = torch.einsum("...mk,...nk->...mn", inp, weight)
                else:
                    res = torch.einsum("...mk,...kn->...mn", inp, weight)
                if bias is not None:
                    res = res + bias
                return res
        return func(*args, **(kwargs or {}))


torch_function_override = TorchFunctionOverride()
torch_function_override.__enter__()


def _router_forward(self, hidden_states):
    """Monkey-patched GptOssTopKRouter.forward that returns full [T, E] sparse
    routing weights (matching 4.57.1 behavior), instead of compact [T, K]."""
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
    router_top_value = torch.nn.functional.softmax(
        router_top_value, dim=1, dtype=router_top_value.dtype
    )
    router_scores = torch.zeros_like(router_logits).scatter_(
        1, router_indices, router_top_value
    )
    return router_logits, router_scores, router_indices


def _experts_forward(self, hidden_states, router_indices=None, routing_weights=None):
    """Monkey-patched GptOssExperts.forward matching 4.57.1 behavior.

    CPU path uses per-expert loop (memory-efficient, serves as PCC golden reference).
    Device path uses dense bmm (static graph for torch.compile).

    Args:
        hidden_states: [T, H] or [B, S, H]
        router_indices: [T, K]
        routing_weights: [T, E] full sparse routing weights
    """
    batch_size = hidden_states.shape[0]
    num_experts = routing_weights.shape[1]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)

    if hidden_states.device.type == "cpu":
        next_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                router_indices, num_classes=num_experts + 1
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == num_experts:
                continue
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = (
                current_state @ self.gate_up_proj[expert_idx]
                + self.gate_up_proj_bias[expert_idx]
            )
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            out = ((up + 1) * glu) @ self.down_proj[expert_idx] + self.down_proj_bias[
                expert_idx
            ]
            weighted_output = out * routing_weights[token_idx, expert_idx, None]
            next_states.index_add_(
                0, token_idx, weighted_output.to(hidden_states.dtype)
            )
        next_states = next_states.view(batch_size, -1, self.hidden_size)
    else:
        hidden_states = hidden_states.repeat(num_experts, 1)
        hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
        gate_up = (
            torch.bmm(hidden_states, self.gate_up_proj)
            + self.gate_up_proj_bias[..., None, :]
        )
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        next_states = torch.bmm(((up + 1) * glu), self.down_proj)
        next_states = next_states + self.down_proj_bias[..., None, :]
        next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        next_states = (
            next_states
            * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[
                ..., None
            ]
        )
        next_states = next_states.sum(dim=0)
    return next_states


def _sparse_mlp_forward(self, hidden_states):
    """Monkey-patched GptOssMLP.forward matching 4.57.1 behavior."""
    _, router_scores, router_indices = self.router(hidden_states)
    routed_out = self.experts(
        hidden_states, router_indices=router_indices, routing_weights=router_scores
    )
    return routed_out, router_scores


# Monkey patch to restore 4.57.1 interfaces:
# - Router returns full [T, E] sparse routing weights
# - Experts has CPU per-expert loop + device dense bmm
# - Bypasses @use_experts_implementation decorator dispatch
try:
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssExperts,
        GptOssMLP,
        GptOssTopKRouter,
    )

    GptOssTopKRouter.forward = _router_forward
    GptOssExperts.forward = _experts_forward
    GptOssMLP.forward = _sparse_mlp_forward
except ImportError:
    pass


def _qwen3_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Monkey-patched Qwen3MoeExperts.forward.

    Replaces grouped_mm_experts_forward (uses torch._grouped_mm, not XLA-aware)
    with a static-shape dense bmm path on device and a per-expert loop on CPU.

    Weight layout (transformers convention):
        gate_up_proj: [E, 2*I, H]  (use .T per expert to project H -> 2*I)
        down_proj:    [E, H,   I]  (use .T per expert to project I -> H)

    Args:
        hidden_states: [T, H] pre-flattened by Qwen3MoeSparseMoeBlock
        top_k_index:   [T, K] expert indices
        top_k_weights: [T, K] routing weights (already normalized)
    """
    T, H = hidden_states.shape
    E = self.num_experts

    if hidden_states.device.type == "cpu":
        next_states = torch.zeros(T, H, dtype=hidden_states.dtype,
                                  device=hidden_states.device)
        for expert_idx in range(E):
            expert_mask = (top_k_index == expert_idx)    # [T, K]
            token_mask = expert_mask.any(dim=-1)          # [T]
            token_indices = token_mask.nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue
            current = hidden_states[token_indices]                           # [m, H]
            gate_up = current @ self.gate_up_proj[expert_idx].T             # [m, 2*I]
            gate, up = gate_up.chunk(2, dim=-1)
            activated = self.act_fn(gate) * up                              # [m, I]
            out = activated @ self.down_proj[expert_idx].T                  # [m, H]
            w = (top_k_weights[token_indices] * expert_mask[token_indices]).sum(-1, keepdim=True)
            next_states.index_add_(0, token_indices,
                                   (out * w).to(hidden_states.dtype))
        return next_states
    else:
        # Build full [T, E] routing weight matrix — static shape, XLA-friendly
        routing_full = torch.zeros(
            T, E, device=hidden_states.device, dtype=hidden_states.dtype
        ).scatter(1, top_k_index.long(), top_k_weights.to(hidden_states.dtype))

        # Tile hidden states: [T, H] -> [E, T, H]
        hidden_tiled = hidden_states.repeat(E, 1).view(E, T, H)

        # gate_up_proj [E, 2*I, H] -> transpose -> [E, H, 2*I]
        gate_up = torch.bmm(hidden_tiled,
                            self.gate_up_proj.transpose(-2, -1))   # [E, T, 2*I]
        gate, up = gate_up.chunk(2, dim=-1)
        activated = self.act_fn(gate) * up                         # [E, T, I]

        # down_proj [E, H, I] -> transpose -> [E, I, H]
        output = torch.bmm(activated,
                           self.down_proj.transpose(-2, -1))       # [E, T, H]

        # Weighted sum over experts
        output = output * routing_full.T.unsqueeze(-1)             # [E, T, H]
        return output.sum(dim=0).to(hidden_states.dtype)           # [T, H]


# Bypass @use_experts_implementation dispatch for Qwen3MoE — grouped_mm path
# is not XLA-aware and segfaults during partition_fx_graph_for_cpu_fallback.
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    Qwen3MoeExperts.forward = _qwen3_moe_experts_forward
except ImportError:
    pass
