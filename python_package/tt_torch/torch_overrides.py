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


def _qwen3moe_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Device-friendly Qwen3MoeExperts.forward that avoids nonzero() and Python loops.

    The original uses nonzero() + a for-loop which produces data-dependent output
    shapes and segfaults during partition_fx_graph_for_cpu_fallback.

    CPU path uses per-expert loop (serves as PCC golden reference).
    Device path uses dense bmm (static graph for torch.compile).

    Args:
        hidden_states: [B, S, H] or [T, H]
        top_k_index:   [B, S, K] or [T, K]  (top-k expert indices, long)
        top_k_weights: [B, S, K] or [T, K]  (top-k routing weights)
    """
    orig_shape = hidden_states.shape
    T = 1
    for d in orig_shape[:-1]:
        T *= d
    H = orig_shape[-1]
    hidden_flat = hidden_states.reshape(T, H)
    index_flat = top_k_index.reshape(T, -1)    # [T, K]
    weights_flat = top_k_weights.reshape(T, -1)  # [T, K]
    E = self.num_experts

    if hidden_flat.device.type == "cpu":
        final = torch.zeros_like(hidden_flat)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(index_flat, num_classes=E)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for idx in expert_hit:
            e = idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[e])
            cur = hidden_flat[token_idx]
            gate, up = torch.nn.functional.linear(
                cur, self.gate_up_proj[e]
            ).chunk(2, dim=-1)
            cur_hidden = self.act_fn(gate) * up
            cur_hidden = torch.nn.functional.linear(cur_hidden, self.down_proj[e])
            cur_hidden = cur_hidden * weights_flat[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur_hidden.to(final.dtype))
    else:
        # Build sparse routing weights [T, E]
        routing = torch.zeros(T, E, dtype=hidden_flat.dtype, device=hidden_flat.device)
        routing.scatter_(1, index_flat, weights_flat)

        # [E, T, H] via expand (no data copy)
        hs = hidden_flat.unsqueeze(0).expand(E, -1, -1)

        # gate_up_proj: [E, 2*I, H] → transpose → [E, H, 2*I]
        gate_up = torch.bmm(hs, self.gate_up_proj.transpose(1, 2))  # [E, T, 2*I]
        gate, up = gate_up.chunk(2, dim=-1)  # each [E, T, I]
        cur_hidden = self.act_fn(gate) * up   # [E, T, I]

        # down_proj: [E, H, I] → transpose → [E, I, H]
        expert_out = torch.bmm(cur_hidden, self.down_proj.transpose(1, 2))  # [E, T, H]

        # Weight by routing: routing.T = [E, T], unsqueeze(-1) → [E, T, 1]
        expert_out = expert_out * routing.T.unsqueeze(-1)
        final = expert_out.sum(dim=0)  # [T, H]

    return final.reshape(orig_shape)


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

# Patch Qwen3MoeExperts to avoid nonzero()/for-loop segfault in
# partition_fx_graph_for_cpu_fallback (data-dependent output shapes).
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    Qwen3MoeExperts.forward = _qwen3moe_experts_forward
except ImportError:
    pass
