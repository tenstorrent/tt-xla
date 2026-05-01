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
    """Device-friendly Qwen3MoeExperts.forward.

    CPU path: original per-expert loop (memory-efficient, golden reference).
    Device path: dense batched matmul over all experts (static graph, no
    data-dependent shapes — avoids nonzero/for-loop that segfaults in
    partition_fx_graph_for_cpu_fallback).

    Weight layout: gate_up_proj [E, 2*I, H], down_proj [E, H, I]
    (F.linear convention: output_features x input_features).
    Routing: top_k_index [T, K], top_k_weights [T, K] (compact top-k).
    """
    import torch.nn.functional as F

    T, H = hidden_states.shape
    num_experts = self.num_experts

    if hidden_states.device.type == "cpu":
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(
                2, dim=-1
            )
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(
                current_hidden_states, self.down_proj[expert_idx]
            )
            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states
    else:
        # Expand all tokens through all experts via batched matmul.
        # gate_up_proj: [E, 2*I, H] → transpose to [E, H, 2*I] for bmm
        # down_proj:    [E, H, I]   → transpose to [E, I, H]  for bmm
        hs = hidden_states.unsqueeze(0).expand(num_experts, -1, -1).contiguous()
        gate_up = torch.bmm(hs, self.gate_up_proj.transpose(-1, -2))  # [E, T, 2*I]
        gate, up = gate_up.chunk(2, dim=-1)  # [E, T, I] each
        expert_out = self.act_fn(gate) * up  # [E, T, I]
        expert_out = torch.bmm(expert_out, self.down_proj.transpose(-1, -2))  # [E, T, H]
        # Aggregate: gather expert outputs by token then weight-sum over K.
        expert_out = expert_out.permute(1, 0, 2)  # [T, E, H]
        K = top_k_index.shape[-1]
        gathered = expert_out.gather(
            1, top_k_index.unsqueeze(-1).expand(-1, -1, H)
        )  # [T, K, H]
        return (gathered * top_k_weights.unsqueeze(-1)).sum(dim=1)  # [T, H]


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

# Monkey-patch Qwen3MoeExperts to replace the per-expert nonzero/for-loop
# with a device-friendly dense batched matmul that the TT compiler can trace.
try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    Qwen3MoeExperts.forward = _qwen3moe_experts_forward
except ImportError:
    pass
