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
            if len(args[0].shape) >= 4 or len(args[1].shape) >= 4:
                if func.__name__ == "linear":
                    # Linear function transposes args[1]
                    res = torch.einsum("...mk,...nk->...mn", args[0], args[1])
                else:
                    res = torch.einsum("...mk,...kn->...mn", args[0], args[1])
                if len(args) > 2 and args[2] is not None:
                    res = res + args[2]
                return res
        return func(*args, **(kwargs or {}))


torch_function_override = TorchFunctionOverride()
torch_function_override.__enter__()


def _dense_experts_forward(
    self, hidden_states, router_indices=None, routing_weights=None
):
    """Static dense experts forward that computes all experts for all tokens.

    Replaces the data-dependent loop in GptOssExperts.forward with a fully
    static dense computation so torch.compile can capture the entire graph.

    Handles both interfaces:
    - Old: routing_weights [T, E] full sparse
    - New (via @use_experts_implementation): routing_weights [T, K] compact
    """
    batch_size = hidden_states.shape[0]
    num_experts = self.num_experts
    hidden_states = hidden_states.reshape(-1, self.hidden_size)  # [T, H]
    num_tokens = hidden_states.shape[0]

    # Build full sparse routing weights [T, E] from compact [T, K] if needed
    if routing_weights.shape[1] != num_experts:
        full_weights = torch.zeros(
            num_tokens,
            num_experts,
            dtype=routing_weights.dtype,
            device=hidden_states.device,
        )
        full_weights.scatter_(1, router_indices.long(), routing_weights)
        routing_weights = full_weights

    # Dense gate+up: [E, T, H] @ [E, H, 2*inter] → [E, T, 2*inter]
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

    # Weighted sum: routing_weights [T, E] → [E, B, S, 1]
    next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
    next_states = (
        next_states
        * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    )
    next_states = next_states.sum(dim=0)
    return next_states


# Register as "dense" experts implementation so it can be selected via
# config._experts_implementation = "dense"
try:
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    ALL_EXPERTS_FUNCTIONS["dense"] = _dense_experts_forward
except ImportError:
    pass
