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


def _router_forward(self, hidden_states):
    """Monkey-patched GptOssTopKRouter.forward that returns full [T, E] sparse
    routing weights (matching 4.57.1 behavior), instead of compact [T, K]."""
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
    router_top_value = torch.nn.functional.softmax(
        router_top_value, dim=1, dtype=router_top_value.dtype
    )
    num_experts = router_logits.shape[-1]
    expert_mask = torch.nn.functional.one_hot(
        router_indices, num_classes=num_experts
    ).to(router_top_value.dtype)
    router_scores = (expert_mask * router_top_value.unsqueeze(-1)).sum(dim=1)
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


def _deepseek_v2_rotary_forward(self, x, position_ids):
    """Monkey-patched ``DeepseekV2RotaryEmbedding.forward`` that emits a
    real-packed ``[cos, sin]`` tensor instead of a complex ``freqs_cis``.

    Upstream produces a complex tensor::

        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        freqs_cis = freqs_cis * self.attention_scaling

    and ``apply_rotary_emb`` then runs ``view_as_complex`` / complex
    multiplication / ``view_as_real`` on it. The TT PJRT plugin does not
    support these complex-tensor codepaths (the original failure was
    "Complex tensor with num_dims == 0 is not supported." from
    ``mul(polar, 1.0)``; downstream ``view_as_complex`` / complex mul also
    crash the device backend).

    This patch (paired with ``_deepseek_v2_apply_rotary_emb``) replaces the
    complex pipeline with mathematically equivalent real arithmetic:
    ``freqs_cis`` becomes a real ``[..., 2]`` tensor whose last dim is
    ``(scaling * cos(theta), scaling * sin(theta))``.
    """
    from transformers.utils.generic import maybe_autocast

    inv_freq_expanded = (
        self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    )
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = (
        x.device.type
        if isinstance(x.device.type, str) and x.device.type != "mps"
        else "cpu"
    )
    with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
        scaling = float(self.attention_scaling)
        cos = torch.cos(freqs) * scaling
        sin = torch.sin(freqs) * scaling
        # Pack as real (..., 2) — last dim is [cos, sin]. Drop-in for the
        # complex freqs_cis since the only consumer (apply_rotary_emb) is
        # patched below.
        freqs_cis = torch.stack((cos, sin), dim=-1)

    return freqs_cis


def _deepseek_v2_apply_rotary_emb(xq, xk, freqs_cis):
    """Monkey-patched ``apply_rotary_emb`` that performs the rotation using
    only real arithmetic.

    Expects ``freqs_cis`` to be a real ``(B, S, D/2, 2)`` tensor whose last
    dim is ``(cos, sin)`` (produced by the patched
    ``DeepseekV2RotaryEmbedding.forward`` above), not a complex tensor.

    For each adjacent ``(a, b)`` pair in the input head_dim:
        out_even = a * cos - b * sin
        out_odd  = a * sin + b * cos
    which is exactly ``(a + b*i) * (cos + i*sin)`` written in real form.
    """

    def _rotate(x):
        # x: (B, H, S, D) -> pairs: (B, H, S, D/2, 2)
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        a = pairs[..., 0]
        b = pairs[..., 1]
        cos = fc[..., 0]
        sin = fc[..., 1]
        out_even = a * cos - b * sin
        out_odd = a * sin + b * cos
        out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
        return out.type_as(x)

    # freqs_cis: (B, S, D/2, 2) -> broadcast over head dim H -> (B, 1, S, D/2, 2)
    fc = freqs_cis.unsqueeze(1).to(xq.device)
    return _rotate(xq), _rotate(xk)


# Monkey patch DeepseekV2RotaryEmbedding.forward + apply_rotary_emb to avoid
# every complex-tensor op (torch.polar, view_as_complex, complex multiply,
# view_as_real), which the TT PJRT plugin does not fully support.
# Re-apply @torch.no_grad and @dynamic_rope_update on the rotary forward so
# dynamic / longrope rope types still recompute inv_freq correctly.
try:
    from transformers.modeling_rope_utils import dynamic_rope_update
    from transformers.models.deepseek_v2 import modeling_deepseek_v2

    modeling_deepseek_v2.DeepseekV2RotaryEmbedding.forward = torch.no_grad()(
        dynamic_rope_update(_deepseek_v2_rotary_forward)
    )
    modeling_deepseek_v2.apply_rotary_emb = _deepseek_v2_apply_rotary_emb
except ImportError:
    pass
