# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.overrides import TorchFunctionMode


def _unflatten_to_shape(
    input_shape: torch.Size, dim: int, sizes: tuple[int, ...] | list[int] | torch.Size
) -> tuple:
    dim = dim if dim >= 0 else len(input_shape) + dim
    if dim < 0 or dim >= len(input_shape):
        raise IndexError(f"Dimension out of range: dim={dim}, rank={len(input_shape)}")

    sizes = tuple(sizes)
    inferred_index = None
    known_product = 1
    normalized_sizes = []

    for index, size in enumerate(sizes):
        if size == -1:
            if inferred_index is not None:
                raise ValueError("Only one dimension can be inferred in unflatten")
            inferred_index = index
            normalized_sizes.append(size)
        else:
            normalized_sizes.append(size)
            known_product *= size

    if inferred_index is not None:
        normalized_sizes[inferred_index] = input_shape[dim] // known_product

    return (
        *input_shape[:dim],
        *normalized_sizes,
        *input_shape[dim + 1 :],
    )


def _embedding_tile_count_triggers_gather_bug(embedding_dim: int) -> bool:
    """Return True for ``embedding_dim`` values whose 32-tile layout trips
    TT PJRT's ``ttnn.embedding`` / gather lowering.

    Empirical D sweep on a single Blackhole chip (small vocab, seq_len=64)
    for both ``F.embedding`` and advanced indexing (``weight[ids]``) --
    both paths end up in the same gather kernel and share the regression:

      D      | tile_count | result
      ------ | ---------- | -------
      256    |      8     | PCC ~ 1.0
      1024   |     32     | PCC ~ 1.0
      2560   |     80     | PCC ~ 1.0
      4096   |    128     | PCC ~ 1.0
      8192   |    256     | PCC ~ 1.0
      10240  |    320     | PCC ~ 1.0
      10752  |    336     | PCC ~ 0.02  <-- Gemma4 per-layer embedding
      16384  |    512     | PCC ~ 1.0

    Power-of-two tile counts (8, 32, 128, 256, 512) all worked, and
    non-power-of-two counts up to 320 also worked. 336 (2^4 * 3 * 7) is
    the first non-power-of-two tile count we've seen mis-lower, so the
    heuristic flags "non-power-of-two tile count above the largest
    known-good one". Any D whose tile layout does not match that
    "unusual" profile stays on the default, fast embedding path.
    """
    if embedding_dim <= 0 or embedding_dim % 32 != 0:
        return False
    tile_count = embedding_dim // 32
    # Clean power-of-two tile counts have never been seen to misbehave.
    if (tile_count & (tile_count - 1)) == 0:
        return False
    # 320 was the largest non-power-of-two tile count observed to work
    # (D=10240). Only rewrite above that to keep the blast radius narrow.
    return tile_count > 320


# The embedding rewrite *must* go through a module-level monkey patch rather
# than the ``TorchFunctionOverride`` below. Dynamo optimises the
# ``has_torch_function_variadic(input, weight)`` fast-path inside
# ``F.embedding`` away at trace time and lowers straight to ``torch.embedding``
# (aten C op) without revisiting any active ``TorchFunctionMode``. That means
# a mode-based intercept only fires for the eager reference execution; the
# FX / stablehlo graph that actually runs on the TT device never sees the
# rewrite, and ``stablehlo.gather`` stays in the compiled module.
#
# Replacing ``torch.nn.functional.embedding`` and ``torch.nn.Embedding.forward``
# with Python wrappers that compute the lookup as ``one_hot @ weight`` makes
# the compile-time trace inline the matmul path directly, which is what
# finally shows up in the compiled graph.
_original_F_embedding = torch.nn.functional.embedding


def _tt_patched_F_embedding(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    if (
        isinstance(weight, torch.Tensor)
        and weight.ndim == 2
        and padding_idx is None
        and max_norm is None
        and _embedding_tile_count_triggers_gather_bug(weight.shape[-1])
    ):
        num_classes = int(weight.shape[0])
        one_hot = torch.nn.functional.one_hot(input, num_classes=num_classes).to(
            weight.dtype
        )
        return torch.einsum("...v,vd->...d", one_hot, weight)
    return _original_F_embedding(
        input,
        weight,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
    )


torch.nn.functional.embedding = _tt_patched_F_embedding


_original_nn_Embedding_forward = torch.nn.Embedding.forward


def _tt_patched_nn_Embedding_forward(self, input):
    # Short-circuits ``nn.Embedding.forward`` on its own so Dynamo sees the
    # one-hot+matmul path even when it inlines the Module's forward directly
    # without going through ``F.embedding``.
    if (
        self.padding_idx is None
        and self.max_norm is None
        and _embedding_tile_count_triggers_gather_bug(self.weight.shape[-1])
    ):
        num_classes = int(self.weight.shape[0])
        one_hot = torch.nn.functional.one_hot(input, num_classes=num_classes).to(
            self.weight.dtype
        )
        return torch.einsum("...v,vd->...d", one_hot, self.weight)
    return _original_nn_Embedding_forward(self, input)


torch.nn.Embedding.forward = _tt_patched_nn_Embedding_forward


class TorchFunctionOverride(TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        kwargs = kwargs or {}

        # NOTE: the embedding rewrite used to live here, but Dynamo bypasses
        # ``TorchFunctionMode`` for ``F.embedding`` (see the module-level
        # ``_tt_patched_F_embedding`` / ``_tt_patched_nn_Embedding_forward``
        # monkey patches above for the actual fix).

        if func.__name__ == "unflatten":
            tensor = args[0]
            dim = kwargs.get("dim", args[1])
            sizes = kwargs.get("sizes", args[2])
            # Bypass Tensor.unflatten -> super().unflatten(), which Dynamo
            # cannot trace in fullgraph mode under TorchFunctionMode.
            return torch.ops.aten.reshape.default(
                tensor, _unflatten_to_shape(tensor.shape, dim, sizes)
            )

        # ttnn.pow_scalar does not support negative exponents.
        # Rewrite pow(x, -0.5) → rsqrt(x), pow(x, -n) → 1/pow(x, n).
        if func is torch.Tensor.pow or func is torch.pow:
            base = args[0]
            exponent = args[1] if len(args) > 1 else kwargs.get("exponent", None)
            if isinstance(exponent, (int, float)) and exponent < 0:
                if exponent == -0.5:
                    return torch.rsqrt(base)
                return torch.reciprocal(func(base, -exponent))

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
        return func(*args, **kwargs)


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
