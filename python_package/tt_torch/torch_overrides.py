# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.overrides import TorchFunctionMode


def _clamp_oob_slice(tensor, idx):
    """Clamp slice start/stop values that lie outside [-size, size].

    CPU silently clamps such endpoints; torch-xla's lazy backend raises
    "Value out of range". Upstream diffusers (AutoencoderKLWan) and
    similar third-party models rely on the CPU behavior — e.g.
    ``x[:, :, -2:, :, :]`` on a size-1 temporal dim. Returns the original
    ``idx`` unchanged if no clamping was needed, otherwise a new index.

    Only applies to slices whose step is None or > 0. For step < 0, CPython
    uses different bounds (``max(-1, start + size)`` instead of
    ``max(0, start + size)``) that do not round-trip through torch-xla's
    canonicalization — silently producing wrong results is worse than the
    raised error, so negative-step slices are left untouched.
    """

    def clamp(s, size):
        # Rewrite a single slice so its endpoints lie in [-size, size], the
        # range torch-xla accepts. Returns the same ``s`` object if nothing
        # needed to change, so the caller can use identity comparison.
        start, stop, step = s.start, s.stop, s.step

        # Negative step has different CPython clamping rules (see docstring).
        # Bail out — leave the caller to surface torch-xla's error if any.
        if isinstance(step, int) and step < 0:
            return s

        changed = False

        # Clamp ``start`` into [-size, size]. The ``isinstance(..., int)``
        # guard skips ``None`` (default start) and any symbolic/dynamic ints.
        if isinstance(start, int):
            if start < -size:
                start, changed = -size, True
            elif start > size:
                start, changed = size, True

        # Same treatment for ``stop``.
        if isinstance(stop, int):
            if stop < -size:
                stop, changed = -size, True
            elif stop > size:
                stop, changed = size, True

        return slice(start, stop, s.step) if changed else s

    def dims_consumed(s):
        # How many of the *input* tensor's dims this indexer eats. Boolean
        # masks broadcast across ``ndim`` dims and collapse them into one
        # output dim; every other indexer (int, int tensor, slice) covers
        # exactly one input dim. Used both to resolve Ellipsis span and to
        # advance the per-dim cursor while walking the index tuple.
        if isinstance(s, torch.Tensor) and s.dtype == torch.bool:
            return s.ndim
        return 1

    # Fast path: ``x[some_slice]`` — a single slice applies to dim 0.
    if isinstance(idx, slice):
        return clamp(idx, tensor.shape[0]) if tensor.ndim else idx

    # Any other scalar indexer (int, tensor, bool mask) — nothing to clamp.
    if not isinstance(idx, tuple):
        return idx

    # Resolve ``Ellipsis``: count the input dims that the tuple explicitly
    # addresses, then anything left over is what ``...`` stands for. Only
    # one Ellipsis is allowed by the subscript grammar, so this is well
    # defined. ``None`` is skipped because it *inserts* an output dim
    # without consuming an input dim.
    explicit = sum(
        dims_consumed(s) for s in idx if s is not Ellipsis and s is not None
    )
    ellipsis_span = tensor.ndim - explicit

    # Walk the tuple, rewriting slices in place and advancing a cursor over
    # input dims so ``tensor.shape[dim]`` lines up with the current entry.
    out = []
    changed = False
    dim = 0
    for s in idx:
        if s is Ellipsis:
            # Ellipsis itself is kept as-is; just skip past the dims it covers.
            out.append(s)
            dim += ellipsis_span
        elif s is None:
            # ``None`` / newaxis inserts an output dim but doesn't consume
            # one from the input tensor, so the cursor stays put.
            out.append(s)
        elif isinstance(s, slice) and dim < tensor.ndim:
            # The one place we may rewrite: a slice positioned on a real
            # input dim. ``clamped is not s`` flips ``changed`` to True only
            # when clamp actually produced a new slice object.
            clamped = clamp(s, tensor.shape[dim])
            out.append(clamped)
            changed = changed or clamped is not s
            dim += 1
        else:
            # Int, int tensor, or bool mask — leave untouched and advance
            # the cursor by the dims it consumes (1 for scalars/int tensors,
            # ``ndim`` for bool masks).
            out.append(s)
            dim += dims_consumed(s)

    # Preserve the original ``idx`` object when nothing changed so the
    # caller's identity check can short-circuit the re-dispatch.
    return tuple(out) if changed else idx


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
        # Intercept ``Tensor.__getitem__`` to make torch-xla match CPU's
        # silent-clamp semantics for out-of-range slice endpoints. The
        # ``isinstance(args[0], torch.Tensor)`` guard avoids touching any
        # non-tensor ``__getitem__`` that happens to share the method name.
        if (
            func.__name__ == "__getitem__"
            and len(args) >= 2
            and isinstance(args[0], torch.Tensor)
        ):
            new_idx = _clamp_oob_slice(args[0], args[1])
            # Identity check: the helper returns the original ``args[1]``
            # object when nothing needed clamping, so the common case pays
            # only one walk and no re-dispatch. When it did clamp, re-issue
            # the call with the rewritten index — and since the index is now
            # in range, the recursive pass walks, finds nothing to change,
            # and falls through to the default dispatch below.
            if new_idx is not args[1]:
                return func(args[0], new_idx, *args[2:], **(kwargs or {}))
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
