# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import threading

import torch
from torch.overrides import TorchFunctionMode

_masked_scatter_inspect_state = threading.local()


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

        # Eager-time mask-scatter inspection. The custom decomposition in
        # tt_torch/backend/decompositions.py runs under AOT autograd with
        # FakeTensors, so it can verify row-constancy by metadata only.
        # Here we sit on the eager path (real CPU/CUDA tensors) and dump
        # the actual row-level mask once per call, so the row-constant
        # claim used by the OPT path can be confirmed against real data.
        # masked_scatter dispatches re-enter __torch_function__ via the
        # function/method/aten chain; the thread-local flag ensures we
        # log exactly once per outer call.
        _already_inspecting = getattr(
            _masked_scatter_inspect_state, "active", False
        )
        if (
            func.__name__ == "masked_scatter"
            and not torch.compiler.is_compiling()
            and not _already_inspecting
            and len(args) >= 2
        ):
            data, mask = args[0], args[1]
            source = args[2] if len(args) > 2 else (kwargs or {}).get("source")
            _masked_scatter_inspect_state.active = True
            try:
                if (
                    isinstance(mask, torch.Tensor)
                    and mask.device.type in ("cpu", "cuda")
                    and mask.dtype == torch.bool
                    and mask.ndim >= 2
                ):
                    H = data.shape[-1] if data.ndim >= 2 else 0
                    n_true = (
                        source.numel() // H
                        if (H > 0 and isinstance(source, torch.Tensor))
                        else 0
                    )
                    print(
                        f"[masked_scatter eager] data={tuple(data.shape)} dtype={data.dtype} | "
                        f"mask={tuple(mask.shape)} dtype={mask.dtype} stride={tuple(mask.stride())} | "
                        f"source={tuple(source.shape) if isinstance(source, torch.Tensor) else None} | "
                        f"H={H} n_true={n_true}",
                        flush=True,
                    )
                    with torch.no_grad():
                        m = mask.detach().cpu()
                        row_const = m.all(dim=-1) == m.any(dim=-1)
                        bad = int((~row_const).sum().item())
                        total = int(row_const.numel())
                        print(
                            f"[masked_scatter eager] row-constant check: rows_uniform={total - bad}/{total}, inconsistent_rows={bad}",
                            flush=True,
                        )
                        # Print each row separately so each row is on its
                        # own line. For uniform rows we summarise (all
                        # True / all False) since the row is huge; for
                        # non-uniform rows we dump the full row to expose
                        # the bad pattern.
                        flat = m.reshape(-1, m.shape[-1])
                        H_ = flat.shape[-1]
                        print(
                            f"[masked_scatter eager] per-row mask dump ({flat.shape[0]} rows, each row length={H_}):",
                            flush=True,
                        )
                        old_opts = torch._tensor_str.PRINT_OPTS
                        torch.set_printoptions(
                            threshold=float("inf"),
                            edgeitems=10000,
                            linewidth=10**9,
                        )
                        try:
                            for i in range(flat.shape[0]):
                                row = flat[i]
                                all_t = bool(row.all().item())
                                any_t = bool(row.any().item())
                                if all_t:
                                    tag = "all_True "
                                elif not any_t:
                                    tag = "all_False"
                                else:
                                    n_true_row = int(row.sum().item())
                                    tag = f"NON-UNIFORM(n_true={n_true_row}/{H_})"
                                print(
                                    f"  row[{i:>4}] all_true={all_t} any_true={any_t} {tag}: {row.tolist()}",
                                    flush=True,
                                )
                        finally:
                            torch.set_printoptions(
                                threshold=old_opts.threshold,
                                edgeitems=old_opts.edgeitems,
                                linewidth=old_opts.linewidth,
                                precision=old_opts.precision,
                                profile="default",
                            )
            except Exception as _e:
                print(f"[masked_scatter eager] inspect failed: {_e}", flush=True)
            try:
                return func(*args, **(kwargs or {}))
            finally:
                _masked_scatter_inspect_state.active = False

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
