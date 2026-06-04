# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TT-compatible override of vLLM's GatedDeltaNetAttention.

Status: FIRST-PASS STUB. The goal of this file is to let the TT compile path
(torch.compile(backend="tt") -> tt-mlir) trace cleanly through Qwen3-Next /
Qwen3.5 DeltaNet layers — i.e. unblock dynamo from the FLA Triton kernels +
no-op fake_impl that crash today.

What this stub does:
  * Replaces the per-layer torch.ops.vllm.gdn_attention_core custom op with a
    pure-PyTorch implementation: causal conv1d + gating + naive token-by-token
    delta-rule recurrence + gated RMSNorm + out-proj. All ops have proper meta
    implementations, so dynamo's FakeTensor shape propagation is consistent
    end-to-end.

What this stub does NOT do (knowingly):
  * Read attn_metadata. The recurrent state is allocated by the plugin's
    initialize_kv_cache (MambaSpec branch) and reachable via self.kv_cache,
    but for first bringup we ignore the request-slot indexing entirely and
    operate as if every forward is a fresh prefill for a single request at
    slot 0. With max_num_seqs=1 this is functionally fine for the first
    forward; subsequent forwards will compute wrong outputs because the state
    is not threaded back in. Numerical correctness across multiple steps is
    a follow-up task.
  * Speculative decoding. We assume spec_sequence_masks is None.
  * Performance. The token-by-token loop will unroll under dynamic=False and
    produce a very large FX graph when num_tokens is large (e.g. 16384). The
    first compile run with --max-num-batched-tokens 16384 may be very slow or
    OOM. If it does, drop --max-model-len / --max-num-batched-tokens to ~128
    for first end-to-end testing, then scale up once the path is validated.
  * Chunkwise-parallel formulation. The FLA kernel uses a chunkwise decomposition
    that's mathematically equivalent but compiles to many fewer ops. Implement
    that as a follow-up once the naive loop validates correctness.
  * Numerical match against the FLA reference. The naive loop should be
    mathematically equivalent at fp32, but bf16 accumulation order differs.

For full context see examples/vllm/Qwen3.6-27B/DELTA_NET_BRINGUP.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from vllm.model_executor.layers.mamba.gdn_linear_attn import (
    GatedDeltaNetAttention,
)

from .logger import tt_init_logger

logger = tt_init_logger(__name__)


class TTGatedDeltaNetAttention(nn.Module):
    """Pure-PyTorch drop-in replacement for GatedDeltaNetAttention.

    Same forward signature as the original: forward(hidden_states, output).
    Reuses the original module's sub-layers (projections, conv1d, norm,
    out_proj, A_log, dt_bias) — only the core recurrence is re-implemented.
    """

    def __init__(self, layer: GatedDeltaNetAttention):
        super().__init__()
        assert isinstance(layer, GatedDeltaNetAttention)

        # Shape constants
        self.tp_size = layer.tp_size
        self.num_k_heads = layer.num_k_heads
        self.num_v_heads = layer.num_v_heads
        self.head_k_dim = layer.head_k_dim
        self.head_v_dim = layer.head_v_dim
        self.key_dim = layer.key_dim
        self.value_dim = layer.value_dim
        self.conv_kernel_size = layer.conv_kernel_size
        self.activation = layer.activation
        self.act = ACT2FN[layer.activation]
        self.gqa_interleaved_layout = layer.gqa_interleaved_layout
        self.prefix = layer.prefix

        # Sub-modules — keep references to the originals, which are standard
        # TP-aware layers that already compile cleanly through the TT path.
        self.in_proj_qkvz = getattr(layer, "in_proj_qkvz", None)
        self.in_proj_qkv = getattr(layer, "in_proj_qkv", None)
        self.in_proj_z = getattr(layer, "in_proj_z", None)
        self.in_proj_ba = layer.in_proj_ba
        self.conv1d = layer.conv1d
        self.norm = layer.norm
        self.out_proj = layer.out_proj
        self.A_log = layer.A_log
        self.dt_bias = layer.dt_bias

        # GQA group ratio (n value-heads per key-head)
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                f"num_v_heads ({self.num_v_heads}) must be divisible by "
                f"num_k_heads ({self.num_k_heads})"
            )
        self.gqa_ratio = self.num_v_heads // self.num_k_heads

        # Per-rank head counts (after TP slicing of the projection outputs)
        self.num_k_heads_per_rank = self.num_k_heads // self.tp_size
        self.num_v_heads_per_rank = self.num_v_heads // self.tp_size

        # Sanity check that the original layer holds the same kv_cache attr
        # we want to read from. It's populated by bind_kv_cache after
        # initialize_kv_cache runs.
        # NOTE: vllm.MambaBase declares kv_cache: tuple[torch.Tensor, ...].

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _project(self, hidden_states: torch.Tensor):
        """Replicates Part 1 of GatedDeltaNetAttention.forward.

        Returns (mixed_qkv, z, b, a) where mixed_qkv is the concatenation of
        (q, k, v) projections pre-conv1d, and z/b/a are the auxiliary
        signals.
        """
        if self.in_proj_qkv is not None:
            # LoRA-style separate projections (rarely hit on Qwen3.5)
            mixed_qkv, _ = self.in_proj_qkv(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            z, _ = self.in_proj_z(hidden_states)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
        else:
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            # Qwen3.5 path: weights are already in [q, k, v, z] order.
            # (Qwen3-Next's gqa_interleaved_layout path is not handled yet.)
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)

        return mixed_qkv.contiguous(), z, b.contiguous(), a.contiguous()

    def _causal_conv1d_prefill(self, mixed_qkv: torch.Tensor) -> torch.Tensor:
        """Depthwise causal conv1d over the (num_tokens, conv_dim/tp) input.

        For first bringup we treat every forward as a fresh prefill with an
        all-zero conv state, i.e. the left context is padded with zeros. We
        do NOT update the conv_state cache; that is a known shortcut tracked
        in DELTA_NET_BRINGUP.md.
        """
        num_tokens = mixed_qkv.size(0)
        conv_w = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )  # (conv_dim/tp, K)

        # Reshape to (N=1, C=conv_dim/tp, L=num_tokens), pad left with K-1
        # zeros for causality, then run depthwise conv via F.conv1d with
        # groups=C.
        K = self.conv_kernel_size
        x = mixed_qkv.transpose(0, 1).unsqueeze(0)  # (1, C, num_tokens)
        if K > 1:
            pad = torch.zeros(
                (1, x.size(1), K - 1), dtype=x.dtype, device=x.device
            )
            x = torch.cat([pad, x], dim=-1)
        w = conv_w.unsqueeze(1)  # (C, 1, K)
        y = F.conv1d(x, w, groups=conv_w.size(0))  # (1, C, num_tokens)
        y = y.squeeze(0).transpose(0, 1)  # (num_tokens, C)
        if self.conv1d.bias is not None:
            y = y + self.conv1d.bias.view(1, -1)
        y = self.act(y)
        return y

    def _split_qkv(self, mixed_qkv: torch.Tensor):
        """Split the post-conv (q, k, v) projection result into per-head tensors."""
        num_tokens = mixed_qkv.size(0)
        q_size = self.key_dim // self.tp_size
        v_size = self.value_dim // self.tp_size
        q, k, v = torch.split(mixed_qkv, [q_size, q_size, v_size], dim=-1)
        q = q.view(num_tokens, self.num_k_heads_per_rank, self.head_k_dim)
        k = k.view(num_tokens, self.num_k_heads_per_rank, self.head_k_dim)
        v = v.view(num_tokens, self.num_v_heads_per_rank, self.head_v_dim)
        return q, k, v

    def _compute_gating(self, a: torch.Tensor, b: torch.Tensor):
        """Pure-PyTorch equivalent of fused_gdn_gating.

        a, b: (num_tokens, num_v_heads/tp)
        Returns (g, beta) of the same shape. g is log-decay (will be exp'd at
        use-site), beta is the write strength.
        """
        # softplus_x = softplus(a + dt_bias)
        softplus_x = F.softplus(a.to(torch.float32) + self.dt_bias.to(torch.float32))
        g = -torch.exp(self.A_log.to(torch.float32)) * softplus_x
        beta = torch.sigmoid(b.to(torch.float32))
        return g.to(a.dtype), beta.to(b.dtype)

    def _delta_rule_recurrence(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Naive token-by-token gated delta-rule recurrence.

        Inputs (all per-rank):
          q: (T, num_k_heads/tp, head_k_dim)
          k: (T, num_k_heads/tp, head_k_dim)
          v: (T, num_v_heads/tp, head_v_dim)
          g: (T, num_v_heads/tp)  -- log-decay, will be exp'd
          beta: (T, num_v_heads/tp)

        Returns:
          y: (T, num_v_heads/tp, head_v_dim)

        Math per token t (per head h):
          S_t = exp(g_t) * S_{t-1}
                - beta_t * (S_{t-1} k_t) k_t^T
                + beta_t * v_t k_t^T
          y_t = S_t q_t
        """
        T = q.size(0)
        Hv = self.num_v_heads_per_rank
        Dv = self.head_v_dim
        Dk = self.head_k_dim

        # L2-normalize q, k (the FLA kernels do this internally when
        # use_qk_l2norm_in_kernel=True).
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # GQA expand: q and k from (T, num_k_heads/tp, Dk) →
        # (T, num_v_heads/tp, Dk) by repeating each k-head gqa_ratio times.
        q = q.repeat_interleave(self.gqa_ratio, dim=1)
        k = k.repeat_interleave(self.gqa_ratio, dim=1)

        S = torch.zeros((Hv, Dv, Dk), dtype=v.dtype, device=v.device)
        y = torch.empty((T, Hv, Dv), dtype=v.dtype, device=v.device)

        # NOTE: this loop will be unrolled by dynamo under dynamic=False.
        # For T=16384 the resulting FX graph is enormous. For first bringup
        # run with small max_num_batched_tokens (e.g. 128) and scale up once
        # validated.
        for t in range(T):
            decay = torch.exp(g[t]).view(Hv, 1, 1)  # (Hv, 1, 1)
            S = S * decay
            # r = S @ k_t  → (Hv, Dv)
            r = torch.einsum("hvd,hd->hv", S, k[t])
            delta = (v[t] - r) * beta[t].view(Hv, 1)  # (Hv, Dv)
            # S += delta outer k_t → (Hv, Dv, Dk)
            S = S + torch.einsum("hv,hd->hvd", delta, k[t])
            y[t] = torch.einsum("hvd,hd->hv", S, q[t])

        return y

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
        # The FLA reference assumes 2-D (num_tokens, hidden_size) inputs.
        # The TT plugin's _dummy_run / profile_run feeds 3-D
        # (batch=1, num_tokens, hidden_size) hidden_states; output is
        # torch.empty_like(hidden_states) so it has the same rank.
        # Flatten internally to 2-D, do the work, and write back through a
        # 2-D view on `output` (torch.empty_like produces a contiguous tensor
        # so the reshape is a view and the in-place write propagates).
        if hidden_states.dim() == 3:
            hidden_states_2d = hidden_states.reshape(-1, hidden_states.size(-1))
            output_2d = output.reshape(-1, output.size(-1))
        else:
            hidden_states_2d = hidden_states
            output_2d = output

        num_tokens = hidden_states_2d.size(0)

        # Part 1: input projections
        mixed_qkv, z, b, a = self._project(hidden_states_2d)

        # Part 2a: causal conv1d (fresh-prefill assumption)
        post_conv = self._causal_conv1d_prefill(mixed_qkv)

        # Part 2b: split q/k/v
        q, k, v = self._split_qkv(post_conv)

        # Part 2c: gating signals
        g, beta = self._compute_gating(a, b)

        # Part 2d: delta-rule recurrence
        core_attn_out = self._delta_rule_recurrence(q, k, v, g, beta)
        # core_attn_out: (num_tokens, num_v_heads/tp, head_v_dim)

        # Part 3: gated RMSNorm + output projection
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(num_tokens, -1)
        proj_out, _ = self.out_proj(core_attn_out)
        # In-place write into the (view of the) output tensor.
        output_2d[:num_tokens] = proj_out


def tt_gated_delta_net_module(layer: nn.Module) -> nn.Module:
    """Factory used by replace_modules() to swap in the TT-compatible class."""
    assert isinstance(layer, GatedDeltaNetAttention)
    return TTGatedDeltaNetAttention(layer)
