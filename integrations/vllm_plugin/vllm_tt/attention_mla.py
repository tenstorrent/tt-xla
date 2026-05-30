# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""
MLA (Multi-head Latent Attention) backend for TT devices.

Mirrors the structural shape of `PallasMLAttentionBackendImpl` from
tpu-inference: a single unified `forward()` in the impl with stubbed
`forward_mha` / `forward_mqa` methods. To make the unified forward
reachable in vLLM v1, we replace the standard `MLAAttention` layer with
a subclass that calls `impl.forward(...)` directly, and we install that
replacement via vLLM's `PluggableLayer` OOT registry by registering a
custom `MultiHeadLatentAttentionWrapper` subclass.

Scope: prefill only. The unified forward raises NotImplementedError on
the decode path.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.v1.attention.backend import AttentionBackend, AttentionLayer, MLAAttentionImpl
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from .attention import TTAttentionMetadataBuilder, TTMetadata
from .logger import tt_init_logger

if TYPE_CHECKING:
    # Type-only: importing `vllm.config.VllmConfig` at module top-level
    # races vLLM's own initialization. Plugin discovery calls our
    # `register()` while `vllm.config` is still mid-init, so a real
    # import would raise `ImportError: cannot import name 'VllmConfig'
    # from partially initialized module 'vllm.config'`. Keep it
    # type-only — it's only used in a `get_page_size` annotation.
    from vllm.config import VllmConfig

logger = tt_init_logger(__name__)


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class TTMLAAttentionBackend(AttentionBackend):
    """vLLM attention backend for MLA on TT devices."""

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA"

    @staticmethod
    def get_impl_cls() -> type["TTMLAAttentionBackendImpl"]:
        return TTMLAAttentionBackendImpl

    @staticmethod
    def get_builder_cls():
        # Reuse the same stub builder used by the non-MLA backend; MLA does
        # not need a different metadata class for the prefill-only scope.
        return TTAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # MLA stores a single concatenated latent KV tensor per slot.
        # num_kv_heads is always 1 and head_size = kv_lora_rank + qk_rope_head_dim.
        # Shape convention matches the non-MLA TTAttentionBackend so the
        # existing model_runner KV-cache plumbing works without changes.
        return (num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_page_size(vllm_config: "VllmConfig") -> int:
        return 32

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TT MLA backend.")


# --------------------------------------------------------------------------- #
# Impl
# --------------------------------------------------------------------------- #
class TTMLAAttentionBackendImpl(MLAAttentionImpl):
    """MLA attention impl for TT.

    Mirrors tpu-inference's `PallasMLAttentionBackendImpl`: a single
    unified `forward` does Q-absorption, paged latent KV-cache write, the
    latent attention, and the W_UV projection back to physical space.
    `forward_mha` and `forward_mqa` are stubs since this impl is reached
    via the OOT layer override, not vLLM's standard MLAAttention.forward_impl.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        # MLA-specific arguments
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: int = 0,
        qk_nope_head_dim: int = 0,
        qk_rope_head_dim: int = 0,
        qk_head_dim: int = 0,
        v_head_dim: int = 0,
        kv_b_proj=None,
        indexer: Optional[object] = None,
        q_pad_num_heads: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes are not supported for MLA on TT.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError(
                f"Quantized MLA KV cache ({kv_cache_dtype}) is not yet "
                "supported on TT."
            )

    # ------------------------------------------------------------------ #
    # Abstract stubs — never called because we bypass forward_impl via
    # the OOT TTMLAAttention layer override below.
    # ------------------------------------------------------------------ #
    def forward_mha(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "TTMLAAttentionBackendImpl.forward_mha should never be called; "
            "the TT MLA layer routes through forward() directly. Did the "
            "OOT TTMultiHeadLatentAttentionWrapper fail to register?"
        )

    def forward_mqa(self, *args, **kwargs):
        raise RuntimeError(
            "TTMLAAttentionBackendImpl.forward_mqa should never be called; "
            "the TT MLA layer routes through forward() directly. Did the "
            "OOT TTMultiHeadLatentAttentionWrapper fail to register?"
        )

    # ------------------------------------------------------------------ #
    # Unified forward — the real work happens here.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        q: tuple[torch.Tensor, torch.Tensor],
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TTMetadata,
        layer: "MLAAttention",
        output: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """MLA prefill on TT.

        Shapes (from `TTMLAAttention.forward` after splitting `q`):
            q_nope:      [tokens, num_heads, qk_nope_head_dim]
            q_pe:        [tokens, num_heads, qk_rope_head_dim]
            kv_c_normed: [tokens, kv_lora_rank]
            k_pe:        [tokens, 1, qk_rope_head_dim]
            kv_cache:    [num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim]
            output:      [tokens, num_heads * v_head_dim]   (write target)

        Returns the written output tensor.
        """
        q_nope, q_pe = q
        print("[HET DEBUG][INPUTS] q type: ", type(q))
        print("[HET DEBUG][INPUTS] q length: ", len(q))
        print("[HET DEBUG][INPUTS] q_nope shape: ", q_nope.shape)
        print("[HET DEBUG][INPUTS] q_pe shape: ", q_pe.shape)
        print("[HET DEBUG][INPUTS] k_pe shape: ", k_pe.shape)
        print("[HET DEBUG][INPUTS] kv_cache type: ", type(kv_cache))
        print("[HET DEBUG][INPUTS] kv_cache shape: ", kv_cache.shape)

        # print("[HET DEBUG][INPUTS] kv_cache shape: ", kv_cache.shape)
        # print("[HET DEBUG][INPUTS] attn_metadata: ", attn_metadata)

        is_prefill = self._infer_is_prefill(q_nope, attn_metadata)
        if not is_prefill:
            raise NotImplementedError(
                "Paged MLA decode is not yet implemented on TT — this change "
                "covers prefill only. Add a tt::paged_flash_mla_decode op + "
                "wire it through this branch as a follow-up."
            )

        users = (
            attn_metadata.cache_position.shape[0]
            if attn_metadata is not None and attn_metadata.cache_position is not None
            else 1
        )
        total_tokens = q_nope.shape[0]
        assert (
            total_tokens % users == 0
        ), f"total_tokens ({total_tokens}) not divisible by users ({users})."
        S = total_tokens // users
        N = self.num_heads
        L = self.kv_lora_rank
        R = self.qk_rope_head_dim
        V = self.v_head_dim
        P = self.qk_nope_head_dim

        # -- 1. Reshape inputs to [users, S, ...] --------------------------
        q_nope = q_nope.view(users, S, N, P)
        q_pe = q_pe.view(users, S, N, R)
        kv_c = kv_c_normed.view(users, S, L)
        k_pe_v = k_pe.view(users, S, 1, R)

        # -- 2. Q absorption: q_nope @ W_UK_T  ----------------------------
        # layer.W_UK_T : [num_heads, qk_nope_head_dim, kv_lora_rank].
        # W_UK_T is assigned as a plain tensor attribute (not nn.Parameter
        # or registered buffer) in MLAAttention.process_weights_after_loading
        # (mla_attention.py:797), so `model.to('xla')` doesn't move it —
        # explicit `.to(device=q_nope.device)` is required here.
        act_dtype = q_pe.dtype
        device = q_nope.device
        q_nope_lat = torch.einsum(
            "bsnp,npl->bsnl",
            q_nope.to(torch.float32),
            layer.W_UK_T.to(device=device, dtype=torch.float32),
        ).to(act_dtype)

        # -- 3. Build concatenated latent Q / K ---------------------------
        q_lat = torch.cat([q_nope_lat, q_pe], dim=-1)  # [b, S, N, L+R]
        k_lat = torch.cat([kv_c.unsqueeze(2), k_pe_v], dim=-1)  # [b, S, 1, L+R]

        # -- 4. Latent prefill attention (V = K[..., :L]) -----------------
        # The kernel reads from the local k_lat tensor; it does not touch
        # kv_cache. We attend first and persist after, so that if the
        # kernel raises the cache stays untouched (clean diagnostic
        # rollback). Order is otherwise semantically irrelevant.
        q_for_kernel = q_lat.transpose(1, 2).contiguous()  # [b, N, S, L+R]
        k_for_kernel = k_lat.transpose(1, 2).contiguous()  # [b, 1, S, L+R]
        out_lat = torch.ops.tt.flash_mla_prefill(
            query=q_for_kernel,
            key=k_for_kernel,
            head_dim_v=L,
            value=None,
            is_causal=attn_metadata.is_causal if attn_metadata is not None else True,
            scale=self.scale,
        )  # [b, N, S, L]

        # -- 5. Project latent output back to v_head_dim via W_UV ---------
        # layer.W_UV : [num_heads, kv_lora_rank, v_head_dim].
        # Same plain-tensor caveat as W_UK_T above — explicit device move.
        out = torch.einsum(
            "bnsl,nlv->bnsv",
            out_lat.to(torch.float32),
            layer.W_UV.to(device=device, dtype=torch.float32),
        ).to(
            act_dtype
        )  # [b, N, S, V]

        # -- 6. Reshape to vLLM's output contract: [tokens, N * V] --------
        out = out.transpose(1, 2).reshape(users * S, N * V)

        # -- 7. Persist new tokens into the latent KV cache. --------------
        # paged_fill_cache wants the fill tensor as [b, num_kv_heads, S, head_dim].
        # Skip during profile runs (attn_metadata is None or kv_cache.numel()
        # == 0 — both are vLLM's profile-run sentinels). Without this guard
        # `attn_metadata.fill_page_table` raises AttributeError on None.
        if (
            attn_metadata is not None
            and isinstance(kv_cache, torch.Tensor)
            and kv_cache.numel() > 0
        ):
            k_lat_for_fill = k_lat.transpose(1, 2)  # [b, 1, S, L+R]
            fill_page_table = attn_metadata.fill_page_table
            for batch_idx in range(users):
                kv_cache = torch.ops.tt.paged_fill_cache(
                    kv_cache,
                    k_lat_for_fill[batch_idx : batch_idx + 1],
                    fill_page_table,
                    batch_idx=torch.tensor(
                        [batch_idx], dtype=torch.int32, device=kv_cache.device
                    ),
                )

        # -- 8. Write into the pre-allocated output buffer ----------------
        if output is not None:
            output.copy_(out)
            return output
        return out

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _infer_is_prefill(q_nope: torch.Tensor, attn_metadata: TTMetadata) -> bool:
        """Heuristic: prefill when more than one token per user.

        Mirrors `TTAttentionBackendImpl._prepare_inputs` at attention.py:346.
        """
        if attn_metadata is None or attn_metadata.cache_position is None:
            # Conservative default during profile runs — treat as prefill so
            # the path doesn't hit the NotImplementedError unnecessarily.
            return True
        users = attn_metadata.cache_position.shape[0]
        if users == 0:
            return True
        total_tokens = q_nope.shape[0]
        return (total_tokens // users) > 1


# --------------------------------------------------------------------------- #
# Custom MLAAttention layer that calls impl.forward(...) directly,
# bypassing vLLM's MHA/MQA split + Q-absorb + _v_up_proj inside
# `MLAAttention.forward_impl`.
# --------------------------------------------------------------------------- #
class TTMLAAttention(MLAAttention):
    """`MLAAttention` subclass that calls `impl.forward(...)` directly."""

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: Optional[torch.Size] = None,
    ) -> torch.Tensor:
        # Split q into (q_nope, q_pe). vLLM's standard MLAAttention.forward
        # only does this inside forward_impl's MQA branch; we do it here so
        # the impl sees the same tuple shape PallasMLAttentionBackendImpl
        # expects.
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata.get(self.layer_name)
        kv_cache = self.kv_cache

        if output_shape is None:
            output_shape = (q.shape[0], self.num_heads * self.v_head_dim)
        output = torch.empty(output_shape, dtype=q.dtype, device=q.device)

        self.impl.forward(
            q=(q_nope, q_pe),
            kv_c_normed=kv_c_normed,
            k_pe=k_pe,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            layer=self,
            output=output,
        )
        return output


# --------------------------------------------------------------------------- #
# OOT wrapper replacement.
#
# vLLM's PluggableLayer.__new__ swaps the instantiated class to whatever is
# registered in op_registry_oot under the key cls.__name__. Registering
# this subclass means model code that does
#   self.mla_attn = MultiHeadLatentAttentionWrapper(...)
# actually constructs a TTMultiHeadLatentAttentionWrapper instead.
#
# Inside __init__ we temporarily rebind the module-level `MLAAttention`
# name in `vllm.model_executor.layers.mla` so the upstream wrapper's
# super().__init__ instantiates TTMLAAttention (without the upstream
# wrapper code knowing). This avoids both reimplementing the wrapper's
# entire __init__ and double-registering the layer in
# compilation_config.static_forward_context.
# --------------------------------------------------------------------------- #
@MultiHeadLatentAttentionWrapper.register_oot
class TTMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):
    def __init__(self, *args, **kwargs):
        import vllm.model_executor.layers.mla as _mla_module

        orig_cls = _mla_module.MLAAttention
        _mla_module.MLAAttention = TTMLAAttention
        try:
            super().__init__(*args, **kwargs)
        finally:
            _mla_module.MLAAttention = orig_cls
        logger.info(
            "[TT] Installed TTMLAAttention (prefix=%s) — MLA prefill uses "
            "torch.ops.tt.flash_mla_prefill; decode is NotImplementedError.",
            getattr(self, "prefix", "?"),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The TT model runner passes hidden_states as 3D [users, S, H] and
        # positions as 2D [users, S]. The upstream MultiHeadLatentAttentionWrapper.forward
        # at mla.py:154-155 does:
        #     q = q.view(-1, num_heads, qk_head_dim)   # collapses leading
        #     k_pe = k_pe.unsqueeze(1)                 # assumes 2D input
        # `unsqueeze(1)` is only correct for 2D k_pe `[tokens, rope_dim]`. With
        # 3D k_pe `[users, S, rope_dim]` it produces `[users, 1, S, rope_dim]`,
        # which then broadcasts incorrectly against DeepseekScalingRotaryEmbedding's
        # cos/sin of shape `[users, S, 1, rope_dim]` (deepseek_scaling_rope.py:137-141)
        # — producing `[users, S, S, rope_dim]`, applying every position's
        # cosine to every token.
        #
        # Flatten to vLLM's standard 2D `[total_tokens, hidden]` before the
        # upstream forward, then reshape the output back so downstream layers
        # see the same shape they sent in.
        orig_ndim = hidden_states.ndim
        if orig_ndim == 3:
            orig_users, orig_S, hidden_size = hidden_states.shape
            hidden_states = hidden_states.reshape(orig_users * orig_S, hidden_size)
            positions = positions.reshape(-1)

        out = super().forward(positions, hidden_states, llama_4_scaling)

        if orig_ndim == 3:
            out = out.reshape(orig_users, orig_S, out.shape[-1])
        return out


# --------------------------------------------------------------------------- #
# Register the backend with vLLM's attention registry. Import-time side
# effect (this module is imported by vllm_tt/__init__.py).
# --------------------------------------------------------------------------- #
register_backend(
    backend=AttentionBackendEnum.FLASH_ATTN_MLA,
    class_path="vllm_tt.attention_mla.TTMLAAttentionBackend",
)


# Silence unused-import lints; nn is kept for future use (e.g. when we
# add weight processing for FP8 MLA).
_ = (nn, AttentionLayer)
