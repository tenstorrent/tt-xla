# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""
DeepSeek Sparse Attention (DSA) backend for TT devices.

DSA (introduced by DeepSeek-V3.2-Exp) is plain Multi-head Latent Attention (MLA)
plus a *lightning indexer*: a lightweight, head-independent scorer that, for each
query token, ranks the candidate key positions and keeps only the top-k. Tokens
outside the top-k are dropped from the attention softmax, which bounds attention
cost on long contexts.

This backend reuses the MLA machinery wholesale and realizes the sparsity as an
**additive attention mask**:

    index_score[u, i, j] = sum_h ReLU(q_idx[u, i, h] . k_idx[u, j]) * w[u, i, h]
    keep = top-k_j(index_score) within the causal region
    mask[u, i, j] = 0 if j in keep else -inf   (plus the causal mask)

The mask is shared across the MLA attention heads (the indexer has its own heads,
independent of the 128 attention heads), so it slots straight into the existing
``tt::flash_mla_prefill`` / ``tt::paged_flash_mla_decode`` ops via their optional
``attn_mask`` argument (run with ``is_causal=False``). No new device op is needed —
the only addition over MLA is the indexer's score/top-k computation and its own
small K cache.

Numerically this mirrors the BF16, mask-based reference in
``third_party/tt_forge_models/.../deepseek_v3_2_exp/.../modified_model.py`` (the
``Indexer`` module + ``bf16_index``), rather than the GPU production path
(per-block FP8 ``sparse_attn_indexer`` + a true sparse gather kernel), which is
CUDA-specific.

Wiring:
    * ``TTPlatform.get_attn_backend_cls`` routes ``use_mla and use_sparse`` here.
    * The OOT ``TTMultiHeadLatentAttentionWrapper`` (in ``attention_mla.py``)
      detects sparse layers and calls ``dsa_wrapper_forward`` below, which runs the
      TT indexer and threads the resulting mask into the (shared) MLA layer.
"""

from typing import Optional

import torch
from vllm.forward_context import get_forward_context

from .attention_mla import (
    TTMLAAttentionBackend,
    TTMLAAttentionBackendImpl,
    TTMultiHeadLatentAttentionWrapper,
)
from .logger import tt_init_logger

logger = tt_init_logger(__name__)


# --------------------------------------------------------------------------- #
# Indexer math helpers (BF16, mask-based)
# --------------------------------------------------------------------------- #
def _lin_out(layer, x: torch.Tensor) -> torch.Tensor:
    """Call a (possibly vLLM) linear layer and return just the output tensor.

    vLLM's ``ReplicatedLinear`` returns ``(output, bias)``; plain ``nn.Linear``
    (and the test stand-ins) return a tensor. Normalize to the tensor.
    """
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


def _indexer_project(
    indexer,
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    rope_emb,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project hidden states / query-latent into the indexer's q, k and weights.

    Mirrors ``vllm...deepseek_v2.Indexer.forward`` up to (but excluding) the FP8
    quantization, computing everything in the activation dtype instead.

    Shapes (T = number of tokens this call, NH = index heads, HD = index head dim,
    RD = rope dim):
        hidden_states: [T, hidden]
        qr:            [T, q_lora_rank]      (the q_a_layernorm output, == q_c)
    Returns:
        q:       [T, NH, HD]   rope applied to the leading RD dims
        k:       [T, HD]       rope applied to the leading RD dims (single head)
        weights: [T, NH]       per-head softmax/normalization scale folded in
    """
    NH = indexer.n_head
    HD = indexer.head_dim
    RD = indexer.rope_dim

    q = _lin_out(indexer.wq_b, qr).view(-1, NH, HD)
    q_pe, q_nope = torch.split(q, [RD, HD - RD], dim=-1)

    k = _lin_out(indexer.wk, hidden_states)
    k = indexer.k_norm(k)
    k_pe, k_nope = torch.split(k, [RD, HD - RD], dim=-1)

    # The indexer applies its own (NeoX-style) rope; rope is shape-preserving.
    q_pe, k_pe = rope_emb(positions, q_pe, k_pe.unsqueeze(1))
    q_pe = q_pe.reshape(-1, NH, RD)
    k_pe = k_pe.reshape(-1, 1, RD)

    q = torch.cat([q_pe, q_nope], dim=-1)  # [T, NH, HD]
    k = torch.cat([k_pe.squeeze(1), k_nope], dim=-1)  # [T, HD]

    # weights folds in softmax_scale and the per-head normalization. The GPU path
    # also multiplies by the FP8 q-scale; in BF16 there is no such scale.
    weights = _lin_out(indexer.weights_proj, hidden_states)  # [T, NH]
    weights = weights * (indexer.softmax_scale * (NH**-0.5))
    return q, k, weights


def _index_scores(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Lightning-indexer scores: ReLU-then-weighted-sum over the index heads.

    Shapes:
        q:       [users, Sq, NH, HD]
        k:       [users, Sk, HD]
        weights: [users, Sq, NH]
    Returns:
        scores:  [users, Sq, Sk]
    """
    logits = torch.einsum("ushd,utd->usht", q, k)
    logits = torch.relu(logits)
    # weights_proj may run in fp32 (as in the DeepSeek reference) while q/k are
    # bf16; cast so the einsum dtypes agree and the score stays in the q dtype.
    return torch.einsum("usht,ush->ust", logits, weights.to(logits.dtype))


def _gather_indexer_cache(
    cache: torch.Tensor, page_table: torch.Tensor
) -> torch.Tensor:
    """Gather a dense indexer-K tensor from the paged indexer cache.

    Paged layout (mirrors the MLA latent cache, single KV head):
        cache:      [num_blocks, 1, block_size, HD]
        page_table: [users, num_blocks_per_user]
    Returns:
        [users, num_blocks_per_user * block_size, HD]
    The trailing positions past each user's current position are masked out by
    the causal mask in ``compute_dsa_sparse_mask``.
    """
    num_blocks_per_user = page_table.shape[1]
    block_size = cache.shape[2]
    head_size = cache.shape[3]
    users = page_table.shape[0]

    flat_indices = page_table.reshape(-1).long()
    gathered = torch.index_select(cache, 0, flat_indices)
    # [users, num_blocks_per_user, 1, block_size, HD]
    gathered = gathered.view(users, num_blocks_per_user, 1, block_size, head_size)
    # [users, 1, num_blocks_per_user, block_size, HD]
    gathered = gathered.permute(0, 2, 1, 3, 4).contiguous()
    # [users, num_blocks_per_user * block_size, HD]
    return gathered.reshape(users, num_blocks_per_user * block_size, head_size)


def _indexer_kv_cache(indexer) -> Optional[torch.Tensor]:
    """Fetch the bound indexer-K cache tensor, or None when not yet allocated.

    The vLLM ``Indexer`` owns a ``DeepseekV32IndexerCache`` submodule
    (``indexer.k_cache``); the TT model runner binds the device tensor onto its
    ``kv_cache`` attribute (see ``model_runner.initialize_kv_cache``). Before
    binding (profiling / dummy runs) it is an empty tensor.
    """
    k_cache_mod = getattr(indexer, "k_cache", None)
    cache = getattr(k_cache_mod, "kv_cache", None)
    if isinstance(cache, torch.Tensor) and cache.numel() > 0:
        return cache
    return None


# --------------------------------------------------------------------------- #
# Sparse mask construction
# --------------------------------------------------------------------------- #
def compute_dsa_sparse_mask(
    indexer,
    rope_emb,
    indexer_kv_cache: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    attn_metadata,
    is_prefill: bool,
    users: int,
    seq_len: int,
) -> Optional[torch.Tensor]:
    """Build the additive DSA mask for the MLA kernels, updating the indexer cache.

    Returns ``None`` (→ fall back to plain causal MLA) when a faithful mask can't
    be built: profiling runs without metadata, or a decode step before the
    indexer cache is allocated. Otherwise returns:
        prefill: ``[users, 1, S, S]``                (head-broadcast)
        decode:  ``[users, 1, 1, num_blocks_per_user * block_size]``

    Side effect: writes the freshly computed indexer K into ``indexer_kv_cache``
    (prefill fills, decode updates at the current position), mirroring how the MLA
    impl persists the latent K.
    """
    if attn_metadata is None or attn_metadata.cache_position is None:
        return None

    device = hidden_states.device
    NH = indexer.n_head
    HD = indexer.head_dim
    topk = indexer.topk_tokens

    q, k, weights = _indexer_project(indexer, hidden_states, qr, positions, rope_emb)
    q = q.view(users, seq_len, NH, HD)
    weights = weights.view(users, seq_len, NH)
    score_dtype = q.dtype
    neg_inf = torch.finfo(score_dtype).min

    if is_prefill:
        k_local = k.view(users, seq_len, HD)

        # Persist indexer K for subsequent decode steps (skipped while the cache
        # is unbound, e.g. profiling). Mirrors the MLA latent paged_fill_cache.
        if indexer_kv_cache is not None:
            k_fill = k_local.unsqueeze(1)  # [users, 1, S, HD]
            fill_page_table = attn_metadata.fill_page_table
            filled = indexer_kv_cache
            for b in range(users):
                filled = torch.ops.tt.paged_fill_cache(
                    filled,
                    k_fill[b : b + 1],
                    fill_page_table,
                    batch_idx=torch.tensor([b], dtype=torch.int32, device=device),
                )
            indexer_kv_cache.copy_(filled)

        scores = _index_scores(q, k_local, weights)  # [users, S, S]
        # Causal mask: position i may only see j <= i. Applied before top-k so
        # future positions are never selected.
        causal = torch.triu(
            torch.full((seq_len, seq_len), neg_inf, device=device, dtype=score_dtype),
            diagonal=1,
        )
        scores = scores + causal
        kth = min(topk, seq_len)
        topk_idx = scores.topk(kth, dim=-1).indices  # [users, S, kth]
        sparse = torch.full(
            (users, seq_len, seq_len), neg_inf, device=device, dtype=score_dtype
        ).scatter_(-1, topk_idx, 0.0)
        # Re-add causal so filler picks (when fewer than k valid positions exist)
        # stay masked. -inf + finite/-inf == -inf, so no NaNs are produced.
        mask = sparse + causal
        return mask.unsqueeze(1)  # [users, 1, S, S]

    # ---- decode (S == 1) ----
    if indexer_kv_cache is None:
        # No indexer history yet (dummy/profiling). Plain causal MLA is correct
        # here; sparsity only kicks in once the cache is populated.
        return None

    cur_pos = attn_metadata.cache_position  # [users]
    page_table = attn_metadata.page_table

    # Write the new token's indexer K at its position, then read the full history.
    k_update = k.view(users, 1, 1, HD).transpose(0, 1)  # [1, users, 1, HD]
    updated = torch.ops.tt.paged_update_cache(
        indexer_kv_cache, k_update, cur_pos, page_table
    )
    indexer_kv_cache.copy_(updated)
    k_all = _gather_indexer_cache(indexer_kv_cache, page_table)  # [users, max_seq, HD]
    max_seq = k_all.shape[1]

    scores = _index_scores(q, k_all, weights)  # [users, 1, max_seq]
    pos_range = torch.arange(max_seq, device=device)
    future = pos_range.view(1, 1, max_seq) > cur_pos.view(users, 1, 1)
    causal = torch.zeros(
        (users, 1, max_seq), device=device, dtype=score_dtype
    ).masked_fill(future, neg_inf)
    scores = scores + causal
    kth = min(topk, max_seq)
    topk_idx = scores.topk(kth, dim=-1).indices  # [users, 1, kth]
    sparse = torch.full(
        (users, 1, max_seq), neg_inf, device=device, dtype=score_dtype
    ).scatter_(-1, topk_idx, 0.0)
    mask = sparse + causal  # [users, 1, max_seq]
    return mask.unsqueeze(1)  # [users, 1, 1, max_seq]


# --------------------------------------------------------------------------- #
# Wrapper forward (replaces the upstream sparse path, which is GPU-only)
# --------------------------------------------------------------------------- #
def dsa_wrapper_forward(
    wrapper: TTMultiHeadLatentAttentionWrapper,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MLA + lightning-indexer forward for the TT OOT MLA wrapper.

    This reimplements ``vllm...mla.MultiHeadLatentAttentionWrapper.forward`` for
    sparse (DeepSeek-V3.2) layers: the upstream forward calls the GPU-only indexer
    op and, moreover, never threads the indexer result into ``mla_attn`` (on GPU it
    flows through a shared ``topk_indices_buffer``). Here we instead run the TT
    indexer and pass the resulting additive mask straight into the (shared)
    ``TTMLAAttention`` layer, which forwards it to ``flash_mla_prefill`` /
    ``paged_flash_mla_decode``.

    Only the q-LoRA path is implemented (DeepSeek-V3.2 always sets
    ``q_lora_rank``); the non-LoRA path raises.
    """
    # The TT runner passes 3D [users, S, H] / [users, S]; flatten to vLLM's 2D
    # token stream and restore the shape on the way out.
    orig_ndim = hidden_states.ndim
    orig_users = orig_S = None
    if orig_ndim == 3:
        orig_users, orig_S, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(orig_users * orig_S, hidden_size)
        positions = positions.reshape(-1)

    if wrapper.fused_qkv_a_proj is None or wrapper.q_a_layernorm is None:
        raise NotImplementedError(
            "TT DSA requires the q-LoRA MLA path (fused_qkv_a_proj + "
            "q_a_layernorm); the non-LoRA path is not supported."
        )

    # -- MLA query/kv projection (mirrors MultiHeadLatentAttentionWrapper) -----
    qkv_lora = _lin_out(wrapper.fused_qkv_a_proj, hidden_states)
    q_c, kv_lora = qkv_lora.split(
        [wrapper.q_lora_rank, wrapper.kv_lora_rank + wrapper.qk_rope_head_dim],
        dim=-1,
    )
    q_c = wrapper.q_a_layernorm(q_c)
    q = _lin_out(wrapper.q_b_proj, q_c)

    kv_c, k_pe = kv_lora.split([wrapper.kv_lora_rank, wrapper.qk_rope_head_dim], dim=-1)
    kv_c_normed = wrapper.kv_a_layernorm(kv_c)

    q = q.view(-1, wrapper.num_heads, wrapper.qk_head_dim)
    k_pe = k_pe.unsqueeze(1)
    if wrapper.rotary_emb is not None:
        q[..., wrapper.qk_nope_head_dim :], k_pe = wrapper.rotary_emb(
            positions, q[..., wrapper.qk_nope_head_dim :], k_pe
        )

    # -- Lightning indexer -> additive top-k mask -----------------------------
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata.get(wrapper.mla_attn.layer_name)

    total_tokens = hidden_states.shape[0]
    if attn_metadata is not None and attn_metadata.cache_position is not None:
        users = attn_metadata.cache_position.shape[0]
    else:
        users = orig_users if orig_users is not None else 1
    assert (
        total_tokens % users == 0
    ), f"total_tokens ({total_tokens}) not divisible by users ({users})."
    seq_len = total_tokens // users
    is_prefill = seq_len > 1

    attn_mask = compute_dsa_sparse_mask(
        wrapper.indexer,
        wrapper.indexer_rope_emb,
        _indexer_kv_cache(wrapper.indexer),
        hidden_states,
        q_c,
        positions,
        attn_metadata,
        is_prefill,
        users,
        seq_len,
    )

    if llama_4_scaling is not None:
        q *= llama_4_scaling

    attn_out = wrapper.mla_attn(
        q,
        kv_c_normed,
        k_pe,
        output_shape=(total_tokens, wrapper.num_heads * wrapper.v_head_dim),
        attn_mask=attn_mask,
    )
    out = _lin_out(wrapper.o_proj, attn_out)

    if orig_ndim == 3:
        out = out.reshape(orig_users, orig_S, out.shape[-1])
    return out


# --------------------------------------------------------------------------- #
# Backend / impl
# --------------------------------------------------------------------------- #
class TTDSAAttentionBackendImpl(TTMLAAttentionBackendImpl):
    """DeepSeek Sparse Attention impl for TT.

    Identical to the MLA impl except that the OOT wrapper supplies an additive
    top-k ``attn_mask`` (computed by the indexer) to ``forward``; the inherited
    MLA forward then runs ``tt::flash_mla_prefill`` / ``tt::paged_flash_mla_decode``
    with ``is_causal=False`` and that mask. Exists as its own class so vLLM can
    select it via the sparse backend and so true sparse-gather kernels can be
    slotted in later without touching the dense MLA impl.
    """


class TTDSAAttentionBackend(TTMLAAttentionBackend):
    """vLLM attention backend for DeepSeek Sparse Attention (MLA + indexer) on TT.

    Shares the MLA latent KV-cache layout, page size and (stub) metadata builder
    with ``TTMLAAttentionBackend``; only the impl and identity differ.
    """

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA_SPARSE"

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TTDSAAttentionBackendImpl"]:
        return TTDSAAttentionBackendImpl
