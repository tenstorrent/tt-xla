# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT DeepSeek Sparse Attention (DSA) indexer.

DeepSeek-V3.2 replaces dense MLA with sparse attention: a lightweight "lightning
indexer" scores every past token against the current query, top-k keeps the most
relevant ``index_topk`` of them, and attention runs only over that subset.

``TTIndexer`` replaces vLLM's ``Indexer`` (``deepseek_v2.py``), which is
CUDA-only: it FP8-quantizes the indexer query per 128-element group
(``per_token_group_quant_fp8``), keeps a uint8 fp8+scale K cache, and dispatches a
``SparseAttnIndexer`` custom op. The TT version keeps everything in bf16 and emits

    tt.indexer_score_dsa -> tt.topk_large_indices

publishing the selected indices for ``TTMLAAttentionBackendImpl`` to consume.
Weight construction reuses the upstream submodule classes verbatim, so checkpoint
parameter names (``indexer.wq_b.weight``, ``indexer.wk_weights_proj.weight``,
``indexer.k_norm.{weight,bias}``) and the ``stacked_params_mapping`` fusion entry
keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.distributed.spmd as xs
from tt_torch.custom_ops import (
    dsa_kernels_available,
    topk_large_indices_mask_invalid_slots,
)
from tt_torch.sharding import sharding_constraint_tensor
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import FullAttentionSpec

from ..logger import tt_init_logger

if TYPE_CHECKING:
    from vllm.config import CacheConfig, VllmConfig

logger = tt_init_logger(__name__)

# ``tt.topk_large_indices`` requires k in [16, 2048] and k % 16 == 0;
# ``tt.sparse_sdpa`` additionally requires topk % k_chunk_size == 0 with
# k_chunk_size a multiple of 32. 32 keeps both satisfiable.
_TOPK_MIN = 16
_TOPK_MAX = 2048
_TOPK_ALIGN = 32
# Largest first: fewer kernel iterations. 128 is the tt-metal default.
_K_CHUNK_CANDIDATES = (128, 64, 32)


def dsa_prefill_uses_sparse(seq_len: int, topk_tokens: int) -> bool:
    """Whether sparse prefill is both possible and worthwhile.

    ``tt.topk_large_indices`` requires ``input.shape[-1] >= k``, and for prefill the
    score is ``[.., seq_len, seq_len]``, so ``seq_len < topk`` cannot even be
    expressed. It is also pointless: top-k over a row with at most ``topk`` finite
    (causally visible) scores keeps *every* visible key, so the sparse result equals
    dense causal attention exactly. Below the threshold the dense
    ``tt.flash_mla_prefill`` kernel is both correct and faster.

    Both arguments are Python ints at trace time, so callers get a *static* branch —
    one graph per token bucket, no dynamic control flow.
    """
    return seq_len >= topk_tokens


def dsa_decode_uses_sparse(max_seq_len: int, topk_tokens: int) -> bool:
    """Whether decode needs sparsity at all.

    ``max_seq_len`` is the bucket's key width (``page_table.shape[1] * block_size``),
    a compile-time constant. When it cannot exceed ``topk``, top-k selects every
    causally visible key and dense ``tt.paged_flash_mla_decode`` is exactly
    equivalent — so the dense path is not an approximation.

    Must be computed from the bucket width, never from ``cache_position``: that is a
    runtime value, and branching on it would either graph-break or bake one step's
    answer into every bucket.
    """
    return max_seq_len > topk_tokens


@dataclass(frozen=True, kw_only=True)
class TTDSAIndexerSpec(FullAttentionSpec):
    """KV cache spec for the DSA indexer K cache: one latent vector per token.

    Deliberately a ``FullAttentionSpec`` and NOT an ``MLAAttentionSpec``, even
    though the storage layout matches MLA's single-tensor cache. Reason:
    ``MLAAttentionSpec.merge`` does not verify that the specs it merges are equal
    (unlike ``FullAttentionSpec.merge``, which asserts every ``AttentionSpec``
    field matches) -- it checks only cache_dtype_str / compress_ratio /
    model_version and then takes ``specs[0].head_size``. With the 576-wide MLA
    specs and this 128-wide one in the same model, ``is_kv_cache_spec_uniform``
    therefore reports True and every layer silently gets one merged spec, so the
    MLA caches come out 128 wide and ttir.paged_fill_cache rejects the fill.

    Being a non-MLA FullAttentionSpec makes that merge assert instead (either
    ``MLAAttentionSpec.merge`` rejects a non-MLA spec, or
    ``FullAttentionSpec.merge`` rejects an MLA one), so grouping falls through to
    ``UniformTypeKVCacheSpecs``, which keeps per-layer specs. That still yields a
    SINGLE kv-cache group -- ``is_uniform_type`` only requires every spec to be a
    ``FullAttentionSpec`` -- so there is one block table and the indexer can reuse
    ``attn_metadata.page_table`` directly.
    """

    @property
    def real_page_size_bytes(self) -> int:
        # One tensor, not the K+V pair FullAttentionSpec assumes.
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )


class _TopKSlot:
    """Per-model holder for the current step's top-k indices.

    Upstream passes indices between the indexer and the attention layer through a
    shared ``[max_num_batched_tokens, index_topk]`` int32 device tensor mutated in
    place — 64 MB at ``max_num_batched_tokens=8192``, hostile to tracing, and
    invisible to Shardy. A plain Python attribute read back in the same dynamo frame
    is equivalent and free.
    """

    __slots__ = ("indices",)

    def __init__(self) -> None:
        self.indices: Optional[torch.Tensor] = None


def _topk_slot_for(topk_indices_buffer: Optional[torch.Tensor]) -> _TopKSlot:
    """One slot per model.

    Every layer is handed the same ``topk_indices_buffer`` object, so hanging the
    slot off it gives cross-layer index sharing (``index_topk_freq`` /
    ``skip_topk``) without a global registry.
    """
    if topk_indices_buffer is None:
        return _TopKSlot()
    slot = getattr(topk_indices_buffer, "_tt_dsa_slot", None)
    if slot is None:
        slot = _TopKSlot()
        topk_indices_buffer._tt_dsa_slot = slot
    return slot


def install_tt_indexer() -> None:
    """Rebind ``deepseek_v2.Indexer`` to ``TTIndexer``. Idempotent.

    A module-attribute rebind rather than a post-hoc module-tree swap (as
    ``overrides.py`` does for RMSNorm): ``Indexer.__init__`` constructs a
    ``DeepseekV32IndexerCache``, which registers itself in
    ``compilation_config.static_forward_context`` and raises on a duplicate prefix,
    so the cache cannot be re-created with TT's dtype/head_dim after the fact. This
    mirrors how ``TTMultiHeadLatentAttentionWrapper`` swaps in ``TTMLAAttention``.
    """
    from vllm.model_executor.models import deepseek_v2

    # TTDSAIndexerSpec is a new spec type, and get_manager_for_kv_cache_spec() does
    # an exact-type lookup (spec_manager_map[type(spec)]), so register it or the
    # KV cache manager raises KeyError. FullAttentionManager is the right block
    # accounting: the indexer key cache holds one entry per token over the whole
    # context, exactly like MLAAttentionSpec, which maps here too.
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        spec_manager_map,
    )

    spec_manager_map.setdefault(TTDSAIndexerSpec, FullAttentionManager)

    if deepseek_v2.Indexer is not TTIndexer:
        deepseek_v2.Indexer = TTIndexer
        logger.info(
            "[TT] Installed TTIndexer — DeepSeek Sparse Attention scoring uses "
            "torch.ops.tt.indexer_score_dsa + torch.ops.tt.topk_large_indices."
        )


class TTIndexer(nn.Module):
    """bf16 DSA lightning indexer for TT (drop-in for ``deepseek_v2.Indexer``)."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config,
        cache_config: "CacheConfig",
        topk_indices_buffer: Optional[torch.Tensor],
        prefix: str = "",
    ) -> None:
        # Deliberately NOT super().__init__() on Indexer: that would build the fp8
        # quantization apparatus and a uint8 K cache we cannot use.
        nn.Module.__init__(self)

        self.config = config
        self.prefix = prefix
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.max_model_len = vllm_config.model_config.max_model_len
        self._block_size = vllm_config.cache_config.block_size

        # Upstream submodule classes verbatim, so checkpoint key names and the
        # wk/weights_proj fusion in stacked_params_mapping keep working.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: one GEMM producing [head_dim + n_head].
        # disable_tp=True keeps full weights on every rank; `shard_model` skips it
        # (sharding the 128-wide key across devices would corrupt every key vector).
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)

        self.topk_indices_buffer = topk_indices_buffer
        self._slot = _topk_slot_for(topk_indices_buffer)

        additional_config = getattr(vllm_config, "additional_config", None) or {}
        self.dsa_mode = str(additional_config.get("dsa_mode", "auto"))

        self.k_chunk_size = self._pick_k_chunk_size(self.topk_tokens)
        # Sparse needs a legal top-k for both ops; otherwise stay dense forever.
        self._sparse_supported = self.dsa_mode != "off" and self._topk_is_legal()

        self.k_cache = None
        if self.dsa_mode != "off":
            # Upstream class, TT-correct arguments: bf16 and the plain index head
            # dim, versus upstream's uint8 and head_dim + head_dim//128*4 (its fp8
            # value + scale layout). Registers itself in static_forward_context, so
            # model_runner.get_kv_cache_spec() picks it up.
            from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

            self.k_cache = DeepseekV32IndexerCache(
                head_dim=self.head_dim,
                dtype=torch.bfloat16,
                prefix=f"{prefix}.k_cache",
                cache_config=cache_config,
            )

        # Resolved once here, never inside forward: dsa_kernels_available() calls
        # into torch_xla._XLAC, which dynamo cannot trace (it aborts compilation of
        # the model graph). The architecture is fixed for the process anyway.
        self._kernels_available = dsa_kernels_available()

        # Same reason: xs.get_global_mesh() reaches into _XLAC. The mesh is built
        # once in model_runner before load_model, so it is available by now and
        # fixed for the process. Used by the prefill sequence split.
        self._mesh = xs.get_global_mesh()
        self._mesh_devices = 1
        if self._mesh is not None:
            for dim in self._mesh.mesh_shape:
                self._mesh_devices *= dim

        self._log_advisories()

    # ------------------------------------------------------------------ #
    # Static configuration
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_k_chunk_size(topk_tokens: int) -> int:
        for candidate in _K_CHUNK_CANDIDATES:
            if topk_tokens % candidate == 0:
                return candidate
        return _K_CHUNK_CANDIDATES[-1]

    def _topk_is_legal(self) -> bool:
        topk = self.topk_tokens
        problems = []
        if not (_TOPK_MIN <= topk <= _TOPK_MAX):
            problems.append(f"index_topk must be in [{_TOPK_MIN}, {_TOPK_MAX}]")
        if topk % _TOPK_MIN != 0:
            problems.append(f"index_topk must be a multiple of {_TOPK_MIN}")
        if topk % _TOPK_ALIGN != 0:
            problems.append(
                f"index_topk must be a multiple of {_TOPK_ALIGN} so a legal "
                "sparse_sdpa k_chunk_size exists"
            )
        if problems:
            logger.warning(
                "[TT] DSA disabled (index_topk=%d): %s. Falling back to dense MLA.",
                topk,
                "; ".join(problems),
            )
            return False
        return True

    @property
    def topk_indices(self) -> Optional[torch.Tensor]:
        """This step's selected key indices, or ``None`` when attention is dense."""
        return self._slot.indices

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
    ) -> Optional[torch.Tensor]:
        """Score, select, and publish the top-k key indices for this step.

        Called by ``MultiHeadLatentAttentionWrapper.forward`` before ``mla_attn``.
        Upstream discards the return value (indices travel via the shared buffer);
        it is returned here for tests.
        """
        attn_metadata = self._attn_metadata()
        kv_cache = getattr(self.k_cache, "kv_cache", None) if self.k_cache else None
        cache_ready = isinstance(kv_cache, torch.Tensor) and kv_cache.numel() > 0

        # Profiling / warmup runs have no metadata and no bound cache.
        if not self._sparse_supported or attn_metadata is None or not cache_ready:
            self._slot.indices = None
            return None

        if attn_metadata.dp_size > 1:
            raise NotImplementedError(
                "DSA does not support data parallelism yet: the per-user "
                "paged_fill_cache / gather loops index the batch globally, without "
                "the `% local_batch` rebasing the non-MLA path performs. Set "
                "additional_config['dsa_mode']='off' or use dp_size=1."
            )
        if getattr(attn_metadata, "chunk_start_idx", None) is not None:
            raise NotImplementedError(
                "DSA does not support chunked prefill: tt.indexer_score_dsa takes a "
                "compile-time chunk_start_idx, but TTMetadata.chunk_start_idx is a "
                "device tensor. MLA already forces chunked prefill off; this "
                "indicates it was re-enabled."
            )

        users = attn_metadata.cache_position.shape[0]
        total_tokens = hidden_states.shape[0]
        assert (
            total_tokens % users == 0
        ), f"total_tokens ({total_tokens}) not divisible by users ({users})."
        seq_len = total_tokens // users

        q_op, k_op, w_op = self._project(
            hidden_states, qr, positions, rotary_emb, users, seq_len
        )

        if seq_len > 1:
            indices = self._forward_prefill(
                q_op, k_op, w_op, kv_cache, attn_metadata, users, seq_len
            )
        else:
            indices = self._forward_decode(
                q_op, k_op, w_op, kv_cache, attn_metadata, users
            )

        self._slot.indices = indices
        return indices

    def _attn_metadata(self):
        """The TTMetadata for this step, or None during profiling."""
        md = get_forward_context().attn_metadata
        if isinstance(md, dict):
            # DeepseekV32IndexerCache is an AttentionLayerBase, so its prefix is a
            # key; fall back to any entry since all layers share one TTMetadata.
            key = getattr(self.k_cache, "prefix", None)
            md = (md.get(key) if key else None) or next(iter(md.values()), None)
        return md

    def _project(self, hidden_states, qr, positions, rotary_emb, users, seq_len):
        """Indexer q / k / gate weights in the layouts the DSA ops expect.

        Mirrors upstream's projection exactly except for the FP8 quantization: TT
        stays in bf16, so there is no ``q_scale`` to fold into the gate weights.
        """
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        # Fused wk + weights_proj: one GEMM, then split.
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # RoPE can introduce extra leading dims during compilation; reshape back to
        # token-flattened shapes.
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)  # [tokens, n_head, head_dim]
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)  # [tokens, head_dim]

        # Upstream folds q_scale in here; with no quantization it is absent.
        weights = weights * self.softmax_scale * self.n_head**-0.5

        q_op = (
            q.view(users, seq_len, self.n_head, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )  # [users, n_head, seq_len, head_dim]
        k_op = (
            k.view(users, seq_len, 1, self.head_dim).transpose(1, 2).contiguous()
        )  # [users, 1, seq_len, head_dim]
        w_op = (
            weights.view(users, seq_len, self.n_head, 1).transpose(1, 2).contiguous()
        )  # [users, n_head, seq_len, 1]
        return q_op, k_op, w_op

    # ------------------------------------------------------------------ #
    # Index selection
    # ------------------------------------------------------------------ #
    def _select(self, score: torch.Tensor, visible_count: torch.Tensor):
        """top-k over ``score``, with the architecture-appropriate index repair.

        On Blackhole the TTNN kernel implements the ``-inf`` -> sentinel contract
        itself and ``tt.sparse_sdpa``'s reader derives the valid-key count from the
        first sentinel, so the uint32 output is passed through untouched. Elsewhere
        the composite inlines a plain ``ttir.topk`` that returns ordinary indices
        for ``-inf`` ties, which would let sparse attention read future tokens — so
        the invalid tail is marked explicitly. See
        ``topk_large_indices_mask_invalid_slots``.
        """
        indices = torch.ops.tt.topk_large_indices(score, self.topk_tokens)
        if self._kernels_available:
            return indices
        return topk_large_indices_mask_invalid_slots(indices, visible_count)

    def _prefill_seq_shard_plan(self, seq_len: int) -> tuple[Optional[tuple], int]:
        """``(partition_spec, padded_seq_len)`` for splitting the indexer query.

        A ``None`` spec means "leave the query replicated", which is correct on a
        single device (rank 0, so the op's per-device offset vanishes) and is the
        historical behaviour.

        On a mesh the query MUST be split: ``indexer_score_dsa`` derives a
        per-device rank from q's device coords and requires
        ``T >= (max_rank + 1) * Sq``, so a replicated query aborts at runtime.
        Two divisibility conditions have to hold for a legal split:
          * ``seq_len % devices == 0`` -- the op derives one Sq for every rank, so
            a ragged split would misdescribe the tail device.
          * ``Sq % 32 == 0`` -- Sq is a tile height (the op asserts
            ``Sq % TILE_HEIGHT == 0``).
        Together: ``seq_len`` must be a multiple of ``32 * devices``.

        Buckets that are not are **padded up**, never left replicated. Leaving them
        replicated would emit an op that passes the TTNN verifier (the shapes are
        fine) and then aborts inside the device op -- a hard crash rather than a
        graceful degrade. Padding is not hypothetical: on 8 devices this requires
        multiples of 256, so a 128-token bucket with ``index_topk=128`` clears
        ``dsa_prefill_uses_sparse`` and would otherwise die.

        ``seq_len`` is a Python int at trace time, so this is a static branch: one
        graph per prefill bucket, same as the sparse predicates.
        """
        if self._mesh is None or self._mesh_devices <= 1:
            return None, seq_len
        align = 32 * self._mesh_devices
        padded = ((seq_len + align - 1) // align) * align
        # Compound axis in mesh-axis order => row-major linearization over ALL
        # devices, which is what the op's seq_shard_axes=[] / cluster_axis=None
        # flat device rank assumes.
        return (None, None, tuple(self._mesh.axis_names), None), padded

    def _forward_prefill(
        self, q_op, k_op, w_op, kv_cache, attn_metadata, users, seq_len
    ):
        # Persist this chunk's indexer K. Accumulate into a local so `kv_cache`
        # keeps referencing the bound buffer (mirrors the MLA prefill fill loop).
        filled = kv_cache
        for batch_idx in range(users):
            filled = torch.ops.tt.paged_fill_cache(
                filled,
                k_op[batch_idx : batch_idx + 1],
                attn_metadata.fill_page_table,
                batch_idx=torch.tensor(
                    [batch_idx], dtype=torch.int32, device=kv_cache.device
                ),
            )
        kv_cache.copy_(filled)

        if not dsa_prefill_uses_sparse(seq_len, self.topk_tokens):
            return None

        seq_spec, padded_len = self._prefill_seq_shard_plan(seq_len)
        key_op = k_op
        if padded_len != seq_len:
            # Pad q/weights/key together along seq. Padding the key too is required,
            # not cosmetic: the op checks max_cs + Sq <= T against the KEY length, so
            # padding only the query would push the window past T and abort. Real row
            # s still sees only keys [0, s] (chunk_start_idx == 0), and every padded
            # key column t >= seq_len is causally visible ONLY to padded rows
            # s >= t >= seq_len -- which are sliced off below. So the causal mask
            # already isolates the padding; no extra masking is needed.
            pad = padded_len - seq_len
            q_op = F.pad(q_op, (0, 0, 0, pad))
            w_op = F.pad(w_op, (0, 0, 0, pad))
            key_op = F.pad(k_op, (0, 0, 0, pad))

        # Row s sees keys [0, s] (chunk_start_idx == 0), so its visible count is s+1.
        visible_count = (
            torch.arange(padded_len, dtype=torch.int32, device=q_op.device) + 1
        ).view(1, 1, padded_len, 1)

        # Shard the query sequence across the mesh when it divides evenly. The
        # indexer_score_dsa kernel models q as sequence-parallel: it derives a
        # per-device rank from q's device coords and requires
        # T >= (max_rank + 1) * Sq. A sequence-REPLICATED q on N devices therefore
        # reports rank N-1 with Sq == T and aborts with "fullest-device chunk window
        # ... exceeds T". Sharding seq gives Sq == T / N, satisfying the bound with
        # equality -- exactly the layout the op's chunk_start deduction assumes, so
        # nothing extra has to be plumbed through tt-mlir. K stays replicated
        # (the op needs all T keys) and so do the indexer's heads, since the op sums
        # over the Hi resident on the device.
        #
        # sharding_constraint_tensor, not xs.mark_sharding: this runs inside the
        # traced graph, and mark_sharding/get_global_mesh reach into _XLAC, which
        # dynamo refuses to trace ("Attempted to call function marked as skipped").
        # It is functional, hence the reassignments.
        if seq_spec is not None:
            q_op = sharding_constraint_tensor(q_op, self._mesh, seq_spec)
            w_op = sharding_constraint_tensor(w_op, self._mesh, seq_spec)
            # visible_count is indexed by query row, so it must follow q's split.
            # A sharded iota is materialized per shard with globally-correct values,
            # keeping row s's visible count s+1 in GLOBAL coordinates.
            visible_count = sharding_constraint_tensor(
                visible_count, self._mesh, seq_spec
            )

        # Both DSA ops require batch == 1, so score/select one user at a time.
        per_user = []
        for u in range(users):
            score = torch.ops.tt.indexer_score_dsa(
                query=q_op[u : u + 1],
                key=key_op[u : u + 1],
                weights=w_op[u : u + 1],
                chunk_start_idx=0,
            )  # [1, 1, padded_len/shards, padded_len]
            per_user.append(self._select(score, visible_count))
        indices = torch.cat(per_user, dim=0)  # [users, 1, padded_len, topk]

        if seq_spec is not None:
            # sparse_sdpa is sharded on heads, not sequence, so every device needs
            # indices for every query row: state the replication explicitly rather
            # than leaving the resharding for the partitioner to infer.
            indices = sharding_constraint_tensor(
                indices, self._mesh, (None, None, None, None)
            )
        # Drop the padded rows. Done after the gather so the slice is on a replicated
        # tensor rather than across the sharded seq dim.
        return indices[:, :, :seq_len, :]

    def _forward_decode(self, q_op, k_op, w_op, kv_cache, attn_metadata, users):
        # Append this token's indexer K at the current position.
        updated = torch.ops.tt.paged_update_cache(
            kv_cache,
            k_op.transpose(0, 1),  # [1, users, 1, head_dim]
            attn_metadata.cache_position,
            attn_metadata.page_table,
        )
        kv_cache.copy_(updated)

        max_seq_len = attn_metadata.page_table.shape[1] * self._block_size
        if not dsa_decode_uses_sparse(max_seq_len, self.topk_tokens):
            # Dense paged decode is exactly equivalent here, and faster.
            return None
        if self.dsa_mode == "dense_decode":
            # Explicit opt-OUT: keep dense decode even though the context can
            # exceed index_topk, so all entries participate instead of just the
            # selected top-k. Deviates from the sparsity the model was trained
            # with; _log_advisories() warned at construction (nothing may log from
            # inside the traced forward).
            return None

        cache_position = attn_metadata.cache_position
        positions = torch.arange(
            max_seq_len, dtype=torch.int32, device=q_op.device
        ).view(1, 1, 1, max_seq_len)

        per_user = []
        for u in range(users):
            # Gather this user's indexer K out of the paged cache into logical
            # order; indexer_score_dsa needs a dense [1, 1, T, D] key and batch 1.
            blocks = torch.index_select(
                kv_cache, 0, attn_metadata.page_table[u].to(torch.int64)
            )
            key_u = blocks.reshape(1, 1, max_seq_len, self.head_dim)

            # chunk_start_idx is a compile-time attribute masking t > start + s, but
            # with seq_len == 1 the real bound is the *runtime, per-user*
            # cache_position. Make the op's own mask a no-op and apply the real
            # bound below. Positions past cache_position hold prefill padding, so
            # this mask is load-bearing, not cosmetic.
            score = torch.ops.tt.indexer_score_dsa(
                query=q_op[u : u + 1],
                key=key_u,
                weights=w_op[u : u + 1],
                chunk_start_idx=max_seq_len - 1,
            )  # [1, 1, 1, max_seq_len]

            cur_pos = cache_position[u].view(1, 1, 1, 1)
            visible = positions <= cur_pos
            score = score + torch.where(
                visible,
                torch.zeros((), dtype=score.dtype, device=score.device),
                torch.full((), float("-inf"), dtype=score.dtype, device=score.device),
            )
            per_user.append(self._select(score, (cur_pos + 1).to(torch.int32)))
        return torch.cat(per_user, dim=0)  # [users, 1, 1, topk]

    def _log_advisories(self) -> None:
        """Emit configuration advisories once, at construction.

        Deliberately not from ``forward``: that runs inside the traced graph, and
        dynamo only tolerates the logging methods registered via
        ``ignore_logging_methods`` (``logger.info``) -- a ``logger.warning`` there
        aborts compilation of the whole model. Everything reported here is static
        anyway.
        """
        if not self._sparse_supported:
            return

        if not self._kernels_available:
            logger.warning(
                "[TT] DSA kernels are Blackhole-only; on this architecture the "
                "composites inline their primitive decompositions. Results stay "
                "correct (the top-k index repair supplies the sentinel contract the "
                "kernel would), but performance will be well below the kernel path."
            )

        # Worst-case decode bucket for this config, matching how
        # dsa_decode_uses_sparse computes it from the page table at runtime.
        max_bucket = (-(-self.max_model_len // self._block_size)) * self._block_size
        if not dsa_decode_uses_sparse(max_bucket, self.topk_tokens):
            return

        if self.dsa_mode == "dense_decode":
            logger.warning(
                "[TT] DSA: dsa_mode='dense_decode' with a context window reaching "
                "%d, beyond index_topk (%d). Decode will attend to ALL cached "
                "entries rather than the selected top-k, which deviates from the "
                "sparsity this model was trained with. Use dsa_mode='auto' for "
                "correct sparse decode.",
                max_bucket,
                self.topk_tokens,
            )
        else:
            logger.warning(
                "[TT] DSA: the decode context window can reach %d, beyond "
                "index_topk (%d), so decode will use sparse attention over the "
                "selected top-k. That is correct but currently SLOWER than dense "
                "decode: tt.sparse_sdpa cannot read a paged TILE cache, so each "
                "step gathers the latent cache (O(context) traffic even though the "
                "attention math is O(top-k)). Exposing tt-metal's cache_batch_idx "
                "on TTNN_SparseSdpaOp would remove the gather.",
                max_bucket,
                self.topk_tokens,
            )
