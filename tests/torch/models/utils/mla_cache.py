# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""General MLA cache utilities shared across MLA-based models.

Classes
-------
MLAStaticLayer / MLACache
    4-D cache (Kimi K2 / HuggingFace-compatible):
    tensors shaped [batch, 1, max_seq, dim], two tensors per layer
    (compressed_kv, k_pe).  Designed to satisfy the transformers Cache
    interface so the model can be used with HuggingFace generate().

DeepSeekMLAStaticLayer / DeepSeekMLACache
    3-D cache for the custom DeepSeek Transformer in modified_model.py:
    tensors shaped [batch, max_seq, dim] (no singleton head dim), three
    tensors per layer (kv, pe, optional k for the Indexer).  Does NOT
    inherit from the HuggingFace Cache class because modified_model.py
    uses a bespoke forward() signature.  current_pos is tracked here so
    that DeepSeekV32ForCausalLM.forward() can derive start_pos from it.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import Cache, CacheLayerMixin

# ---------------------------------------------------------------------------
# Kimi K2 / HuggingFace-compatible cache (unchanged from kimi_k2/utils.py)
# ---------------------------------------------------------------------------


class MLAStaticLayer(CacheLayerMixin):
    """
    A static cache layer for Multi-head Latent Attention (MLA) that stores compressed KV latents
    and rotary position embeddings as static tensors of shape `[batch_size, 1, max_cache_len, dim]`.

    Unlike standard MHA caches which store separate K and V tensors of equal dimension,
    MLA caches the low-rank compressed KV representation and the decoupled RoPE keys separately,
    which have different head dimensions.

    It lazily allocates its full backing tensors, and then mutates them in-place.
    Built for `torch.compile` support.

    Args:
        max_cache_len (`int`):
            Maximum number of tokens that can be stored, used for tensor preallocation.
    """

    is_compileable = True
    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len

    def lazy_initialization(self, compressed_kv: torch.Tensor, k_pe: torch.Tensor):
        """
        Lazy initialization of the compressed KV and RoPE key tensors.
        """
        self.max_batch_size, _, _, self.kv_lora_rank = compressed_kv.shape
        self.max_batch_size, _, _, self.pe_rank = k_pe.shape
        self.dtype, self.device = compressed_kv.dtype, compressed_kv.device

        # Compressed KV latent: [batch, 1, max_seq, kv_lora_rank]
        self.compressed_kv = torch.zeros(
            (self.max_batch_size, 1, self.max_cache_len, self.kv_lora_rank),
            dtype=self.dtype,
            device=self.device,
        )
        # Decoupled RoPE keys: [batch, 1, max_seq, qk_rope_head_dim]
        self.k_pe = torch.zeros(
            (self.max_batch_size, 1, self.max_cache_len, self.pe_rank),
            dtype=self.dtype,
            device=self.device,
        )

        self.keys = self.compressed_kv
        self.values = self.k_pe

        if not torch.compiler.is_compiling():
            torch._dynamo.mark_static_address(self.compressed_kv)
            torch._dynamo.mark_static_address(self.k_pe)

        self.is_initialized = True

    def update(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the key and value caches in-place."""
        if not self.is_initialized:
            self.lazy_initialization(compressed_kv, k_pe)

        cache_position = (
            cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        )
        cache_position = (
            cache_position
            if cache_position is not None
            else torch.arange(compressed_kv.shape[-2], device=self.device)
        )

        try:
            self.compressed_kv.index_copy_(2, cache_position, compressed_kv)
            self.k_pe.index_copy_(2, cache_position, k_pe)
        except NotImplementedError:
            self.compressed_kv[:, :, cache_position, :] = compressed_kv
            self.k_pe[:, :, cache_position, :] = k_pe
        return self.compressed_kv, self.k_pe

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return (
            (self.compressed_kv[0, 0].any(dim=-1)).sum() if self.is_initialized else 0
        )

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len


class MLACache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        max_cache_len: int,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
        **kwargs,
    ):
        config = config.get_text_config(decoder=True)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            sliding_window = getattr(config, "sliding_window", None)
            chunked_attention = getattr(config, "attention_chunk_size", None)
            assert (
                sliding_window is None and chunked_attention is None
            ), "Sliding window and chunked attention are not supported for MLA"
            layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]
        if hasattr(config, "num_kv_shared_layers"):
            layer_types = layer_types[: -config.num_kv_shared_layers]

        layers = []
        for _ in layer_types:
            layer = MLAStaticLayer(max_cache_len=max_cache_len)
            layers.append(layer)

        super().__init__(
            layers=layers,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor]]:
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += (
                layer.keys,
                layer.values,
            )
        return legacy_cache


# ---------------------------------------------------------------------------
# DeepSeek extension — 3-D tensors, optional Indexer k cache, current_pos
# ---------------------------------------------------------------------------


class DeepSeekMLAStaticLayer(MLAStaticLayer):
    """MLA static cache layer for the custom DeepSeek Transformer.

    Extends ``MLAStaticLayer`` with two differences required by
    modified_model.py:

    1. **3-D tensors** — [batch, max_seq, dim] rather than the 4-D
       [batch, 1, max_seq, dim] used by Kimi K2.  The Indexer / MLA
       attention reads/writes via ``[:bsz, start:end]`` slices, not
       ``[:bsz, :, start:end]``, so the singleton head dimension is omitted.

    2. **Optional third tensor** (``k``) — the Indexer caches its own
       key tensor separately from the MLA kv/pe caches.  ``index_head_dim``
       must be set at construction time (via ``from_model_args``) so that
       ``lazy_initialization`` allocates ``k`` on the first MLA update,
       before the Indexer writes into it.

    ``update()`` uses ``index_copy_`` on dim 1 (sequence dim in 3-D)
    instead of dim 2 used by the parent.
    """

    def __init__(self, max_cache_len: int, index_head_dim: Optional[int] = None):
        super().__init__(max_cache_len)
        # Stored at construction so lazy_initialization can allocate k without
        # needing the Indexer to pass k on the first update call.
        self._index_head_dim = index_head_dim
        self.k = None

    def lazy_initialization(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        k: Optional[torch.Tensor] = None,
    ):
        """Allocate 3-D backing buffers inferred from the first update call.

        ``k`` is always allocated when ``index_head_dim`` was supplied at
        construction, even if no ``k`` tensor is passed here, because
        ``MLA.forward()`` triggers this call before the Indexer runs.

        Args:
            compressed_kv: [batch, seqlen, kv_lora_rank]  (3-D)
            k_pe:           [batch, seqlen, qk_rope_head_dim]  (3-D)
            k:              [batch, seqlen, index_head_dim] or None
        """
        batch_size, _, kv_lora_rank = compressed_kv.shape
        _, _, pe_rank = k_pe.shape
        self.dtype, self.device = compressed_kv.dtype, compressed_kv.device

        self.compressed_kv = torch.zeros(
            batch_size,
            self.max_cache_len,
            kv_lora_rank,
            dtype=self.dtype,
            device=self.device,
        )
        self.k_pe = torch.zeros(
            batch_size,
            self.max_cache_len,
            pe_rank,
            dtype=self.dtype,
            device=self.device,
        )

        # Allocate k if index_head_dim is known (set at construction) OR inferred.
        index_head_dim = (
            self._index_head_dim
            if self._index_head_dim is not None
            else (k.shape[-1] if k is not None else None)
        )
        self.k = (
            torch.zeros(
                batch_size,
                self.max_cache_len,
                index_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            if index_head_dim is not None
            else None
        )

        # Keep parent aliases so get_seq_length() still works.
        self.keys = self.compressed_kv
        self.values = self.k_pe

        if not torch.compiler.is_compiling():
            torch._dynamo.mark_static_address(self.compressed_kv)
            torch._dynamo.mark_static_address(self.k_pe)
            if self.k is not None:
                torch._dynamo.mark_static_address(self.k)

        self.is_initialized = True

    def update(
        self,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
        k: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Write kv/pe (and optionally k) and return the full 3-D cache tensors.

        Args:
            compressed_kv:  [batch, seqlen, kv_lora_rank]
            k_pe:           [batch, seqlen, qk_rope_head_dim]
            cache_kwargs:   dict with ``cache_position`` (1-D int64, length seqlen)
            k:              [batch, seqlen, index_head_dim] or None

        Returns:
            (compressed_kv_full, k_pe_full, k_full) — read ``[:bsz, :end_pos]``
        """
        if not self.is_initialized:
            self.lazy_initialization(compressed_kv, k_pe, k)

        cache_position = (
            cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        )
        if cache_position is None:
            cache_position = torch.arange(compressed_kv.shape[1], device=self.device)

        # index_copy_ on dim=1 (sequence dim in 3-D tensors).
        self.compressed_kv.index_copy_(1, cache_position, compressed_kv)
        self.k_pe.index_copy_(1, cache_position, k_pe)

        return self.compressed_kv, self.k_pe, self.k

    def get_seq_length(self) -> int:
        return (self.compressed_kv[0].any(dim=-1)).sum() if self.is_initialized else 0

    def to(self, device) -> "DeepSeekMLAStaticLayer":
        if self.is_initialized:
            self.compressed_kv = self.compressed_kv.to(device)
            self.k_pe = self.k_pe.to(device)
            if self.k is not None:
                self.k = self.k.to(device)
            self.keys = self.compressed_kv
            self.values = self.k_pe
        return self

    def zero_(self) -> "DeepSeekMLAStaticLayer":
        if self.is_initialized:
            self.compressed_kv.zero_()
            self.k_pe.zero_()
            if self.k is not None:
                self.k.zero_()
        return self


class DeepSeekMLACache:
    """External KV cache for the custom DeepSeek Transformer.

    Passed as ``past_key_values`` to ``DeepSeekV32ForCausalLM.forward()``.
    The loader detects it via ``isinstance(past_key_values, DeepSeekMLACache)``
    and uses ``current_pos`` as ``start_pos`` for the underlying Transformer
    call, advancing it by ``seqlen`` after each forward.

    Does not inherit from HuggingFace ``Cache`` because the custom model
    does not implement the HuggingFace ``forward()`` signature.
    """

    def __init__(self, layers: list[DeepSeekMLAStaticLayer]):
        self.layers = layers
        self.current_pos: int = 0

    def reset(self) -> "DeepSeekMLACache":
        """Zero all buffers and reset the position counter."""
        self.current_pos = 0
        for layer in self.layers:
            layer.zero_()
        return self

    def to(self, device) -> "DeepSeekMLACache":
        for layer in self.layers:
            layer.to(device)
        return self

    @staticmethod
    def from_model_args(
        args,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ) -> "DeepSeekMLACache":
        """Build a cache sized for the given ModelArgs.

        Args:
            args:        ModelArgs instance (available as ``loader._args``
                         after calling ``loader.load_model()``).
            batch_size:  Defaults to ``args.max_batch_size``.
            max_seq_len: Defaults to ``args.max_seq_len``.
        """
        batch_size = batch_size if batch_size is not None else args.max_batch_size
        max_seq_len = max_seq_len if max_seq_len is not None else args.max_seq_len
        index_head_dim = args.index_head_dim if args.index_n_heads > 0 else None
        layers = [
            DeepSeekMLAStaticLayer(
                max_cache_len=max_seq_len,
                index_head_dim=index_head_dim,
            )
            for _ in range(args.n_layers)
        ]
        return DeepSeekMLACache(layers)
