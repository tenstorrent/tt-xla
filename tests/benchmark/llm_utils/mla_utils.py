# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MLACache implementation and initializer for MLA (Multi-head Latent Attention) models.

Copied from tests/torch/models/kimi_k2/utils.py with an added init_mla_cache helper.
"""

from typing import Any, Optional

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import Cache, CacheLayerMixin


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

        # Parent class expects keys and values, so we alias the compressed KV and RoPE key tensors to them
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
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def build_causal_mask(
        self,
        cache_position: torch.Tensor,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a 4D causal attention mask of shape (batch, 1, q_len, max_cache_len)."""
        key_idx = torch.arange(self.max_cache_len, dtype=cache_position.dtype, device=device)
        fill_value = torch.finfo(dtype).min
        causal = (key_idx.unsqueeze(0) > cache_position.unsqueeze(1)).to(dtype)
        return (causal * fill_value).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        return (
            (self.compressed_kv[0, 0].any(dim=-1)).sum() if self.is_initialized else 0
        )

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class MLACache(Cache):
    """
    Static cache for MLA (Multi-head Latent Attention) models such as Kimi K2 / DeepSeek.

    Stores compressed KV latents and decoupled RoPE keys instead of standard K/V pairs.
    Compatible with `torch.compile`.

    Args:
        config (`PretrainedConfig`): Model config.
        max_cache_len (`int`): Maximum sequence length to cache.
        offloading (`bool`): Whether to offload layers to CPU.
        offload_only_non_sliding (`bool`): Offload only non-sliding layers when offloading.
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

        layers = [MLAStaticLayer(max_cache_len=max_cache_len) for _ in layer_types]

        super().__init__(
            layers=layers,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor]]:
        """Converts the `MLACache` instance into its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += (
                layer.keys,
                layer.values,
            )
        return legacy_cache


def init_mla_cache(
    *,
    config,
    batch_size: int,
    max_cache_len: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> MLACache:
    """Initialize an MLACache with pre-allocated backing tensors on the given device.

    Analogous to init_static_cache + early_initialization for StaticCache.
    Pre-initializes each MLAStaticLayer so that transfer_to_device can move
    the tensors before any model forward pass runs.

    Args:
        config: Model config (PretrainedConfig).
        batch_size: Batch size.
        max_cache_len: Maximum sequence length to cache.
        device: Device to allocate tensors on.
        dtype: Tensor dtype.

    Returns:
        Fully initialized MLACache instance.
    """
    cache = MLACache(config=config, max_cache_len=max_cache_len)

    text_config = config.get_text_config(decoder=True)
    kv_lora_rank = text_config.kv_lora_rank
    qk_rope_head_dim = text_config.qk_rope_head_dim

    dummy_kv = torch.zeros(
        (batch_size, 1, 1, kv_lora_rank), dtype=dtype, device=device
    )
    dummy_pe = torch.zeros(
        (batch_size, 1, 1, qk_rope_head_dim), dtype=dtype, device=device
    )

    for layer in cache.layers:
        layer.lazy_initialization(dummy_kv, dummy_pe)

    return cache
