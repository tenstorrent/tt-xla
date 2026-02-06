# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility classes for Kimi K2 model testing, including MLA cache implementations."""

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
        Lazy initialization of the compressed KV and RoPE key tensors. This allows capturing
        all properties (dtype, device, dimensions) at runtime, which avoids device/dtype
        movements later that could break static dynamo addresses.

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly,
        which will call this function ahead-of-time (required for `torch.export`).

        For `compile`, as we internally don't compile prefill, this is guaranteed to have
        been called already when compiling decode. Compiling prefill is supported but without
        guarantees for certain options (e.g. `mode="reduce-overhead"` with cuda graphs).
        """
        self.max_batch_size, _, _, self.kv_lora_rank = compressed_kv.shape
        self.max_batch_size, _, _, self.pe_rank = k_pe.shape
        self.dtype, self.device = compressed_kv.dtype, compressed_kv.device

        # Compressed KV latent: [batch, 1, max_seq, kv_lora_rank]
        # The head dim is 1 because MLA uses a single shared latent across all heads
        self.compressed_kv = torch.zeros(
            (self.max_batch_size, 1, self.max_cache_len, self.kv_lora_rank),
            dtype=self.dtype,
            device=self.device,
        )
        # Decoupled RoPE keys: [batch, 1, max_seq, qk_rope_head_dim]
        # Stored separately because RoPE is applied only to this component
        self.k_pe = torch.zeros(
            (self.max_batch_size, 1, self.max_cache_len, self.pe_rank),
            dtype=self.dtype,
            device=self.device,
        )

        # Parent class expects keys and values, so we alias the compressed KV and RoPE key tensors to them
        self.keys = self.compressed_kv
        self.values = self.k_pe

        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer, preventing compiled graph
        # breaks when updating the cache. However, it is not supported when tracing the graph, so we skip it in this case.
        # As prefill should never be compiled, this is not an issue and it will still be run (except when users compile
        # prefill explicitly, but this should be avoided!)
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
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            compressed_kv (`torch.Tensor`): The new compressed key and value states to cache.
            k_pe (`torch.Tensor`): The new key position embedding states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The compressed key and value states and key position embedding states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(compressed_kv, k_pe)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (compressed_kv.shape[-2] == self.max_cache_len)
        cache_position = (
            cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        )
        cache_position = (
            cache_position
            if cache_position is not None
            else torch.arange(compressed_kv.shape[-2], device=self.device)
        )

        # Update the cache
        try:
            self.compressed_kv.index_copy_(2, cache_position, compressed_kv)
            self.k_pe.index_copy_(2, cache_position, k_pe)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.compressed_kv[:, :, cache_position, :] = compressed_kv
            self.k_pe[:, :, cache_position, :] = k_pe
        return self.compressed_kv, self.k_pe

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        return (
            (self.compressed_kv[0, 0].any(dim=-1)).sum() if self.is_initialized else 0
        )

    def get_max_cache_shape(self) -> int:
        """Return the maximum cache shape of the cache"""
        return self.max_cache_len


class MLACache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`. It will check the `config`
    for potential hybrid cache structure, and initialize each layer accordingly.

    See `Cache` for details on common methods that are implemented by all cache classes.

    Args:
        config (`PretrainedConfig`):
            The config of the model for which this Cache will be used. It will be used to check for sliding
            or hybrid layer structure, and initialize each layer accordingly.
        max_cache_len (`int`):
            The maximum number of tokens that this Cache should hold.
        offloading (`bool`, *optional*, defaults to `False`):
            Whether to perform offloading of the layers to `cpu`, to save GPU memory.
        offload_only_non_sliding (`bool`, *optional*, defaults to `True`):
            If `offloading` is `True`, this further decides if only the non-sliding layers will be offloaded (because
            usually the sliding layers are small in size, so there is no need to offload them, and skipping it is faster).

    Example:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

    >>> # Prepare a cache class and pass it to model's forward
    >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
    >>> max_generated_length = inputs.input_ids.shape[1] + 10
    >>> past_key_values = StaticCache(config=model.config, max_cache_len=max_generated_length)
    >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
    >>> outputs.past_key_values # access cache filled with key/values from generation
    StaticCache()
    ```
    """

    # Pass-in kwargs as well to avoid crashing for BC (it used more arguments before)
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
        # If `layer_types` is not explicitly provided, infer if the model is fully sliding
        if layer_types is None:
            sliding_window = getattr(config, "sliding_window", None)
            chunked_attention = getattr(config, "attention_chunk_size", None)
            assert (
                sliding_window is None and chunked_attention is None
            ), "Sliding window and chunked attention are not supported for MLA"
            layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]
        # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
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
        """Converts the `MLACache` instance into its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer in self.layers:
            legacy_cache += (
                layer.keys,
                layer.values,
            )
        return legacy_cache
