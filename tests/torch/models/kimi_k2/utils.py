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
    A static cache layer that stores the key and value states as static tensors of shape `[batch_size, num_heads, max_cache_len), head_dim]`.
    It lazily allocates its full backing tensors, and then mutates them in-place. Built for `torch.compile` support.

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
        Lazy initialization of the keys and values tensors. This allows to get all properties (dtype, device,
        num_heads in case of TP etc...) at runtime directly, which is extremely practical as it avoids moving
        devices, dtypes etc later on for each `update` (which could break the static dynamo addresses as well).

        If this is unwanted, one can call `early_initialization(...)` on the Cache directly, which will call this
        function ahead-of-time (this is required for `torch.export` for example). Note that for `compile`, as we
        internally don't compile the prefill, this is guaranteed to have been called already when compiling.
        If compiling the prefill as well, e.g. calling `model.compile(...)` before `generate` with a static cache,
        it is still supported in general, but without guarantees depending on the compilation options (e.g. cuda graphs,
        i.e. `mode="reduce-overhead"` is known to fail). But it will in general work correctly, and prefill should
        not be compiled anyway for performances!
        """
        self.max_batch_size, _, _, self.kv_lora_rank = compressed_kv.shape
        self.max_batch_size, _, _, self.pe_rank = k_pe.shape
        self.dtype, self.device = compressed_kv.dtype, compressed_kv.device

        self.keys = torch.zeros(
            (self.max_batch_size, 1, self.max_cache_len, self.kv_lora_rank),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_batch_size, 1, self.max_cache_len, self.pe_rank),
            dtype=self.dtype,
            device=self.device,
        )
        # Note: `mark_static_address` is used to tag the cache as a fixed data pointer, preventing compiled graph
        # breaks when updating the cache. However, it is not supported when tracing the graph, so we skip it in this case.
        # As prefill should never be compiled, this is not an issue and it will still be run (except when users compile
        # prefill explicitly, but this should be avoided!)
        if not torch.compiler.is_compiling():
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)

        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = (
            cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        )
        cache_position = (
            cache_position
            if cache_position is not None
            else torch.arange(key_states.shape[-2], device=self.device)
        )

        # Update the cache
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.keys[:, cache_position, :] = key_states
            self.values[:, :, cache_position, :] = value_states
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Return the length and offset of the cache, used to generate the attention mask"""
        kv_offset = 0
        kv_length = self.max_cache_len
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the sequence length of the cached states."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        return (self.keys[0, 0].any(dim=-1)).sum() if self.is_initialized else 0

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
            if getattr(config, "sliding_window", None) is not None:
                layer_types = [
                    "sliding_attention" for _ in range(config.num_hidden_layers)
                ]
            elif getattr(config, "attention_chunk_size", None) is not None:
                layer_types = [
                    "chunked_attention" for _ in range(config.num_hidden_layers)
                ]
            else:
                layer_types = [
                    "full_attention" for _ in range(config.num_hidden_layers)
                ]
        # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
        if hasattr(config, "num_kv_shared_layers"):
            layer_types = layer_types[: -config.num_kv_shared_layers]

        layers = []
        for layer_type in layer_types:
            assert layer_type not in ["sliding_attention", "chunked_attention"]
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
