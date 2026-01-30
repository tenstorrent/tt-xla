# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
import numpy as np
from torch_xla.distributed.spmd import Mesh
from configuration_deepseek import DeepseekV3Config
from modeling_deepseek import DeepseekV3ForCausalLM, DeepseekV3Attention, DeepseekV3DecoderLayer
from transformers import PretrainedConfig

# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# import accelerate
# from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM, DeepseekV3Attention, DeepseekV3DecoderLayer

from infra import Framework, run_graph_test
from tests.utils import failed_ttmlir_compilation
from transformers.cache_utils import Cache, CacheLayerMixin

from typing import Any, Optional


@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "'ttir.concat' op Output tensor dimension 0 does not match the sum of input tensor dimensions: 1 vs. 32. "
    )
)
def test_kimi_k2_single_layer():
    xr.set_device_type("TT")

    # Load full Kimi K2 config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)

    # Override for single layer testing
    config.num_hidden_layers = 1
    config.use_cache = False

    model = DeepseekV3ForCausalLM(config)

    model = model.to(torch.bfloat16)
    model = model.eval()

    compiled_model = torch.compile(model, backend="tt")

    batch_size = 1
    seq_len = 32
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    device = torch_xla.device()
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    with torch.no_grad():
        output = compiled_model(tokens)
        output.to("cpu")


def test_kimi_k2_attention_prefill():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    # Load full Kimi K2 config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)


    attention = DeepseekV3Attention(config, layer_idx=0)
    attention = attention.to(torch.bfloat16)

    batch_size = 1
    seq_len = 32
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    device = torch_xla.device()
    attention = attention.to(device)

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    hidden_states = hidden_states.to(device)
    attention_mask = attention_mask.to(device)


    def get_shard_spec(attention, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = (None, None, "batch")
        shard_specs[attention.q_b_proj.weight] = ("model", None)
        shard_specs[attention.kv_b_proj.weight] = ("model", None)
        shard_specs[attention.o_proj.weight] = ("batch", "model")

        # Consume hidden states, TP on batch dimension
        shard_specs[attention.q_a_proj.weight] = (None, "batch")
        shard_specs[attention.kv_a_proj_with_mqa.weight] = (None, "batch")
        return shard_specs


    run_graph_test(
        attention,
        [hidden_states, attention_mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


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
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
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
                layer_types = ["sliding_attention" for _ in range(config.num_hidden_layers)]
            elif getattr(config, "attention_chunk_size", None) is not None:
                layer_types = ["chunked_attention" for _ in range(config.num_hidden_layers)]
            else:
                layer_types = ["full_attention" for _ in range(config.num_hidden_layers)]
        # Some models have shared layers thus no cache is needed for them (e.g. Gemma3n)
        if hasattr(config, "num_kv_shared_layers"):
            layer_types = layer_types[: -config.num_kv_shared_layers]

        layers = []
        for layer_type in layer_types:
            assert layer_type not in ["sliding_attention", "chunked_attention"]
            layer = MLAStaticLayer(max_cache_len=max_cache_len)
            layers.append(layer)

        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

def test_kimi_k2_attention_decode():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    # breakpoint()
    # config = AutoConfig.from_pretrained("moonshotai/Kimi-K2-Base",  trust_remote_code=True)
    # with accelerate.init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config,  trust_remote_code=True)

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)
    config.num_hidden_layers = 1

    attention = DeepseekV3Attention(config, layer_idx=0)
    attention = attention.to(torch.bfloat16)

    max_cache_len = 512
    batch_size = 32
    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16)


    num_devices = xr.global_runtime_device_count()
    mesh_shape = (8, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("model", "batch"))

    position_ids = torch.arange(seq_len)
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache
    # outs = attention(hidden_states, attention_mask, position_ids, past_key_states)

    def get_shard_spec(attention, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = (None, None, "batch")
        shard_specs[attention.q_b_proj.weight] = ("model", None)
        shard_specs[attention.kv_b_proj.weight] = ("model", None)
        shard_specs[attention.o_proj.weight] = ("batch", "model")

        # Consume hidden states, TP on batch dimension
        shard_specs[attention.q_a_proj.weight] = (None, "batch")
        shard_specs[attention.kv_a_proj_with_mqa.weight] = (None, "batch")
        return shard_specs


    run_graph_test(
        attention,
        [hidden_states, attention_mask, position_ids, past_key_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )



def test_kimi_k2_layer():
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    # Load full Kimi K2 config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = DeepseekV3Config.from_json_file(config_path)
    config._attn_implementation = "eager"


    layer = DeepseekV3DecoderLayer(config, layer_idx=0)
    layer = layer.to(torch.bfloat16)

    max_cache_len = 512
    batch_size = 512
    seq_len = 1
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16)


    num_devices = xr.global_runtime_device_count()
    mesh_shape = (8, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("model", "batch"))

    position_ids = torch.arange(seq_len)
    static_cache = MLACache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    past_key_states = static_cache

    device = torch_xla.device()
    layer = layer.to(device)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))

    hidden_states = hidden_states.to(device)
    attention_mask = attention_mask.to(device)


    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("model", None, "batch")
        
        # Main attention weights, TP across model and batch dimensions
        shard_specs[layer.self_attn.q_b_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.kv_b_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        # Consume hidden states, TP on batch dimension
        shard_specs[layer.self_attn.q_a_proj.weight] = (None, "batch")
        shard_specs[layer.self_attn.kv_a_proj_with_mqa.weight] = (None, "batch")


        shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

        shard_specs[layer.input_layernorm.weight] = ("batch",)
        shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs


    run_graph_test(
        layer,
        [hidden_states, attention_mask, position_ids, past_key_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
