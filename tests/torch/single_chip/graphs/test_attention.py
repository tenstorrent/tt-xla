# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla
import torch_xla.runtime as xr
import pytest
from transformers.cache_utils import StaticCache
from transformers import CacheConfig
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)
from typing import Callable


from tests.infra.comparators.comparison_config import (
    ComparisonConfig,
    AtolConfig,
)
from infra.comparators.torch_comparator import TorchComparator
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader as LlamaModelLoader,
)


# To see all available models and variants, run:
# pytest -s tests/torch/single_chip/graphs/test_attention.py::test_display_available_variants

MODEL_LOADER_MAP = {
    "llama": LlamaModelLoader,
}


def get_available_variants(model_name):
    ModelLoader = MODEL_LOADER_MAP[model_name]
    available_variants = ModelLoader.query_available_variants()
    return available_variants


@pytest.mark.parametrize("model_name", list(MODEL_LOADER_MAP.keys()))
def test_display_available_variants(model_name):
    print(
        f"\nAvailable variants for {model_name}: ",
        [str(k) for k in get_available_variants(model_name)],
    )


"""Llama attention tests"""


@pytest.mark.push
@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_attention_prefill(seq_len, variant, variant_config):
    xr.set_device_type("TT")

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    hidden_states = torch.randn(
        (1, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(1, seq_len, model.config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(1, 1, seq_len, seq_len, dtype=torch.bfloat16)

    past_key_states = None
    golden = attention(
        hidden_states, position_embeddings, attention_mask, past_key_states
    )

    device = torch_xla.device()
    compiled_fn = torch.compile(attention.to(device), backend="tt")

    output = attention(
        hidden_states.to(device), position_embeddings, attention_mask, past_key_states
    )

    comparator = TorchComparator(
        ComparisonConfig(
            # atol=AtolConfig(required_atol=0.02),
        )
    )
    comparator.compare(output, golden)


@pytest.mark.push
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_attention_decode(variant, variant_config):
    xr.set_device_type("TT")

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    seq_len = 1
    hidden_states = torch.randn(
        (1, seq_len, model.config.hidden_size), dtype=torch.bfloat16
    )
    cos_sin = torch.rand(1, seq_len, model.config.head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos_sin, cos_sin)
    attention_mask = torch.rand(1, 1, seq_len, seq_len, dtype=torch.bfloat16)

    batch_size = 1
    max_cache_len = 16
    static_cache: StaticCache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    cache_position = torch.tensor([0])
    past_key_states = static_cache

    golden = attention(
        hidden_states, position_embeddings, attention_mask, past_key_states
    )

    device = torch_xla.device()
    past_key_states.key_cache = [k.to(device) for k in static_cache.key_cache]
    past_key_states.value_cache = [v.to(device) for v in static_cache.value_cache]
    compiled_fn = torch.compile(attention.to(device), backend="tt")

    output = attention(
        hidden_states.to(device), position_embeddings, attention_mask, past_key_states
    )

    comparator = TorchComparator(
        ComparisonConfig(
            # atol=AtolConfig(required_atol=0.02),
        )
    )
    comparator.compare(output, golden)


@pytest.mark.push
@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_concat_heads(variant, variant_config, seq_len):
    xr.set_device_type("TT")

    def concat_heads(attn_output, input_shape):
        return attn_output.reshape(*input_shape, -1).contiguous()

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)

    batch_size = 1
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim
    hidden_size = model.config.hidden_size

    attn_output = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    input_shape = (batch_size, seq_len)

    golden = concat_heads(attn_output, input_shape)

    device = torch_xla.device()
    compiled_fn = torch.compile(concat_heads, backend="tt")
    output = compiled_fn(attn_output.to(device), input_shape)

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.02),
        )
    )
    comparator.compare(output.cpu(), golden)


@pytest.mark.push
@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_create_heads(variant, variant_config, seq_len):
    xr.set_device_type("TT")

    def create_heads(hidden_states, hidden_shape, q_proj, k_proj, v_proj):
        query_states = q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    batch_size = 1
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim

    hidden_states = torch.randn(
        (batch_size, seq_len, hidden_size), dtype=torch.bfloat16
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    q_proj = attention.q_proj
    k_proj = attention.k_proj
    v_proj = attention.v_proj

    golden = create_heads(hidden_states, hidden_shape, q_proj, k_proj, v_proj)

    device = torch_xla.device()
    compiled_fn = torch.compile(create_heads, backend="tt")

    output = compiled_fn(
        hidden_states.to(device),
        hidden_shape,
        q_proj.to(device),
        k_proj.to(device),
        v_proj.to(device),
    )

    comparator = TorchComparator(
        ComparisonConfig(
            # atol=AtolConfig(required_atol=0.02),
        )
    )
    for i, (out_tensor, golden_tensor) in enumerate(zip(output, golden)):
        comparator.compare(out_tensor.cpu(), golden_tensor)


@pytest.mark.push
@pytest.mark.parametrize("seq_len", [1024, 2048])  # 4096 causes OOM on CPU
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_sdpa(variant, variant_config, seq_len):
    xr.set_device_type("TT")

    def sdpa(
        attention_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout,
        scaling,
    ):
        attention_interface: Callable = eager_attention_forward
        if attention_module.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                attention_module.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            attention_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
        )
        return attn_output, attn_weights

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    attention = model.model.layers[0].self_attn

    batch_size = 1
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_key_value_heads = getattr(model.config, "num_key_value_heads", num_heads)
    head_dim = model.config.head_dim

    query_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    value_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )

    attention_mask = torch.rand(1, 1, seq_len, seq_len, dtype=torch.bfloat16)

    dropout = 0.0
    scaling = attention.scaling

    golden = sdpa(
        attention,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout,
        scaling,
    )

    device = torch_xla.device()
    compiled_fn = torch.compile(sdpa, backend="tt")

    output = compiled_fn(
        attention.to(device),
        query_states.to(device),
        key_states.to(device),
        value_states.to(device),
        attention_mask.to(device),
        dropout,
        scaling,
    )

    comparator = TorchComparator(
        ComparisonConfig(
            # atol=AtolConfig(required_atol=0.02),
        )
    )

    for i, (out_tensor, golden_tensor) in enumerate(zip(output, golden)):
        if (
            out_tensor is not None and golden_tensor is not None
        ):  # attn_weights might be None
            comparator.compare(out_tensor.cpu(), golden_tensor)
