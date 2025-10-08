# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra.comparators.torch_comparator import TorchComparator
from transformers import CacheConfig
from transformers.cache_utils import StaticCache
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

from infra import run_graph_test, Framework

from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)
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


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_attention_prefill(seq_len, variant, variant_config):
    # Xfail 70B models that don't fit on device
    if "70b" in str(variant):
        pytest.xfail("70B models don't fit on device")

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

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_attention_decode(variant, variant_config):

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
    past_key_states = static_cache

    run_graph_test(
        attention,
        [hidden_states, position_embeddings, attention_mask, past_key_states],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_concat_heads(variant, variant_config, seq_len):

    def concat_heads(attn_output, input_shape):
        attn_output = attn_output.transpose(1, 2).contiguous()
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

    run_graph_test(concat_heads, [attn_output, input_shape], framework=Framework.TORCH)


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_create_heads(variant, variant_config, seq_len):

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

    run_graph_test(
        create_heads,
        [hidden_states, hidden_shape, q_proj, k_proj, v_proj],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])  # 4096 causes OOM on CPU
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_sdpa(variant, variant_config, seq_len):

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

    run_graph_test(
        sdpa,
        [
            attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout,
            scaling,
        ],
        framework=Framework.TORCH,
    )
