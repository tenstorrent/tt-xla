# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader as LlamaModelLoader,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelLoader as Qwen3ModelLoader,
)

MODEL_LOADER_MAP = {
    "llama": LlamaModelLoader,
    "qwen3": Qwen3ModelLoader,
}

AVAILABLE_VARIANT_MAP = {
    "llama": [
        "llama_3_8b",
        "llama_3_1_8b",
        "llama_3_2_1b",
        "llama_3_2_3b",
        "huggyllama_7b",
        "TinyLlama_v1.1",
    ],
    "qwen3": ["0_6b", "1_7b", "4b", "8b", "14b"],
}


def get_available_variants(model_name):
    ModelLoader = MODEL_LOADER_MAP[model_name]
    available_variants = ModelLoader.query_available_variants()

    # Filter to only include variants that match names in AVAILABLE_VARIANT_MAP
    if model_name in AVAILABLE_VARIANT_MAP:
        allowed_variant_names = set(AVAILABLE_VARIANT_MAP[model_name])
        return {
            variant_key: variant_config
            for variant_key, variant_config in available_variants.items()
            if str(variant_key) in allowed_variant_names
        }

    return available_variants


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
@pytest.mark.parametrize("seq_len", [1024])
def test_llama_RoPE(seq_len, variant, variant_config):
    loader = LlamaModelLoader(variant=variant)
    config = loader.load_config()
    RoPE = LlamaRotaryEmbedding(config).to(torch.bfloat16)

    # Create query tensors and position_ids for RoPE to operate on
    num_query_heads = config.num_attention_heads
    head_dim = config.head_dim

    query_states = torch.randn(
        (1, num_query_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    position_ids = torch.arange(seq_len, dtype=torch.bfloat16).unsqueeze(0)

    run_graph_test(RoPE, [query_states, position_ids], framework=Framework.TORCH)


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
@pytest.mark.parametrize("seq_len", [1024])
def test_llama_apply_rotary_emb(seq_len, variant, variant_config):
    loader = LlamaModelLoader(variant=variant)
    config = loader.load_config()

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    num_query_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim

    query_states = torch.randn(
        (1, num_query_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (1, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    cos = torch.randn((1, seq_len, head_dim), dtype=torch.bfloat16)
    sin = torch.randn((1, seq_len, head_dim), dtype=torch.bfloat16)

    run_graph_test(
        apply_rotary_pos_emb,
        [query_states, key_states, cos, sin],
        framework=Framework.TORCH,
    )


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
@pytest.mark.parametrize("seq_len", [1024])
def test_qwen_3_RoPE(seq_len, variant, variant_config):
    loader = Qwen3ModelLoader(variant=variant)
    config = loader.load_config()
    RoPE = Qwen3RotaryEmbedding(config).to(torch.bfloat16)

    # Create query tensors and position_ids for RoPE to operate on
    num_query_heads = config.num_attention_heads
    head_dim = config.head_dim

    query_states = torch.randn(
        (1, num_query_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    position_ids = torch.arange(seq_len, dtype=torch.bfloat16).unsqueeze(0)

    run_graph_test(RoPE, [query_states, position_ids], framework=Framework.TORCH)


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
@pytest.mark.parametrize("seq_len", [1024])
def test_qwen_3_apply_rotary_emb(seq_len, variant, variant_config):
    loader = Qwen3ModelLoader(variant=variant)
    config = loader.load_config()

    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    num_query_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim

    query_states = torch.randn(
        (1, num_query_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (1, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    cos = torch.randn((1, seq_len, head_dim), dtype=torch.bfloat16)
    sin = torch.randn((1, seq_len, head_dim), dtype=torch.bfloat16)

    run_graph_test(
        apply_rotary_pos_emb,
        [query_states, key_states, cos, sin],
        framework=Framework.TORCH,
    )
