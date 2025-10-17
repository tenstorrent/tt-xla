# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.comparators.torch_comparator import TorchComparator

from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)
from third_party.tt_forge_models.bert.masked_lm.pytorch.loader import (
    ModelLoader as BertModelLoader,
)
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader as LlamaModelLoader,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelLoader as QwenModelLoader,
)

llama_available_variants = LlamaModelLoader.query_available_variants()
qwen_available_variants = QwenModelLoader.query_available_variants()
bert_available_variants = BertModelLoader.query_available_variants()


@pytest.mark.push
@pytest.mark.parametrize(
    "variant, variant_config",
    llama_available_variants.items(),
    ids=[str(k) for k in llama_available_variants.keys()],
)
@pytest.mark.parametrize("seq_len", [1024])
def test_llama_rotary_emb(seq_len, variant, variant_config):
    # Xfail 70B models that don't fit on device
    if "70b" in str(variant):
        pytest.xfail("70B models don't fit on device")

    loader = LlamaModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)

    # extract RoPE module from model
    RoPE = model.model.rotary_emb

    # Create query tensors and position_ids for RoPE to operate on
    hidden_size = model.config.hidden_size  # Should be 128 for Llama 3.2 3B
    num_heads = model.config.num_attention_heads
    query_states = torch.randn(
        (1, num_heads, seq_len, hidden_size), dtype=torch.bfloat16
    )
    position_ids = torch.arange(seq_len, dtype=torch.bfloat16).unsqueeze(0)

    run_graph_test(RoPE, [query_states, position_ids], framework=Framework.TORCH)


@pytest.mark.push
@pytest.mark.parametrize(
    "variant, variant_config",
    qwen_available_variants.items(),
    ids=[str(k) for k in qwen_available_variants.keys()],
)
@pytest.mark.parametrize("seq_len", [1024])
def test_qwen_3_rotary_emb(seq_len, variant, variant_config):
    # Xfail 32B and 30B models that don't fit on device
    if "32b" in str(variant):
        pytest.xfail("32B models don't fit on device")
    if "30b" in str(variant):
        pytest.xfail("30B models don't fit on device")

    loader = QwenModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)

    # extract RoPE module from model
    RoPE = model.model.rotary_emb

    # Create query tensors and position_ids for RoPE to operate on
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    query_states = torch.randn(
        (1, num_heads, seq_len, hidden_size), dtype=torch.bfloat16
    )
    position_ids = torch.arange(seq_len, dtype=torch.bfloat16).unsqueeze(0)

    run_graph_test(RoPE, [query_states, position_ids], framework=Framework.TORCH)


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    qwen_available_variants.items(),
    ids=[str(k) for k in qwen_available_variants.keys()],
)
def test_qwen_3_create_heads(variant, variant_config, seq_len):
    # Xfail 32B and 30B models that don't fit on device
    if "32b" in str(variant):
        pytest.xfail("32B models don't fit on device")
    if "30b" in str(variant):
        pytest.xfail("30B models don't fit on device")

    def create_heads(
        hidden_states, hidden_shape, q_proj, k_proj, v_proj, q_norm, k_norm
    ):
        # Qwen3 applies normalization to query and key states after projection
        query_states = q_norm(q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = k_norm(k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    loader = QwenModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)

    # Force attention_bias to True to ensure bias is added
    model.config.attention_bias = True

    # Recreate the attention layer with bias
    attention = model.model.layers[0].self_attn

    # If the projections don't have bias, recreate them with bias
    if attention.q_proj.bias is None:
        # Recreate projection layers with bias
        import torch.nn as nn

        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        head_dim = attention.head_dim
        num_key_value_heads = model.config.num_key_value_heads

        # Create new projection layers with bias=True
        attention.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True).to(
            dtype=torch.bfloat16
        )
        attention.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=True
        ).to(dtype=torch.bfloat16)
        attention.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=True
        ).to(dtype=torch.bfloat16)

        # Initialize the bias tensors (they will be random, but that's fine for testing)
        nn.init.zeros_(attention.q_proj.bias)
        nn.init.zeros_(attention.k_proj.bias)
        nn.init.zeros_(attention.v_proj.bias)

    batch_size = 1
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = attention.head_dim  # Qwen3 stores head_dim as an attribute

    hidden_states = torch.randn(
        (batch_size, seq_len, hidden_size), dtype=torch.bfloat16
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    q_proj = attention.q_proj
    k_proj = attention.k_proj
    v_proj = attention.v_proj
    q_norm = attention.q_norm
    k_norm = attention.k_norm

    run_graph_test(
        create_heads,
        [hidden_states, hidden_shape, q_proj, k_proj, v_proj, q_norm, k_norm],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    bert_available_variants.items(),  # You'll need to define bert_available_variants
    ids=[str(k) for k in bert_available_variants.keys()],
)
def test_bert_create_heads(variant, variant_config, seq_len):

    def create_heads(hidden_states, hidden_shape, query_proj, key_proj, value_proj):
        # BERT uses straightforward projection without normalization
        query_states = query_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = key_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = value_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    loader = BertModelLoader(variant=variant)  # Assumes you have a BertModelLoader
    model = loader.load_model(dtype_override=torch.bfloat16)

    # Force all projections to have bias for testing matmul + add pattern
    # BERT typically doesn't have bias in attention by default
    attention = model.bert.encoder.layer[0].attention.self

    batch_size = 1
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = attention.attention_head_size

    hidden_states = torch.randn(
        (batch_size, seq_len, hidden_size), dtype=torch.bfloat16
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    query_proj = attention.query
    key_proj = attention.key
    value_proj = attention.value

    # Verify that bias is present
    assert query_proj.bias is not None, "query projection should have bias"
    assert key_proj.bias is not None, "key projection should have bias"
    assert value_proj.bias is not None, "value projection should have bias"

    run_graph_test(
        create_heads,
        [hidden_states, hidden_shape, query_proj, key_proj, value_proj],
        framework=Framework.TORCH,
    )
