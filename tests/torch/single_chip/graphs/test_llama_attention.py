# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla
import torch_xla.runtime as xr
import pytest

from tests.infra.comparators.comparison_config import (
    ComparisonConfig,
    AtolConfig,
)
from infra.comparators.torch_comparator import TorchComparator
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

available_variants = ModelLoader.query_available_variants()
print("Available variants: ", [str(k) for k in available_variants.keys()])

@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
@pytest.mark.parametrize(
    "variant,variant_config",
    available_variants.items(),
    ids=[str(k) for k in available_variants.keys()],
)
def test_llama_attention_prefill(seq_len, variant, variant_config):
    xr.set_device_type("TT")

    loader = ModelLoader(ModelVariant.LLAMA_3_2_3B)
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
        ComparisonConfig()
    )

    comparator.compare(output, golden)


@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
def test_concat_heads(seq_len):
    xr.set_device_type("TT")

    # Test parameters based on LLaMA 3.2 3B config
    # TODO: place breakpoint when running attention layer and double check these values
    batch_size = 1
    num_heads = 24  # LLaMA 3.2 3B has 24 attention heads
    head_dim = 128  # 3072 hidden_size / 24 heads = 128
    hidden_size = num_heads * head_dim
    
    # Create input tensor (output from attention computation)
    attn_output = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), 
        dtype=torch.bfloat16
    )
    
    # Input shape for reshape (batch_size, seq_len)
    input_shape = (batch_size, seq_len)

    def concat_heads(attn_output, input_shape):
        return attn_output.reshape(*input_shape, -1).contiguous()
    
    # Run on CPU for golden output
    golden = concat_heads(attn_output, input_shape)
    
    # Test compiled version on TT device
    device = torch_xla.device()
    compiled_fn = torch.compile(concat_heads, backend="tt")
    output = compiled_fn(attn_output.to(device), input_shape)
    
    # Compare TT compiled output with CPU reference
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.01),
        )
    )
    
    comparator.compare(output.cpu(), golden)

