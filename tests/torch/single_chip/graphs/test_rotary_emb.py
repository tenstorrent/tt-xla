# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)
import torch
import torch_xla
import torch_xla.runtime as xr

import pytest
from infra.comparators.torch_comparator import TorchComparator

from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader as LlamaModelLoader,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelLoader as QwenModelLoader,
)

llama_available_variants = LlamaModelLoader.query_available_variants()
qwen_available_variants = QwenModelLoader.query_available_variants()


@pytest.mark.parametrize(
    "variant, variant_config",
    llama_available_variants.items(),
    ids=[str(k) for k in llama_available_variants.keys()],
)
@pytest.mark.parametrize("seq_len", [256, 512, 1024, 2048, 4096])
def test_llama_rotary_emb(seq_len, variant, variant_config):
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
    
    # CPU for golden
    cos, sin = RoPE(query_states, position_ids)
    
    # Compile RoPE module and run on device
    xr.set_device_type("TT")
    device = torch_xla.device()
    compiled_RoPE = torch.compile(RoPE.to(device), backend="tt")
    
    # Run on device
    device_cos, device_sin = compiled_RoPE(
        query_states.to(device), 
        position_ids.to(device)
    )
    
    # Compare results
    comparator = TorchComparator(
        ComparisonConfig(
            #atol=AtolConfig(required_atol=0.02),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    
    # Compare both cos and sin outputs
    comparator.compare(device_cos, cos)
    comparator.compare(device_sin, sin)


@pytest.mark.parametrize(
    "variant, variant_config",
    qwen_available_variants.items(),
    ids=[str(k) for k in qwen_available_variants.keys()],
)
@pytest.mark.parametrize("seq_len", [256, 512, 1024, 2048, 4096])
def test_qwen_3_rotary_emb(seq_len, variant, variant_config):
    loader = QwenModelLoader(variant=variant)
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
    
    # CPU for golden
    cos, sin = RoPE(query_states, position_ids)
    
    # Compile RoPE module and run on device
    xr.set_device_type("TT")
    device = torch_xla.device()
    compiled_RoPE = torch.compile(RoPE.to(device), backend="tt")
    
    # Run on device
    device_cos, device_sin = compiled_RoPE(
        query_states.to(device), 
        position_ids.to(device)
    )
    
    # Compare results
    comparator = TorchComparator(
        ComparisonConfig(
            #atol=AtolConfig(required_atol=0.02),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    
    # Compare both cos and sin outputs
    comparator.compare(device_cos, cos)
    comparator.compare(device_sin, sin)