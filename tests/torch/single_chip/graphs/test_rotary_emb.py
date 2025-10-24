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
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader as LlamaModelLoader,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelLoader as QwenModelLoader,
)

llama_available_variants = LlamaModelLoader.query_available_variants()
qwen_available_variants = QwenModelLoader.query_available_variants()


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
