# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)
import torch
import torch_xla.core.xla_model as xm

import pytest

from infra.comparators.torch_comparator import TorchComparator
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)


@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
def test_llama_3b_attention_prefill(seq_len):
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

    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt")

    output = model(
        hidden_states.to(device), position_embeddings, attention_mask, past_key_states
    )
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.02),
        )
    )

    comparator.compare(output, golden)
