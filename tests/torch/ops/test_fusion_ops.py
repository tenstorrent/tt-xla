# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra.utilities.types import Framework
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from tests.infra.comparators.comparison_config import ComparisonConfig
from tests.infra.testers.single_chip.graph.graph_tester import run_graph_test


@pytest.mark.single_device
@pytest.mark.push
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [(1, 32, 32), (1, 128, 768), (1, 1024, 768)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_llama_rms_norm_fusion(batch_size, seq_len, hidden_size, dtype):

    options = {
        "tt_enable_fusion_passes": True,
        "tt_enable_composite_ops": True,
    }

    model = LlamaRMSNorm(hidden_size)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )
