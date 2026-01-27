# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from configuration_deepseek import DeepseekV3Config
from modeling_deepseek import DeepseekV3ForCausalLM

from tests.utils import failed_ttmlir_compilation


@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "'ttir.concat' op Output tensor dimension 0 does not match the sum of input tensor dimensions: 1 vs. 32. "
    )
)
def test_kimi_k2_single_layer():
    xr.set_device_type("TT")

    # Create model config with a single layer for testing
    config = DeepseekV3Config(
        num_hidden_layers=1,
        use_cache=False,
    )

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
