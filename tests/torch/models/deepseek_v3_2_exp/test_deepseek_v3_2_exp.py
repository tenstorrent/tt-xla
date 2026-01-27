# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from model_modified import ModelArgs, Transformer
from model_modified_non_complex import Transformer as TransformerNonComplex

from tests.utils import failed_ttmlir_compilation

# Both models are modified from deepseek_v3_2_exp/model.py to:
# 1. Use scipy.linalg.hadamard instead of fast_hadamard_transform
# 2. Disable FP8 quantization features (act_quant, fp8_gemm, fp8_index) with stubs
# The non-complex model additionally avoids torch.view_as_complex/view_as_real operations


@pytest.mark.xfail(
    reason="TT_THROW: Statically allocated circular buffers on core range [(x=7,y=6) - (x=7,y=6)] grow to 16897152 B which is beyond max L1 size of 1499136 B"
)
def test_deepseek_v3_2_exp_non_complex():
    xr.set_device_type("TT")

    # Create model args with a single layer for testing
    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
    )

    # Non-complex model is adjusted to avoid complex number operations
    model = TransformerNonComplex(args)

    model = model.to(torch.bfloat16)

    model = model.eval()
    compiled_model = torch.compile(model, backend="tt")

    batch_size = 1
    seq_len = 32
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    device = torch_xla.device()
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    with torch.no_grad():
        output = compiled_model(tokens)


@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'stablehlo.complex'"
    )
)
def test_deepseek_v3_2_exp():
    xr.set_device_type("TT")

    # Create model args with a single layer for testing
    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
    )

    # This model uses complex number operations
    model = Transformer(args)

    model = model.to(torch.bfloat16)

    model = model.eval()
    compiled_model = torch.compile(model, backend="tt")

    batch_size = 1
    seq_len = 32
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    device = torch_xla.device()
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    with torch.no_grad():
        output = compiled_model(tokens)
