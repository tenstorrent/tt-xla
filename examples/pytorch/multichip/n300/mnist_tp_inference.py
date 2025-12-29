# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.comparators.comparison_config import AtolConfig, ComparisonConfig, PccConfig
from infra.comparators.torch_comparator import TorchComparator
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from utils import tensor_parallel_inference_mnist

from examples.pytorch.multichip.n300.mnist_dp_inference import MNISTLinear


# these examples should be moved to the test infra once this issue is closed: https://github.com/tenstorrent/tt-xla/issues/1822
@pytest.mark.parametrize(
    "batch_size,input_size",
    [
        pytest.param(
            32,
            784,
            marks=pytest.mark.xfail(
                reason="Replicating bias has issues. Tracking issue: https://github.com/tenstorrent/tt-mlir/issues/5290",
                strict=True,
            ),
        ),
        pytest.param(
            128,
            784,
            marks=pytest.mark.xfail(
                reason="Replicating bias has issues. Tracking issue: https://github.com/tenstorrent/tt-mlir/issues/5290",
                strict=True,
            ),
        ),
    ],
)
def test_mnist_inference_tensor_parallel(batch_size: int, input_size: int):
    """Test MNIST linear inference with tensor parallel mode across multiple chips."""
    xr.set_device_type("TT")
    enable_spmd()

    torch.manual_seed(42)

    model = MNISTLinear(input_size, 512, 10, bias=True).to(torch.bfloat16)

    # Random input
    input_data = torch.randn(batch_size, input_size, dtype=torch.bfloat16)

    # CPU baseline (same weights)
    model_cpu = model.cpu().eval()
    with torch.no_grad():
        cpu_output = model_cpu(input_data)

    # Run on devices (tensor parallel)
    tt_output = tensor_parallel_inference_mnist(model=model, inputs=input_data)

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.05),
            pcc=PccConfig(required_pcc=0.95),
        )
    )
    comparator.compare(tt_output.cpu(), cpu_output)
