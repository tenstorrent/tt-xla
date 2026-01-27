# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from utils import MNISTLinear, data_parallel_inference_generic


@pytest.mark.parametrize(
    "batch_size,input_size",
    [
        (32, 784),
        (128, 784),
    ],
)
def test_mnist_inference_data_parallel(batch_size: int, input_size: int):
    """Test MNIST linear inference with data parallel across multiple chips."""
    xr.set_device_type("TT")
    enable_spmd()

    # set seed for reproducibility
    torch.manual_seed(42)

    model = MNISTLinear(input_size, 512, 10, bias=True).to(torch.bfloat16)

    # Create random input data
    input_data = torch.randn(batch_size, input_size, dtype=torch.bfloat16)

    # Run multichip inference first to get device count
    tt_output = data_parallel_inference_generic(
        model=model, inputs=input_data, batch_dim=0
    )

    # model to CPU for comparison
    model_cpu = model.cpu()

    # CPU inference
    model_cpu.eval()
    with torch.no_grad():
        cpu_output = model_cpu(input_data)

    assert torch.allclose(
        tt_output.cpu(), cpu_output, atol=0.05
    ), f"MNIST inference mismatch. Max diff: {(tt_output.cpu() - cpu_output).abs().max()}"
