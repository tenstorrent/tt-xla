# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from utils import MNISTLinear, tensor_parallel_inference_mnist


# these examples should be moved to the test infra once this issue is closed: https://github.com/tenstorrent/tt-xla/issues/1822
@pytest.mark.parametrize(
    "batch_size,input_size",
    [
        (32, 784),
        (128, 784),
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

    assert torch.allclose(
        tt_output.cpu(), cpu_output, atol=0.05
    ), f"MNIST TP inference mismatch. Max diff: {(tt_output.cpu() - cpu_output).abs().max()}"
