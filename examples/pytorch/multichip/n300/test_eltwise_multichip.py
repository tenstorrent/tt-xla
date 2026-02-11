# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
import torch.nn as nn
from utils import data_parallel_inference_generic

class CustomModule(nn.Module):
    """Simple custom model for inference testing."""

    def __init__(
        self
    ):
        super().__init__()
    def forward(self, inputs):
        return torch.add(inputs[0], inputs[1])


@pytest.mark.parametrize(
    "dim1_size,dim2_size",
    [
        (128, 512),
    ],
)
def test_eltwise_data_parallel(dim1_size: int, dim2_size: int):
    """Test eltwise addition inference with data parallel across multiple chips."""
    xr.set_device_type("TT")
    enable_spmd()

    # set seed for reproducibility
    torch.manual_seed(42)

    model = CustomModule().to(torch.bfloat16)

    print("Setting custom compile options...")
    import os
    os.environ["TT_RUNTIME_TRACE_REGION_SIZE"] = "10000000"
    # Set relevant compiler options.
    torch_xla.set_custom_compile_options(
        {
            # Enable runtime trace.
            "enable_trace": "true",
        }
    )

    # model to CPU for comparison
    model_cpu = model.cpu()

    for i in range(5):
        inputs = (
            torch.randn(10, dim1_size, dim2_size, dtype=torch.bfloat16),
            torch.randn(10, dim1_size, dim2_size, dtype=torch.bfloat16),
        )
        print(f"Running iteration {i+1}...")
        tt_output = data_parallel_inference_generic(
            model=model, inputs=(inputs), batch_dim=0
        )

        # CPU inference
        model_cpu.eval()
        with torch.no_grad():
            cpu_output = model_cpu(inputs)

        assert torch.allclose(
            tt_output.cpu(), cpu_output, atol=0.05
        ), f"MNIST inference mismatch. Max diff: {(tt_output.cpu() - cpu_output).abs().max()}"
