# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace with data parallel

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import ResNetForImageClassification
from examples.pytorch.multichip.n300.utils import tensor_parallel_inference_mnist
from infra.utilities.torch_multichip_utils import enable_spmd

# Set up TT backend
xr.set_device_type("TT")
enable_spmd()

class MNISTLinear(nn.Module):
    """Simple linear MNIST model for inference testing."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 512,
        num_classes: int = 10,
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3 = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)
        
# Set up codegen options
options = {
    "backend": "codegen_py",
    "export_path": "mnist_codegen_tp",
}
torch_xla.set_custom_compile_options(options)

torch.manual_seed(42)

batch_size = 32
input_size = 784

model = MNISTLinear(input_size, 512, 10, bias=True).to(torch.bfloat16)

# Create random input data
input_data = torch.randn(batch_size, input_size, dtype=torch.bfloat16)

# Compile the model for TT backend 
model.compile(backend="tt")

# Run multichip tensor parallel inference. This triggers code generation.
tt_output = tensor_parallel_inference_mnist(
    model=model, 
    inputs=input_data
)