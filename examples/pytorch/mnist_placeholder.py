# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MNIST model tracing with placeholder tensors.

This demonstrates offline compilation by tracing the model graph without
actual weight values - only shapes and dtypes are needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.core.xla_builder as xb

# Use the create_placeholder_tensor from xla_builder
from torch_xla.core.xla_builder import create_placeholder_tensor


def replace_module_params_with_placeholders(module: nn.Module, dtype: torch.dtype):
    """
    Replace all parameters and buffers in a module with placeholder tensors.
    """
    # Collect parameters to replace (can't modify during iteration)
    params_to_replace = []
    for name, param in module.named_parameters(recurse=False):
        params_to_replace.append((name, param.shape, dtype))

    buffers_to_replace = []
    for name, buffer in module.named_buffers(recurse=False):
        if buffer is not None:
            buffers_to_replace.append((name, buffer.shape, dtype))

    # Replace parameters with placeholders
    for name, shape, dt in params_to_replace:
        placeholder = create_placeholder_tensor(shape, dt)
        # Delete the parameter and register as buffer (placeholders aren't trainable)
        delattr(module, name)
        module.register_buffer(name, placeholder)

    # Replace buffers with placeholders
    for name, shape, dt in buffers_to_replace:
        placeholder = create_placeholder_tensor(shape, dt)
        setattr(module, name, placeholder)

    # Recurse into child modules
    for child in module.children():
        replace_module_params_with_placeholders(child, dtype)


class MNISTLinearModel(nn.Module):
    """Simple linear MNIST model (no convolutions for simplicity)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def trace_with_placeholders():
    """
    Trace MNIST model using placeholder tensors.
    No actual weight values are loaded - only shapes are used.
    """
    dtype = torch.bfloat16
    batch_size = 4
    device = xm.xla_device()
    # Create model structure (weights will be replaced with placeholders)
    model = MNISTLinearModel().to(dtype=dtype).eval()

    print("Original model parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape} {param.dtype}")

    # Replace all weights with placeholders
    replace_module_params_with_placeholders(model, dtype)

    print("\nAfter placeholder replacement:")
    for name, buffer in model.named_buffers():
        print(f"  {name}: {buffer.shape} (placeholder)")

    # Create placeholder input
    input_shape = (batch_size, 1, 28, 28)
    input_placeholder = create_placeholder_tensor(input_shape, dtype)

    print(f"\nInput placeholder: {input_placeholder.shape}")

    # Compile the model with tt backend
    # This should trace the graph structure without needing actual values
    compiled_model = torch.compile(model, backend="tt")

    # print("\nModel compiled successfully with placeholder tensors")
    # print("The compiled graph captures the computation structure without weight values")
    input_tensor = torch.ones((batch_size, 1, 28, 28), dtype=dtype, device=device)

    output = compiled_model(input_tensor)
    
    torch_xla.sync()
    return compiled_model, input_placeholder


def run_with_real_tensors():
    """
    For comparison: run with real tensors to verify the model works.
    """
    dtype = torch.bfloat16
    batch_size = 4
    device = xm.xla_device()

    # Create model with real tensors
    torch.manual_seed(42)
    model = MNISTLinearModel().to(dtype=dtype).eval().to(device)

    # Create real input
    input_tensor = torch.ones((batch_size, 1, 28, 28), dtype=dtype, device=device)

    # Compile and run
    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(input_tensor)

    xm.mark_step()

    print(f"Output shape: {output.shape}")
    print(f"Output (first sample): {output[0].cpu()}")

    return output


if __name__ == "__main__":
    xr.set_device_type("TT")

    print("=" * 60)
    print("Test 1: Placeholder-based tracing (no weight values)")
    print("=" * 60)

    try:
        compiled_model, input_placeholder = trace_with_placeholders()
        print("\n✓ Placeholder tracing succeeded!")
    except Exception as e:
        print(f"\n✗ Placeholder tracing failed: {e}")
        import traceback
        traceback.print_exc()

    # print("\n" + "=" * 60)
    # print("Test 2: Normal execution with real tensors")
    # print("=" * 60)

    # try:
    #     output = run_with_real_tensors()
    #     print("\n✓ Real tensor execution succeeded!")
    # except Exception as e:
    #     print(f"\n✗ Real tensor execution failed: {e}")
    #     import traceback
    #     traceback.print_exc()
