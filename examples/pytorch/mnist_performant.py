# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import time

# Required to enable runtime tracing.
os.environ["TT_RUNTIME_TRACE_REGION_SIZE"] = "10000000"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


class MNISTCNNDropoutModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def mnist_performant():
    """Minimal example of running MNIST CNN model with all performance options enabled."""
    # Initialize model.
    model = MNISTCNNDropoutModel()

    # Put it in inference mode.
    model = model.eval()

    # Convert weights and ops to bfloat16.
    model = model.to(dtype=torch.bfloat16)

    # Set relevant compiler options.
    torch_xla.set_custom_compile_options(
        {
            # Set to highest optimization level.
            "optimization_level": 2,
            # Enable runtime trace.
            "enable_trace": "true",
            # Cast weights and ops to bfloat8_b.
            "enable_bfp8_conversion": "true",
        }
    )

    # Compile the model for TT backend.
    model.compile(backend="tt")

    # Connect the device.
    device = xm.xla_device()

    # Move model to device.
    model = model.to(device)

    # Set batch size to optimal value.
    batch_size = 64

    # Warmup the device with 3 runs. This is needed as first 2 iterations are slow.
    warmup_input = generate_input(batch_size, torch.bfloat16)
    run_inference(model, device, warmup_input, loop_count=3, verbose=False)

    # Run fast inference loop and measure performance.
    inference_input = generate_input(batch_size, torch.bfloat16)
    run_inference(model, device, inference_input, loop_count=128, verbose=True)


def run_inference(model, device, input, loop_count, verbose=True):
    """Run inference and measure performance."""
    iteration_times = []
    # Run fast inference loop.
    with torch.no_grad():
        for i in range(loop_count):
            start = time.perf_counter_ns()

            # Move input to device.
            device_input = input.to(device)
            # Run the model.
            output = model(device_input)
            # Move output back to CPU.
            output.to("cpu")

            end = time.perf_counter_ns()

            iteration_times.append(end - start)
            if verbose:
                print(f"Iteration {i} took:\t{iteration_times[-1] / 1_000_000} ms")

    # Calculate and print average throughput.
    batch_size = input.shape[0]
    total_time = sum(iteration_times)
    samples_per_second = batch_size * loop_count / (total_time / 1_000_000_000)
    if verbose:
        print(f"Average throughput: {round(samples_per_second)} samples/second")


def generate_input(batch_size, dtype):
    """Helper to generate random inputs for inference."""
    return torch.randn((batch_size, 1, 28, 28), dtype=dtype)


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    mnist_performant()
