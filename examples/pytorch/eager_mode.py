# tests/torch/single_chip/models/mnist/ff/test_small_mnist_like.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch_xla.debug import metrics as xla_metrics
from torch_xla.experimental.eager import eager_mode_context
from tests.utils import (
    failed_ttmlir_compilation,
)


# ---- Force TT device for this test session ----
xr.set_device_type("TT")


# -----------------------
# Minimal MNIST-like MLP
# -----------------------
class SmallMNISTLike(nn.Module):
    """
    A minimal feedforward NN for MNIST-like data.
    Input shape (N, 1, 28, 28) -> Output shape (N, 10)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (N, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        ret = F.softmax(x, dim=1)
        return ret  # (N, 10)


class SmallConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.fc = nn.Linear(8 * 14 * 14, 10)  # after pooling 28x28 -> 14x14

    def forward(self, x):
        x = self.conv1(x)  # (N, 8, 28, 28)
        print("After conv:", x)
        x = self.pool(x)  # (N, 8, 14, 14) <-- MaxPool2d, multi-output op
        print("After pool:", x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # (N, 10)
        return F.softmax(x, dim=1)


class SmallLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)

    def forward(self, x):
        # Same input shape passed through same linear layers multiple times
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc1(x)  # repeat fc1
        x = self.fc2(x)
        x = self.fc2(x)  # repeat fc2
        x = self.fc3(x)
        x = self.fc3(x)  # repeat fc3
        return x


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture(scope="module")
def device():
    return xm.xla_device()


@pytest.fixture
def dummy_input(device):
    x = torch.randn(1, 1, 28, 28)
    return x.to(device)


@pytest.fixture
def model():
    m = SmallMNISTLike().eval()
    return m


@pytest.mark.push
def test_small_mnist_like_lazy_tensor(model, dummy_input, device):
    # Eager mode through LazyTensor is working by default.
    # The only thing user needs to do is to set print
    # (or some other materialization trigger) inside the model's forward
    # to see the output. Graph will break in that place and everything up until
    # materialization trigger will be compiled as one graph and executed on the device.
    model.to(device)

    with torch.no_grad():
        out = model(dummy_input)
        print("Final output:", out)


@pytest.mark.push
def test_small_mnist_like_eager(model, dummy_input, device):
    # True eager mode is enabled via eager_mode_context.
    # Everything inside the context will be executed eagerly on the device.
    # Each op within model's forward will be compiled and executed separately.
    # This is useful for debugging and model development, but it can be slower due to
    # lack of whole-graph optimizations, and compile/run overhead for each op.
    model.to(device)
    with eager_mode_context(True):
        with torch.no_grad():
            out = model(dummy_input)
            print("Final output:", out)


@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "failed to legalize operation 'stablehlo.select_and_scatter' "
    )
)
@pytest.mark.push
def test_small_convnet_eager(device, dummy_input):
    model = SmallConvNet().eval().to(device)

    with eager_mode_context(True):
        with torch.no_grad():
            out = model(dummy_input)
            print("Final output:", out)


@pytest.mark.push
def test_eager_cache(
    device,
):
    """
    Shows that running the same model multiple times in eager mode
    benefits from compilation cache, making subsequent runs faster.
    """
    model = SmallLinearNet().eval().to(device)

    x = torch.randn(4, 128).to(device)

    for i in range(5):
        t0 = time.perf_counter()
        with eager_mode_context(True):
            y1 = model(x)
            _ = y1.cpu()  # force execution
        t1 = time.perf_counter()
        print(f"Iteration {i} took {t1 - t0:.4f} seconds")
