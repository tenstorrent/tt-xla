# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch._dynamo.exc
import torch_xla
import tt_torch
from torch.utils._python_dispatch import TorchDispatchMode
from tt_torch.backend.backend import xla_backend
from utils import Category

# Tests that experimental compile indeed bypasses the fx->XLA retracing overhead by asserting that no fx op dispatching occurs after the first run.

WARMUP_DONE = False


class AssertingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert (
            not WARMUP_DONE
        ), "No fx op dispatching after the first run when experimental compile is on"
        return func(*args, **(kwargs or {}))


def custom_backend(gm, example_inputs, options=None):
    result = xla_backend(gm, example_inputs, options)

    class WrappedCallable:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __call__(self, *args, **kwargs):
            with AssertingMode():
                return self._wrapped(*args, **kwargs)

    return WrappedCallable(result)


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x @ x.T


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_old_compile_dispatch_asserts():
    global WARMUP_DONE
    WARMUP_DONE = False
    model = Dummy().to(torch_xla.device())
    model.compile(backend=custom_backend, options={"tt_experimental_compile": False})
    input = torch.randn(2, 2).to(torch_xla.device())
    model(input)
    WARMUP_DONE = True
    with pytest.raises(AssertionError):
        model(input)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_new_compile_dispatch():
    global WARMUP_DONE
    WARMUP_DONE = False
    model = Dummy().to(torch_xla.device())
    model.compile(backend=custom_backend, options={"tt_experimental_compile": True})
    input = torch.randn(3, 3).to(torch_xla.device())
    model(input)
    WARMUP_DONE = True
    model(input)
