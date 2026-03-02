# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests that tensors persisting across tests (module-level globals) remain valid
after _release_dynamo_bridge_tensors() runs in the run_around_tests fixture.

This verifies that the memory leak workaround (#3507) does not corrupt or
invalidate tensors that outlive a single test. Covers:
- User input tensors persisted across tests
- Model parameters/weights (graph constants cached by GraphInputMatcher)
- Compiled model outputs reused as inputs
- A persistent model re-compiled and re-run after cache clearing
"""

import pytest
import torch
import torch_xla.core.xla_model as xm
from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator

# Module-level persistent state — survives across tests.
# The run_around_tests fixture (which calls _release_dynamo_bridge_tensors)
# runs between each test.
_persistent_cpu_tensor = torch.randn(32, 32, dtype=torch.float32)
_persistent_device_tensor = None
_persistent_compiled_output = None

_comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))


class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2.0 + 1.0


class AnotherModel(torch.nn.Module):
    def forward(self, x):
        return x + 10.0


# Persistent model with parameters — weights are exactly what
# GraphInputMatcher caches as "constant" graph inputs.
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)


# Module-level model instance — parameters persist across tests.
_persistent_linear_model = LinearModel()
_persistent_linear_expected = None


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_first_compile_and_persist():
    """Compile a model, run it, and persist the output tensor as a global."""
    global _persistent_device_tensor, _persistent_compiled_output

    device = xm.xla_device()
    model = torch.compile(SimpleModel(), backend="tt")
    model = model.to(device)

    input_on_device = _persistent_cpu_tensor.to(device)
    output = model(input_on_device)

    # Persist both the input (on device) and output for the next test
    _persistent_device_tensor = input_on_device
    _persistent_compiled_output = output

    # Verify correctness
    expected = _persistent_cpu_tensor * 2.0 + 1.0
    comparator = TorchComparisonEvaluator(_comparison_config)
    comparator.evaluate(output.cpu(), expected)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_second_reuse_persistent_tensor():
    """Use the persisted tensors from the first test in a new compilation.

    Between test_first and test_second, the run_around_tests fixture runs
    torch._dynamo.reset() and _release_dynamo_bridge_tensors(). This test
    verifies the persistent tensors are still valid.
    """
    global _persistent_device_tensor, _persistent_compiled_output

    assert _persistent_device_tensor is not None

    device = xm.xla_device()

    # Use the persisted output from test_first as input to a new model
    model = torch.compile(AnotherModel(), backend="tt")
    model = model.to(device)
    output = model(_persistent_compiled_output)

    # Verify correctness: AnotherModel(SimpleModel(x)) = (x * 2.0 + 1.0) + 10.0
    expected = _persistent_cpu_tensor * 2.0 + 1.0 + 10.0
    comparator = TorchComparisonEvaluator(_comparison_config)
    comparator.evaluate(output.cpu(), expected)

    # Also verify the persisted device tensor itself is still readable
    comparator.evaluate(_persistent_device_tensor.cpu(), _persistent_cpu_tensor)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_third_fresh_compile_with_same_input():
    """Compile yet another model using the same persistent input tensor.

    This verifies that after two rounds of _release_dynamo_bridge_tensors(),
    the original persistent tensor is still valid.
    """
    assert _persistent_device_tensor is not None

    device = xm.xla_device()

    class YetAnotherModel(torch.nn.Module):
        def forward(self, x):
            return x * 3.0 - 5.0

    model = torch.compile(YetAnotherModel(), backend="tt")
    model = model.to(device)
    output = model(_persistent_device_tensor)

    expected = _persistent_cpu_tensor * 3.0 - 5.0
    comparator = TorchComparisonEvaluator(_comparison_config)
    comparator.evaluate(output.cpu(), expected)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_fourth_compile_persistent_model_with_parameters():
    """Compile and run a persistent model that has parameters (nn.Linear).

    Model parameters are the graph constants that GraphInputMatcher caches
    in graph_input_xla_values. This test verifies that the model's weights
    survive _release_dynamo_bridge_tensors() and produce correct results
    when the model is re-compiled in a later test.
    """
    global _persistent_linear_expected

    device = xm.xla_device()

    with torch.no_grad():
        _persistent_linear_expected = _persistent_linear_model(_persistent_cpu_tensor)

    compiled = torch.compile(_persistent_linear_model, backend="tt")
    compiled = compiled.to(device)
    input_on_device = _persistent_cpu_tensor.to(device)
    output = compiled(input_on_device)

    comparator = TorchComparisonEvaluator(_comparison_config)
    comparator.evaluate(output.cpu(), _persistent_linear_expected)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_fifth_recompile_persistent_model_after_cache_clear():
    """Re-compile and re-run the same persistent model after cache clearing.

    Between test_fourth and test_fifth, _release_dynamo_bridge_tensors() clears
    the GraphInputMatcher cache (which held this model's weight tensors).
    torch._dynamo.reset() also clears dynamo's compilation cache. This test
    verifies that re-compiling and re-running the same model still produces
    correct results — i.e., the model's parameter tensors are not corrupted.
    """
    assert _persistent_linear_expected is not None

    device = xm.xla_device()

    # Re-compile the same persistent model (dynamo will re-trace since reset cleared caches)
    compiled = torch.compile(_persistent_linear_model, backend="tt")
    compiled = compiled.to(device)
    input_on_device = _persistent_cpu_tensor.to(device)
    output = compiled(input_on_device)

    # Should produce the same result as test_fourth — same weights, same input
    comparator = TorchComparisonEvaluator(_comparison_config)
    comparator.evaluate(output.cpu(), _persistent_linear_expected)
