# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for tensor persistence and reuse across multiple graph executions,
with various reuse chains and multi-graph topologies.

These tests verify that output tensors from one computation are properly
retained when they need to be reused as inputs to multiple subsequent
computations that execute serially.

These tests are expected to produce program crashes (eg. Buffer is not allocated)
rather than numerical errors if tensor persistence is not handled correctly.
"""

import threading

import pytest
import torch
import torch_xla.core.xla_model as xm
from infra.comparators.comparison_config import ComparisonConfig, PccConfig
from infra.comparators.torch_comparator import TorchComparator

"""
A test suite checking various multi-graph tensor persistence scenarios.

Computations are done in fp32 to allow simple use of torch.allclose to validate.
"""


def run_model_on_device(model, inputs):
    """
    Helper to compile a model, move it and inputs to TT device, execute, and return output.

    Args:
        model: torch.nn.Module to execute
        inputs: List of input tensors

    Returns:
        Output tensor(s) from the model execution on device
    """
    # Compile the model for TT device
    compiled_model = torch.compile(model, backend="tt")

    # Move model and inputs to TT device
    device = xm.xla_device()
    model_on_device = compiled_model.to(device)
    inputs_on_device = [
        inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs
    ]

    # Execute on device
    output = model_on_device(*inputs_on_device)

    # Return output (still on device)
    return output


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_output_reused_in_two_serial_graphs():
    """
    Test the scenario: A(I) -> O, B(O) -> P, C(O) -> Q

    Program A produces output O from input I.
    O is reused as input to B (produces P) and C (produces Q).
    B and C run serially, so O must persist after B completes.

    This was a known bug where B would deallocate O before C could use it.
    """

    class ProgramA(torch.nn.Module):
        def forward(self, x):
            return x * 2.0 + 1.0

    class ProgramB(torch.nn.Module):
        def forward(self, o):
            return o + 10.0

    class ProgramC(torch.nn.Module):
        def forward(self, o):
            return o * 3.0

    # Create input and run on CPU to get golden results
    input_i_cpu = torch.randn(32, 32, dtype=torch.float32)

    program_a = ProgramA()
    program_b = ProgramB()
    program_c = ProgramC()

    # CPU execution for golden results
    expected_o = program_a(input_i_cpu)
    expected_p = program_b(expected_o)
    expected_q = program_c(expected_o)

    # Device execution
    output_o = run_model_on_device(program_a, [input_i_cpu])
    output_p = run_model_on_device(program_b, [output_o])
    output_q = run_model_on_device(program_c, [output_o])

    # Compare results using PCC
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparator(comparison_config)
    comparator.compare(output_o.cpu(), expected_o)
    comparator.compare(output_p.cpu(), expected_p)
    comparator.compare(output_q.cpu(), expected_q)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_output_reused_in_three_serial_graphs():
    """
    Test extended scenario: A(I) -> O, B(O) -> P, C(O) -> Q, D(O) -> R

    Output O is reused across three subsequent serial computations.
    """

    class ProgramA(torch.nn.Module):
        def forward(self, x):
            return x + 5.0

    class ProgramB(torch.nn.Module):
        def forward(self, o):
            return o * 2.0

    class ProgramC(torch.nn.Module):
        def forward(self, o):
            return o - 3.0

    class ProgramD(torch.nn.Module):
        def forward(self, o):
            return o / 2.0

    input_i_cpu = torch.randn(16, 16, dtype=torch.float32)

    program_a = ProgramA()
    program_b = ProgramB()
    program_c = ProgramC()
    program_d = ProgramD()

    # CPU execution for golden results
    expected_o = program_a(input_i_cpu)
    expected_p = program_b(expected_o)
    expected_q = program_c(expected_o)
    expected_r = program_d(expected_o)

    # Device execution
    output_o = run_model_on_device(program_a, [input_i_cpu])
    output_p = run_model_on_device(program_b, [output_o])
    output_q = run_model_on_device(program_c, [output_o])
    output_r = run_model_on_device(program_d, [output_o])

    # Compare results using PCC
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparator(comparison_config)
    comparator.compare(output_o.cpu(), expected_o)
    comparator.compare(output_p.cpu(), expected_p)
    comparator.compare(output_q.cpu(), expected_q)
    comparator.compare(output_r.cpu(), expected_r)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_multiple_outputs_reused_independently():
    """
    Test scenario: A(I) -> (O1, O2), B(O1) -> P, C(O2) -> Q, D(O1) -> R

    Program A produces two outputs.
    O1 is reused by B and D.
    O2 is reused by C.
    All programs run serially.
    """

    class ProgramA(torch.nn.Module):
        def forward(self, x):
            o1 = x * 2.0
            o2 = x + 10.0
            return o1, o2

    class ProgramB(torch.nn.Module):
        def forward(self, o1):
            return o1 + 5.0

    class ProgramC(torch.nn.Module):
        def forward(self, o2):
            return o2 * 3.0

    class ProgramD(torch.nn.Module):
        def forward(self, o1):
            return o1 - 2.0

    input_i_cpu = torch.randn(24, 24, dtype=torch.float32)

    program_a = ProgramA()
    program_b = ProgramB()
    program_c = ProgramC()
    program_d = ProgramD()

    # CPU execution for golden results
    expected_o1, expected_o2 = program_a(input_i_cpu)
    expected_p = program_b(expected_o1)
    expected_q = program_c(expected_o2)
    expected_r = program_d(expected_o1)

    # Device execution
    outputs = run_model_on_device(program_a, [input_i_cpu])
    output_o1, output_o2 = outputs
    output_p = run_model_on_device(program_b, [output_o1])
    output_q = run_model_on_device(program_c, [output_o2])
    output_r = run_model_on_device(program_d, [output_o1])

    # Compare results using PCC
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparator(comparison_config)
    comparator.compare(output_o1.cpu(), expected_o1)
    comparator.compare(output_o2.cpu(), expected_o2)
    comparator.compare(output_p.cpu(), expected_p)
    comparator.compare(output_q.cpu(), expected_q)
    comparator.compare(output_r.cpu(), expected_r)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_diamond_dependency_pattern():
    """
    Test diamond pattern: A(I) -> O, B(O) -> P, C(O) -> Q, D(P, Q) -> R

    Program A produces O.
    B and C both consume O (serial execution).
    D consumes both P and Q.

    This tests that O persists through both B and C, and that P and Q
    both persist until D completes.
    """

    class ProgramA(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    class ProgramB(torch.nn.Module):
        def forward(self, o):
            return o + 5.0

    class ProgramC(torch.nn.Module):
        def forward(self, o):
            return o - 3.0

    class ProgramD(torch.nn.Module):
        def forward(self, p, q):
            return p + q

    input_i_cpu = torch.randn(32, 32, dtype=torch.float32)

    program_a = ProgramA()
    program_b = ProgramB()
    program_c = ProgramC()
    program_d = ProgramD()

    # CPU execution for golden results
    expected_o = program_a(input_i_cpu)
    expected_p = program_b(expected_o)
    expected_q = program_c(expected_o)
    expected_r = program_d(expected_p, expected_q)

    # Device execution
    output_o = run_model_on_device(program_a, [input_i_cpu])
    output_p = run_model_on_device(program_b, [output_o])
    output_q = run_model_on_device(program_c, [output_o])
    output_r = run_model_on_device(program_d, [output_p, output_q])

    # Compare results using PCC
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparator(comparison_config)
    comparator.compare(output_o.cpu(), expected_o)
    comparator.compare(output_p.cpu(), expected_p)
    comparator.compare(output_q.cpu(), expected_q)
    comparator.compare(output_r.cpu(), expected_r)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_chain_with_multiple_reuses():
    """
    Test complex chain: A(I) -> O, B(O) -> P, C(O, P) -> Q, D(P) -> R

    O is used by B and C.
    P is used by C and D.
    Tests overlapping lifetimes and multiple dependency patterns.
    """

    class ProgramA(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    class ProgramB(torch.nn.Module):
        def forward(self, o):
            return o * 2.0

    class ProgramC(torch.nn.Module):
        def forward(self, o, p):
            return o + p

    class ProgramD(torch.nn.Module):
        def forward(self, p):
            return p - 5.0

    input_i_cpu = torch.randn(16, 16, dtype=torch.float32)

    program_a = ProgramA()
    program_b = ProgramB()
    program_c = ProgramC()
    program_d = ProgramD()

    # CPU execution for golden results
    expected_o = program_a(input_i_cpu)
    expected_p = program_b(expected_o)
    expected_q = program_c(expected_o, expected_p)
    expected_r = program_d(expected_p)

    # Device execution
    output_o = run_model_on_device(program_a, [input_i_cpu])
    output_p = run_model_on_device(program_b, [output_o])
    output_q = run_model_on_device(program_c, [output_o, output_p])
    output_r = run_model_on_device(program_d, [output_p])

    # Compare results using PCC
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparator(comparison_config)
    comparator.compare(output_o.cpu(), expected_o)
    comparator.compare(output_p.cpu(), expected_p)
    comparator.compare(output_q.cpu(), expected_q)
    comparator.compare(output_r.cpu(), expected_r)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_output_reused_with_matrix_operations():
    """
    Test with more realistic operations (matmul, etc.)
    A(I) -> O, B(O) -> P, C(O) -> Q

    Uses actual matrix operations instead of simple arithmetic.
    """

    class ProgramA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 64, dtype=torch.float32)

        def forward(self, x):
            return self.linear(x)

    class ProgramB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(64, 32, dtype=torch.float32)

        def forward(self, o):
            return self.linear(o)

    class ProgramC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(64, 16, dtype=torch.float32)

        def forward(self, o):
            return self.linear(o)

    input_i_cpu = torch.randn(8, 32, dtype=torch.float32)

    program_a = ProgramA()
    program_b = ProgramB()
    program_c = ProgramC()

    # CPU execution for golden results
    with torch.no_grad():
        expected_o = program_a(input_i_cpu)
        expected_p = program_b(expected_o)
        expected_q = program_c(expected_o)

    # Device execution
    output_o = run_model_on_device(program_a, [input_i_cpu])
    output_p = run_model_on_device(program_b, [output_o])
    output_q = run_model_on_device(program_c, [output_o])

    # Compare results using PCC
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparator(comparison_config)
    comparator.compare(output_o.cpu(), expected_o)
    comparator.compare(output_p.cpu(), expected_p)
    comparator.compare(output_q.cpu(), expected_q)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_input_moved_to_device_then_used_in_graph():
    """
    Test scenario: Input A is moved to device via .to(), printed/accessed (returning it to CPU) then used in graph G.

    This verifies that tensors moved to device and accessed before graph execution
    remain valid and usable as inputs.

    The print of A happens before it is moved to device during execution, so it has neither a host or runtime tensor yet.
    """

    class ProgramG(torch.nn.Module):
        def forward(self, a):
            return a * 2.0 + 5.0

    input_a_cpu = torch.randn(16, 16, dtype=torch.float32)

    program_g = ProgramG()

    # CPU execution for golden result
    expected = program_g(input_a_cpu)

    # Move input to device explicitly
    device = xm.xla_device()
    input_a_device = input_a_cpu.to(device)

    # Access the tensor (this shouldn't invalidate it for later use)
    # Note: Printing XLA tensors should trigger materialization, unless no graph
    # is traced yet which is the case here.
    input_a_cpu_back = input_a_device.cpu()
    print("Input A on CPU after moving to device and back:", input_a_cpu_back)
    assert torch.allclose(input_a_cpu_back, input_a_cpu, rtol=1e-5, atol=1e-5)

    # Compare using PCC
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparator(comparison_config)
    comparator.compare(input_a_cpu_back, input_a_cpu)

    # Now use the input in a graph
    compiled_g = torch.compile(program_g, backend="tt")
    model_on_device = compiled_g.to(device)
    output_g = model_on_device(input_a_device)

    # Compare result
    comparator.compare(output_g.cpu(), expected)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
def test_input_not_modified_reused_in_another_graph():
    """
    Test scenario: Input A participates in graph G (not modified/not returned),
    then A is reused in graph H.

    This verifies that inputs that are used but not modified or returned by a graph
    remain valid for subsequent graphs.

    This is a simple caching case, where input_a_device should persist across both graphs.
    """

    class ProgramG(torch.nn.Module):
        def forward(self, a):
            # Use 'a' but don't modify it in-place, and don't return it
            return a + 10.0

    class ProgramH(torch.nn.Module):
        def forward(self, a):
            # Reuse the same input 'a'
            return a * 3.0

    input_a_cpu = torch.randn(32, 32, dtype=torch.float32)

    program_g = ProgramG()
    program_h = ProgramH()

    # CPU execution for golden results
    expected_g = program_g(input_a_cpu)
    expected_h = program_h(input_a_cpu)

    # Device execution
    device = xm.xla_device()
    compiled_g = torch.compile(program_g, backend="tt")
    model_g = compiled_g.to(device)
    input_a_device = input_a_cpu.to(device)
    output_g = model_g(input_a_device)

    # Now reuse input_a_device in program H
    compiled_h = torch.compile(program_h, backend="tt")
    model_h = compiled_h.to(device)
    output_h = model_h(input_a_device)

    # Compare results using PCC
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.9999))
    comparator = TorchComparator(comparison_config)
    comparator.compare(output_g.cpu(), expected_g)
    comparator.compare(output_h.cpu(), expected_h)


@pytest.mark.push
@pytest.mark.nightly
def test_concurrent_buffer_instance_transfer():
    """
    Test scenario: Input A participates in some graph, and is concurrently copied to host
    by multiple framework threads.

    This tests for race conditions in the copyToHost thread instance mutex management.
    """

    class ProgramA(torch.nn.Module):
        def forward(self, A):
            return A + 1

    input_a_cpu = torch.randn(32, 32, dtype=torch.float32)

    program_a = ProgramA()

    result = run_model_on_device(program_a, [input_a_cpu])

    # Create multiple threads that all print the same result concurrently
    def print_result(thread_id):
        print(f"Thread {thread_id}: {result}")
        # time.sleep(0.1)  # Small delay to increase chance of concurrent access
        print(f"Thread {thread_id}: Shape = {result.shape}")

    threads = []
    num_threads = 10

    # Start multiple threads
    for i in range(num_threads):
        thread = threading.Thread(target=print_result, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()


@pytest.mark.push
@pytest.mark.nightly
def test_concurrent_multi_buffer_instance_transfer():
    """
    Test scenario: Inputs A and B participates in some graph, and are concurrently copied to host
    by multiple framework threads.

    This tests for race conditions in the copyToHost thread instance mutex management, and that
    there are not multiple concurrent calls to tt::runtime::submit triggering metal race conditions,
    as guarded by the static copyToHost internal mutex.
    """

    class ProgramAB(torch.nn.Module):
        def forward(self, A, B):
            return A + 1, B + 1

    input_a_cpu = torch.randn(32, 32, dtype=torch.float32)
    input_b_cpu = torch.randn(32, 32, dtype=torch.float32)

    program_ab = ProgramAB()

    res_a, res_b = run_model_on_device(program_ab, [input_a_cpu, input_b_cpu])

    def print_result(thread_id, _result):
        print(f"Result from thread_id {thread_id} = {_result}")

    threads = []
    num_threads = 10
    for i in range(num_threads):
        thread_a = threading.Thread(target=print_result, args=(i, res_a))
        thread_b = threading.Thread(target=print_result, args=(i, res_b))

        threads.append(thread_a)
        threads.append(thread_b)
        thread_a.start()
        thread_b.start()

    for thread in threads:
        thread.join()
