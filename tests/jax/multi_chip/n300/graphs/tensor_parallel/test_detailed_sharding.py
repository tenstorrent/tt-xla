# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Detailed tests for tensor sharding with various mesh configurations.

These tests exercise different sharding patterns including:
- Basic sharded matmul
- Chain of operations with different shardings
- Output-to-input pipelines (buffer reuse)
- Multiple sequential operations reusing buffers
- Different tensor sizes
- Replicated sharding
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

jax.config.update("jax_use_shardy_partitioner", True)


def create_test_matrices(n, scale=0.01):
    """Create test matrices with normalized values to avoid overflow.

    Args:
        n: Matrix size (n x n)
        scale: Scaling factor for values

    Returns:
        Tuple of (ones_matrix, sequential_matrix) as jax arrays
    """
    ones = jnp.ones((n, n), dtype=jnp.float32) * scale
    seq = (jnp.arange(n * n).reshape(n, n).astype(jnp.float32) / (n * n)) * 2
    return ones, seq


def create_test_matrices_np(n, scale=0.01):
    """Create test matrices with normalized values (numpy version for expected).

    Args:
        n: Matrix size (n x n)
        scale: Scaling factor for values

    Returns:
        Tuple of (ones_matrix, sequential_matrix) as numpy arrays
    """
    ones = np.ones((n, n), dtype=np.float32) * scale
    seq = (np.arange(n * n).reshape(n, n).astype(np.float32) / (n * n)) * 2
    return ones, seq


def verify_result(name, actual, expected, rtol=2e-2, atol=0.5):
    """Verify result by copying to CPU and comparing with numpy.

    Note: Using relaxed tolerances (rtol=2e-2, atol=0.5) because TT hardware
    has lower precision than CPU for floating point operations.
    """
    actual_np = np.array(actual)
    assert np.allclose(
        actual_np, expected, rtol=rtol, atol=atol
    ), f"{name}: max diff: {np.max(np.abs(actual_np - expected))}"


@pytest.fixture
def tt_devices():
    """Get TT devices for tests."""
    return jax.devices("tt")


@pytest.fixture
def mesh_1x2(tt_devices):
    """Create a 1x2 mesh for tensor parallel tests."""
    return jax.make_mesh((1, 2), ("x", "y"), devices=tt_devices)


@pytest.mark.nightly
@pytest.mark.push
def test_basic_sharded_matmul(mesh_1x2):
    """
    Test basic sharding with mesh (1, 2).

    Verifies that a simple sharded matmul works correctly with:
    - Input A sharded on 'x' axis
    - Input B sharded on 'y' axis
    - Output sharded on both axes
    """

    @partial(
        jax.jit,
        in_shardings=(
            NamedSharding(mesh_1x2, P("x", None)),
            NamedSharding(mesh_1x2, P(None, "y")),
        ),
        out_shardings=NamedSharding(mesh_1x2, P("x", "y")),
    )
    def matmul_sharded(a, b):
        return a @ b

    N = 64
    a, b = create_test_matrices(N)
    a_np, b_np = create_test_matrices_np(N)

    a_sharded = jax.device_put(a, NamedSharding(mesh_1x2, P("x", None)))
    b_sharded = jax.device_put(b, NamedSharding(mesh_1x2, P(None, "y")))

    result = matmul_sharded(a_sharded, b_sharded)

    expected = a_np @ b_np
    verify_result("basic_sharded_matmul", result, expected)


@pytest.mark.nightly
@pytest.mark.push
def test_triple_matmul_chain(mesh_1x2):
    """
    Test chain of operations (x @ y @ z) with different shardings.

    Verifies that multiple chained operations work correctly with
    different sharding patterns on each input.
    """

    @partial(
        jax.jit,
        in_shardings=(
            NamedSharding(mesh_1x2, P("x", None)),
            NamedSharding(mesh_1x2, P(None, "y")),
            NamedSharding(mesh_1x2, P(None, None)),
        ),
        out_shardings=NamedSharding(mesh_1x2, P("x", "y")),
    )
    def triple_matmul(x, y, z):
        t = x @ y
        t = t @ z
        return t

    N = 64
    x, y = create_test_matrices(N)
    _, z = create_test_matrices(N)
    x_np, y_np = create_test_matrices_np(N)
    _, z_np = create_test_matrices_np(N)

    x_sharded = jax.device_put(x, NamedSharding(mesh_1x2, P("x", None)))
    y_sharded = jax.device_put(y, NamedSharding(mesh_1x2, P(None, "y")))
    z_sharded = jax.device_put(z, NamedSharding(mesh_1x2, P(None, None)))

    result = triple_matmul(x_sharded, y_sharded, z_sharded)

    expected = (x_np @ y_np) @ z_np
    verify_result("triple_matmul_chain", result, expected)


@pytest.mark.nightly
@pytest.mark.push
def test_output_reused_as_input(mesh_1x2):
    """
    Test reusing output from one computation as input to another.

    This exercises the on-device sharding path where buffers are created
    via device-to-device copy (copyFromBuffer) and then used as inputs to
    subsequent executions.
    """

    @partial(
        jax.jit,
        in_shardings=(
            NamedSharding(mesh_1x2, P("x", None)),
            NamedSharding(mesh_1x2, P(None, "y")),
        ),
        out_shardings=NamedSharding(mesh_1x2, P("x", "y")),
    )
    def first_matmul(x, y):
        t = x @ y
        t = t @ y
        return t

    @partial(
        jax.jit,
        in_shardings=(
            NamedSharding(mesh_1x2, P("x", "y")),
            NamedSharding(mesh_1x2, P(None, None)),
        ),
        out_shardings=NamedSharding(mesh_1x2, P("x", "y")),
    )
    def continue_computation(prev_result, w):
        return prev_result @ w

    N = 64
    x, y = create_test_matrices(N)
    x_np, y_np = create_test_matrices_np(N)

    x_sharded = jax.device_put(x, NamedSharding(mesh_1x2, P("x", None)))
    y_sharded = jax.device_put(y, NamedSharding(mesh_1x2, P(None, "y")))

    # First computation
    result_1 = first_matmul(x_sharded, y_sharded)
    expected_1 = (x_np @ y_np) @ y_np

    # Second computation reusing result_1
    w = jnp.eye(N, dtype=jnp.float32) * 0.5
    w_sharded = jax.device_put(w, NamedSharding(mesh_1x2, P(None, None)))
    result_2 = continue_computation(result_1, w_sharded)

    expected_2 = expected_1 * 0.5
    verify_result("output_reused_as_input_first", result_1, expected_1)
    verify_result("output_reused_as_input_second", result_2, expected_2)


@pytest.mark.nightly
@pytest.mark.push
def test_multiple_sequential_operations_reusing_buffers(mesh_1x2):
    """
    Test multiple sequential operations reusing the same input buffers.

    Verifies that input buffers can be reused across multiple executions
    without being invalidated.
    """

    @partial(
        jax.jit,
        in_shardings=(
            NamedSharding(mesh_1x2, P("x", None)),
            NamedSharding(mesh_1x2, P(None, "y")),
        ),
        out_shardings=NamedSharding(mesh_1x2, P("x", "y")),
    )
    def matmul_add(a, b):
        return a @ b + 0.1

    N = 64
    a, b = create_test_matrices(N)
    a_np, b_np = create_test_matrices_np(N)

    a_sharded = jax.device_put(a, NamedSharding(mesh_1x2, P("x", None)))
    b_sharded = jax.device_put(b, NamedSharding(mesh_1x2, P(None, "y")))

    # First computation
    result_1 = matmul_add(a_sharded, b_sharded)

    # Second computation with same inputs (should reuse buffers)
    result_2 = matmul_add(a_sharded, b_sharded)

    expected = (a_np @ b_np) + 0.1
    verify_result("sequential_ops_first", result_1, expected)
    verify_result("sequential_ops_second", result_2, expected)


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize("size", [32, 64, 128])
def test_different_tensor_sizes(mesh_1x2, size):
    """
    Test sharding with different tensor sizes.

    Parametrized test to verify sharding works correctly across
    various matrix dimensions.
    """

    @partial(
        jax.jit,
        in_shardings=(
            NamedSharding(mesh_1x2, P("x", None)),
            NamedSharding(mesh_1x2, P(None, "y")),
        ),
        out_shardings=NamedSharding(mesh_1x2, P("x", "y")),
    )
    def matmul_sized(a, b):
        return a @ b

    a, b = create_test_matrices(size)
    a_np, b_np = create_test_matrices_np(size)

    a_sharded = jax.device_put(a, NamedSharding(mesh_1x2, P("x", None)))
    b_sharded = jax.device_put(b, NamedSharding(mesh_1x2, P(None, "y")))

    result = matmul_sized(a_sharded, b_sharded)

    expected = a_np @ b_np
    verify_result(f"tensor_size_{size}", result, expected)


@pytest.mark.nightly
@pytest.mark.push
def test_replicated_sharding(mesh_1x2):
    """
    Test replicated sharding where tensors are fully replicated across devices.

    Verifies that replicated (non-sharded) tensors work correctly.
    """

    @partial(
        jax.jit,
        in_shardings=(
            NamedSharding(mesh_1x2, P(None, None)),  # Replicated
            NamedSharding(mesh_1x2, P(None, None)),  # Replicated
        ),
        out_shardings=NamedSharding(mesh_1x2, P(None, None)),  # Replicated
    )
    def matmul_replicated(a, b):
        return a @ b

    N = 64
    a, b = create_test_matrices(N)
    a_np, b_np = create_test_matrices_np(N)

    a_rep = jax.device_put(a, NamedSharding(mesh_1x2, P(None, None)))
    b_rep = jax.device_put(b, NamedSharding(mesh_1x2, P(None, None)))

    result = matmul_replicated(a_rep, b_rep)

    expected = a_np @ b_np
    verify_result("replicated_sharding", result, expected)
