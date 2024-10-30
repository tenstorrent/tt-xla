# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from typing import List, Sequence

from utils import fullname, passed, failed


def random_input_tensor(
    shape: Sequence[int], key: int = 42, on_device: bool = False, dtype=jnp.float32
):
    device_cpu = jax.devices("cpu")[0]
    with jax.default_device(device_cpu):
        tensor = jax.random.uniform(jax.random.PRNGKey(key), shape=shape, dtype=dtype)

    # Although the random tensor is generated on cpu but it is not committed to
    # cpu; so this tensor can be moved to the device and subsequent code will
    # execute on the device. Placing the generated tensor explicitly to cpu or
    # device to avoid unwanted behavior.
    if on_device:
        tensor = jax.device_put(tensor, jax.devices()[0])
    else:
        tensor = jax.device_put(tensor, device_cpu)

    return tensor


def compare_tensor_to_golden(
    tensor: jax.Array,
    golden: jax.Array,
    required_pcc: float = 0.99,
    required_atol: float = 1e-2,
    assert_on_error: bool = True,
):
    ret = True
    if tensor.device != golden.device:
        tensor = jax.device_put(tensor, golden.device)

    ret = ret and tensor.shape == golden.shape
    if assert_on_error:
        assert ret, "Shapes do not match"

    if not tensor.flatten().size == 1:  # pcc invalid for scalar values
        pcc = jnp.min(jnp.corrcoef(tensor.flatten(), golden.flatten()))
        ret = ret and pcc >= required_pcc
        if assert_on_error:
            assert ret, f"PCC is {pcc} which is less than {required_pcc}"

    atol = jnp.max(jnp.abs(tensor - golden))
    ret = ret and atol <= required_atol
    if assert_on_error:
        assert ret, f"ATOL is {atol} which is greater than {required_atol}"

    return ret


def verify_test_module(
    *input_shapes: List[List[Sequence[int]]],
    key: int = 42,
    required_pcc: float = 0.99,
    required_atol: float = 1e-2,
    dtype=jnp.float32,
):
    """
    Decorator for tests which runs test op on device and compares results with golden.

    Note that in `*input_shapes` is defined in such a way that it supports testing the
    op with multiple different input shapes. For example:

    ```python
    @verify([(32, 32), (32, 32)])
    def test_concat_dim_0(x, y):
        return jnp.concatenate([x, y], axis=0)
    ```

    will produce test with x and y being (32, 32) tensors. You can also do

    ```python
    @verify([(32, 32), (32, 32)], [(64, 64), (64, 64)])
    def test_concat_dim_0(x, y):
        return jnp.concatenate([x, y], axis=0)
    ```

    to produce/run one test with (32, 32) shape for inputs, and one with (64, 64) shape
    for inputs.

    Also note that since `*inputs_shapes` is defined with a `*`, every other argument
    of `verify` must be keyworded like

    ```python
    @verify([(32, 32), (32, 32)], [(64, 64), (64, 64)], required_pcc=0.8)
    def test_concat_dim_0(x, y):
        return jnp.concatenate([x, y], axis=0)
    ```
    """

    def decorator(test_fn):
        def wrapper():
            for in_shapes in input_shapes:
                assert len(jax.devices()) != 0, "No TT devices found!"
                tt_device = jax.devices()[0]
                cpu_inputs = [
                    random_input_tensor(shape, key + i, dtype=dtype)
                    for i, shape in enumerate(in_shapes)
                ]
                tt_inputs = [jax.device_put(cpu_input, tt_device) for cpu_input in cpu_inputs]
                graph = jax.jit(test_fn)
                res = graph(*tt_inputs)
                res_cpu = graph(*cpu_inputs)

                if compare_tensor_to_golden(res, res_cpu, required_pcc, required_atol):
                    passed(f"{fullname(test_fn)}{in_shapes}")
                else:
                    failed(f"{fullname(test_fn)}{in_shapes}")

        return wrapper

    return decorator
