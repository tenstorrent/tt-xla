# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp
import numpy

from infrastructure import verify_module


@pytest.mark.parametrize("input_shapes", [[(3, 3)], [(3, 3, 3)]])
def test_abs_op(input_shapes):
    def module_abs(a):
        return jnp.abs(a)

    verify_module(module_abs, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(2, 1)]])
@pytest.mark.skip(
    "Broadcasted values are incorrect. "
    "Fails with: AssertionError: PCC is 0.37796446681022644 which is less than 0.99"
)
def test_broadcast_op(input_shapes):
    def module_broadcast(a):
        return jnp.broadcast_to(a, (2, 4))

    verify_module(module_broadcast, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(3, 3)], [(3, 3, 3)]])
def test_cbrt_op(input_shapes):
    def module_cbrt(a):
        return jax.lax.cbrt(a)

    verify_module(module_cbrt, input_shapes, required_atol=2e-2)


def test_concat_op():
    def module_concat_dim_0(x, y):
        return jnp.concatenate([x, y], axis=0)

    def module_concat_dim_1(x, y):
        return jnp.concatenate([x, y], axis=1)

    def module_concat_dim_2(x, y):
        return jnp.concatenate([x, y], axis=2)

    def module_concat_dim_3(x, y):
        return jnp.concatenate([x, y], axis=3)

    verify_module(module_concat_dim_0, [(32, 32), (64, 32)])  # output shape: (96, 32)
    verify_module(
        module_concat_dim_0, [(32, 32, 32), (64, 32, 32)]
    )  # output shape: (96, 32, 32)
    verify_module(module_concat_dim_1, [(32, 32), (32, 64)])  # output shape: (32, 96)
    verify_module(
        module_concat_dim_1, [(32, 32, 32), (32, 32, 32)]
    )  # output shape: (32, 64, 32)
    verify_module(
        module_concat_dim_2, [(32, 32, 32), (32, 32, 64)]
    )  # output shape: (32, 32, 96)
    verify_module(
        module_concat_dim_2, [(32, 32, 32, 32), (32, 32, 64, 32)]
    )  # output shape: (32, 32, 96, 32)
    verify_module(
        module_concat_dim_3, [(32, 32, 32, 32), (32, 32, 32, 64)]
    )  # output shape: (32, 32, 32, 96)


@pytest.mark.parametrize("input_shapes", [[(3, 3)]])
@pytest.mark.skip("AssertionError: ATOL is 21574.4375 which is greater than 0.01")
def test_constant_op(input_shapes):
    def module_constant_zeros(a):
        zeros = jnp.zeros(a.shape)
        return zeros

    def module_constant_ones(a):
        ones = jnp.ones(a.shape)
        return ones

    verify_module(module_constant_zeros, input_shapes)
    verify_module(module_constant_ones, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(3, 3)]])
@pytest.mark.skip("Fails due to: error: failed to legalize operation 'ttir.constant'")
def test_constant_op_multi_dim(input_shapes):
    def module_constant_multi(a):
        multi = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
        return multi

    verify_module(module_constant_multi, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(2, 2)], [(4, 4, 4)]])
def test_convert_op(input_shapes):
    def module_convert(a):
        return jax.lax.convert_element_type(a, jnp.bfloat16)

    verify_module(module_convert, input_shapes)


@pytest.mark.parametrize(
    ["input_shapes", "required_atol"],
    [([(3, 3), (3, 3)], 0.01), ([(3, 3, 3), (3, 3, 3)], 35e-2)],
)
def test_div_op(input_shapes, required_atol):
    def module_div(a, b):
        return a / b

    verify_module(module_div, input_shapes, required_atol=required_atol)


@pytest.mark.parametrize(
    "input_shapes",
    [[(2, 1), (1, 2)], [(1, 2), (2, 1)]],
)
def test_dot_general_op(input_shapes):
    def module_dot_general(a, b):
        return jnp.dot(a, b)

    verify_module(module_dot_general, input_shapes)


# Exponential generate slightly different values, so using higher ATOL value.
# see tt-mlir issue https://github.com/tenstorrent/tt-mlir/issues/1199)
@pytest.mark.parametrize(
    ["input_shapes", "required_atol"], [([(3, 3)], 20e-2), ([(3, 3, 3)], 25e-2)]
)
def test_exp_op(input_shapes, required_atol):
    def module_exp(a):
        return jnp.exp(a)

    verify_module(module_exp, input_shapes, required_atol=required_atol)


@pytest.mark.parametrize("input_shapes", [[(3, 3), (3, 3)], [(3, 3, 3), (3, 3, 3)]])
def test_maximum_op(input_shapes):
    def module_maximum(a, b):
        return jnp.maximum(a, b)

    verify_module(module_maximum, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(3, 3), (3, 3)], [(3, 3, 3), (3, 3, 3)]])
def test_multiply_op(input_shapes):
    def module_multiply(a, b):
        return a * b

    verify_module(module_multiply, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(3, 3)], [(3, 3, 3)]])
def test_negate_op(input_shapes):
    def module_negate(a):
        return -a

    verify_module(module_negate, input_shapes)


# Reduce is failing due to error in constant.
@pytest.mark.parametrize("input_shapes", [[(3, 3)], [(3, 3, 3)]])
@pytest.mark.skip("keepdim=False is not supported")
def test_reduce_op(input_shapes):
    def module_reduce_max(a):
        return jnp.max(a)

    def module_reduce_sum(a):
        return jnp.sum(a)

    verify_module(module_reduce_max, input_shapes)
    verify_module(module_reduce_sum, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(3, 3)], [(3, 3, 3)]])
def test_rsqrt_op(input_shapes):
    def module_rsqrt(a):
        return jax.lax.rsqrt(a)

    verify_module(module_rsqrt, input_shapes)


# Needs to have a bigger atol due to inaccuracies in the exp op on tt-metal
# see tt-mlir issue https://github.com/tenstorrent/tt-mlir/issues/1199)
@pytest.mark.parametrize("input_shapes", [[(3, 3)], [(3, 3, 3)]])
def test_expm1_op(input_shapes):
    def module_expm1(a):
        return jax.lax.expm1(a)

    verify_module(module_expm1, input_shapes, required_atol=20e-2)


@pytest.mark.parametrize("input_shapes", [[(3, 3)], [(3, 3, 3)]])
def test_log1p_op(input_shapes):
    def module_log1p(a):
        return jax.lax.log1p(a)

    verify_module(module_log1p, input_shapes, required_atol=2e-2)


@pytest.mark.parametrize("input_shapes", [[(3, 3)], [(3, 3, 3)]])
def test_sign_op(input_shapes):
    def module_sign(a):
        return jax.lax.sign(a)

    verify_module(module_sign, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(3, 3)], [(3, 3, 3)]])
def test_sqrt_op(input_shapes):
    def module_sqrt(a):
        return jnp.sqrt(a)

    verify_module(module_sqrt, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(3, 3), (3, 3)], [(3, 3, 3), (3, 3, 3)]])
def test_sub_op(input_shapes):
    def module_sub(a, b):
        return a - b

    verify_module(module_sub, input_shapes)


@pytest.mark.parametrize("input_shapes", [[(3, 3)], [(3, 3, 3)]])
def test_transpose_op(input_shapes):
    def module_transpose(a):
        return jnp.transpose(a)

    verify_module(module_transpose, input_shapes)


def test_scalar_type():
    def module_scalar_type(a):
        return a.shape[0]

    verify_module(module_scalar_type, [(3, 3)])


dim0_cases = []
for begin in numpy.arange(10).tolist():
    for end in numpy.arange(90, 100).tolist():
        dim0_cases.append((begin, end, 0))

dim1_cases = []
for begin in numpy.arange(10).tolist():
    for end in numpy.arange(90, 100).tolist():
        dim1_cases.append((begin, end, 1))

dim2_cases = []
for begin in numpy.arange(0, 64, 32).tolist():
    for end in numpy.arange(64, 128, 32).tolist():
        dim2_cases.append((begin, end, 2))

dim3_cases = []
for begin in numpy.arange(0, 64, 32).tolist():
    for end in numpy.arange(64, 128, 32).tolist():
        dim3_cases.append((begin, end, 3))


@pytest.mark.parametrize(
    ["begin", "end", "dim"], [*dim2_cases, *dim3_cases, *dim0_cases, *dim1_cases]
)
def test_slice(begin, end, dim):
    def module_slice(a):
        if dim == 0:
            return a[begin:end, :, :, :]
        elif dim == 1:
            return a[:, begin:end, :, :]
        elif dim == 2:
            return a[:, :, begin:end, :]
        else:
            return a[:, :, :, begin:end]

    shape = [10, 10, 10, 10]
    shape[dim] = 128
    verify_module(module_slice, [shape])


@pytest.mark.parametrize(
    "input_shapes",
    [
        [(32, 32), (32, 32)],
        pytest.param(
            [(3, 3), (3, 3)],
            marks=pytest.mark.skip(
                reason="Fails due to https://github.com/tenstorrent/tt-xla/issues/70"
            ),
        ),
        pytest.param(
            [(3, 3, 3), (3, 3, 3)],
            marks=pytest.mark.skip(
                reason="Fails due to https://github.com/tenstorrent/tt-xla/issues/70"
            ),
        ),
    ],
)
def test_remainder_op_lax(input_shapes):
    def module_remainder_lax(a, b):
        return jax.lax.rem(a, b)

    verify_module(module_remainder_lax, input_shapes, required_atol=0.02)


@pytest.mark.parametrize(
    "input_shapes",
    [
        pytest.param(
            [(32, 32), (32, 32)],
            marks=pytest.mark.skip(
                reason="Fails due to https://github.com/tenstorrent/tt-xla/issues/71"
            ),
        ),
        pytest.param(
            [(3, 3), (3, 3)],
            marks=pytest.mark.skip(
                reason="Fails due to https://github.com/tenstorrent/tt-xla/issues/70"
            ),
        ),
        pytest.param(
            [(3, 3, 3), (3, 3, 3)],
            marks=pytest.mark.skip(
                reason="Fails due to https://github.com/tenstorrent/tt-xla/issues/70"
            ),
        ),
    ],
)
def test_remainder_op_jnp(input_shapes):
    # `jnp.remainder` generates a more complex stablehlo graph than `jax.lax.rem` with
    # implicit broadcasts, etc. That's why we have both.
    def module_remainder_jnp(a, b):
        return jnp.remainder(a, b)

    verify_module(module_remainder_jnp, input_shapes, required_atol=0.02)
