# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp

from infrastructure import verify_test_module


@verify_test_module([(3, 3)], [(3, 3, 3)])
def test_abs_op(a):
    return jnp.abs(a)


@pytest.mark.skip("Broadcasted values are incorrect")
@verify_test_module([(2, 1)])
def test_broadcast_op(a):
    return jnp.broadcast_to(a, (2, 4))


@verify_test_module([(3, 3)], [(3, 3, 3)], required_atol=2e-2)
def test_cbrt_op(a):
    return jax.lax.cbrt(a)


def test_concat_op():
    @verify_test_module([(32, 32), (64, 32)], [(32, 32, 32), (64, 32, 32)])
    def test_concat_op_dim_0(x, y):
        return jnp.concatenate([x, y], axis=0)

    @verify_test_module([(32, 32), (32, 64)], [(32, 32, 32), (32, 32, 32)])
    def test_concat_op_dim_1(x, y):
        return jnp.concatenate([x, y], axis=1)

    @verify_test_module([(32, 32, 32), (32, 32, 64)], [(32, 32, 32, 32), (32, 32, 64, 32)])
    def test_concat_op_dim_2(x, y):
        return jnp.concatenate([x, y], axis=2)

    @verify_test_module([(32, 32, 32, 32), (32, 32, 32, 64)])
    def test_concat_op_dim_3(x, y):
        return jnp.concatenate([x, y], axis=3)

    test_concat_op_dim_0()
    test_concat_op_dim_1()
    test_concat_op_dim_2()
    test_concat_op_dim_3()


# error: 'ttir.constant' op failed to verify that all of {value, result} have same shape
@pytest.mark.skip(
    "Index is out of bounds for the rank, should be between 0 and 0 however is 18446744073709551615"
)
def test_constant_op():
    @verify_test_module([(3, 3)])
    def test_constant_zeros(a):
        zeros = jnp.zeros(a.shape)
        return zeros

    @verify_test_module([(3, 3)])
    def test_constant_ones(a):
        ones = jnp.ones(a.shape)
        return ones

    @verify_test_module([(3, 3)])
    def test_constant_multi(a):
        multi = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
        return multi

    test_constant_zeros()
    test_constant_ones()
    test_constant_multi()


@verify_test_module([(2, 2)], [(4, 4, 4)])
def test_convert_op(a):
    return jax.lax.convert_element_type(a, jnp.bfloat16)


def test_div_op():
    @verify_test_module([(3, 3), (3, 3)])
    def test_div_2d(a, b):
        return a / b

    @verify_test_module([(3, 3, 3), (3, 3, 3)], required_atol=35e-2)
    def test_div_3d(a, b):
        return a / b

    test_div_2d()
    test_div_3d()


@pytest.mark.skip("VHLO Legalization failed.")
@verify_test_module([(2, 1), (1, 2)], [(1, 2), (2, 1)])
def test_dot_general_op(a, b):
    return jnp.dot(a, b)


# Exponential generate slightly different values, so using higher ATOL value.
def test_exp_op():
    @verify_test_module([(3, 3)], required_atol=20e-2)
    def test_exp_2d(a):
        return jnp.exp(a)

    @verify_test_module([(3, 3, 3)], required_atol=25e-2)
    def test_exp_3d(a):
        return jnp.exp(a)

    test_exp_2d()
    test_exp_3d()


@verify_test_module([(3, 3), (3, 3)], [(3, 3, 3), (3, 3, 3)])
def test_maximum_op(a, b):
    return jnp.maximum(a, b)


@verify_test_module([(3, 3), (3, 3)], [(3, 3, 3), (3, 3, 3)])
def test_multiply_op(a, b):
    return a * b


@verify_test_module([(3, 3)], [(3, 3, 3)])
def test_negate_op(a):
    return -a


# Reduce is failing due to error in constant.
@pytest.mark.skip("keepdim=False is not supported")
def test_reduce_op():
    @verify_test_module([(3, 3)], [(3, 3, 3)])
    def test_reduce_max(a):
        return jnp.max(a)

    @verify_test_module([(3, 3)], [(3, 3, 3)])
    def test_reduce_sum(a):
        return jnp.sum(a)

    test_reduce_max()
    test_reduce_sum()


@verify_test_module([(3, 3)], [(3, 3, 3)])
def test_rsqrt_op(a):
    return jax.lax.rsqrt(a)


@verify_test_module([(3, 3)], [(3, 3, 3)])
def test_sqrt_op(a):
    return jnp.sqrt(a)


@verify_test_module([(3, 3), (3, 3)], [(3, 3, 3), (3, 3, 3)])
def test_sub_op(a, b):
    return a - b


@verify_test_module([(3, 3)])
def test_transpose_op_2d(a):
    return jnp.transpose(a)


# Transpose op failing for higher ranks/dimensions.
@pytest.mark.skip("Transpose op failing for higher ranks/dimensions.")
@verify_test_module([(3, 3, 3)])
def test_transpose_op_3d(a):
    return jnp.transpose(a)


dim0_cases = [(begin, end, 0) for begin in range(10) for end in range(90, 100)]
dim1_cases = [(begin, end, 1) for begin in range(10) for end in range(90, 100)]
dim2_cases = [(begin, end, 2) for begin in range(0, 64, 32) for end in range(64, 128, 32)]
dim3_cases = [(begin, end, 3) for begin in range(0, 64, 32) for end in range(64, 128, 32)]


@pytest.mark.parametrize("begin, end, dim", [*dim2_cases, *dim3_cases, *dim0_cases, *dim1_cases])
def test_slice_op(begin, end, dim):
    shape = [10, 10, 10, 10]
    shape[dim] = 128

    @verify_test_module([shape])
    def test_slice(a):
        if dim == 0:
            return a[begin:end, :, :, :]
        elif dim == 1:
            return a[:, begin:end, :, :]
        elif dim == 2:
            return a[:, :, begin:end, :]
        else:
            return a[:, :, :, begin:end]

    test_slice()
