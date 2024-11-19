# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import jax
import jax.numpy as jnp
import numpy
from jax import export


from infrastructure import verify_module


def test_abs_op():
    def module_abs(a):
        return jnp.abs(a)

    verify_module(module_abs, [(3, 3)])
    verify_module(module_abs, [(3, 3, 3)])


# Broadcasted values are incorrect
@pytest.mark.skip("Broadcasted values are incorrect")
def test_broadcast_op():
    def module_broadcast(a):
        return jnp.broadcast_to(a, (2, 4))

    verify_module(module_broadcast, [(2, 1)])


def test_cbrt_op():
    def module_cbrt(a):
        return jax.lax.cbrt(a)

    verify_module(
        module_cbrt, [(3, 3)], required_atol=2e-2
    )  # ATOL is 0.010040640830993652
    verify_module(module_cbrt, [(3, 3, 3)], required_atol=2e-2)


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


# error: 'ttir.constant' op failed to verify that all of {value, result} have same shape
@pytest.mark.skip(
    "Index is out of bounds for the rank, should be between 0 and 0 however is 18446744073709551615"
)
def test_constant_op():
    def module_constant_zeros(a):
        zeros = jnp.zeros(a.shape)
        return zeros

    def module_constant_ones(a):
        ones = jnp.ones(a.shape)
        return ones

    def module_constant_multi(a):
        multi = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
        return multi

    verify_module(module_constant_zeros, [(3, 3)])
    verify_module(module_constant_ones, [(3, 3)])
    verify_module(module_constant_multi, [(3, 3)])


def test_convert_op():
    def module_convert(a):
        return jax.lax.convert_element_type(a, jnp.bfloat16)

    verify_module(module_convert, [(2, 2)])
    verify_module(module_convert, [(4, 4, 4)])


def test_div_op():
    def module_div(a, b):
        return a / b

    verify_module(module_div, [(3, 3), (3, 3)])
    verify_module(module_div, [(3, 3, 3), (3, 3, 3)], required_atol=35e-2)


@pytest.mark.skip("VHLO Legalization failed.")
def test_dot_general_op():
    def module_dot_general(a, b):
        return jnp.dot(a, b)

    verify_module(module_dot_general, [(2, 1), (1, 2)])
    verify_module(module_dot_general, [(1, 2), (2, 1)])


# Exponential generate slightly different values, so using higher ATOL value.
# see tt-mlir issue https://github.com/tenstorrent/tt-mlir/issues/1199)
def test_exp_op():
    def module_exp(a):
        return jnp.exp(a)

    verify_module(module_exp, [(3, 3)], required_atol=20e-2)
    verify_module(module_exp, [(3, 3, 3)], required_atol=25e-2)


def test_maximum_op():
    def module_maximum(a, b):
        return jnp.maximum(a, b)

    verify_module(module_maximum, [(3, 3), (3, 3)])
    verify_module(module_maximum, [(3, 3, 3), (3, 3, 3)])


def test_multiply_op():
    def module_multiply(a, b):
        return a * b

    verify_module(module_multiply, [(3, 3), (3, 3)])
    verify_module(module_multiply, [(3, 3, 3), (3, 3, 3)])


def test_negate_op():
    def module_negate(a):
        return -a

    verify_module(module_negate, [(3, 3)])
    verify_module(module_negate, [(3, 3, 3)])


# Reduce is failing due to error in constant.
@pytest.mark.skip("keepdim=False is not supported")
def test_reduce_op():
    def module_reduce_max(a):
        return jnp.max(a)

    def module_reduce_sum(a):
        return jnp.sum(a)

    verify_module(module_reduce_max, [(3, 3)])
    verify_module(module_reduce_max, [(3, 3, 3)])

    verify_module(module_reduce_sum, [(3, 3)])
    verify_module(module_reduce_sum, [(3, 3, 3)])


def test_rsqrt_op():
    def module_rsqrt(a):
        return jax.lax.rsqrt(a)

    verify_module(module_rsqrt, [(3, 3)])
    verify_module(module_rsqrt, [(3, 3, 3)])


# Needs to have a bigger atol due to inaccuracies in the exp op on tt-metal
# see tt-mlir issue https://github.com/tenstorrent/tt-mlir/issues/1199)
def test_expm1_op():
    def module_expm1(a):
        return jax.lax.expm1(a)

    verify_module(module_expm1, [(3, 3)], required_atol=20e-2)
    verify_module(module_expm1, [(3, 3, 3)], required_atol=20e-2)


def test_log1p_op():
    def module_log1p(a):
        return jax.lax.log1p(a)

    verify_module(module_log1p, [(3, 3)], required_atol=2e-2)
    verify_module(module_log1p, [(3, 3, 3)], required_atol=2e-2)


def test_sign_op():
    def module_sign(a):
        return jax.lax.sign(a)

    verify_module(module_sign, [(3, 3)])
    verify_module(module_sign, [(3, 3, 3)])


def test_sqrt_op():
    def module_sqrt(a):
        return jnp.sqrt(a)

    verify_module(module_sqrt, [(3, 3)])
    verify_module(module_sqrt, [(3, 3, 3)])


def test_sub_op():
    def module_sub(a, b):
        return a - b

    verify_module(module_sub, [(3, 3), (3, 3)])
    verify_module(module_sub, [(3, 3, 3), (3, 3, 3)])


def test_transpose_op_2d():
    def module_transpose(a):
        return jnp.transpose(a)

    verify_module(module_transpose, [(3, 3)])

def test_shape_scalar():
    def module_shape_scalar(a):
        return a.shape[0]

    verify_module(module_shape_scalar, [(3, 3)])


# Transpose op failing for higher ranks/dimensions.
@pytest.mark.skip("Transpose op failing for higher ranks/dimensions.")
def test_transpose_op_3d():
    def module_transpose(a):
        return jnp.transpose(a)

    verify_module(module_transpose, [(3, 3, 3)])


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
    "begin, end, dim", [*dim2_cases, *dim3_cases, *dim0_cases, *dim1_cases]
)
@pytest.mark.skip("Requires tt-metal uplift.")
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
