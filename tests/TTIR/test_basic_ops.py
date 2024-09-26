# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import jax
import jax.numpy as jnp

from infrastructure import verify_module


def test_abs_op():
  def module_abs(a):
    return jnp.abs(a)

  verify_module(module_abs, [(3, 3)])
  verify_module(module_abs, [(3, 3, 3)])


#Broadcasted values are incorrect
@pytest.mark.skip("Broadcasted values are incorrect")
def test_broadcast_op():
  def module_broadcast(a):
    return jnp.broadcast_to(a, (2, 4))

  verify_module(module_broadcast, [(2, 1)])


#error: 'ttir.constant' op failed to verify that all of {value, result} have same shape
@pytest.mark.skip("Index is out of bounds for the rank, should be between 0 and 0 however is 18446744073709551615")
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


def test_dot_general_op():
  def module_dot_general(a, b):
    return jnp.dot(a, b)

  verify_module(module_dot_general, [(2, 1), (1, 2)])
  verify_module(module_dot_general, [(1, 2), (2, 1)])


# Exponential generate slightly different values, so using higher ATOL value.
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
  

#Reduce is failing due to error in constant.
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


# Transpose op failing for higher ranks/dimensions.
@pytest.mark.skip("Transpose op failing for higher ranks/dimensions.")
def test_transpose_op_3d():
  def module_transpose(a):
    return jnp.transpose(a)

  verify_module(module_transpose, [(3, 3, 3)])

