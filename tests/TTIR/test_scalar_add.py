# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp

def test_scalar_add():
  pytest.skip("Not working")
  def module_add(a, b):
    return a + b

  a = jnp.float32(5.)
  b = jnp.array(6.)
  tt_graph = jax.jit(module_add)
  res = tt_graph(a, b)
  cpu_graph = jax.jit(module_add, backend='cpu')
  res_cpu = cpu_graph(a, b)
  assert jnp.allclose(res, res_cpu)
