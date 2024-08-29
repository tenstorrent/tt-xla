# RUN: %PYTHON %s | FileCheck %s

import os
import jax._src.xla_bridge as xb
import jax
import jax.numpy as jnp

def initialize():
  path = os.path.join(os.path.dirname(__file__), "../../build/src/tt/pjrt_plugin_tt.so")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

  print("Loading tt_pjrt C API plugin")
  plugin = xb.register_plugin('tt', library_path=path, options=None)
  print("Loaded")
  jax.config.update("jax_platforms", "tt")

print("Starting Test")
initialize()
def module_add(a, b):
  return a + b
print("Starting test")
a = jnp.array([[1., 2.], [3., 4.]])
b = jnp.array([[5., 6.], [7., 8.]])
graph = jax.jit(module_add)
res = graph(a, b)
print(res)

# CHECK: [ 6. 8.]
# CHECK: [10. 12.]
