import jax
import jax.numpy as jnp
from jax import random

import os
import sys
import jax._src.xla_bridge as xb

def init_device():
    backend = "tt"
    path = os.path.join(os.path.dirname(__file__), "/proj_sw/user_dev/ajakovljevic/work_new/tt-xla/build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    print("Loading tt_pjrt C API plugin", file=sys.stderr)
    xb.discover_pjrt_plugins()

    plugin = xb.register_plugin("tt", priority=500, library_path=path, options=None)
    print("Loaded", file=sys.stderr)
    jax.config.update("jax_platforms", "tt,cpu")

init_device()
current_device = jax.devices()[0]

def f(x):
    return x**2

@jax.jit
def sgd_optimization(x, learning_rate):
    grad = jax.grad(f)(x)
    x = x - learning_rate * grad
    return x

x_init = 5.0  # Starting point
learning_rate = 0.1
num_iterations = 10

x = x_init

for i in range(num_iterations):
    x_old = x
    x = sgd_optimization(x, learning_rate)
    print(f"Iteration {i}, x = {x}, f(x) = {f(x)}")

# Final result
print(f"Optimized x = {x}, f(x) = {f(x)}")