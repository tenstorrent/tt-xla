# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import jax
import jax.numpy as jnp

# Importing Jax functions useful for tracing/interpreting.
from functools import wraps

from jax.extend import core
from jax._src.util import safe_map, unzip2

import numpy as np

import pprint

#########################################
# Register cpu and tt plugin. tt plugin is registered with higher priority; so
# program will execute on tt device if not specified otherwise.
import os
import jax._src.xla_bridge as xb
import sys


def initialize():
    backend = "tt"
    path = os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    print("Loading tt_pjrt C API plugin", file=sys.stderr)
    xb.discover_pjrt_plugins()

    plugin = xb.register_plugin("tt", priority=500, library_path=path, options=None)
    print("Loaded", file=sys.stderr)
    jax.config.update("jax_platforms", "tt,cpu")


initialize()
########################################


@dataclass
class GoldenJaxprTracer(jax.core.Tracer):
    device_data: jnp.array
    golden_data: jnp.array

    def lift(val):
        return GoldenJaxprTracer(val, val)

    def __repr__(self):
        return f"GoldenJaxprTracer(\n{pprint.pformat((self.device_data, self.golden_data))}\n)"


def M(x, y):
    with jax.default_device(jax.devices("cpu")[0]):
        print(f"Linf/ATOL\t: {jnp.max(jnp.abs(x - y))}", end="\t")
        print(f"PCC\t\t: {jnp.corrcoef(x.flatten(), y.flatten())[0, 1]}")
        # print(f"RMSE\t\t: {jnp.sqrt(jnp.mean((x - y) ** 2))}")
        # print(f"MAE\t\t: {jnp.mean(jnp.abs(x - y))}")
        # print(f"Median AE\t: {jnp.median(jnp.abs(x - y))}")


def eval_jaxpr_golden(jaxpr, consts, *args):
    # Mapping from variable -> value
    env = {}

    def read(var):
        # Literals are values baked into the Jaxpr
        if type(var) is core.Literal:
            return GoldenJaxprTracer.lift(var.val)
        return env[var]

    # dead, left in for convenience
    def read_final(var):
        if type(var) is core.Literal:
            return var.val
        val = env[var]
        if type(val) is GoldenJaxprTracer:
            return val.device_data
        return val

    def write(var, val):
        if type(val) is GoldenJaxprTracer:
            env[var] = val
        else:
            env[var] = GoldenJaxprTracer.lift(val)

    # Bind args and consts to environment
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # Loop through equations and evaluate primitives using `bind`
    for eqn in jaxpr.eqns:
        # Read inputs to equation from environment
        invals = safe_map(read, eqn.invars)
        # `bind` is how a primitive is called
        ###############################################
        # Here is the juicy bit
        # Logic:
        #   On input we have tracers carrying both CPU side and device side data.
        #   We also have flexibility to freely migrate values between devices
        #   This allows us to implement both "Parallel" and "CPU prop" modes from tt-blacksmith accuracy debugger, at the same time
        #   1) Parallel:
        #      There are no migrations, device input is used for the OP on device; likewise for CPU
        #      We get device and golden and we compare the two. This gives us accuracy up to that point in the model
        #   2) CPU prop:
        #      We migrate the golden to the device and run the OP with CPU inputs on both the CPU and the device
        #      That means that for every op in the model, we execute it with identical data, giving us the accuracy of just that OP in isolation

        # Note that 1) and 2) are not mutually exclusive, we can do both.
        # In the future, we could implement customizable policies for what gets passed on where, for example for "op bypass"

        device_invals = []
        golden_invals = []
        for val in invals:
            if type(val) is GoldenJaxprTracer:
                device_invals.append(val.device_data)
                golden_invals.append(val.golden_data)
            else:
                assert False, "Not a GoldenJaxprTracer"

        with jax.default_device(jax.devices("tt")[0]):
            y = eqn.primitive.bind(*device_invals, **eqn.params)
            y1 = eqn.primitive.bind(*golden_invals, **eqn.params)

        with jax.default_device(jax.devices("cpu")[0]):
            yp = eqn.primitive.bind(*golden_invals, **eqn.params)

        # For metric M, M(y, yp) implements 1)"Parallel" and M(y, y1) implements 2)"CPU prop".
        print(f"Op {eqn.primitive} with params {eqn.params}")
        print(f"Parallel/Whole model", end="\t")
        M(y, yp)
        print(f"Each op CPU prop", end="\t")
        M(y, y1)

        outvals = GoldenJaxprTracer(y, yp)

        #################################################
        # Primitives may return multiple outputs or not
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        # Write the results of the primitive into the environment
        safe_map(write, eqn.outvars, outvals)
    # Read the final result of the Jaxpr from the environment
    return safe_map(read, jaxpr.outvars)


DIMA = 512
DIMB = 128032
DTYPE = jnp.float32

print("Now let's get started for real")

np.random.seed(4747)

up1 = np.random.randn(DIMA, DIMB) * DIMA**-0.5
down1 = np.random.randn(DIMB, DIMA) * DIMB**-0.5
up2 = np.random.randn(DIMA, DIMB) * DIMA**-0.5
down2 = np.random.randn(DIMB, DIMA) * DIMB**-0.5
up3 = np.random.randn(DIMA, DIMB) * DIMA**-0.5
down3 = np.random.randn(DIMB, DIMA) * DIMB**-0.5

x = np.random.randn(1, DIMA)

d_up1 = jnp.array(up1, dtype=DTYPE)
d_down1 = jnp.array(down1, dtype=DTYPE)
d_up2 = jnp.array(up2, dtype=DTYPE)
d_down2 = jnp.array(down2, dtype=DTYPE)
d_up3 = jnp.array(up3, dtype=DTYPE)
d_down3 = jnp.array(down3, dtype=DTYPE)
d_x = jnp.array(x, dtype=DTYPE)


def testcase(x, up1, down1, up2, down2, up3, down3):
    x = x @ up1
    x = x @ down1
    x = x @ up2
    x = x @ down2
    x = x @ up3
    x = x @ down3
    return x


problematic_jaxpr = jax.make_jaxpr(testcase)(
    x, d_up1, d_down1, d_up2, d_down2, d_up3, d_down3
)
print(
    eval_jaxpr_golden(
        problematic_jaxpr.jaxpr,
        problematic_jaxpr.literals,
        x,
        up1,
        down1,
        up2,
        down2,
        up3,
        down3,
    )
)
# Example output
# Op dot_general with params {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': dtype('float32'), 'out_sharding': None}
# Parallel/Whole model    Linf/ATOL       : 1.6614782810211182    PCC             : 0.9263313412666321
# Each op CPU prop        Linf/ATOL       : 0.015605747699737549  PCC             : 0.9999812245368958
# Op dot_general with params {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': dtype('float32'), 'out_sharding': None}
# Parallel/Whole model    Linf/ATOL       : 2.246687412261963     PCC             : 0.9265462160110474
# Each op CPU prop        Linf/ATOL       : 2.2378649711608887    PCC             : 0.9265226721763611
# Op dot_general with params {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': dtype('float32'), 'out_sharding': None}
# Parallel/Whole model    Linf/ATOL       : 2.0891482830047607    PCC             : 0.8588035106658936
# Each op CPU prop        Linf/ATOL       : 0.879144549369812     PCC             : 0.9278896450996399
# Op dot_general with params {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': dtype('float32'), 'out_sharding': None}
# Parallel/Whole model    Linf/ATOL       : 3.1679539680480957    PCC             : 0.8586235046386719
# Each op CPU prop        Linf/ATOL       : 3.159365177154541     PCC             : 0.8586151003837585
# Op dot_general with params {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': dtype('float32'), 'out_sharding': None}
# Parallel/Whole model    Linf/ATOL       : 2.443096160888672     PCC             : 0.8002186417579651
# Each op CPU prop        Linf/ATOL       : 1.1728925704956055    PCC             : 0.8617667555809021
