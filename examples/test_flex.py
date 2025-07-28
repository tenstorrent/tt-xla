# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
from jax._src.lib import xla_extension as xe
import jax._src.xla_bridge as xb
import os
import sys
from jax.extend import core
from jax.interpreters import mlir
from jax.interpreters.mlir import ir, register_lowering


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

# -----------------------------
# Primitive: mark_weight
# -----------------------------
mark_weight_p = core.Primitive("mark_weight")


def mark_weight(x):
    return mark_weight_p.bind(x)


mark_weight_p.def_impl(lambda x: x)
mark_weight_p.def_abstract_eval(lambda x: x)


def lowering_mark_weight(ctx, x):
    """MLIR lowering for mark_weight primitive."""
    x_type = ir.RankedTensorType(x.type)
    with ir.Location.current:
        # Get the current module from the insertion point
        current_op = ir.InsertionPoint.current.block.owner
        print("Current module1:", current_op)
        while current_op and current_op.name != "builtin.module":
            print("Current module3:", current_op)
            current_op = current_op.parent
        print("Current module2:", current_op)
        if current_op:
            tt_mark_defined_attr = "tt.mark_function_defined"

            if tt_mark_defined_attr not in current_op.attributes:
                print("ADDING STUFF HEREEE")
                # Define tt.mark function once per MLIR module as a nop
                func_type = ir.FunctionType.get([x_type], [x_type])

                # Insert function at module level
                with ir.InsertionPoint.at_block_begin(current_op.regions[0].blocks[0]):
                    func_op = ir.Operation.create(
                        "func.func",
                        attributes={
                            "sym_name": ir.StringAttr.get("tt.mark"),
                            "function_type": ir.TypeAttr.get(func_type),
                            "sym_visibility": ir.StringAttr.get("private"),
                        },
                        regions=1,
                    )

                    # Add function body that returns input unchanged
                    entry_block = func_op.regions[0].blocks.append(x_type)
                    with ir.InsertionPoint(entry_block):
                        ir.Operation.create(
                            "func.return", operands=[entry_block.arguments[0]]
                        )

                # Mark that tt.mark function has been defined in this module
                current_op.attributes[tt_mark_defined_attr] = ir.BoolAttr.get(True)

        # Create the custom call to tt.mark
        op = ir.Operation.create(
            "stablehlo.custom_call",
            results=[x_type],
            operands=[x],
            attributes={
                "call_target_name": ir.StringAttr.get("tt.mark"),
                "tt.role": ir.StringAttr.get("weight"),
            },
        )
    return [op.result]


register_lowering(mark_weight_p, lowering_mark_weight)

# -----------------------------
# Primitive: mark_input
# -----------------------------
mark_input_p = core.Primitive("mark_input")


def mark_input(x):
    return mark_input_p.bind(x)


mark_input_p.def_impl(lambda x: x)
mark_input_p.def_abstract_eval(lambda x: x)


def lowering_mark_input(ctx, x):
    x_type = ir.RankedTensorType(x.type)
    with ir.Location.current:
        op = ir.Operation.create(
            "stablehlo.custom_call",
            results=[x_type],
            operands=[x],
            attributes={
                "call_target_name": ir.StringAttr.get("tt.mark"),
                "tt.role": ir.StringAttr.get("input"),
            },
        )
    return [op.result]


register_lowering(mark_input_p, lowering_mark_input)

# Define a simple model
class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=4)(x)  # Dense layer has weights


# Create model instance
model = SimpleModel()

# Initialize parameters with dummy input
dummy_input = jnp.ones((1, 3))
# Manual parameter initialization instead of random
params = freeze(
    {
        "params": {
            "Dense_0": {
                "kernel": jnp.ones((3, 4)),  # Weight matrix initialized to ones
                "bias": jnp.zeros((4,)),  # Bias vector initialized to zeros
            }
        }
    }
)

# Store original apply method
original_apply = nn.Module.apply

# Create JIT-compiled function that marks weights
def run_model(self, variables, *args, **kwargs):
    # Apply mark_weight to all leaf tensors in variables
    marked_variables = jax.tree.map(mark_weight, variables)
    return original_apply(self, marked_variables, *args, **kwargs)


run_model = jax.jit(run_model, static_argnums=0)

# Monkeypatch the apply method
nn.Module.apply = run_model

# Run the model using the monkeypatched apply function
real_input = jnp.array([[2.0, -1.0, 0.5]])
output = model.apply(params, real_input)

print("Output:", output)
