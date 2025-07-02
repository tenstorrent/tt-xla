import jax
import jax.numpy as jnp
from jax.extend import core
from jax.interpreters import mlir
from jax.interpreters.mlir import ir, register_lowering
import jax._src.xla_bridge as xb
import os

def register_pjrt_plugin():
    """Registers TT PJRT plugin."""

    plugin_path = os.path.join(
        os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so"
    )
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"Could not find TT PJRT plugin at {plugin_path}")

    xb.register_plugin("tt", library_path=plugin_path)
    jax.config.update("jax_platforms", "tt,cpu")

register_pjrt_plugin()

# -----------------------------
# Primitive: mark_weight
# -----------------------------
mark_weight_p = core.Primitive("mark_weight")

def mark_weight(x):
    return mark_weight_p.bind(x)

mark_weight_p.def_impl(lambda x: x)
mark_weight_p.def_abstract_eval(lambda x: x)

def lowering_mark_weight(ctx, x):
    x_type = ir.RankedTensorType(x.type)
    with ir.Location.current:
        op = ir.Operation.create(
            "stablehlo.custom_call",
            results=[x_type],
            operands=[x],
            attributes={
                "call_target_name": ir.StringAttr.get("tt.mark"),
                "tt.role": ir.StringAttr.get("weight")
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
                "tt.role": ir.StringAttr.get("input")
                },
        )
    return [op.result]

register_lowering(mark_input_p, lowering_mark_input)

# -----------------------------
# Example JAX function
# -----------------------------
def f(w, x):
    w = mark_weight(w)
    x = mark_input(x)
    return w * x

# -----------------------------
# Test and inspect StableHLO
# -----------------------------
w = jnp.array([2.0, 3.0])
x = jnp.array([4.0, 5.0])

compiled = jax.jit(f)
print(compiled(w, x))
