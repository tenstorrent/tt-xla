import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.core import ShapedArray
from jax.interpreters.mlir import register_lowering, ir
import numpy as np
import jax._src.xla_bridge as xb
import os
from jax.tree_util import register_pytree_node_class

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
# Primitive: mark_role
# -----------------------------
mark_role_p = Primitive("mark_role")

def mark_role(x, *, role):
    return mark_role_p.bind(x, role=role)

mark_role_p.def_impl(lambda x, *, role: x)
mark_role_p.def_abstract_eval(lambda x, *, role: x)

def lowering_mark_role(ctx, x, *, role):
    x_type = ir.RankedTensorType(x.type)
    with ir.Location.current:
        op = ir.Operation.create(
            "stablehlo.custom_call",
            results=[x_type],
            operands=[x],
            attributes={
                "call_target_name": ir.StringAttr.get("tt.mark"),
                "tt.role": ir.StringAttr.get(role)
            },
        )
    return [op.result]

register_lowering(mark_role_p, lowering_mark_role)

@register_pytree_node_class
class RoleAnnotated:
    def __init__(self, value, role):
        self.value = value
        self.role = role

    def tree_flatten(self):
        # Return the value to be traced and the auxiliary data (role)
        return (self.value,), self.role

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        value, = children
        return cls(value, aux_data)

    def as_jax_array(self):
        # Inject the annotation here
        return mark_role(jnp.asarray(self.value), role=self.role)

# -----------------------------
# Function wrapper to unwrap RoleAnnotated inputs
# -----------------------------
def unwrap_and_annotate(fn):
    def wrapped(*args):
        new_args = [
            a.as_jax_array() if isinstance(a, RoleAnnotated) else a
            for a in args
        ]
        return fn(*new_args)
    return wrapped

def f(w, x):
    return w * x

# ---- Mark roles without changing function ----
w = RoleAnnotated(jnp.array([2.0, 3.0]), role="param")
x = RoleAnnotated(jnp.array([4.0, 5.0]), role="not param")

compiled = jax.jit(unwrap_and_annotate(f))
print(compiled(w, x))  # will lower with tt.role="weight" and "input"
