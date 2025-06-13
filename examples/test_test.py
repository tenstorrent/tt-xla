import jax
import jax.numpy as jnp
import os
import jax._src.xla_bridge as xb
from jax._src.lib import xla_extension as xe
from jax.experimental import serialize_executable
from jax.experimental.serialize_executable import _JaxPjrtUnpickler
import io

def register_pjrt_plugin():
    """Registers TT PJRT plugin."""

    plugin_path = os.path.join(
        os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so"
    )
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"Could not find TT PJRT plugin at {plugin_path}")

    xb.register_plugin("tt", library_path=plugin_path)
    jax.config.update("jax_platforms", "tt,cpu")


@jax.jit
def my_func(x):
    return x ** 2 + 1

register_pjrt_plugin()

# Call once to trigger compilation
x = jnp.array([1.0, 2.0, 3.0])


# Grab the compiled executable from the JIT function
compiled = my_func.lower(x).compile()

# Working example
# runtime_exec = compiled.runtime_executable()
# client = xe.get_c_api_client('tt')
# serialized = client.serialize_executable(runtime_exec)

# # Fancy Example
# Serialize the compiled executable
payload, in_tree, out_tree = serialize_executable.serialize(compiled)

# Deserialize the payload into a callable
exec_fn = serialize_executable.deserialize_and_load(payload, in_tree, out_tree)

# Prepare inputs that match the original input structure
x2 = jnp.array([1.0, 2.0, 3.0])  # Must match dtype/shape of original x

# Call the deserialized executable
result = exec_fn(x2)

print("Result:", result)
# with open("compiled_executable.ttnn", "wb") as f:
#     f.write(serialized)

# print(f"Serialized binary written to compiled_executable.ttnn ({len(serialized)} bytes)")