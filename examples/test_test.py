# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import os
import jax._src.xla_bridge as xb
from jax._src.lib import xla_extension as xe
from jax.experimental import serialize_executable
from jax.experimental.serialize_executable import _JaxPjrtUnpickler
import io
import pickle


def register_pjrt_plugin():
    """Registers TT PJRT plugin."""

    plugin_path = os.path.join(
        os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so"
    )
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"Could not find TT PJRT plugin at {plugin_path}")

    xb.register_plugin("tt", library_path=plugin_path)
    jax.config.update("jax_platforms", "tt,cpu")


def persistent_load(pid):
    # We're only interested in the 'XlaSerializedExecutable', which comes first.
    # pid is a tuple like: ('xla_serialized_executable', XlaSerializedExecutable(...))
    if (len(pid) < 2):
        return pid[0]
    print("pid_size=", len(pid))
    print("pid=", pid[0])
    print("pid[1]=", pid[1])
    return pid[1]

@jax.jit
def my_func(x):
    return x**2 + 1


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

# Extract the binary from the payload using the generic unpickler
payload_io = io.BytesIO(payload)
unpickler = pickle.Unpickler(payload_io)
unpickler.persistent_load = persistent_load
unloaded_executable, _, _ = unpickler.load()
flatbuffer_binary = unloaded_executable

# Deserialize the payload into a callable
#exec_fn = serialize_executable.deserialize_and_load(payload, in_tree, out_tree)

# Prepare inputs that match the original input structure
#x2 = jnp.array([1.0, 2.0, 3.0])  # Must match dtype/shape of original x

# Call the deserialized executable
#result = exec_fn(x2)
print(dir(unloaded_executable))
#print("Result:", result)
with open("compiled_executable_2.ttnn", "wb") as f:
    f.write(flatbuffer_binary.xla_executable)

# print(f"Serialized binary written to compiled_executable.ttnn ({len(serialized)} bytes)")
