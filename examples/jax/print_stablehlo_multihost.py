# Modified version of test_multihost.py to print StableHLO code

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np
import os
import jax._src.xla_bridge as xb
import fcntl
import time

# in this example, get multi-process parameters from sys.argv
import sys

if len(sys.argv) < 3:
    print("Usage: python print_stablehlo_multihost.py <proc_id> <num_procs>")
    sys.exit(1)

proc_id = int(sys.argv[1])
num_procs = int(sys.argv[2])

def initialize():
    # Use file-based locking to serialize plugin initialization across processes
    lock_file_path = "/tmp/tt_plugin_init.lock"
    
    # Acquire file lock for initialization - each process will wait its turn
    print(f"Process {proc_id}: Attempting to acquire initialization lock", file=sys.stderr)
    with open(lock_file_path, 'w') as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            print(f"Process {proc_id}: Acquired initialization lock", file=sys.stderr)
            
            path = os.path.join(os.path.dirname(__file__), "../../build/src/tt/pjrt_plugin_tt.so")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

            print(f"Process {proc_id}: Loading tt_pjrt C API plugin", file=sys.stderr)
            xb.discover_pjrt_plugins()

            plugin = xb.register_plugin("tt", priority=500, library_path=path, options=None)
            print(f"Process {proc_id}: TT plugin loaded successfully", file=sys.stderr)
            jax.config.update("jax_platforms", "tt,cpu")
            
            # Add a small delay to ensure complete initialization before next process
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Process {proc_id}: Error during initialization: {e}", file=sys.stderr)
            raise
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print(f"Process {proc_id}: Released initialization lock", file=sys.stderr)


# initialize TT plugin
initialize()

# initialize the distributed system
jax.distributed.initialize('localhost:10000', num_procs, proc_id)

# this example assumes 8 devices total
#print(f"Process {proc_id}: Device count = {jax.device_count()}")
#print(f"Process {proc_id}: All devices = {jax.devices()}")
#print(f"Process {proc_id}: TT devices = {jax.devices('tt')}")

# make a 2D mesh that refers to devices from all processes
mesh = jax.make_mesh((1, 8), ('i', 'j'))

# create some toy data
global_data = np.arange(32).reshape((1, 32))

# make a process- and device-spanning array from our toy data
sharding = NamedSharding(mesh, P('i', 'j'))
global_array = jax.device_put(global_data, sharding)

print(f"Process {proc_id}: Global array shape: {global_array.shape}")

# each process has different shards of the global array
for shard in global_array.addressable_shards:
    print(f"Process {proc_id}: device {shard.device} has local data {shard.data}")

# Define the computation function with explicit input and output shardings
def multihost_computation_fn(x):
    return jax.lax.psum(jnp.sin(x), axis_name='j')

# Apply shard_map to bind axis names, then jit
multihost_computation = jax.jit(
    shard_map(
        multihost_computation_fn,
        mesh=mesh,
        in_specs=P('i', 'j'),
        out_specs=P('i')  # psum reduces along 'j', keeps 'i' dimension
    ),
    out_shardings=NamedSharding(mesh, P('i'))
)

# Print the StableHLO/HLO representation
if proc_id == 0:  # Only print from process 0 to avoid duplication
    print("\n=== StableHLO Code 0 ===")
    lowered = multihost_computation.lower(global_array)
    print("Lowered representation:")
    print(lowered.as_text())

if proc_id == 1:  # Only print from process 0 to avoid duplication
    print("\n=== StableHLO Code 1 ===")
    lowered = multihost_computation.lower(global_array)
    print("Lowered representation:")
    print(lowered.as_text())


print("\n=== Executing Computation ===")

# apply the computation - shard_map handles the mesh context
global_result = multihost_computation(global_array)
print(f'Process {proc_id}: got result: {global_result}')