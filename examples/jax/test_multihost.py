# call this file toy.py, to be run in each process simultaneously

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

# in this example, get multi-process parameters from sys.argv
import sys
proc_id = int(sys.argv[1])
num_procs = int(sys.argv[2])

# initialize the distributed system
jax.distributed.initialize('localhost:10000', num_procs, proc_id)

# this example assumes 8 devices total
assert jax.device_count() == 8

# make a 2D mesh that refers to devices from all processes
mesh = jax.make_mesh((4, 2), ('i', 'j'))

# create some toy data
global_data = np.arange(32).reshape((4, 8))

# make a process- and device-spanning array from our toy data
sharding = NamedSharding(mesh, P('i', 'j'))
global_array = jax.device_put(global_data, sharding)
assert global_array.shape == global_data.shape

# each process has different shards of the global array
for shard in global_array.addressable_shards:
  print(f"device {shard.device} has local data {shard.data}")

# apply a simple computation, automatically partitioned
global_result = jnp.sum(jnp.sin(global_array))
print(f'process={proc_id} got result: {global_result}')