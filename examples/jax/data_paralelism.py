import jax
import tensorflow as tf
import numpy as np

################################################################################
# Step 1: Set up the Dataset with a different data shard per-process (do once)
#         (same as for pure data parallelism)
################################################################################
# Fake example data (replace with your Dataset)
per_process_batches = [np.ones((16, 3)) * i for i in range(100)]
ds = tf.data.Dataset.from_tensor_slices(per_process_batches)

ds = ds.shard(num_shards=jax.process_count(), index=jax.process_index())

################################################################################
# Step 2: Create a jax.Array of per-replica batches from the per-process batch
# produced from the Dataset (repeat every step)
################################################################################
# Grab just the first batch from the Dataset for this example
per_process_batch = ds.as_numpy_iterator().next()

num_model_replicas_per_process = 2 # set according to your parallelism strategy
num_model_replicas_total = num_model_replicas_per_process * jax.process_count()

# Create an example `Mesh` for per-process data parallelism. Make sure all devices
# are grouped by process, and then resize so each row is a model replica.
mesh_devices = np.array([jax.local_devices(process_idx)
                         for process_idx in range(jax.process_count())])
mesh_devices = mesh_devices.reshape(num_model_replicas_total, -1)
# Double check that each replica's devices are on a single process.
for replica_devices in mesh_devices:
  num_processes = len(set(d.process_index for d in replica_devices))
  assert num_processes == 1
mesh = jax.sharding.Mesh(mesh_devices, ["model_replicas", "data_parallelism"])

# Shard the data across model replicas. You don't shard across the
# data_parallelism mesh axis, meaning each per-replica shard will be replicated
# across that axis.
sharding = jax.sharding.NamedSharding(
    mesh, jax.sharding.PartitionSpec("model_replicas"))

global_batch_array = jax.make_array_from_process_local_data(
    sharding, per_process_batch)