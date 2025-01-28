import os
import jax
import jax.numpy as jnp
import jax._src.xla_bridge as xb
from jax.sharding import Mesh, PartitionSpec, NamedSharding, SingleDeviceSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from functools import partial
def random_input_tensor(shape, key=42, on_device=False, dtype=jnp.float32):
  device_cpu = jax.devices('cpu')[0]
  with jax.default_device(device_cpu):
    tensor = jax.random.uniform(jax.random.PRNGKey(key), shape=shape, dtype=dtype)
  # Although the random tensor is generated on cpu but it is not committed to
  # cpu; so this tensor can be moved to the device and subsequent code will
  # execute on the device. Placing the generated tensor explicitly to cpu or
  # device to avoid unwanted behavior.
  if on_device:
    tensor = jax.device_put(tensor, jax.devices('tt')[0])
  else:
    tensor = jax.device_put(tensor, device_cpu)
  return tensor

def initializePJRT():
  path = os.path.join(os.path.dirname(__file__), "/localdev/ajakovljevic/tt-xla/build/src/tt/pjrt_plugin_tt.so")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}, have you compiled the project?")
  plugin = xb.register_plugin('tt', priority=10, library_path=path, options=None)
  jax.config.update("jax_platforms", "cpu,tt")
  #jax.config.update("jax_use_shardy_partitioner", True)

def negative_basic(a_block):
    b_block = jnp.negative(a_block)
    #stitched_result = jax.lax.all_gather(b_block, axis_name='x', tiled=True)
    stitched_result = jax.lax.psum(b_block, ('x', 'y'))
    return stitched_result

def mesh_negative(act):
    device_tt = jax.devices('tt')
    print("device:: ", device_tt)
    #devices = mesh_utils.create_device_mesh((1, 1), [jax.devices('tt')[0]])
    devices = mesh_utils.create_device_mesh((1, 2), device_tt)
    #devices = mesh_utils.create_device_mesh((2, 4), device_tt)
    #devices = mesh_utils.create_device_mesh((1, 8), device_tt)
    #devices = mesh_utils.create_device_mesh((4, 2), device_tt)

    mesh = Mesh(devices=devices, axis_names=('x', 'y'))

    in_spec = PartitionSpec('x','y')  # Partition along 'x', even though x=1
    out_spec = PartitionSpec(None, None)

    module_sharded = shard_map(
        negative_basic,
        mesh=mesh,
        in_specs=(in_spec,),  # Partition inputs along 'x'
        out_specs=out_spec   # Partition outputs along 'x'
    )

    output_sharding = NamedSharding(mesh, out_spec)
    with mesh: 
        act_sharded = jax.device_put(act, NamedSharding(mesh, in_spec), may_alias=True)
        graph = jax.jit(module_sharded, out_shardings=output_sharding)
        result = graph(act_sharded)
        print("--------------------------")
        print(act)
        print("--------------------------")
        print("sharding=", result.sharding)
        #jax.numpy.set_printoptions(threshold = 128*128)
        print(result)
        print(result.shape)

initializePJRT()
n_m = 8192
n_k = 784
n_n = 8192*2

batch_size = 1
layer_sizes = [784, 8192, 8192, 8192, 10]
key = jax.random.key(0)
key, *keys = jax.random.split(key, len(layer_sizes))
k1 = keys[0]
act = jax.numpy.ones((256, 256))*2
#act = random_input_tensor((n_m, n_k), on_device=True)
#W = random_input_tensor((n_k, n_n), on_device=True)
mesh_negative(act)