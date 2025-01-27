import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P, NamedSharding
import jax.numpy as jnp
import os
import jax._src.xla_bridge as xb

def initializePJRT():
  path = os.path.join(os.path.dirname(__file__), "/localdev/ajakovljevic/tt-xla/build/src/tt/pjrt_plugin_tt.so")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}, have you compiled the project?")
  plugin = xb.register_plugin('tt', priority=10, library_path=path, options=None)
  jax.config.update("jax_platforms", "cpu,tt")

def negative_basic(a_block):
    b_block = jnp.negative(a_block)
    stitched_result = jax.lax.psum(b_block, 'y')
    return stitched_result

negative_pmapped = jax.pmap(negative_basic, axis_name='y')
def mesh_negative(act):
    devices = jax.devices('tt')
    mesh_devices = mesh_utils.create_device_mesh((1, 2), devices)
    with Mesh(mesh_devices, ('x', 'y')):
        act_sharded = jax.device_put(act, NamedSharding(Mesh, P('x','y')), may_alias=True)
        result = negative_pmapped(act_sharded)
    return result

if __name__ == "__main__":
    initializePJRT()
    sample_data = jax.numpy.ones((128, 128))
    result = mesh_negative(sample_data)
    print("Result:", result) 