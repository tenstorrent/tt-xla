import os
import jax
import jax._src.xla_bridge as xb
from jax.experimental import mesh_utils
from jax.sharding import Mesh

TT_PJRT_PLUGIN_RELPATH = "src/tt/pjrt_plugin_tt.so"

def initialize(use_shardy=True, backend="tt,cpu", priority=10):
    if( "TTXLA_BUILD_DIR" in os.environ and os.path.exists(os.environ["TTXLA_BUILD_DIR"])):
      TTXLA_BUILD_DIR = os.environ["TTXLA_BUILD_DIR"]
    else:
      #assume build directory is at root of tt-xla directory
      TTXLA_BUILD_DIR = os.path.join(os.path.dirname(__file__), "../..")

    plugin_path = os.path.join(TTXLA_BUILD_DIR, TT_PJRT_PLUGIN_RELPATH)
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(
            f"Could not find tt_pjrt C API plugin at {plugin_path}"
        )
    plugin = xb.register_plugin('tt', priority=priority, library_path=plugin_path, options=None)
    jax.config.update("jax_platforms", backend)
    jax.config.update("jax_use_shardy_partitioner", use_shardy)

def open_device(mesh_shape=None, axis=["x", "y"]):
    device_tt = jax.devices('tt')
    device_count = len(device_tt)
    if mesh_shape is None:
      mesh_shape = (1, device_count)
    mesh = Mesh(
      mesh_utils.create_device_mesh(mesh_shape, device_tt),
      axis_names=(axis[0], axis[1]),
    )
    return mesh, device_count, device_tt

def get_device():
  return jax.devices('tt')

def get_num_devices():
  return len(jax.devices('tt'))
