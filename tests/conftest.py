import pytest
import os
import jax
import jax._src.xla_bridge as xb

def initialize():
  path = os.path.join(os.path.dirname(__file__), "../install/lib/pjrt_plugin_tt.so")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}, have you compiled the project?")

  plugin = xb.register_plugin('tt', priority=500, library_path=path, options=None)
  jax.config.update("jax_platforms", "tt,cpu")


@pytest.fixture(scope="session", autouse=True)
def setup_session():
    initialize()
