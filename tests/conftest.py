import pytest
import os
import jax
import jax._src.xla_bridge as xb

def initialize():
  path = os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

  print("Loading tt_pjrt C API plugin")
  plugin = xb.register_plugin('tt', priority=500, library_path=path, options=None)
  print("Loaded")
  jax.config.update("jax_platforms", "tt")



@pytest.fixture(scope="session", autouse=True)
def setup_session():
    initialize()
    # Code to run once per session, like initializing a database connection
    print("Running session initialization code")

@pytest.fixture(scope="module", autouse=True)
def setup_module():
    # Code to run once per module
    print("Running module initialization code")

@pytest.fixture(scope="function", autouse=True)
def setup_function():
    # Code to run before each test function
    print("Running function initialization code")