import os

# Set environment variables
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

# Import JAX and set config
import jax
NUM_VIRTUAL_DEVICES = 8  
jax.config.update("jax_num_cpu_devices", NUM_VIRTUAL_DEVICES)
cpu_devices = jax.devices("cpu")
axis_name = "X"
num_devices = len(cpu_devices)
device_mesh = jax.make_mesh((num_devices,), (axis_name), devices=cpu_devices)
