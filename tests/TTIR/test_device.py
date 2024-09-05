import pytest
import jax
import jax.numpy as jnp

from infrastructure import cpu_random_input

def test_num_devices():
  devices = jax.devices()
  assert len(devices) == 1


def test_to_device():
  cpu_array = cpu_random_input((2, 2), 42)
  device = jax.devices()[0]
  tt_array = jax.device_put(cpu_array, device)
  assert tt_array.device.device_kind == "wormhole"