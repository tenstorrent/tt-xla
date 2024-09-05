import pytest
import jax
import jax.numpy as jnp

from infrastructure import cpu_random_input

def test_num_devices():
  devices = jax.devices()
  assert len(devices) == 1


def test_to_device():
  cpu_array = cpu_random_input((32, 32))
  device = jax.devices()[0]
  tt_array = jax.device_put(cpu_array, device)
  assert tt_array.device.device_kind == "wormhole"


def test_input_on_device():
  def module_add(a, b):
    return a + b
  
  tt_device = jax.devices()[0]
  cpu_param = cpu_random_input((32, 32))
  tt_param = jax.device_put(cpu_param, tt_device)

  graph = jax.jit(module_add)
  cpu_activation_0 = cpu_random_input((32, 32))
  cpu_activation_1 = cpu_random_input((32, 32))
  tt_activation_0 = jax.device_put(cpu_activation_0, tt_device)
  tt_activation_1 = jax.device_put(cpu_activation_1, tt_device)
  
  res0 = graph(tt_activation_0, tt_param)
  res1 = graph(tt_activation_1, tt_param)

  res0_cpu = graph(cpu_activation_0, cpu_param)
  res1_cpu = graph(cpu_activation_1, cpu_param)

  res0 = jax.device_put(res0, res0_cpu.device)
  res1 = jax.device_put(res1, res1_cpu.device)

  assert jnp.allclose(res0, res0_cpu, atol=1e-2)
  assert jnp.allclose(res1, res1_cpu, atol=1e-2)