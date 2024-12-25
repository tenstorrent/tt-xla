# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@contextmanager
def run_on_cpu():
    devices = jax.local_devices(backend="cpu")
    assert len(devices) > 0
    cpu = devices[0]

    with jax.default_device(cpu):
        yield


# TODO(issue #80): Make this creation more explicit regarding the originating devices.
def random_input_tensor(shape, key=42, on_device=False, dtype=jnp.float32):
    device_cpu = jax.devices("cpu")[0]
    with jax.default_device(device_cpu):
        tensor = jax.numpy.ones(shape=shape, dtype=dtype)

    # Although the random tensor is generated on cpu but it is not committed to
    # cpu; so this tensor can be moved to the device and subsequent code will
    # execute on the device. Placing the generated tensor explicitly to cpu or
    # device to avoid unwanted behavior.
    if on_device:
        tensor = jax.device_put(tensor, jax.devices()[0])
    else:
        tensor = jax.device_put(tensor, device_cpu)

    return tensor


def compare_tensor_to_golden(
    tensor, golden, required_pcc=0.99, required_atol=1e-2, assert_on_error=True
):
    ret = True

    # TODO (issue #81): Remove these reshapes once the PJRT can handle scalars.
    if tensor.ndim == 0:
        tensor = tensor.reshape((1,))
    if golden.ndim == 0:
        with run_on_cpu():
            golden = golden.reshape((1,))

    if tensor.device != golden.device:
        tensor = jax.device_put(tensor, golden.device)

    ret = ret and tensor.shape == golden.shape
    if assert_on_error:
        assert ret, "Shapes do not match"

    if not tensor.flatten().size == 1:  # pcc invalid for scalar values
        pcc = jnp.min(jnp.corrcoef(tensor.flatten(), golden.flatten()))
        ret = ret and (
            pcc >= required_pcc or (tensor.flatten() == golden.flatten()).all()
        )
        if assert_on_error:
            assert ret, f"PCC is {pcc} which is less than {required_pcc}"

    atol = jnp.max(jnp.abs(tensor - golden))
    ret = ret and atol <= required_atol
    if assert_on_error:
        assert ret, f"ATOL is {atol} which is greater than {required_atol}"

    return ret


def verify_module(
    module,
    input_shapes,
    key=42,
    required_pcc=0.99,
    required_atol=1e-2,
    dtype=jnp.float32,
):
    cpu_inputs = [
        random_input_tensor(input_shapes[i], key + i, dtype=dtype)
        for i in range(len(input_shapes))
    ]

    #device_tt = jax.devices('cpu')
    device_tt = jax.devices('tt')
    #print("device_tt:", device_tt)
    devices = device_tt[0]  # 1D list of devices
    #devices = device_tt
    print("device:", devices)
    mesh = Mesh(devices=(devices,), axis_names=('x',))  # Match dimensionality
    print("Mesh devices:", mesh.devices)
    print("Mesh axis names:", mesh.axis_names)
    # PartitionSpec for axis 'x' (size=1)
    in_spec = PartitionSpec('x')  # Partition along 'x', even though x=1
    out_spec = PartitionSpec('x')
    module_sharded = shard_map(
        module,
        mesh=mesh,
        in_specs=(in_spec,),  # Partition inputs along 'x'
        out_specs=out_spec   # Partition outputs along 'x'
    )
    sharding = NamedSharding(mesh, in_spec)
    #tt_inputs = [jax.device_put(cpu_input, devices) for cpu_input in cpu_inputs]
    tt_inputs = [jax.device_put(cpu_input, sharding) for cpu_input in cpu_inputs]
    # Check data placement
    print("help_me=", sharding.num_devices)
    print("Input devices:", [inp.device for inp in tt_inputs])

    # Compile and execute
    graph = jax.jit(module_sharded)
    cpu_graph = jax.jit(module)
    # lowered_single = graph.lower(*tt_inputs)
    # print("Lowered HLO:")
    # print(lowered_single.as_text())

    # Execute the graph
    with mesh: 
        res = graph(*tt_inputs)
    with run_on_cpu():
        res_cpu = cpu_graph(*cpu_inputs)
    compare_tensor_to_golden(res, res_cpu, required_pcc, required_atol)
