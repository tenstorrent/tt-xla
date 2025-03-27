# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import jax
import jax._src.xla_bridge as xb
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial


def initializePJRT():
    path = os.path.join(
        os.path.dirname(__file__),
        "/localdev/ajakovljevic/tt-xla/build/src/tt/pjrt_plugin_tt.so",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find tt_pjrt C API plugin at {path}, have you compiled the project?"
        )
    plugin = xb.register_plugin("tt", priority=10, library_path=path, options=None)
    jax.config.update("jax_platforms", "cpu,tt")
    # jax.config.update("jax_use_shardy_partitioner", True)


def test_one():
    device_tt = jax.devices("tt")
    print("device:: ", device_tt)
    mesh = jax.make_mesh((1, 2), ("batch", "model"), devices=device_tt)
    batch = jax.numpy.ones((256, 256))
    out_spec = P("batch")

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("batch", "model")),
        out_specs=out_spec,
    )
    def fwd(batch):
        act = jax.numpy.negative(batch)
        act = jax.lax.psum(act, "model")
        return act

    output_sharding = NamedSharding(mesh, out_spec)
    batch_sharded = jax.device_put(batch, NamedSharding(mesh, P("batch", "model")), may_alias=True)
    fwd_jit = jax.jit(fwd, out_shardings=output_sharding)
    output = fwd_jit(batch_sharded).block_until_ready()
    print(output)
    print(output.shape)


initializePJRT()
test_one()
